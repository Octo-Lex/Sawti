"""Typer CLI: `sawti transcribe`, `sawti eval`, `sawti listen`.

`transcribe` supports --engine stub (default, hermetic) | m4t (real SeamlessM4T).
`listen` (M2) is the live-microphone edge: MicSource -> the SAME production
graph as `transcribe --engine m4t`; transcript to stdout, capture stats to
stderr, Ctrl+C = clean session end.
"""
from __future__ import annotations

from pathlib import Path

import typer

from sawti.config import SawtiConfig, load_config
from sawti.engine import EngineManager, StubEngine
from sawti.logging_setup import configure_logging
from sawti.pipeline import Pipeline
from sawti.postprocess import StubPostProcessor
from sawti.postprocess_real import RealPostProcessor
from sawti.quality_gate import StubQualityGate
from sawti.quality_gate_balanced import BalancedQualityGate
from sawti.segmenter import StubSegmenter
from sawti.sources import StubAudioSource

app = typer.Typer(add_completion=False, help="Sawti multilingual STT-translation.")


def _stub_pipeline(on_decision=None) -> Pipeline:
    return Pipeline(
        segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
        engine=EngineManager(engine=StubEngine("hello", 0.9)),
        gate=StubQualityGate(),
        postprocessor=StubPostProcessor(),
        on_decision=on_decision,
    )


def _real_pipeline(config: SawtiConfig, on_decision=None) -> Pipeline:
    """The ONE production graph (sawti/build.py): full recovery stack with
    conservative retry, rechunker, and the real ASR+MT provider."""
    from sawti.build import build_real_pipeline

    return build_real_pipeline(config, on_decision=on_decision)


@app.command()
def transcribe(
    file: Path = typer.Argument(None, help="Audio file to transcribe (omit for stub demo)"),
    target: str = typer.Option("eng", help="Target language: eng|ara|fra"),
    engine: str = typer.Option("stub", help="stub | m4t"),
    config_path: Path = typer.Option(Path("config/default.yaml"), help="Config YAML"),
) -> None:
    """Transcribe audio to the target language."""
    from sawti.env import load_env

    load_env()  # entry edge: fills absent vars only — OS environment wins
    if engine not in ("stub", "m4t"):
        raise typer.BadParameter(
            f"unsupported engine {engine!r} — expected 'stub' or 'm4t'")
    configure_logging()
    config = load_config(config_path) if config_path.exists() else SawtiConfig()
    if engine == "m4t" and file is not None:
        from sawti.audio_io import FileSource
        pipe = _real_pipeline(config)
        src = FileSource(file, frame_samples=16000)
    else:
        pipe = _stub_pipeline()
        src = StubAudioSource(n_frames=4, samples_per_frame=16000)
    for seg in pipe.run(src, target_lang=target):
        typer.echo(f"[{seg.start_time:.2f}-{seg.end_time:.2f}] {seg.text}")


@app.command()
def eval(
    eval_set: Path = typer.Argument(..., help="Eval set directory"),
    target: str = typer.Option("eng", help="Target language: eng|ara|fra"),
    engine: str = typer.Option(
        "stub", help="stub: real execution of stub components (no models); "
                     "real: the full production graph (loads M4T/Whisper)"),
    config_path: Path = typer.Option(Path("config/default.yaml"), help="Config YAML"),
) -> None:
    """Run the evaluation harness through a real pipeline."""
    from sawti.env import load_env

    load_env()  # entry edge: fills absent vars only — OS environment wins
    if engine not in ("stub", "real"):
        raise typer.BadParameter(
            f"unsupported engine {engine!r} — expected 'stub' or 'real'")
    from eval.harness import run_eval
    from eval.transcribers import make_pipeline_transcriber

    if engine == "real":
        config = load_config(config_path) if config_path.exists() else SawtiConfig()
        factory = lambda on_decision=None: _real_pipeline(  # noqa: E731
            config, on_decision=on_decision)
    else:
        # Explicit stub mode: a real execution of stub components — the
        # same seam, no models. Not a stubbed hypothesis.
        factory = lambda on_decision=None: _stub_pipeline(  # noqa: E731
            on_decision=on_decision)
    transcriber = make_pipeline_transcriber(factory, target)
    report = run_eval(eval_set, target_lang=target, transcriber=transcriber)
    typer.echo(f"Wrote report: {report}")


@app.command()
def listen(
    target: str = typer.Option("eng", help="Target language: eng|ara|fra"),
    device: str = typer.Option(
        None, help="Microphone: numeric PortAudio device id or name "
                   "substring (omit for the backend default)"),
    list_devices: bool = typer.Option(
        False, "--list-devices",
        help="List capture devices and exit (no models loaded)"),
    config_path: Path = typer.Option(Path("config/default.yaml"), help="Config YAML"),
) -> None:
    """Live microphone -> timestamped target-language text (M2).

    Same production graph as `transcribe --engine m4t`; only the audio
    source changes (MicSource instead of FileSource). Ctrl+C ends the
    session cleanly. Transcript goes to stdout; capture stats to stderr.
    """
    from sawti.env import load_env

    load_env()  # entry edge: fills absent vars only — OS environment wins
    if target not in ("eng", "ara", "fra"):
        raise typer.BadParameter(
            f"unsupported target {target!r} — expected eng, ara or fra")
    if list_devices:
        _list_input_devices()  # sounddevice ONLY — never the ML graph
        raise typer.Exit()

    configure_logging()
    config = load_config(config_path) if config_path.exists() else SawtiConfig()
    # Typer delivers --device as text: an all-digit value is a PortAudio
    # device id (int), everything else a name substring. Normalized here
    # so "3" is id 3, not a substring search (reviewer pin, Task 4).
    dev = int(device) if device is not None and device.isdigit() else device

    from sawti.mic_source import MicError, MicSource

    src = MicSource(device=dev)
    try:
        pipe = _real_pipeline(config)
        for seg in pipe.run(src, target_lang=target):
            typer.echo(f"[{seg.start_time:.2f}-{seg.end_time:.2f}] {seg.text}")
    except KeyboardInterrupt:
        pass  # Ctrl+C: normal session end — clean teardown below, no traceback
    except MicError as e:
        typer.echo(f"listen: {e}", err=True)
        raise typer.Exit(code=1)
    finally:
        src.close()
        _print_capture_stats(src)


def _list_input_devices() -> None:
    """--list-devices: sounddevice only. Must never import or build any
    part of the ML graph (plan Task 4)."""
    try:
        import sounddevice as sd
    except Exception as e:
        typer.echo(f"listen: microphone backend unavailable: {e}\n"
                   f"install it with: uv sync --extra mic", err=True)
        raise typer.Exit(code=1)
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            typer.echo(f"{i}: {d.get('name')} "
                       f"({d.get('max_input_channels')} in, "
                       f"default {d.get('default_samplerate')} Hz)")


def _print_capture_stats(src) -> None:
    """Session capture summary — stderr only, transcript stdout stays
    machine-friendly (plan Task 6)."""
    s = src.stats()
    typer.echo(
        "[capture] {captured_seconds}s captured / {emitted_seconds}s "
        "emitted | queue high-water {queue_high_water} "
        "(now {queue_depth}) | input overflows {input_overflow_count} | "
        "queue overflows {queue_overflow_count} | device "
        "{selected_device} @ {sample_rate} Hz x {block_ms} ms".format(**s),
        err=True)


if __name__ == "__main__":
    app()
