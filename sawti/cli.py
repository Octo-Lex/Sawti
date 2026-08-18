"""Typer CLI: `sawti transcribe` and `sawti eval`.

`transcribe` supports --engine stub (default, hermetic) | m4t (real SeamlessM4T).
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


if __name__ == "__main__":
    app()
