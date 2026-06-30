"""Typer CLI: `sawti transcribe` and `sawti eval`.

`transcribe` supports --engine stub (default, hermetic) | m4t (real SeamlessM4T).
"""
from __future__ import annotations

from pathlib import Path

import typer

import sawti.env  # noqa: F401  loads .env into os.environ before HF imports
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


def _stub_pipeline() -> Pipeline:
    return Pipeline(
        segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
        engine=EngineManager(engine=StubEngine("hello", 0.9)),
        gate=StubQualityGate(),
        postprocessor=StubPostProcessor(),
    )


def _real_pipeline(config: SawtiConfig) -> Pipeline:
    from sawti.engine_m4t import SeamlessM4TEngine
    from sawti.segmenter_silero import RealSegmenter
    from sawti.vad import SileroVad

    device = config.s2tt.device
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large")
    # SeamlessM4TEngine moves the model to `device` itself.
    engine = SeamlessM4TEngine(processor=processor, model=model, device=device)
    return Pipeline(
        segmenter=RealSegmenter(vad=SileroVad(), config=config.segmentation),
        engine=EngineManager(engine=engine, config=config.s2tt),
        gate=BalancedQualityGate(config=config.quality_gate),
        postprocessor=RealPostProcessor(config=config.postprocess),
    )


@app.command()
def transcribe(
    file: Path = typer.Argument(None, help="Audio file to transcribe (omit for stub demo)"),
    target: str = typer.Option("eng", help="Target language: eng|ara|fra"),
    engine: str = typer.Option("stub", help="stub | m4t"),
    config_path: Path = typer.Option(Path("config/default.yaml"), help="Config YAML"),
) -> None:
    """Transcribe audio to the target language."""
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
) -> None:
    """Run the evaluation harness."""
    from eval.harness import run_eval

    report = run_eval(eval_set, target_lang=target)
    typer.echo(f"Wrote report: {report}")


if __name__ == "__main__":
    app()
