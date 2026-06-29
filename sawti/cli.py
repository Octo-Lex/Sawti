"""Typer CLI: `sawti transcribe` and `sawti eval` (spec §6.3, §7.6).

M0: both subcommands wire stub components. M1 swaps `transcribe` to the
real FileSource + pipeline, and `eval` to the real harness.
"""
from __future__ import annotations

from pathlib import Path

import typer

from sawti.engine import EngineManager, StubEngine
from sawti.logging_setup import configure_logging
from sawti.pipeline import Pipeline
from sawti.postprocess import StubPostProcessor
from sawti.quality_gate import StubQualityGate
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


@app.command()
def transcribe(
    target: str = typer.Option("eng", help="Target language: eng|ara|fra"),
) -> None:
    """Transcribe audio to the target language (stub in M0)."""
    configure_logging()
    pipe = _stub_pipeline()
    src = StubAudioSource(n_frames=4, samples_per_frame=16000)
    for seg in pipe.run(src, target_lang=target):
        typer.echo(f"[{seg.start_time:.2f}-{seg.end_time:.2f}] {seg.text}")


@app.command()
def eval(
    eval_set: Path = typer.Argument(..., help="Eval set directory"),
    target: str = typer.Option("eng", help="Target language: eng|ara|fra"),
) -> None:
    """Run the evaluation harness (skeleton in M0)."""
    from eval.harness import run_eval

    report = run_eval(eval_set, target_lang=target)
    typer.echo(f"Wrote report: {report}")


if __name__ == "__main__":
    app()
