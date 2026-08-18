"""Pipeline transcribers for the evaluator (Commit 5).

Adapts the REAL execution path (Pipeline over FileSource) into the
evaluator's per-clip seam. The Pipeline.on_decision callback is the
observability instrument: it receives each chunk's FINAL GateDecision
(accepted or exhausted, post-fallback), whose .log is the structured
stage trace. No orchestration is duplicated here — primary/retry/rechunk/
ASR+MT belong to Pipeline + FallbackHandler exclusively.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Transcription:
    """Per-clip result the evaluator consumes."""

    hypothesis: str
    segments: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)      # structured stages
    trace_text: str = ""                                  # human-readable
    low_confidence: bool = False
    fallback_paths: list[str] = field(default_factory=list)


def make_pipeline_transcriber(
    pipeline_factory: Callable[..., "object"],
    target_lang: str,
    frame_samples: int = 16000,
):
    """Build a transcriber from a pipeline factory.

    ``pipeline_factory(on_decision=...) -> Pipeline`` must construct a
    FRESH pipeline per clip (decisions are collected per run). Hermetic
    by construction: nothing here imports or loads models — whatever the
    factory builds is what runs.
    """
    from sawti.audio_io import FileSource

    def transcribe(wav_path: str) -> Transcription:
        from sawti.fallback import format_trace

        decisions: dict[str, object] = {}

        def on_decision(d) -> None:
            decisions[d.chunk_id] = d

        pipe = pipeline_factory(on_decision=on_decision)
        segments = list(
            pipe.run(FileSource(wav_path, frame_samples=frame_samples), target_lang)
        )
        trace: list[dict] = []
        paths: list[str] = []
        trace_texts: list[str] = []
        for s in segments:
            d = decisions.get(s.chunk_id)
            if d is not None:
                # Raw entries preserved verbatim, but every entry leaves
                # with chunk attribution: entries lacking chunk_id (e.g.
                # a plain gate evaluate record on primary-success chunks)
                # get it filled in a COPY — no stage synthesis, no
                # orchestration here.
                for e in d.log:
                    if isinstance(e, dict):
                        e = dict(e)
                        e.setdefault("chunk_id", s.chunk_id)
                        trace.append(e)
                if getattr(d, "fallback_path", None):
                    paths.append(d.fallback_path)
                trace_texts.append(format_trace(d))
        return Transcription(
            hypothesis=" ".join(s.text for s in segments).strip(),
            segments=[
                {
                    "chunk_id": s.chunk_id,
                    "text": s.text,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "low_confidence": s.low_confidence,
                }
                for s in segments
            ],
            trace=trace,
            trace_text="\n\n".join(t for t in trace_texts if t),
            low_confidence=any(s.low_confidence for s in segments),
            fallback_paths=sorted(set(paths)),
        )

    return transcribe
