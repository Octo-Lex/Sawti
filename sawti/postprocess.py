"""Post-processor protocol + stub (spec §4, §5.2).

M1 replaces StubPostProcessor with the 6-step deterministic pipeline
(dedupe/repeat-collapse/normalize/punctuate/emit/log). Protocol unchanged.
"""
from __future__ import annotations

from typing import Iterable, Protocol

from sawti.config import PostprocessConfig
from sawti.types import GateDecision, OutputSegment


class PostProcessor(Protocol):
    def process(
        self, decisions: Iterable[GateDecision], target_lang: str
    ) -> Iterable[OutputSegment]: ...


class StubPostProcessor:
    """Passes result text through unchanged (M0 only)."""

    def __init__(self, config: PostprocessConfig | None = None) -> None:
        self.config = config or PostprocessConfig()

    def process(
        self, decisions: Iterable[GateDecision], target_lang: str
    ) -> Iterable[OutputSegment]:
        for d in decisions:
            yield OutputSegment(
                chunk_id=d.chunk_id,
                text=d.result.raw_text,
                start_time=d.result.timing_ms.get("_start", 0.0) if isinstance(
                    d.result.timing_ms.get("_start"), float
                ) else 0.0,
                end_time=0.0,
                low_confidence=d.low_confidence,
            )
