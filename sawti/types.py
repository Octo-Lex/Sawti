"""Core data types shared across all pipeline components (spec §5.1)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioChunk:
    """A segment of audio emitted by the segmentation layer."""

    id: str
    audio: np.ndarray  # float32 mono PCM
    sample_rate: int  # 16000
    start_time: float  # seconds from session start
    end_time: float
    overlap_from_prev_s: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time


@dataclass
class EngineResult:
    """Output of the S2TT engine for one chunk."""

    chunk_id: str
    raw_text: str
    confidence: float  # 0..1
    source_lang_guess: str | None
    timing_ms: dict
    target_lang: str


@dataclass
class GateDecision:
    """The quality gate's verdict on an EngineResult.

    start_time/end_time echo the source AudioChunk's timing so the
    postprocessor can emit correctly timestamped OutputSegments without
    needing direct access to the chunk.
    """

    chunk_id: str
    accepted: bool
    result: EngineResult
    checks: dict
    start_time: float = 0.0
    end_time: float = 0.0
    fallback_path: str | None = None  # None | "retry" | "rechunk" | "asr_mt"
    low_confidence: bool = False
    needs_retry: bool = False
    log: list[dict] = field(default_factory=list)


@dataclass
class OutputSegment:
    """Emitted unit: timestamp-aligned target-language text."""

    chunk_id: str
    text: str
    start_time: float
    end_time: float
    low_confidence: bool = False
