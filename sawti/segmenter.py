"""Segmenter protocol + stub (spec §2, §5.2).

M1 replaces StubSegmenter with a Silero-VAD-backed real segmenter; the
protocol and orchestrator are unchanged.
"""
from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np

from sawti.config import SegmentationConfig
from sawti.sources import AudioFrame
from sawti.types import AudioChunk


class Segmenter(Protocol):
    """Consumes audio frames, yields coherent AudioChunks."""

    def process(self, frames: Iterable[AudioFrame]) -> Iterable[AudioChunk]: ...


class StubSegmenter:
    """Groups frames into fixed-size chunks (M0 only).

    Ignores VAD/pause logic entirely — just packs `chunk_frames` frames
    per chunk so the orchestrator has something to consume.
    """

    def __init__(
        self,
        chunk_frames: int = 2,
        sample_rate: int = 16000,
        config: SegmentationConfig | None = None,
    ) -> None:
        self.chunk_frames = chunk_frames
        self.sample_rate = sample_rate
        self._counter = 0

    def process(self, frames: Iterable[AudioFrame]) -> Iterable[AudioChunk]:
        buffer: list[np.ndarray] = []
        start_ts: float | None = None
        last_ts = 0.0
        for frame in frames:
            if start_ts is None:
                start_ts = frame.timestamp_s
            buffer.append(frame.audio)
            last_ts = frame.timestamp_s + len(frame.audio) / self.sample_rate
            if len(buffer) >= self.chunk_frames:
                yield self._emit(buffer, start_ts or 0.0, last_ts)
                buffer = []
                start_ts = None
        if buffer:
            yield self._emit(buffer, start_ts or 0.0, last_ts)

    def _emit(self, buffer: list[np.ndarray], start: float, end: float) -> AudioChunk:
        chunk_id = f"c{self._counter}"
        self._counter += 1
        return AudioChunk(
            id=chunk_id,
            audio=np.concatenate(buffer).astype(np.float32),
            sample_rate=self.sample_rate,
            start_time=start,
            end_time=end,
            overlap_from_prev_s=0.0,
            meta={},
        )
