"""Rechunking seam for the fallback escalation path (spec §3.6, stage 2).

A Rechunker splits a rejected chunk into tighter sub-chunks so the engine
can retry shorter spans. FixedSplitRechunker divides evenly by duration;
smarter implementations (micro-pause-aware) slot in behind the same
Protocol without touching the handler.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from sawti.types import AudioChunk


class Rechunker(Protocol):
    def rechunk(self, chunk: AudioChunk) -> list[AudioChunk]: ...


class FixedSplitRechunker:
    """Splits a chunk into contiguous sub-chunks, each <= max_sub_duration_s.

    Sub-chunk ids are ``{parent_id}.r{i}``; timestamps subdivide the parent
    span; audio is sliced correspondingly (contiguous, no overlap — overlap
    is a segmenter concern, not a fallback concern).
    """

    def __init__(self, max_sub_duration_s: float = 3.0) -> None:
        self.max_sub_duration_s = max_sub_duration_s

    def rechunk(self, chunk: AudioChunk) -> list[AudioChunk]:
        dur = chunk.end_time - chunk.start_time
        n = max(1, int(np.ceil(dur / max(self.max_sub_duration_s, 1e-9))))
        total = len(chunk.audio)
        out: list[AudioChunk] = []
        for i in range(n):
            a = i * total // n
            b = (i + 1) * total // n
            start = chunk.start_time + a / chunk.sample_rate
            end = chunk.start_time + b / chunk.sample_rate
            out.append(AudioChunk(
                id=f"{chunk.id}.r{i}",
                audio=np.ascontiguousarray(chunk.audio[a:b], dtype=np.float32),
                sample_rate=chunk.sample_rate,
                start_time=start,
                end_time=end,
                overlap_from_prev_s=0.0,
                meta={"rechunk_parent": chunk.id},
            ))
        return out
