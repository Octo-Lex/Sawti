"""Rechunking seam for the fallback escalation path (spec §3.6, stage 2).

A Rechunker splits a rejected chunk into tighter sub-chunks so the engine
can retry shorter spans. FixedSplitRechunker divides evenly by duration;
smarter implementations (micro-pause-aware) slot in behind the same
Protocol without touching the handler.

Timeline semantics: sub-chunk timestamps are derived PROPORTIONALLY from
the parent's declared span — start + (sample_offset / total_samples) *
span — never from sample-rate arithmetic. This keeps the timeline coherent
even while the segmenter's audio-duration/span mismatch exists (repaired
in the segmentation commit): first sub starts at parent start, last sub
ends at parent end, adjacent boundaries are contiguous, and the
concatenated sub-audio exactly reconstructs the parent audio.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from sawti.types import AudioChunk


class Rechunker(Protocol):
    def rechunk(self, chunk: AudioChunk) -> list[AudioChunk]: ...


class FixedSplitRechunker:
    """Splits a chunk into contiguous sub-chunks, each APPROXIMATELY
    bounded by max_sub_duration_s on the declared timeline (integer
    sample boundaries on awkward lengths can exceed the bound by a tiny
    rounding epsilon; the invariant tests tolerate this).

    Sub-chunk ids are ``{parent_id}.r{i}``; audio is sliced by samples
    (contiguous, no overlap — overlap is a segmenter concern, not a
    fallback concern); timestamps subdivide the parent span proportionally.
    """

    def __init__(self, max_sub_duration_s: float = 3.0) -> None:
        self.max_sub_duration_s = max_sub_duration_s

    def with_tighter(self, factor: float = 2.0) -> "FixedSplitRechunker":
        """Returns a tighter rechunker (max_sub_duration_s / factor, floored
        at 0.25s) for multi-round rechunk escalation. The FallbackHandler
        calls this between rounds when more rounds remain."""
        return FixedSplitRechunker(
            max_sub_duration_s=max(0.25, self.max_sub_duration_s / factor)
        )

    def rechunk(self, chunk: AudioChunk) -> list[AudioChunk]:
        total = len(chunk.audio)
        if total == 0:
            return []
        span = chunk.end_time - chunk.start_time
        n = max(1, int(np.ceil(span / max(self.max_sub_duration_s, 1e-9))))
        out: list[AudioChunk] = []
        for i in range(n):
            a = i * total // n
            b = (i + 1) * total // n
            start = chunk.start_time + (a / total) * span
            end = chunk.start_time + (b / total) * span
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
