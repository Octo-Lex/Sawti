"""FixedSplitRechunker timeline + audio invariants (Commit 1 correction 2).

The parent deliberately carries a span that does NOT match its audio
duration (2s of samples spanning a declared 4s) — exactly the segmenter
mismatch that exists until the segmentation repair — to prove the
proportional-timestamp semantics hold regardless.
"""
from __future__ import annotations

import numpy as np
import pytest

from sawti.rechunk import FixedSplitRechunker
from sawti.types import AudioChunk


def _parent(n: int = 32001, start: float = 1.0, end: float = 5.0) -> AudioChunk:
    # Odd sample count (uneven split) + span != audio duration.
    return AudioChunk(
        id="c0",
        audio=np.linspace(-1.0, 1.0, n).astype(np.float32),
        sample_rate=16000,
        start_time=start,
        end_time=end,
    )


def test_timeline_invariants_ends_and_contiguity():
    p = _parent()
    subs = FixedSplitRechunker(max_sub_duration_s=1.0).rechunk(p)  # 4s span -> 4
    assert len(subs) == 4
    assert subs[0].start_time == pytest.approx(p.start_time)
    assert subs[-1].end_time == pytest.approx(p.end_time)
    for a, b in zip(subs, subs[1:]):
        assert a.end_time == pytest.approx(b.start_time)
    # Proportional to the SPAN (~1s each), not sample-rate derived. Integer
    # sample boundaries on an odd count make these approximate by ~1e-4.
    for s in subs:
        assert (s.end_time - s.start_time) == pytest.approx(1.0, abs=2e-3)


def test_audio_reconstruction_exact():
    p = _parent()
    subs = FixedSplitRechunker(1.0).rechunk(p)
    rebuilt = np.concatenate([s.audio for s in subs])
    assert rebuilt.shape == p.audio.shape
    assert np.array_equal(rebuilt, p.audio)


def test_ids_meta_and_even_split():
    p = _parent()
    subs = FixedSplitRechunker(2.0).rechunk(p)  # 2 subs over the span
    assert [s.id for s in subs] == ["c0.r0", "c0.r1"]
    assert all(s.meta.get("rechunk_parent") == "c0" for s in subs)
    assert all(s.sample_rate == p.sample_rate for s in subs)


def test_short_chunk_returns_single_subchunk():
    p = _parent(n=1600, start=0.0, end=0.1)
    subs = FixedSplitRechunker(3.0).rechunk(p)
    assert len(subs) == 1
    assert subs[0].id == "c0.r0"
    assert subs[0].start_time == pytest.approx(0.0)
    assert subs[0].end_time == pytest.approx(0.1)
    assert np.array_equal(subs[0].audio, p.audio)


def test_zero_audio_returns_empty():
    p = AudioChunk(id="c0", audio=np.zeros(0, np.float32),
                   sample_rate=16000, start_time=0.0, end_time=1.0)
    assert FixedSplitRechunker(3.0).rechunk(p) == []
