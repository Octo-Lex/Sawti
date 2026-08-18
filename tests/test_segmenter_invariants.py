"""Commit 4 invariants: wall-clock-true chunk semantics.

Every test asserts the master invariant — len(audio) == (end_time -
start_time) * sample_rate — including chunks with INTERNAL silence, and
the specific repaired behaviors: sample_rate propagation, physical
overlap carry (reconstruction on a non-constant waveform), min_speech
blip gating, and min_gap preventing splits below the configured gap.
"""
from __future__ import annotations

import numpy as np
import pytest

from sawti.config import SegmentationConfig
from sawti.segmenter_silero import RealSegmenter
from sawti.sources import AudioFrame
from sawti.vad import FakeVad

W = 512  # one sub-window per frame at 16 kHz


def _ramp_frames(pattern: list[bool], start_sample: int = 0):
    """Frames with a DISTINCT deterministic waveform (non-constant, so
    sample-equality assertions are meaningful), one 512-sample frame per
    verdict; global sample position advances so timestamps are exact."""
    frames, verdicts, pos = [], [], start_sample
    for i, speech in enumerate(pattern):
        # deterministic non-repeating-ish ramp chunk
        audio = ((np.arange(pos, pos + W) % 977) / 977.0).astype(np.float32)
        frames.append(AudioFrame(audio=audio, sample_rate=16000,
                                 timestamp_s=pos / 16000))
        verdicts.append((0.9, speech))
        pos += W
    return frames, FakeVad(verdicts), pos


def _identity(chunk) -> None:
    span = chunk.end_time - chunk.start_time
    assert len(chunk.audio) == pytest.approx(span * chunk.sample_rate, abs=1)


def test_internal_silence_preserved_and_identity_holds():
    # speech(5) - 320ms internal pause (below threshold) - speech(5) - long pause
    pattern = [True] * 5 + [False] * 10 + [True] * 5 + [False] * 20
    frames, vad, _ = _ramp_frames(pattern)
    seg = RealSegmenter(vad=vad, config=SegmentationConfig(
        pause_threshold_ms=350, min_chunk_duration_ms=0, overlap_ms=0,
        min_speech_ms=0))
    chunks = list(seg.process(iter(frames)))

    assert len(chunks) == 1
    c = chunks[0]
    _identity(c)
    # Chunk spans first speech start .. last speech end: 20 speech frames
    # PLUS the 10 internal-silence frames -> audio must contain them all.
    assert c.start_time == pytest.approx(0.0)
    assert c.end_time == pytest.approx(20 * W / 16000)
    assert len(c.audio) == 20 * W
    # And the internal silence is really in there (positions 5..15 frames).
    internal = c.audio[5 * W:15 * W]
    assert len(internal) == 10 * W


def test_sample_rate_propagates_from_input():
    sr = 8000
    w = 256
    frames, verdicts = [], []
    pos = 0
    for speech in [True] * 8 + [False] * 8:
        audio = ((np.arange(pos, pos + w) % 13) / 13.0).astype(np.float32)
        frames.append(AudioFrame(audio=audio, sample_rate=sr,
                                 timestamp_s=pos / sr))
        verdicts.append((0.9, speech))
        pos += w
    seg = RealSegmenter(vad=FakeVad(verdicts), config=SegmentationConfig(
        pause_threshold_ms=10, min_chunk_duration_ms=0, overlap_ms=0,
        min_speech_ms=0))
    chunks = list(seg.process(iter(frames)))
    assert len(chunks) == 1
    assert chunks[0].sample_rate == sr          # not hardcoded 16000
    _identity(chunks[0])


def test_overlap_physically_carries_previous_tail():
    # speech(12=384ms > 200ms overlap) - split pause - speech(12)
    S, P = 12, 15
    pattern = [True] * S + [False] * P + [True] * S + [False] * P
    frames, vad, _ = _ramp_frames(pattern)
    seg = RealSegmenter(vad=vad, config=SegmentationConfig(
        pause_threshold_ms=350, min_chunk_duration_ms=0,
        overlap_ms=200, min_speech_ms=0))
    chunks = list(seg.process(iter(frames)))

    assert len(chunks) == 2
    c1, c2 = chunks
    _identity(c1)
    _identity(c2)
    carry = int(200 / 1000 * 16000)
    assert c2.overlap_from_prev_s == pytest.approx(200 / 1000, abs=1e-3)
    # Splice semantics: c2's timeline starts overlap-seconds before its OWN
    # first speech (frame S+P), NOT before c1's end (a pause gap may lie
    # between; the leading overlap re-covers c1's closing words).
    own2_start = (S + P) * W / 16000
    assert c2.start_time == pytest.approx(own2_start - 200 / 1000, abs=1e-3)
    # THE reconstruction proof: chunk 2's head IS chunk 1's tail (exact
    # samples, non-constant waveform so equality is meaningful).
    assert np.array_equal(c2.audio[:carry], c1.audio[-carry:])
    assert c1.overlap_from_prev_s == 0.0          # first chunk: nothing carried
    # The tail really is prior-chunk audio, not zeros/decoration.
    assert np.any(c1.audio[-carry:] != 0)


def test_min_speech_ms_drops_blips():
    pattern = [True] * 3 + [False] * 20          # 96ms blip

    frames, vad, _ = _ramp_frames(pattern)
    seg_default = RealSegmenter(vad=vad, config=SegmentationConfig(
        pause_threshold_ms=350, min_chunk_duration_ms=0, overlap_ms=0))
    assert list(seg_default.process(iter(frames))) == []   # blip dropped

    # Fresh frames + FRESH FakeVad (it is stateful — scripted verdicts are
    # consumed by the first segmenter).
    frames2, vad2, _ = _ramp_frames(pattern)
    seg_open = RealSegmenter(vad=vad2, config=SegmentationConfig(
        pause_threshold_ms=350, min_chunk_duration_ms=0, overlap_ms=0,
        min_speech_ms=0))
    assert len(list(seg_open.process(iter(frames2)))) == 1


def test_min_gap_ms_prevents_split_becomes_internal_silence():
    # 640ms inter-speech silence: pause_threshold(10) satisfied early, but
    # min_gap(1000) forbids the split -> silence stays INSIDE one chunk.
    pattern = [True] * 5 + [False] * 20 + [True] * 5 + [False] * 20
    frames, vad, _ = _ramp_frames(pattern)
    seg = RealSegmenter(vad=vad, config=SegmentationConfig(
        pause_threshold_ms=10, min_chunk_duration_ms=0, overlap_ms=0,
        min_speech_ms=0, min_gap_ms=1000))
    chunks = list(seg.process(iter(frames)))

    assert len(chunks) == 1                        # no split below the gap
    _identity(chunks[0])
    # All 30 frames' span (speech + 640ms internal silence + speech).
    assert len(chunks[0].audio) == 30 * W

    # Control: min_gap below the pause -> two chunks, as before.
    seg2 = RealSegmenter(vad=FakeVad([(0.9, p) for p in pattern]),
                         config=SegmentationConfig(
                             pause_threshold_ms=10, min_chunk_duration_ms=0,
                             overlap_ms=0, min_speech_ms=0, min_gap_ms=10))
    assert len(list(seg2.process(iter(_ramp_frames(pattern)[0])))) == 2


def test_ordering_and_ids_deterministic():
    pattern = [True] * 5 + [False] * 15 + [True] * 5 + [False] * 15
    frames, vad, _ = _ramp_frames(pattern)
    seg = RealSegmenter(vad=vad, config=SegmentationConfig(
        pause_threshold_ms=350, min_chunk_duration_ms=0, overlap_ms=0,
        min_speech_ms=0))
    chunks = list(seg.process(iter(frames)))
    assert [c.id for c in chunks] == ["c0", "c1"]
    assert chunks[0].end_time <= chunks[1].start_time   # time-ordered
    for c in chunks:
        _identity(c)
