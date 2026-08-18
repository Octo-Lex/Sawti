"""Commit 4 invariants: source-time-true chunk semantics.

Master oracle: every emitted chunk's audio must be the LITERAL source
waveform for [start_time, end_time] — asserted via exact array equality
against the reconstructed source stream. Additional regressions pin:
sample_rate propagation, overlap carry ACROSS the real bridged gap (tail
| actual intervening silence | new speech), max-span enforcement during
retained silence, carry-drop when bridging would exceed max span,
min_speech blip gating, and min_gap preventing splits below the gap.
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
    for speech in enumerate(pattern):
        i, sp = speech
        audio = ((np.arange(pos, pos + W) % 977) / 977.0).astype(np.float32)
        frames.append(AudioFrame(audio=audio, sample_rate=16000,
                                 timestamp_s=pos / 16000))
        verdicts.append((0.9, sp))
        pos += W
    return frames, FakeVad(verdicts), pos


def _source(frames) -> np.ndarray:
    return np.concatenate([f.audio for f in frames])


def _source_equal(chunk, frames) -> None:
    src = _source(frames)
    a = int(round(chunk.start_time * chunk.sample_rate))
    b = int(round(chunk.end_time * chunk.sample_rate))
    assert np.array_equal(chunk.audio, src[a:b]), (
        f"chunk {chunk.id} audio is not the literal source slice "
        f"[{chunk.start_time:.4f}, {chunk.end_time:.4f}]"
    )


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
    _source_equal(c, frames)
    assert c.start_time == pytest.approx(0.0)
    assert c.end_time == pytest.approx(20 * W / 16000)
    # The internal silence is really in there (positions 5..15 frames).
    assert len(c.audio[5 * W:15 * W]) == 10 * W


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
    _source_equal(chunks[0], frames)


def test_overlap_bridges_the_real_gap_source_true():
    # speech(12=384ms) - pause(15=480ms) - speech(12); overlap 200ms.
    S, P = 12, 15
    pattern = [True] * S + [False] * P + [True] * S + [False] * P
    frames, vad, _ = _ramp_frames(pattern)
    seg = RealSegmenter(vad=vad, config=SegmentationConfig(
        pause_threshold_ms=350, min_chunk_duration_ms=0,
        overlap_ms=200, min_speech_ms=0))
    chunks = list(seg.process(iter(frames)))

    assert len(chunks) == 2
    c1, c2 = chunks
    src = _source(frames)
    carry = int(200 / 1000 * 16000)
    prev_end = S * W / 16000

    # start_time is the tail's TRUE source time (not own_start - overlap).
    assert c2.start_time == pytest.approx(prev_end - 200 / 1000, abs=1e-3)
    assert c2.overlap_from_prev_s == pytest.approx(200 / 1000, abs=1e-3)
    assert c2.end_time == pytest.approx((2 * S + P) * W / 16000, abs=1e-3)

    # THE strong oracle: c2's audio is the literal source slice, i.e.
    # previous tail | ACTUAL intervening silence | new speech, in true
    # source order — the waveform makes each region distinguishable.
    _source_equal(c2, frames)
    _source_equal(c1, frames)

    # Explicit decomposition on top of the oracle: head IS c1's tail...
    assert np.array_equal(c2.audio[:carry], c1.audio[-carry:])
    # ...and the samples following the tail are the REAL source-gap
    # samples (the 480ms between prev_end and the new speech start).
    gap_start_i = int(round(prev_end * 16000))
    new_speech_i = int(round(((S + P) * W / 16000) * 16000))
    assert np.array_equal(
        c2.audio[carry:new_speech_i - gap_start_i + carry],
        src[gap_start_i:new_speech_i],
    )
    assert c1.overlap_from_prev_s == 0.0          # first chunk: nothing carried


def test_max_span_enforced_during_retained_silence():
    # max=1s, min_gap huge (no pause-close): 384ms speech then 1920ms
    # retained silence — the chunk MUST close at ~1s span while SILENT,
    # not wait for speech to resume.
    pattern = [True] * 12 + [False] * 60 + [True] * 6 + [False] * 20
    frames, vad, _ = _ramp_frames(pattern)
    seg = RealSegmenter(vad=vad, config=SegmentationConfig(
        pause_threshold_ms=350, min_chunk_duration_ms=0, overlap_ms=0,
        min_speech_ms=0, min_gap_ms=10000, max_chunk_duration_s=1.0))
    chunks = list(seg.process(iter(frames)))

    assert len(chunks) == 2
    c1, c2 = chunks
    # The max-span close fired WHILE SILENT (wall span ~1.02s), so c1
    # emits through last_speech_end (384ms) and the resumed speech opens
    # a FRESH chunk at its own start. Under the old code (speech-branch
    # check only) the chunk would have stayed open through all 1.9s of
    # retained silence and swallowed the resumed speech into c1.
    assert c1.end_time == pytest.approx(12 * W / 16000, abs=1e-6)
    assert len(c1.audio) == 12 * W
    assert c2.start_time == pytest.approx(72 * W / 16000, abs=1e-6)
    assert c2.overlap_from_prev_s == 0.0            # overlap_ms=0 here
    for c in chunks:
        _identity(c)
        _source_equal(c, frames)


def test_long_gap_drops_carry_rather_than_exceeding_max():
    # overlap 200ms but a 3s gap with max=1s: bridging would exceed the
    # span ceiling, so the carry is DROPPED — chunk 2 opens fresh with
    # truthful timestamps and no overlap.
    S, P = 12, 94  # gap = 94*32ms ≈ 3.0s
    pattern = [True] * S + [False] * P + [True] * S + [False] * 15
    frames, vad, _ = _ramp_frames(pattern)
    seg = RealSegmenter(vad=vad, config=SegmentationConfig(
        pause_threshold_ms=350, min_chunk_duration_ms=0,
        overlap_ms=200, min_speech_ms=0, max_chunk_duration_s=1.0))
    chunks = list(seg.process(iter(frames)))

    assert len(chunks) == 2
    c1, c2 = chunks
    assert c2.overlap_from_prev_s == 0.0          # carry dropped
    assert c2.start_time == pytest.approx((S + P) * W / 16000, abs=1e-3)
    _source_equal(c1, frames)
    _source_equal(c2, frames)


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
    _source_equal(chunks[0], frames)
    # All 30 frames' span (speech + 640ms internal silence + speech).
    assert len(chunks[0].audio) == 30 * W

    # Control: min_gap below the pause -> two chunks, as before.
    frames2, _, _ = _ramp_frames(pattern)
    seg2 = RealSegmenter(vad=FakeVad([(0.9, p) for p in pattern]),
                         config=SegmentationConfig(
                             pause_threshold_ms=10, min_chunk_duration_ms=0,
                             overlap_ms=0, min_speech_ms=0, min_gap_ms=10))
    assert len(list(seg2.process(iter(frames2)))) == 2


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
        _source_equal(c, frames)


def test_rejected_blip_does_not_consume_overlap_carry():
    """emitted A -> gap -> blip below min_speech (rejected) -> gap ->
    emitted B. The carry relationship is between EMITTED chunks: B must
    still carry A's tail across the real bridge (which now contains the
    rejected blip's source audio too), subject to the carry budget."""
    sr = 16000
    pattern = ([True] * 12            # A: 384ms speech
               + [False] * 15         # gap 480ms (>= pause: A closes)
               + [True] * 2           # blip: 64ms < min_speech_ms(100)
               + [False] * 15         # gap 480ms
               + [True] * 12           # B: 384ms speech
               + [False] * 15)         # close B
    frames, vad, _ = _ramp_frames(pattern)
    seg = RealSegmenter(vad=vad, config=SegmentationConfig(
        pause_threshold_ms=350, min_chunk_duration_ms=0,
        overlap_ms=200, min_speech_ms=100))
    chunks = list(seg.process(iter(frames)))

    assert len(chunks) == 2
    a, b = chunks
    carry = int(200 / 1000 * sr)
    src = _source(frames)

    assert b.overlap_from_prev_s == pytest.approx(200 / 1000, abs=1e-3)
    # Strong oracle: B's audio is the literal source slice — the bridge
    # contains the real gaps AND the rejected blip's source samples.
    _source_equal(b, frames)
    _source_equal(a, frames)
    assert b.start_time == pytest.approx(a.end_time - 200 / 1000, abs=1e-3)
    # Head IS A's tail, on a non-constant waveform.
    assert np.array_equal(b.audio[:carry], a.audio[-carry:])


def test_rejected_blip_then_budget_overflow_still_drops_carry():
    """Same sequence, but the accumulated bridge (gaps + blip) exceeds the
    max-span carry budget: dropping the carry remains the correct outcome
    — B opens fresh with truthful timestamps and no overlap."""
    # Budget variant: wider gaps (640ms each) so carried(0.2s) + bridge
    # (~1.35s incl. the blip and residues) exceeds max span 1s (int type
    # per the frozen config) -> carry dropped, B opens fresh.
    pattern = ([True] * 12
               + [False] * 20
               + [True] * 2
               + [False] * 20
               + [True] * 12
               + [False] * 15)
    frames, vad, _ = _ramp_frames(pattern)
    seg = RealSegmenter(vad=vad, config=SegmentationConfig(
        pause_threshold_ms=350, min_chunk_duration_ms=0,
        overlap_ms=200, min_speech_ms=100, max_chunk_duration_s=1))
    chunks = list(seg.process(iter(frames)))

    assert len(chunks) == 2
    a, b = chunks
    assert b.overlap_from_prev_s == 0.0            # carry dropped by budget
    assert b.start_time == pytest.approx(54 * W / 16000, abs=1e-3)  # own speech
    _source_equal(a, frames)
    _source_equal(b, frames)
