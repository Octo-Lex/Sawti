import numpy as np
import pytest

from sawti.config import SegmentationConfig
from sawti.segmenter_silero import RealSegmenter
from sawti.sources import AudioFrame
from sawti.vad import FakeVad

# Each frame is exactly one Silero VAD window (512 samples = 0.032s at 16kHz),
# so FakeVad is called once per frame — matching how a real VAD sees the audio.
_FRAME_SAMPLES = 512
_FRAME_S = 0.032  # 512 / 16000


def _frames(probs):
    """Build frames of silence tagged with scripted VAD probs (one per frame)."""
    return (
        [
            AudioFrame(
                audio=np.zeros(_FRAME_SAMPLES, np.float32),
                sample_rate=16000,
                timestamp_s=i * _FRAME_S,
            )
            for i in range(len(probs))
        ],
        probs,
    )


def test_segmenter_emits_one_chunk_for_continuous_speech():
    frames, probs = _frames([True] * 10 + [False, False, False, False])
    seg = RealSegmenter(
        vad=FakeVad([(0.9, p) for p in probs]),
        config=SegmentationConfig(pause_threshold_ms=10, min_chunk_duration_ms=0),
    )
    chunks = list(seg.process(iter(frames)))
    assert len(chunks) == 1
    assert chunks[0].start_time == 0.0
    assert chunks[0].end_time > 0.0


def test_segmenter_splits_on_long_pause():
    # 5 speech frames, 5 silence (pause), 5 speech frames
    pattern = [True] * 5 + [False] * 5 + [True] * 5
    frames, probs = _frames(pattern)
    seg = RealSegmenter(
        vad=FakeVad([(0.9, p) for p in probs]),
        config=SegmentationConfig(
            pause_threshold_ms=10, min_chunk_duration_ms=0, overlap_ms=0,
        ),
    )
    chunks = list(seg.process(iter(frames)))
    assert len(chunks) == 2


def test_segmenter_force_closes_at_max_duration():
    # 100 continuous speech frames = 3.2s; max_chunk_duration_s=1 forces close.
    pattern = [True] * 100
    frames, probs = _frames(pattern)
    seg = RealSegmenter(
        vad=FakeVad([(0.9, p) for p in probs]),
        config=SegmentationConfig(
            pause_threshold_ms=99999, max_chunk_duration_s=1,
            min_chunk_duration_ms=0, overlap_ms=0,
        ),
    )
    chunks = list(seg.process(iter(frames)))
    # 3.2s total / 1s max -> at least 3 chunks (force-close at ~1s, ~2s, ~3.2s).
    assert len(chunks) >= 3


def test_segmenter_skips_too_short_chunk():
    # 1 speech frame (~0.032s) then long pause — below min_chunk_duration_ms=100.
    pattern = [True] + [False] * 10
    frames, probs = _frames(pattern)
    seg = RealSegmenter(
        vad=FakeVad([(0.9, p) for p in probs]),
        config=SegmentationConfig(
            pause_threshold_ms=10, min_chunk_duration_ms=100, overlap_ms=0,
        ),
    )
    chunks = list(seg.process(iter(frames)))
    assert len(chunks) == 0  # 0.032s of speech is below 100ms minimum


def test_segmenter_windows_large_frames_for_vad():
    """When a frame is larger than the VAD window (e.g. FileSource's 16000-sample
    1s frames), the segmenter must sub-window it for the VAD rather than passing
    the whole frame (Silero raises on anything other than 512 samples). This test
    uses a fake VAD that records call sizes to confirm windowing happens."""
    calls = []

    class RecordingVad:
        def prob(self, frame, sample_rate=16000):
            from sawti.vad import VadResult

            calls.append(len(frame))
            # treat all calls as speech so the buffer stays open
            return VadResult(probability=0.9, is_speech=True)

    # One big 16000-sample frame (1s), max_chunk_duration forces a close.
    big_frame = AudioFrame(
        audio=np.zeros(16000, np.float32), sample_rate=16000, timestamp_s=0.0
    )
    seg = RealSegmenter(
        vad=RecordingVad(),
        config=SegmentationConfig(
            pause_threshold_ms=99999, max_chunk_duration_s=1,
            min_chunk_duration_ms=0, overlap_ms=0,
        ),
    )
    list(seg.process(iter([big_frame])))
    # The 16000-sample frame must be split into 512-sample VAD calls, never
    # passed whole.
    assert calls, "VAD was never called"
    assert all(c == 512 for c in calls), f"VAD got non-512-sample calls: {calls}"


# --- P2 #2 regression tests: sub-window granularity prevents timestamp inflation ---


def _big_frame_with_scripted_vad(frame_samples, frame_start_s, vad_verdicts):
    """Build ONE large AudioFrame and a FakeVad that returns `vad_verdicts`
    (one per 512-sample sub-window inside the frame). Returns (frame, vad).

    This lets us test sparse-speech-inside-a-big-frame scenarios precisely.
    """
    sr = 16000
    frame = AudioFrame(
        audio=np.zeros(frame_samples, np.float32), sample_rate=sr,
        timestamp_s=frame_start_s,
    )
    vad = FakeVad([(0.9, v) for v in vad_verdicts])
    return frame, vad


def test_sparse_speech_in_big_frame_does_not_inflate_start_time():
    """Regression (P2 #2): a 1s frame with speech only in its LAST 512-sample
    sub-window must produce a chunk whose start_time is that sub-window's
    timestamp (~0.936s), NOT the frame's start (0.0s). Before the fix, the
    whole frame was marked speech and start_time was inflated to 0.0."""
    # 1s frame = 16000 samples = 31 full 512-windows + 128-sample tail (32nd).
    # Speech only in the last full sub-window (index 30), then silence after.
    n_subs = 32
    verdicts = [False] * 30 + [True] + [False] * 30
    frame, vad = _big_frame_with_scripted_vad(16000, 0.0, verdicts)
    seg = RealSegmenter(
        vad=vad,
        config=SegmentationConfig(
            pause_threshold_ms=10, min_chunk_duration_ms=0, overlap_ms=0,
        ),
    )
    chunks = list(seg.process(iter([frame])))
    assert len(chunks) == 1
    # Sub-window 30 starts at 30 * 512/16000 = 0.96s.
    assert chunks[0].start_time == pytest.approx(0.96, abs=0.01), \
        f"start_time inflated: {chunks[0].start_time}"
    # Speech span is one sub-window = 0.032s, so end ≈ 0.992s.
    assert chunks[0].end_time <= 1.0


def test_min_duration_gates_on_actual_speech_span_not_frame_span():
    """Regression (P2 #2): a 1s frame where only 1 sub-window (0.032s) is
    speech must be DROPPED by min_chunk_duration_ms=100, because the actual
    speech span (0.032s) is below the threshold — not the frame span (1s)."""
    # 1s frame, speech in only the middle sub-window, then enough silence to
    # trigger a pause-close attempt.
    verdicts = [False] * 15 + [True] + [False] * 45  # 1 speech sub in the middle
    frame, vad = _big_frame_with_scripted_vad(16000, 0.0, verdicts)
    seg = RealSegmenter(
        vad=vad,
        config=SegmentationConfig(
            pause_threshold_ms=10, min_chunk_duration_ms=100, overlap_ms=0,
        ),
    )
    chunks = list(seg.process(iter([frame])))
    assert len(chunks) == 0, \
        f"should be dropped (0.032s speech < 100ms min), got {len(chunks)} chunks"


def test_contiguous_speech_in_big_frame_produces_accurate_span():
    """Sanity: a 1s frame that is entirely speech produces a chunk spanning
    the whole frame (~0.0 to ~1.0s), confirming the sub-window logic doesn't
    UNDER-report speech either."""
    verdicts = [True] * 32  # all sub-windows speech
    frame, vad = _big_frame_with_scripted_vad(16000, 0.0, verdicts)
    seg = RealSegmenter(
        vad=vad,
        config=SegmentationConfig(
            pause_threshold_ms=99999, max_chunk_duration_s=2,
            min_chunk_duration_ms=0, overlap_ms=0,
        ),
    )
    chunks = list(seg.process(iter([frame])))
    assert len(chunks) == 1
    assert chunks[0].start_time == pytest.approx(0.0, abs=0.01)
    assert chunks[0].end_time == pytest.approx(1.0, abs=0.05), \
        f"end_time should be ~1.0s, got {chunks[0].end_time}"
