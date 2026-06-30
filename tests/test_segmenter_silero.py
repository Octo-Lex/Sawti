import numpy as np

from sawti.config import SegmentationConfig
from sawti.segmenter_silero import RealSegmenter
from sawti.sources import AudioFrame
from sawti.vad import FakeVad


def _frames(probs):
    """Build frames of silence tagged with scripted VAD probs."""
    # Each "frame" is 0.1s = 1600 samples at 16kHz.
    return [
        AudioFrame(audio=np.zeros(1600, np.float32), sample_rate=16000,
                   timestamp_s=i * 0.1)
        for i in range(len(probs))
    ], probs


def test_segmenter_emits_one_chunk_for_continuous_speech():
    frames, probs = _frames([True] * 10 + [False, False, False, False])  # 1s speech + pause
    seg = RealSegmenter(
        vad=FakeVad([(0.9, p) for p in probs]),
        config=SegmentationConfig(pause_threshold_ms=300, min_chunk_duration_ms=0),
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
            pause_threshold_ms=300, min_chunk_duration_ms=0, overlap_ms=0,
        ),
    )
    chunks = list(seg.process(iter(frames)))
    assert len(chunks) == 2


def test_segmenter_force_closes_at_max_duration():
    # 100 continuous speech frames = 10s; max_chunk_duration_s=2 forces close
    pattern = [True] * 100
    frames, probs = _frames(pattern)
    seg = RealSegmenter(
        vad=FakeVad([(0.9, p) for p in probs]),
        config=SegmentationConfig(
            pause_threshold_ms=99999, max_chunk_duration_s=2,
            min_chunk_duration_ms=0, overlap_ms=0,
        ),
    )
    chunks = list(seg.process(iter(frames)))
    assert len(chunks) == 5  # 100 frames / 20 frames-per-2s


def test_segmenter_skips_too_short_chunk():
    # 1 speech frame then long pause — below min_chunk_duration_ms
    pattern = [True] + [False] * 10
    frames, probs = _frames(pattern)
    seg = RealSegmenter(
        vad=FakeVad([(0.9, p) for p in probs]),
        config=SegmentationConfig(
            pause_threshold_ms=300, min_chunk_duration_ms=500, overlap_ms=0,
        ),
    )
    chunks = list(seg.process(iter(frames)))
    assert len(chunks) == 0  # the 0.1s of speech is below 500ms minimum
