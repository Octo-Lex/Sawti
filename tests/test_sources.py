import numpy as np

from sawti.sources import StubAudioSource, AudioFrame


def test_stub_audio_source_yields_frames():
    src = StubAudioSource(n_frames=3, samples_per_frame=16000)
    frames = list(src.iter_frames())
    assert len(frames) == 3
    assert all(isinstance(f, AudioFrame) for f in frames)
    assert frames[0].audio.shape == (16000,)
    assert frames[0].audio.dtype == np.float32
    # timestamps are monotonic
    assert frames[1].timestamp_s > frames[0].timestamp_s
