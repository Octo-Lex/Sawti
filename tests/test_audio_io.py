import numpy as np
import pytest

from sawti.audio_io import FileSource, load_audio_mono_16k


@pytest.fixture
def wav_file(tmp_path):
    """Write 2s of 16kHz silence to a WAV file."""
    import soundfile as sf
    path = tmp_path / "test.wav"
    sf.write(path, np.zeros(32000, dtype=np.float32), 16000)
    return path


def test_load_audio_returns_mono_16k(wav_file):
    audio, sr = load_audio_mono_16k(wav_file)
    assert sr == 16000
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert len(audio) == 32000


def test_filesource_yields_frames(wav_file):
    src = FileSource(str(wav_file), frame_samples=16000)
    frames = list(src.iter_frames())
    assert len(frames) == 2
    assert frames[0].audio.shape == (16000,)
    assert frames[0].sample_rate == 16000
    assert frames[0].timestamp_s == 0.0
    assert frames[1].timestamp_s == pytest.approx(1.0)


def test_filesource_handles_partial_last_frame(wav_file):
    # 32000 samples / 16000 frame size = exactly 2 frames; make an odd-length file
    import soundfile as sf
    odd = wav_file.parent / "odd.wav"
    sf.write(odd, np.zeros(24000, dtype=np.float32), 16000)  # 1.5 frames worth
    src = FileSource(str(odd), frame_samples=16000)
    frames = list(src.iter_frames())
    assert len(frames) == 2  # last frame is partial (8000 samples)
    assert frames[1].audio.shape == (16000,)  # zero-padded to full frame
