"""Audio file I/O: load + resample to 16kHz mono, and a FileSource that
yields fixed-size AudioFrames (spec §5.2, §8.3). M1's offline input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from sawti.sources import AudioFrame


TARGET_SR = 16000


def load_audio_mono_16k(path: str | Path) -> tuple[np.ndarray, int]:
    """Load any audio file as mono float32 PCM at 16kHz.

    Uses librosa (handles WAV/MP3/FLAC + resampling).
    """
    import librosa

    audio, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return np.ascontiguousarray(audio, dtype=np.float32), TARGET_SR


class FileSource:
    """AudioSource backed by an audio file, split into fixed-size frames.

    The final partial frame is zero-padded to `frame_samples` so downstream
    processing always sees full-size arrays.
    """

    def __init__(self, path: str | Path, frame_samples: int = 16000) -> None:
        self.path = str(path)
        self.frame_samples = frame_samples
        self._audio, self._sr = load_audio_mono_16k(path)

    def iter_frames(self) -> Iterable[AudioFrame]:
        step = self.frame_samples
        total = len(self._audio)
        n = (total + step - 1) // step  # ceil
        for i in range(n):
            chunk = self._audio[i * step : (i + 1) * step]
            if len(chunk) < step:
                chunk = np.pad(chunk, (0, step - len(chunk)))
            yield AudioFrame(
                audio=np.ascontiguousarray(chunk, dtype=np.float32),
                sample_rate=self._sr,
                timestamp_s=i * step / self._sr,
            )
