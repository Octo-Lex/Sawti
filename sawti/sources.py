"""Audio source protocol + stubs (spec §5.2, §8.3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


@dataclass
class AudioFrame:
    """A fixed-size frame of PCM audio with a timestamp."""

    audio: np.ndarray  # float32 mono
    sample_rate: int
    timestamp_s: float


class AudioSource(Protocol):
    """Anything that yields timestamped audio frames."""

    def iter_frames(self) -> Iterable[AudioFrame]: ...


class StubAudioSource:
    """Yields `n_frames` frames of synthetic silence (M0 only)."""

    def __init__(self, n_frames: int = 5, samples_per_frame: int = 16000,
                 sample_rate: int = 16000) -> None:
        self.n_frames = n_frames
        self.samples_per_frame = samples_per_frame
        self.sample_rate = sample_rate

    def iter_frames(self) -> Iterable[AudioFrame]:
        for i in range(self.n_frames):
            yield AudioFrame(
                audio=np.zeros(self.samples_per_frame, dtype=np.float32),
                sample_rate=self.sample_rate,
                timestamp_s=i * self.samples_per_frame / self.sample_rate,
            )
