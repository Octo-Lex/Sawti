"""Voice Activity Detection abstraction (spec §2.2).

The VAD is separated from the segmentation *policy* so the policy can be
unit-tested with a FakeVad that returns scripted probabilities. The real
SileroVad loads the Silero model lazily (only in integration tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass
class VadResult:
    probability: float  # 0..1 speech probability for the frame
    is_speech: bool


class VAD(Protocol):
    """A frame-level voice activity detector."""

    def prob(self, frame: np.ndarray, sample_rate: int = 16000) -> VadResult: ...


class FakeVad:
    """Returns a scripted sequence of probabilities (for unit tests)."""

    def __init__(self, scripted: Sequence[tuple[float, bool]]) -> None:
        self._scripted = list(scripted)
        self._i = 0

    def prob(self, frame: np.ndarray, sample_rate: int = 16000) -> VadResult:
        if self._i >= len(self._scripted):
            return VadResult(0.0, False)
        p, is_speech = self._scripted[self._i]
        self._i += 1
        return VadResult(p, is_speech)


class SileroVad:
    """Real Silero VAD wrapper. Loads the model lazily on first use.

    Only instantiated in integration tests / production. The model is held
    resident after first load.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from silero_vad import load_silero_vad  # type: ignore

            self._model = load_silero_vad()
        return self._model

    def prob(self, frame: np.ndarray, sample_rate: int = 16000) -> VadResult:
        import torch

        model = self._ensure_model()
        # Silero expects a torch tensor of shape (N,) float32.
        t = torch.as_tensor(frame, dtype=torch.float32)
        p = float(model(t, sample_rate).item())
        return VadResult(probability=p, is_speech=p >= self.threshold)
