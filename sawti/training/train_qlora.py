"""QLoRA training for Saudi Whisper (spec §2) — SA Tasks 4-5.

Task 3 scope: SetEpochCallback (dataset epoch advancement wired to the
Trainer). The config factories, DevEvalCallback, and main() arrive with
Tasks 4-5 per the reconciled plan.
"""
from __future__ import annotations


class SetEpochCallback:
    """Advances the dataset's augmentation stream per epoch so
    (seed, epoch, index) determinism actually varies across epochs in
    training — not merely in isolation tests."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kw) -> None:
        if state is not None and getattr(state, "epoch", None) is not None:
            self.dataset.set_epoch(int(state.epoch))
