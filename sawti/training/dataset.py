"""SA Task 3: SadaDataset + WhisperCollator + deterministic augmentation.

Manifest-driven selection (no corpus policy here — label decisions live
in materialization; the dataset loads whatever the manifest contains).
Augmentation randomness derives from (seed, epoch, sample index) via
numpy seed sequences: no shared mutable RNG, so dataloader worker
ordering cannot change what any sample receives.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def augment(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Deterministic-seeded augmentation (spec §2.6): random gain,
    additive gaussian noise at ~15-30 dB SNR, speed perturbation
    {0.9, 1.0, 1.1} via linear resampling. The augmentation class
    plausibly behind oddadmix's zero-loop robustness (Addendum 4).
    Music/reverb: deferred extension."""
    out = audio.copy()
    out *= float(rng.uniform(0.5, 1.0))
    p_signal = float(np.mean(out ** 2)) + 1e-12
    snr_db = float(rng.uniform(15.0, 30.0))
    p_noise = p_signal / (10 ** (snr_db / 10))
    out = out + rng.normal(0.0, p_noise ** 0.5, size=out.shape).astype(np.float32)
    speed = float(rng.choice([0.9, 1.0, 1.1]))
    if speed != 1.0:
        n_out = max(1, int(len(out) / speed))
        out = np.interp(
            np.linspace(0.0, len(out) - 1, n_out),
            np.arange(len(out)), out).astype(np.float32)
    return out.astype(np.float32)


class SadaDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str | Path, max_text_len: int = 448,
                 augment_enabled: bool = False, seed: int = 0,
                 epoch: int = 0,
                 manifest_name: str = "manifest.jsonl") -> None:
        self.dir = Path(data_dir)
        # Manifest-selection SEAM (not dialect policy): the caller names
        # which view to train/evaluate on (manifest.jsonl, manifest_core,
        # manifest_diagnostic...). Missing manifest = caller error.
        manifest_path = self.dir / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        self.rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
        ]
        self.max_text_len = max_text_len
        self.augment_enabled = augment_enabled
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation stream deterministically per epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    EXPECTED_SR = 16000  # Whisper's feature extractor contract

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        audio, sr = sf.read(self.dir / f"{r['clip_id']}.wav", dtype="float32")
        if sr != self.EXPECTED_SR:
            # Materialization preserved each source file's ORIGINAL rate;
            # a non-16k file processed as 16k would be time-scaled wrong
            # with no error. Fail loudly rather than corrupt training.
            raise ValueError(
                f"{r['clip_id']}.wav sample_rate={sr}; expected "
                f"{self.EXPECTED_SR} — rematerialize or resample the corpus "
                f"explicitly (see data_prep)")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if self.augment_enabled:
            # Deterministic PER SAMPLE (and epoch): no shared mutable RNG,
            # so dataloader worker ordering cannot change augmentation.
            rng = np.random.default_rng([self.seed, self.epoch, i])
            audio = augment(audio, rng)
        cleaned = (r.get("cleaned_text") or "").strip()
        text = cleaned if cleaned else (r.get("text") or "").strip()
        return {"audio": np.ascontiguousarray(audio, np.float32),
                "text": text[: self.max_text_len]}


class WhisperCollator:
    """Standard Whisper fine-tuning collator: 30s-padded features + masked
    labels. The tokenizer is invoked as a callable; the DECODER-START
    token (<|startoftranscript|>, model.config.decoder_start_token_id —
    DISTINCT from the tokenizer BOS <|endoftext|>) is stripped from
    labels when the tokenizer prepended it; padded positions are -100.

    decoder_start_token_id MUST be supplied explicitly (Whisper training
    shifts labels by the model's decoder start, not the tokenizer BOS)."""

    def __init__(self, processor, decoder_start_token_id: int) -> None:
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: list[dict]) -> dict:
        inputs = self.processor(
            [f["audio"] for f in features], sampling_rate=16000,
            return_tensors="pt")
        tok = self.processor.tokenizer
        batch = tok(
            [f["text"] for f in features], padding=True,
            truncation=True, max_length=448, return_tensors="pt")
        # Real tokenizers return a mapping (BatchEncoding); some fakes
        # return attribute objects. Support both.
        input_ids = batch["input_ids"] if isinstance(
            batch, dict) else batch.input_ids
        attention_mask = batch["attention_mask"] if isinstance(
            batch, dict) else batch.attention_mask
        labels = input_ids.masked_fill(attention_mask.ne(1), -100)
        # Strip the DECODER-START token when uniformly prepended — NOT the
        # tokenizer BOS (they differ in Whisper: BOS=<|endoftext|>,
        # decoder_start=<|startoftranscript|>). The model re-adds
        # decoder_start internally during label shifting.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        out = dict(input_features=inputs.input_features)
        if hasattr(inputs, "attention_mask"):
            out["attention_mask"] = inputs.attention_mask
        out["labels"] = labels
        return out
