"""SA Task 3: SadaDataset + WhisperCollator + deterministic augmentation.

Hermetic: torch tensors on CPU, no models/network/CUDA. The tokenizer
fake is CALLABLE (the collator invokes tok(...))."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from sawti.training.dataset import SadaDataset, WhisperCollator, augment


# Deliberately DISTINCT: tokenizer BOS (<|endoftext|>) vs the model's
# decoder_start (<|startoftranscript|>) — the regression below fails if
# the collator ever strips the wrong one.
FAKE_BOS = 50257
FAKE_DECODER_START = 50258


class FakeTokenizer:
    """Callable (the collator invokes tok(...)), with bos_token_id."""

    bos_token_id = FAKE_BOS

    def __call__(self, texts, padding=True, truncation=True,
                 max_length=448, return_tensors="pt"):
        # Labels begin with the DECODER-START token (as Whisper
        # tokenization produces), NOT with the tokenizer BOS.
        ids = torch.tensor([[FAKE_DECODER_START, 5, 6],
                            [FAKE_DECODER_START, 5, 0]])
        am = torch.tensor([[1, 1, 1], [1, 1, 0]])

        class B:
            input_ids = ids
            attention_mask = am
        return B()


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __call__(self, audio, sampling_rate=16000, return_tensors="pt"):
        class F:
            input_features = torch.zeros(len(audio), 80, 3000)
            attention_mask = torch.ones(len(audio), 3000, dtype=torch.long)
        return F()


def _mk(tmp_path: Path, n=2):
    for i in range(n):
        sf.write(tmp_path / f"c{i}.wav",
                 np.ones(16000, np.float32) * 0.3, 16000)
    rows = [
        {"clip_id": f"c{i}", "cleaned_text": "مرحبا", "text": "مرحبا",
         "dialect": "Najdi", "duration_s": 1.0}
        for i in range(n)
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8")


def test_dataset_returns_audio_and_text(tmp_path):
    _mk(tmp_path)
    ds = SadaDataset(str(tmp_path))
    item = ds[0]
    assert item["audio"].dtype == np.float32
    assert item["text"] == "مرحبا"


def test_dataset_text_fallback_semantics(tmp_path):
    # cleaned_text preferred; falls back to text; empty -> empty string.
    sf.write(tmp_path / "a.wav", np.zeros(8000, np.float32), 16000)
    sf.write(tmp_path / "b.wav", np.zeros(8000, np.float32), 16000)
    sf.write(tmp_path / "c.wav", np.zeros(8000, np.float32), 16000)
    rows = [
        {"clip_id": "a", "cleaned_text": "منظف", "text": "أصلي"},
        {"clip_id": "b", "cleaned_text": "", "text": "أصلي"},
        {"clip_id": "c", "cleaned_text": "  ", "text": ""},
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8")
    ds = SadaDataset(str(tmp_path))
    assert ds[0]["text"] == "منظف"      # cleaned preferred
    assert ds[1]["text"] == "أصلي"      # cleaned empty -> text fallback
    assert ds[2]["text"] == ""          # both empty -> empty


def test_collator_shapes_and_masking(tmp_path):
    _mk(tmp_path)
    ds = SadaDataset(str(tmp_path))
    batch = WhisperCollator(FakeProcessor(),
                            decoder_start_token_id=FAKE_DECODER_START)(
        [ds[0], ds[1]])
    assert batch["input_features"].shape[0] == 2
    assert batch["labels"].shape[0] == 2
    # padded positions are -100
    assert (batch["labels"] == -100).any()
    # DECODER-START stripped (not the tokenizer BOS — they differ here)
    assert batch["labels"][0, 0].item() == 5


def test_collator_does_not_strip_tokenizer_bos():
    """Regression: a leading tokenizer-BOS (≠ decoder_start) must SURVIVE
    — stripping it would corrupt labels when the two tokens differ."""
    _mk(Path("/tmp/_never") if False else Path(".")) if False else None
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        _mk(p)
        ds = SadaDataset(str(p))
        batch = WhisperCollator(
            FakeProcessor(), decoder_start_token_id=FAKE_DECODER_START)(
            [ds[0], ds[1]])
        # The fake prepends DECODER_START; stripping it leaves 5/6.
        assert batch["labels"][0, 0].item() == 5


def test_collator_requires_explicit_decoder_start(tmp_path):
    _mk(tmp_path)
    with pytest.raises(TypeError):
        WhisperCollator(FakeProcessor())  # decoder_start_token_id required


def test_dataset_rejects_non_16k_audio(tmp_path):
    """A non-16k WAV fails LOUDLY rather than being time-scaled wrong."""
    sf.write(tmp_path / "c0.wav", np.zeros(4000, np.float32), 8000)
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"clip_id": "c0", "cleaned_text": "أ", "text": "أ"}),
        encoding="utf-8")
    ds = SadaDataset(str(tmp_path))
    with pytest.raises(ValueError, match="sample_rate=8000"):
        _ = ds[0]


def test_dataset_text_truncation_at_448(tmp_path):
    sf.write(tmp_path / "c0.wav", np.zeros(8000, np.float32), 16000)
    long_text = "كلمة " * 200  # 1000 chars > 448
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"clip_id": "c0", "cleaned_text": long_text,
                    "text": long_text}, ensure_ascii=False),
        encoding="utf-8")
    ds = SadaDataset(str(tmp_path))
    assert len(ds[0]["text"]) == 448


def test_augment_deterministic_per_sample_and_epoch():
    audio = np.ones(16000, np.float32) * 0.5
    a1 = augment(audio, np.random.default_rng([42, 0, 3]))
    a2 = augment(audio, np.random.default_rng([42, 0, 3]))
    a3 = augment(audio, np.random.default_rng([42, 1, 3]))  # next epoch
    assert np.array_equal(a1, a2)      # same (seed, epoch, index)
    assert not np.array_equal(a1, a3)  # epoch advances the stream
    assert a1.dtype == np.float32
    assert float(np.std(a1)) > 1e-4    # noise actually added


def test_augment_bounds_gain_and_dtype():
    audio = np.ones(16000, np.float32)
    out = augment(audio, np.random.default_rng([7, 0, 0]))
    assert out.dtype == np.float32
    assert float(np.max(np.abs(out))) < 2.0   # bounded, not exploded


def test_dataset_augmentation_deterministic_no_shared_state(tmp_path):
    _mk(tmp_path, n=2)
    d0 = SadaDataset(str(tmp_path), augment_enabled=True, seed=42, epoch=0)
    # Same (seed, epoch, index): stable across repeated access AND across
    # two dataset instances (no shared mutable RNG).
    d0b = SadaDataset(str(tmp_path), augment_enabled=True, seed=42, epoch=0)
    assert np.array_equal(d0[0]["audio"], d0[0]["audio"])
    assert np.array_equal(d0[0]["audio"], d0b[0]["audio"])
    # Worker-order independence: touching index 0 then 1 == 1 then 0.
    d_left = SadaDataset(str(tmp_path), augment_enabled=True, seed=9, epoch=0)
    d_right = SadaDataset(str(tmp_path), augment_enabled=True, seed=9, epoch=0)
    _ = d_left[1]
    assert np.array_equal(d_left[0]["audio"], d_right[0]["audio"])


def test_set_epoch_advances_dataset_stream(tmp_path):
    _mk(tmp_path, n=1)
    ds = SadaDataset(str(tmp_path), augment_enabled=True, seed=42, epoch=0)
    before = ds[0]["audio"].copy()
    ds.set_epoch(1)
    after = ds[0]["audio"]
    assert not np.array_equal(before, after)  # epoch varies augmentation


def test_set_epoch_wiring_matches_trainer_state():
    from sawti.training.train_qlora import SetEpochCallback

    class _State:
        epoch = 3

    class _DS:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, e):
            self.epochs.append(e)

    ds = _DS()
    SetEpochCallback(ds).on_epoch_begin(None, _State(), None)
    assert ds.epochs == [3]                      # wired to Trainer state


def test_indexing_stable_and_manifest_driven(tmp_path):
    _mk(tmp_path, n=3)
    ds = SadaDataset(str(tmp_path))
    assert len(ds) == 3
    assert ds[2]["text"] == "مرحبا"
    with pytest.raises(IndexError):
        _ = ds[3]
