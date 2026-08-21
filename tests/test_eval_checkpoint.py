"""Post-hoc evaluator contract tests (Run 2).

Hermetic: no models, no CUDA, no network. Heavy pieces (model, tokenizer,
feature_extractor) are injected — run_validation/transcribe_wavs are
tested against fakes, pinning the properties that made Run 1 fail:
decoding regime explicitness, batch-order alignment, progress
observability, and atomic record writes.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from sawti.training.eval_checkpoint import (
    BEAM5_KWARGS,
    GREEDY_KWARGS,
    SELECTION_BATCH_SIZE,
    adapter_identity,
    atomic_write_json,
    build_record,
    evaluator_commit,
    manifest_identity,
    run_validation,
    sha256_file,
    stride_sample,
    transcribe_wavs,
)


def test_selection_batch_size_pinned_at_4():
    """Reviewer corrective (2026-08-20): the authoritative selection regime
    is batch_size=4 — bs=8 flips near-tie hypotheses per the benchmark, so
    8 must never silently come back as the CLI default."""
    assert SELECTION_BATCH_SIZE == 4


def test_greedy_kwargs_pin_the_decoding_regime():
    """The single decoding authority: greedy by CONSTRUCTION. The HF ASR
    pipeline defaults num_beams=5 (transformers 4.57.6) — the Run 1
    callback inherited that silently and was not running the regime its
    documentation claimed."""
    assert GREEDY_KWARGS == {"language": "arabic", "task": "transcribe",
                             "num_beams": 1, "do_sample": False}


def test_beam5_kwargs_exist_only_as_benchmark_probe():
    assert BEAM5_KWARGS["num_beams"] == 5 and BEAM5_KWARGS["do_sample"] is False


def test_stride_sample_deterministic_and_spread():
    assert stride_sample(10, 4) == [0, 2, 5, 7]
    assert stride_sample(3, 10) == [0, 1, 2]        # k >= n -> everything
    assert stride_sample(3423, 24) == stride_sample(3423, 24)  # no RNG
    idx = stride_sample(3423, 24)
    assert len(set(idx)) == 24 and min(idx) == 0    # spread + no dupes
    assert max(idx) < 3423


def test_atomic_write_json_leaves_no_tmp_and_replaces(tmp_path):
    out = tmp_path / "record.json"
    atomic_write_json(out, {"a": 1})
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))
    atomic_write_json(out, {"a": 2})                # overwrite via replace
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 2}
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_json_fails_before_touching_target(tmp_path):
    """Non-serializable record: exception must leave NO file behind —
    an interrupted evaluation never masquerades as a valid score."""
    out = tmp_path / "record.json"
    with pytest.raises(TypeError):
        atomic_write_json(out, {"bad": object()})
    assert not out.exists()
    assert not list(tmp_path.glob("*.tmp"))


# ---- fakes: the injected surface of the batched evaluator ----

class _FakeFE:
    """Returns stacked 'features' (the raw waveform matrix) as a real torch
    tensor plus an attention_mask, mirroring return_attention_mask=True —
    the model can see exactly which clips the slice contained (rows)."""

    def __call__(self, audios, sampling_rate=None, return_tensors=None,
                 return_attention_mask=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            input_features=torch.from_numpy(np.stack(audios)),
            attention_mask=torch.ones(len(audios), 1, dtype=torch.long))


class _FakeModel:
    """generate(input_features, attention_mask, **kwargs) -> token 'ids'
    one per clip in batch order; records decode kwargs and received masks
    separately so tests can pin BOTH the regime and the mask plumbing."""

    def __init__(self):
        self.calls = []
        self.masks = []

    def generate(self, input_features, attention_mask=None, **kwargs):
        self.calls.append(kwargs)
        self.masks.append(attention_mask)
        # id encodes (mean amplitude, n_clips_in_batch): distinct per clip,
        # so misalignment between hyps and manifest rows is detectable.
        # round() absorbs PCM-16 quantization wobble (0.001 -> 0.000976).
        return [round(float(x.abs().mean()) * 1000) * 100 + len(input_features)
                for x in input_features]


class _FakeTok:
    def batch_decode(self, ids, skip_special_tokens=None):
        return [f"hyp{i:06d}" for i in ids]


def _make_val_dir(tmp_path, n=5):
    for i in range(n):
        sf.write(tmp_path / f"c{i}.wav",
                 np.full(1600, (i + 1) / 1000, np.float32), 16000)
    rows = [{"clip_id": f"c{i}", "dialect": ["Najdi", "Hijazi", "Khaliji"][i % 3],
             "cleaned_text": "كلام واحد", "duration_s": 2.0}
            for i in range(n)]
    (tmp_path / "manifest.jsonl").write_text(
        chr(10).join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8")
    return rows


def test_transcribe_wavs_passes_greedy_kwargs_and_batches(tmp_path):
    paths = [str(tmp_path / f"c{i}.wav") for i in range(3)]
    _make_val_dir(tmp_path)
    m = _FakeModel()
    hyps = transcribe_wavs(m, _FakeTok(), _FakeFE(), paths, "cpu")
    # amplitudes .001-.003 -> int(mean*1000)=1..3, one batch of 3:
    assert hyps == [f"hyp{i * 100 + 3:06d}" for i in (1, 2, 3)]
    # decode regime forwarded verbatim; attention_mask ALWAYS plumbed
    # through (mandatory batched-Whisper reviewer check):
    assert m.calls == [GREEDY_KWARGS]
    assert len(m.masks) == 1 and m.masks[0] is not None


def test_run_validation_preserves_manifest_order_across_batches(tmp_path):
    """THE batching hazard: hyps must align to manifest rows through every
    slice boundary. Fakes make any misalignment observable via clip-specific
    ids; duration 2.0s keeps all rows non-degenerate so wer is computed."""
    _make_val_dir(tmp_path, n=5)
    captured = []
    m = _FakeModel()
    rows = run_validation(m, _FakeTok(), _FakeFE(), "cpu", str(tmp_path),
                          batch_size=2, progress_every=100, echo=captured.append)
    assert len(rows) == 5
    assert [r["clip_id"] for r in rows] == [f"c{i}" for i in range(5)]
    # ids encode mean-amplitude*1000*100 + batch size: batches are 2/2/1,
    # so the FINAL PARTIAL BATCH has size 1 — alignment must track exactly.
    expected_ids = [102, 202, 302, 402, 501]
    for i, r in enumerate(rows):
        assert r["hyp"] == f"hyp{expected_ids[i]:06d}"
        assert r["wer"] is not None                    # jiwer ran per row
        assert r["dialect"] == ["Najdi", "Hijazi", "Khaliji"][i % 3]
    assert all(c == GREEDY_KWARGS for c in m.calls) and len(m.calls) == 3
    assert all(mask is not None for mask in m.masks)  # mask on every batch
    assert captured                                    # progress observable


def test_run_validation_limit_takes_prefix(tmp_path):
    _make_val_dir(tmp_path, n=5)
    rows = run_validation(_FakeModel(), _FakeTok(), _FakeFE(), "cpu",
                          str(tmp_path), batch_size=8, limit=3, echo=lambda *_: None)
    assert [r["clip_id"] for r in rows] == ["c0", "c1", "c2"]


def test_build_record_stock_mode_skips_selection():
    """baselines=None (stock zero-shot baseline mode): a model cannot be
    screened against guards derived from itself."""
    rows = [{"dialect": d, "wer": 0.5, "n_ref_words": 5, "valid_ref": True,
             "loop": False, "degenerate": False}
            for d in ("Najdi", "Hijazi", "Khaliji")]
    rec = build_record(rows, None, {"model": "stock"})
    assert rec["selection"] is None
    assert rec["aggregate"]["n"] == 3
    assert rec["clips"] == rows


def test_build_record_checkpoint_mode_wires_selection():
    # wer rows are FRACTIONS (jiwer convention); aggregate() scales to %.
    rows = [{"dialect": d, "wer": 0.10, "n_ref_words": 5, "valid_ref": True,
             "loop": False, "degenerate": False}
            for d in ("Najdi", "Hijazi", "Khaliji")]
    baselines = {"Najdi": 46.685, "Hijazi": 49.928, "Khaliji": 56.399}
    cfg = {"generate_kwargs": dict(GREEDY_KWARGS), "batch_size": 8}
    rec = build_record(rows, baselines, cfg)
    assert rec["selection"]["eligible"] is True       # 10 << baselines
    assert rec["selection"]["selection_score"] == 10.0
    assert rec["config"] == cfg                       # regime echoed in record


def test_run_validation_rejects_wrong_sample_rate(tmp_path):
    """Non-16k audio is a materializer-contract violation; fail loudly."""
    sf.write(tmp_path / "c48.wav", np.zeros(4800, np.float32), 48000)
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"clip_id": "c48", "dialect": "Najdi",
                    "cleaned_text": "أ", "duration_s": 2.0}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="16000"):
        run_validation(_FakeModel(), _FakeTok(), _FakeFE(), "cpu",
                       str(tmp_path), batch_size=1, echo=lambda *_: None)


# ---- regime persistence: artifacts must be reproducible evidence ----

def test_manifest_identity_pins_selected_manifest(tmp_path):
    """The seam: identity hashes the manifest ACTUALLY selected — a
    diagnostic-view run must never be attributed to manifest.jsonl."""
    _make_val_dir(tmp_path)                       # writes manifest.jsonl
    (tmp_path / "manifest_diagnostic.jsonl").write_text(
        json.dumps({"clip_id": "d0", "dialect": "Shamali",
                    "cleaned_text": "ش", "duration_s": 2.0}) + "\n",
        encoding="utf-8")
    official = manifest_identity(tmp_path)
    diag = manifest_identity(tmp_path, "manifest_diagnostic.jsonl")
    assert official["name"] == "manifest.jsonl"
    assert diag["name"] == "manifest_diagnostic.jsonl"
    assert official["sha256"] != diag["sha256"]
    with pytest.raises(FileNotFoundError):
        manifest_identity(tmp_path, "manifest_ghost.jsonl")


def test_run_validation_reads_selected_manifest_view(tmp_path):
    """run_validation evaluates the named view's clips — the fake ids make
    any manifest confusion (official vs diagnostic rows) detectable."""
    _make_val_dir(tmp_path, n=5)                  # c0..c4 in manifest.jsonl
    sf.write(tmp_path / "d0.wav", np.full(1600, 0.006, np.float32), 16000)
    (tmp_path / "manifest_diagnostic.jsonl").write_text(
        json.dumps({"clip_id": "d0", "dialect": "Janubi",
                    "cleaned_text": "جملة", "duration_s": 2.0}) + "\n",
        encoding="utf-8")
    rows = run_validation(_FakeModel(), _FakeTok(), _FakeFE(), "cpu",
                          str(tmp_path), batch_size=4,
                          manifest_name="manifest_diagnostic.jsonl",
                          echo=lambda *_: None)
    assert [r["clip_id"] for r in rows] == ["d0"]
    assert rows[0]["dialect"] == "Janubi"
    assert rows[0]["hyp"] == "hyp000601"          # amplitude .006, batch of 1


def test_manifest_identity_pins_content_hash(tmp_path):
    _make_val_dir(tmp_path)
    ident = manifest_identity(tmp_path)
    assert ident["path"].endswith("manifest.jsonl")
    assert len(ident["sha256"]) == 64
    assert ident["sha256"] == sha256_file(tmp_path / "manifest.jsonl")
    (tmp_path / "manifest.jsonl").write_text("changed", encoding="utf-8")
    assert manifest_identity(tmp_path)["sha256"] != ident["sha256"]
    bogus = tmp_path / "nowhere"
    bogus.mkdir()
    with pytest.raises(FileNotFoundError):
        manifest_identity(bogus)


def test_adapter_identity_stock_none_and_checkpoint_hashed(tmp_path):
    assert adapter_identity(None) is None          # stock baseline mode
    ckpt = tmp_path / "checkpoint-2000"
    ckpt.mkdir()
    with pytest.raises(FileNotFoundError):        # empty dir -> fail loud
        adapter_identity(ckpt)
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ckpt / "adapter_model.safetensors").write_bytes(b"\x00" * 10)
    ident = adapter_identity(ckpt)
    assert ident["path"] == str(ckpt)
    assert set(ident) == {"path", "adapter_config.json",
                          "adapter_model.safetensors"}
    assert all(len(v) == 64 for k, v in ident.items() if k != "path")


def test_evaluator_commit_is_sha_or_unknown():
    import re

    sha = evaluator_commit()
    assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}", sha)
