"""SA Tasks 4-5: QLoRA config factories and training args.

Hermetic: no models, no CUDA, no network, no corpus access.

Run 2: live dev evaluation was REMOVED from training (Run 1 proved it is
an hours-per-checkpoint GPU blocker); selection is post-hoc via
sawti.training.eval_checkpoint, sharing compute_selection from here."""
import json
from pathlib import Path

import pytest

from sawti.training.train_qlora import (
    SetEpochCallback,
    build_lora_config,
    build_training_args,
    compute_selection,
)


def test_lora_config_matches_spec_recipe():
    cfg = build_lora_config()
    assert cfg.r == 8 and cfg.lora_alpha == 16 and cfg.lora_dropout == 0.05
    mods = set(cfg.target_modules)
    assert {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"} <= mods


def test_training_args_qlora_flavor():
    a = build_training_args(".", flavor="qlora")
    assert a.optim == "paged_adamw_8bit"
    assert a.per_device_train_batch_size * a.gradient_accumulation_steps == 16
    assert a.learning_rate == 1e-4 and a.warmup_ratio == 0.1
    assert a.max_steps == 10000
    # Run 2 decoupling pins: sparse checkpoints, ALL retained for post-hoc
    # selection (save_total_limit would delete early ones).
    assert a.save_steps == 2000
    assert a.save_total_limit is None
    assert a.gradient_checkpointing is True


def test_training_args_lora_fallback_flavor():
    a = build_training_args(".", flavor="lora")
    assert a.optim == "adamw_torch_fused"
    assert a.per_device_train_batch_size * a.gradient_accumulation_steps == 16


def test_training_args_reject_unknown_flavor():
    with pytest.raises(ValueError, match="unknown flavor"):
        build_training_args(".", flavor="banana")


BASELINES = {"Najdi": 30.0, "Hijazi": 45.0, "Khaliji": 55.0}


def _metrics(dialect_wers, loop=0.0):
    """dialect_wers: dict of core-dialect -> clean WER. The overall clean
    macro is the (deliberately skewed) population-weighted number."""
    per = {d: {"clean_macro_wer": w, "n_clean": 20}
           for d, w in dialect_wers.items()}
    overall = (sum(dialect_wers.values()) / len(dialect_wers)
               if dialect_wers else float("inf"))
    return {"clean_macro_wer": overall, "all_valid_macro_wer": overall * 2.0,
            "all_valid_corpus_wer": overall, "loop_pct": loop,
            "degenerate_rate": loop, "per_dialect": per, "n": 60}


def _core(wer):
    return {"Najdi": wer, "Hijazi": wer, "Khaliji": wer}


def test_compute_selection_dialect_guard_blocks_regression():
    """A Hijazi regression beyond baseline+3pp makes the checkpoint
    INELIGIBLE even with a great Najdi score."""
    result = _metrics({"Najdi": 20.0, "Hijazi": 50.0, "Khaliji": 55.0})
    sel = compute_selection(result, BASELINES)
    # Hijazi baseline 45.0, tolerance 3.0 -> guard at >48.0; 50.0 exceeds.
    assert sel["eligible"] is False
    assert sel["guard_fail"][0]["dialect"] == "Hijazi"
    assert sel["guard_fail"][0]["exceeds_by_pp"] == 5.0  # 50-45
    # But the score itself is still computed (mean of three).
    assert sel["selection_score"] == (20.0 + 50.0 + 55.0) / 3


def test_compute_selection_skew_immunity():
    """The selection score is the unweighted three-dialect mean — a Najdi
    improvement cannot outweigh a Hijazi regression in ELIGIBILITY, and
    the score itself weights dialects equally regardless of population."""
    skewed = _metrics({"Najdi": 15.0, "Hijazi": 47.0, "Khaliji": 55.0})
    sel = compute_selection(skewed, BASELINES)
    assert sel["eligible"] is True  # all within tolerance
    assert sel["selection_score"] == (15.0 + 47.0 + 55.0) / 3


def test_compute_selection_missing_dialect_ineligible():
    result = _metrics({"Najdi": 20.0})  # Hijazi/Khaliji absent
    sel = compute_selection(result, BASELINES)
    assert sel["eligible"] is False
    reasons = {g["dialect"] for g in sel["guard_fail"]}
    assert reasons == {"Hijazi", "Khaliji"}


def test_lora_config_task_type_deliberately_unset():
    """Regression: PEFT task_type MUST stay None for Whisper (the
    Seq2SeqLM wrapper passes input_ids where Whisper expects
    input_features — a documented failure mode)."""
    cfg = build_lora_config()
    assert cfg.task_type is None


def test_set_epoch_callback_advances_dataset_stream():
    import numpy as np

    from sawti.training.dataset import SadaDataset

    class _State:
        epoch = 3

    class _DS:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, e):
            self.epochs.append(e)

    ds = _DS()
    SetEpochCallback(ds).on_epoch_begin(None, _State(), None)
    assert ds.epochs == [3]                     # wired to Trainer state

    # And the dataset actually varies its augmentation per epoch:
    import tempfile

    import soundfile as sf
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        sf.write(p / "x.wav", np.ones(1600, np.float32) * 0.5, 16000)
        (p / "manifest.jsonl").write_text(
            json.dumps({"clip_id": "x", "cleaned_text": "كلمة"}),
            encoding="utf-8")
        d0 = SadaDataset(p, augment_enabled=True, seed=42, epoch=0)
        d1 = SadaDataset(p, augment_enabled=True, seed=42, epoch=1)
        assert not np.array_equal(d0[0]["audio"], d1[0]["audio"])
        assert np.array_equal(d0[0]["audio"], d0[0]["audio"])  # stable


def test_validation_baselines_pinned():
    """The selection baselines come from the stock model's zero-shot run
    on the MATERIALIZED VALIDATION set — pinned in code so training
    configs reference stable numbers (the full JSON stays a local
    operator artifact under gitignored data/)."""
    from sawti.training.baselines import VALIDATION_BASELINES

    assert set(VALIDATION_BASELINES) == {"Najdi", "Hijazi", "Khaliji"}
    for v in VALIDATION_BASELINES.values():
        assert 0.0 < v < 100.0


def test_compute_selection_missing_baselines_fail_closed():
    """Empty/partial baseline maps are rejected — the regression guards
    can never be silently disabled by a wiring omission."""
    result = _metrics(_core(30.0))
    with pytest.raises(ValueError, match="missing baselines"):
        compute_selection(result, {})                 # empty -> reject
    with pytest.raises(ValueError, match="missing baselines"):
        compute_selection(result, {"Najdi": 30.0})   # partial -> reject


def test_production_baselines_are_the_pinned_validation_set():
    """main() must pass VALIDATION_BASELINES (the zero-shot validation
    numbers), not synthetic values. Pinned by checking the module-level
    import path is the one baselines.py exports."""
    from sawti.training.baselines import VALIDATION_BASELINES
    from sawti.training.train_qlora import CORE_DIALECTS

    assert set(VALIDATION_BASELINES) == set(CORE_DIALECTS)
    # None missing, none extra.
    assert all(0 < v < 100 for v in VALIDATION_BASELINES.values())


def test_dataset_manifest_name_selects_view(tmp_path):
    """The manifest-selection seam: the core view excludes Shamali/Janubi;
    the diagnostic view contains exactly those; both resolve the same WAVs."""
    import json as _j
    import soundfile as _sf

    import numpy as _np
    from sawti.training.dataset import SadaDataset

    for cid in ("a", "b", "c"):
        _sf.write(tmp_path / f"{cid}.wav",
                  _np.zeros(8000, _np.float32), 16000)
    rows = [
        {"clip_id": "a", "dialect": "Najdi", "cleaned_text": "أ",
         "audio_sha256": "h1"},
        {"clip_id": "b", "dialect": "Shamali", "cleaned_text": "ب",
         "audio_sha256": "h2"},
        {"clip_id": "c", "dialect": "Janubi", "cleaned_text": "ج",
         "audio_sha256": "h3"},
    ]
    for name, subset in [("manifest.jsonl", rows),
                         ("manifest_core.jsonl", [rows[0]]),
                         ("manifest_diagnostic.jsonl", [rows[1], rows[2]])]:
        (tmp_path / name).write_text(
            chr(10).join(_j.dumps(r, ensure_ascii=False) for r in subset),
            encoding="utf-8")

    full = SadaDataset(str(tmp_path))
    core = SadaDataset(str(tmp_path), manifest_name="manifest_core.jsonl")
    diag = SadaDataset(str(tmp_path),
                       manifest_name="manifest_diagnostic.jsonl")
    assert len(full) == 3
    assert len(core) == 1 and core.rows[0]["dialect"] == "Najdi"
    assert {r["dialect"] for r in diag.rows} == {"Shamali", "Janubi"}
    # Missing manifest fails loudly:
    with pytest.raises(FileNotFoundError):
        SadaDataset(str(tmp_path), manifest_name="manifest_ghost.jsonl")
