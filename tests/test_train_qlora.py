"""SA Tasks 4-5: QLoRA config factories, DevEvalCallback, training args.

Hermetic: no models, no CUDA, no network, no corpus access."""
import json
from pathlib import Path

import pytest

from sawti.training.train_qlora import (
    DevEvalCallback,
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
    assert a.save_steps == 500
    assert a.gradient_checkpointing is True


def test_training_args_lora_fallback_flavor():
    a = build_training_args(".", flavor="lora")
    assert a.optim == "adamw_torch_fused"
    assert a.per_device_train_batch_size * a.gradient_accumulation_steps == 16


def test_training_args_reject_unknown_flavor():
    with pytest.raises(ValueError, match="unknown flavor"):
        build_training_args(".", flavor="banana")


class _Ctl:
    def __init__(self):
        self.should_training_stop = False


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


def test_dev_callback_best_tracking_and_stop_after_3_regressions(tmp_path):
    seq = iter([_metrics(_core(40.0)), _metrics(_core(30.0)),
                _metrics(_core(35.0)), _metrics(_core(36.0)),
                _metrics(_core(37.0))])
    cb = DevEvalCallback(eval_fn=lambda m: next(seq),
                         log_path=str(tmp_path / "dev_log.jsonl"), patience=3,
                         baselines=BASELINES)
    for _ in range(5):
        cb.on_save(args=None, state=None, control=_Ctl(), model=None)
    log = [json.loads(l) for l in
           (tmp_path / "dev_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[1]["selection_score"] == 30.0 and log[1]["is_best"] is True
    assert log[1]["all_valid_macro_wer"] == 60.0  # all four metrics logged
    assert log[1]["baselines"] == BASELINES       # baselines logged
    stops = [l for l in log if l.get("stop")]
    assert stops and stops[0]["eval_index"] == 5  # regressions at evals 3,4,5


def test_dev_callback_loop_constraint_blocks_ineligible_best(tmp_path):
    # All dialects within baselines+3pp so only LOOP drives ineligibility.
    e1 = {"Najdi": 30.0, "Hijazi": 40.0, "Khaliji": 50.0}  # score 40.0
    e2 = {"Najdi": 20.0, "Hijazi": 30.0, "Khaliji": 40.0}  # score 30.0, loop 9
    e3 = {"Najdi": 25.0, "Hijazi": 35.0, "Khaliji": 45.0}  # score 35.0
    seq = iter([_metrics(e1, loop=0.0), _metrics(e2, loop=9.0),
                _metrics(e3, loop=0.0)])
    cb = DevEvalCallback(eval_fn=lambda m: next(seq),
                         log_path=str(tmp_path / "dev_log.jsonl"), patience=3,
                         baselines=BASELINES)
    for _ in range(3):
        cb.on_save(args=None, state=None, control=_Ctl(), model=None)
    log = [json.loads(l) for l in
           (tmp_path / "dev_log.jsonl").read_text(encoding="utf-8").splitlines()]
    # eval 2: best score (30.0) but loop 9% > limit -> ineligible, not best
    assert log[1]["eligible"] is False and log[1]["is_best"] is False
    assert log[1]["loop_ok"] is False
    # eval 3: eligible (35.0 < standing best 40.0) -> new best
    assert log[2]["is_best"] is True and log[2]["best_selection_score"] == 35.0


def test_dev_callback_eligibility_boundary_at_limit(tmp_path):
    """loop == limit is ELIGIBLE (<=, not <)."""
    seq = iter([_metrics(_core(50.0), loop=0.0),
                _metrics(_core(30.0), loop=5.0)])
    cb = DevEvalCallback(eval_fn=lambda m: next(seq),
                         log_path=str(tmp_path / "dev_log.jsonl"), patience=3,
                         baselines=BASELINES)
    cb.on_save(args=None, state=None, control=_Ctl(), model=None)
    cb.on_save(args=None, state=None, control=_Ctl(), model=None)
    log = [json.loads(l) for l in
           (tmp_path / "dev_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[1]["eligible"] is True and log[1]["is_best"] is True


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


def test_dev_callback_per_dialect_logged(tmp_path):
    m = _metrics({"Najdi": 25.0, "Hijazi": 40.0, "Khaliji": 55.0})
    cb = DevEvalCallback(eval_fn=lambda model: m,
                         log_path=str(tmp_path / "dev_log.jsonl"), patience=3)
    cb.on_save(args=None, state=None, control=_Ctl(), model=None)
    log = [json.loads(l) for l in
           (tmp_path / "dev_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[0]["per_dialect"]["Hijazi"]["clean_macro_wer"] == 40.0


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
