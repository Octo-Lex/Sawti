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


def _metrics(wer, loop=0.0):
    return {"clean_macro_wer": wer, "all_valid_macro_wer": wer * 2.0,
            "all_valid_corpus_wer": wer, "loop_pct": loop,
            "degenerate_rate": loop, "per_dialect": {}, "n": 59}


def test_dev_callback_best_tracking_and_stop_after_3_regressions(tmp_path):
    seq = iter([_metrics(40.0), _metrics(30.0), _metrics(35.0),
                _metrics(36.0), _metrics(37.0)])
    cb = DevEvalCallback(eval_fn=lambda m: next(seq),
                         log_path=str(tmp_path / "dev_log.jsonl"), patience=3)
    for _ in range(5):
        cb.on_save(args=None, state=None, control=_Ctl(), model=None)
    log = [json.loads(l) for l in
           (tmp_path / "dev_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[1]["clean_macro_wer"] == 30.0 and log[1]["is_best"] is True
    assert log[1]["all_valid_macro_wer"] == 60.0  # all four metrics logged
    stops = [l for l in log if l.get("stop")]
    assert stops and stops[0]["eval_index"] == 5  # regressions at evals 3,4,5


def test_dev_callback_loop_constraint_blocks_ineligible_best(tmp_path):
    seq = iter([_metrics(40.0, loop=0.0), _metrics(20.0, loop=9.0),
                _metrics(35.0, loop=0.0)])
    cb = DevEvalCallback(eval_fn=lambda m: next(seq),
                         log_path=str(tmp_path / "dev_log.jsonl"), patience=3)
    for _ in range(3):
        cb.on_save(args=None, state=None, control=_Ctl(), model=None)
    log = [json.loads(l) for l in
           (tmp_path / "dev_log.jsonl").read_text(encoding="utf-8").splitlines()]
    # eval 2: best raw WER (20.0) but loop 9% > limit -> ineligible, not best
    assert log[1]["eligible"] is False and log[1]["is_best"] is False
    # eval 3: eligible and better than the standing best (40.0) -> new best
    assert log[2]["is_best"] is True and log[2]["best_clean_macro_wer"] == 35.0


def test_dev_callback_eligibility_boundary_at_limit(tmp_path):
    """loop == limit is ELIGIBLE (<=, not <)."""
    seq = iter([_metrics(50.0, loop=0.0), _metrics(30.0, loop=5.0)])
    cb = DevEvalCallback(eval_fn=lambda m: next(seq),
                         log_path=str(tmp_path / "dev_log.jsonl"), patience=3)
    cb.on_save(args=None, state=None, control=_Ctl(), model=None)
    cb.on_save(args=None, state=None, control=_Ctl(), model=None)
    log = [json.loads(l) for l in
           (tmp_path / "dev_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[1]["eligible"] is True and log[1]["is_best"] is True


def test_dev_callback_per_dialect_logged(tmp_path):
    m = _metrics(30.0)
    m["per_dialect"] = {"Najdi": {"clean_macro_wer": 25.0, "n_clean": 40},
                        "Hijazi": {"clean_macro_wer": 40.0, "n_clean": 15}}
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
