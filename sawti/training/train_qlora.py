"""QLoRA training for Saudi Whisper (spec §2) — SA Tasks 4-5.

Run (OPERATOR):
  uv run python -m sawti.training.train_qlora \\
    --train data/sada_training/train --dev data/sada_training/val \\
    --out checkpoints/sa_qlora --flavor qlora

TRAIN_FLAVOR decided by the Task 0 probe: qlora (bitsandbytes 0.50.1
verified on this workstation). The lora fallback (fp16 + adamw_torch_fused)
remains selectable for environments where the probe fails.

Selection regime (DevEvalCallback): a checkpoint is ELIGIBLE only when
its n-gram loop-rate <= loop_limit_pct (default 5); among eligible
checkpoints the lowest clean macro WER wins. All four headline metrics
(clean macro, all-valid macro, all-valid corpus, loop-rate) plus
per-dialect metrics are logged on every evaluation. Early stop after
`patience` consecutive non-improving evaluations.
"""
from __future__ import annotations

import json
from pathlib import Path

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def build_lora_config():
    """The adopted recipe (spec §2.2): r=8, α=16, dropout 0.05, attention
    projections + FFN in both encoder and decoder.

    task_type is DELIBERATELY UNSET (generic PEFTModel path). PEFT's
    Seq2SeqLM wrapper passes input_ids into WhisperForConditionalGeneration,
    whose speech input is input_features — a documented failure mode.
    Do NOT "fix" this to SEQ_2_SEQ_LM."""
    from peft import LoraConfig

    return LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, target_modules=LORA_TARGETS,
    )


def build_training_args(out_dir: str, flavor: str = "qlora",
                        max_steps: int = 10000):
    """TrainingArguments per flavor. Both hold effective batch 16.

    qlora: NF4 4-bit + paged_adamw_8bit, 8 x 2 (Task 0 decision).
    lora:  fp16 + adamw_torch_fused, 4 x 4 (probe-failure fallback)."""
    from transformers import TrainingArguments

    if flavor == "qlora":
        optim, bs, accum = "paged_adamw_8bit", 8, 2
    elif flavor == "lora":
        optim, bs, accum = "adamw_torch_fused", 4, 4
    else:
        raise ValueError(f"unknown flavor {flavor!r} (expected qlora|lora)")
    return TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=accum,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        max_steps=max_steps,
        logging_steps=50,
        save_steps=500,
        save_total_limit=4,
        eval_strategy="no",
        optim=optim,
        gradient_checkpointing=True,
        per_device_eval_batch_size=8,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        save_safetensors=True,
    )


class SetEpochCallback:
    """Advances the dataset's augmentation stream per epoch so
    (seed, epoch, index) determinism actually varies across epochs in
    training — not merely in isolation tests."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kw) -> None:
        if state is not None and getattr(state, "epoch", None) is not None:
            self.dataset.set_epoch(int(state.epoch))


CORE_DIALECTS = ("Najdi", "Hijazi", "Khaliji")


def compute_selection(result: dict, baselines: dict,
                      loop_limit_pct: float = 5.0,
                      dialect_tolerance_pp: float = 3.0) -> dict:
    """The balanced selection regime (locked after review):

    eligible = loop_pct <= limit AND every core dialect metric exists
               and is finite AND each dialect WER <= baseline + tolerance

    selection_score = unweighted mean of the three core-dialect
                      clean_macro_wer values (NOT the overall clean macro,
                      which inherits the validation population skew).

    baselines come from the stock model's ZERO-SHOT run on the exact
    materialized validation set — never from the test-derived spike."""
    import math

    loop_ok = result["loop_pct"] <= loop_limit_pct
    guard_fail = []
    dialect_wers = []
    for d in CORE_DIALECTS:
        entry = result.get("per_dialect", {}).get(d)
        wer = entry.get("clean_macro_wer") if entry else None
        if wer is None or not math.isfinite(wer):
            guard_fail.append({"dialect": d, "reason": "missing_or_nonfinite"})
            continue
        dialect_wers.append(wer)
        if d in baselines and wer > baselines[d] + dialect_tolerance_pp:
            guard_fail.append({"dialect": d, "wer": wer,
                               "baseline": baselines[d],
                               "exceeds_by_pp": wer - baselines[d]})
    eligible = loop_ok and not guard_fail and len(dialect_wers) == len(CORE_DIALECTS)
    score = sum(dialect_wers) / len(dialect_wers) if dialect_wers else float("inf")
    return {"eligible": eligible, "selection_score": score,
            "guard_fail": guard_fail, "loop_ok": loop_ok}


class DevEvalCallback:
    """Evaluates on the VALIDATION dev set at each save point.

    Selection regime (locked): eligibility requires loop-rate <= limit
    AND no core-dialect regression beyond baseline+tolerance; the
    RANKING metric is the unweighted three-dialect selection_score,
    deliberately NOT the overall clean macro (population skew). All
    four headline metrics + per-dialect + selection diagnostics are
    logged every evaluation. Early stop after patience consecutive
    non-improving evaluations.

    dev is the VALIDATION split — NEVER test-derived data."""

    def __init__(self, eval_fn, log_path: str, patience: int = 3,
                 loop_limit_pct: float = 5.0,
                 baselines: dict | None = None,
                 dialect_tolerance_pp: float = 3.0) -> None:
        self.eval_fn = eval_fn
        self.log_path = log_path
        self.patience = patience
        self.loop_limit_pct = loop_limit_pct
        self.baselines = baselines or {}
        self.dialect_tolerance_pp = dialect_tolerance_pp
        self.best = float("inf")
        self.regress = 0
        self.eval_index = 0

    def _log(self, record: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_save(self, args, state, control, model=None, **kw) -> None:
        self.eval_index += 1
        result = self.eval_fn(model)
        sel = compute_selection(result, self.baselines,
                                self.loop_limit_pct, self.dialect_tolerance_pp)
        eligible = sel["eligible"]
        score = sel["selection_score"]
        is_best = eligible and score < self.best
        if is_best:
            self.best, self.regress = score, 0
        else:
            self.regress += 1
        stop = self.regress >= self.patience
        if control is not None and stop:
            control.should_training_stop = True
        self._log({
            "eval_index": self.eval_index,
            "clean_macro_wer": result["clean_macro_wer"],
            "all_valid_macro_wer": result["all_valid_macro_wer"],
            "all_valid_corpus_wer": result["all_valid_corpus_wer"],
            "loop_pct": result["loop_pct"],
            "selection_score": score,
            "eligible": eligible,
            "guard_fail": sel["guard_fail"],
            "loop_ok": sel["loop_ok"],
            "baselines": self.baselines,
            "is_best": is_best,
            "best_selection_score": self.best,
            "consecutive_regressions": self.regress,
            "stop": stop,
            "per_dialect": result.get("per_dialect", {}),
        })
        print(f"[dev-eval {self.eval_index}] score {score:.1f} "
              f"loop {result['loop_pct']:.1f}% "
              f"{'ELIGIBLE' if eligible else 'INELIGIBLE'} "
              f"best {self.best:.1f} regress {self.regress}/{self.patience}"
              f"{' STOP' if stop else ''}"
              + (f" guards={[g['dialect'] for g in sel['guard_fail']]}"
                 if sel["guard_fail"] else ""))


def main() -> None:
    import argparse

    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import (Trainer, WhisperForConditionalGeneration,
                              WhisperProcessor)

    from sawti.env import load_env
    load_env(override=True)  # operator entry edge (see env.py policy)
    from sawti.training.dataset import SadaDataset, WhisperCollator

    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True,
                   help="materialized TRAIN split dir (manifest-driven)")
    p.add_argument("--dev", default="data/sada_training/val",
                   help="checkpoint-selection dev set — MUST be the "
                        "validation split, never test-derived data")
    p.add_argument("--out", required=True)
    p.add_argument("--flavor", default="qlora", choices=["qlora", "lora"])
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--base", default="openai/whisper-large-v3")
    p.add_argument("--loop-limit-pct", type=float, default=5.0)
    a = p.parse_args()

    dtype = torch.float16
    load_kw = dict(dtype=dtype)
    if a.flavor == "qlora":
        from transformers import BitsAndBytesConfig

        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = WhisperForConditionalGeneration.from_pretrained(a.base, **load_kw)
    if a.flavor == "qlora":
        model = prepare_model_for_kbit_training(model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model = get_peft_model(model, build_lora_config())
    model.print_trainable_parameters()
    # WhisperCollator REQUIRES the decoder-start token id — sourced from
    # the model config (NOT the tokenizer BOS; they differ in Whisper).
    decoder_start_token_id = model.config.decoder_start_token_id

    # Multilingual Whisper contract: the processor MUST be constructed
    # with language+task so training labels carry the Arabic/transcribe
    # prefix tokens (matching dev inference's generate_kwargs).
    processor = WhisperProcessor.from_pretrained(
        a.base, language="arabic", task="transcribe")
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    train_ds = SadaDataset(a.train, augment_enabled=True, seed=42)
    targs = build_training_args(a.out, flavor=a.flavor,
                                max_steps=a.max_steps)

    def dev_eval_fn(m) -> dict:
        from transformers import pipeline

        from sawti.training.eval_utils import aggregate, run_eval

        # NO explicit device: the QLoRA path assigns an Accelerate
        # device_map automatically, and Pipeline REJECTS a device when
        # the model has one. Omitting lets both paths resolve normally.
        asr = pipeline("automatic-speech-recognition", model=m,
                       tokenizer=processor.tokenizer,
                       feature_extractor=processor.feature_extractor,
                       torch_dtype=dtype, chunk_length_s=30)
        rows = run_eval(lambda w: asr(w, generate_kwargs={
            "language": "arabic", "task": "transcribe"})["text"].strip(),
            a.dev)
        return aggregate(rows)

    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds,
        data_collator=WhisperCollator(
            processor, decoder_start_token_id=decoder_start_token_id),
        callbacks=[
            SetEpochCallback(train_ds),
            DevEvalCallback(dev_eval_fn,
                            str(Path(a.out) / "dev_log.jsonl"),
                            loop_limit_pct=a.loop_limit_pct),
        ],
    )
    trainer.train()
    model.save_pretrained(str(Path(a.out) / "last_adapter"))
    processor.save_pretrained(str(Path(a.out) / "last_adapter"))


if __name__ == "__main__":
    main()
