"""QLoRA training for Saudi Whisper (spec §2) — SA Tasks 4-5.

Run (OPERATOR):
  uv run python -m sawti.training.train_qlora \\
    --train data/sada_training/train --out checkpoints/sa_qlora_run2 \\
    --flavor qlora

TRAIN_FLAVOR decided by the Task 0 probe: qlora (bitsandbytes 0.50.1
verified on this workstation). The lora fallback (fp16 + adamw_torch_fused)
remains selectable for environments where the probe fails.

Run 2 contract (reviewer-approved 2026-08-20): training is DECOUPLED from
validation. This module only trains and saves sparse adapter checkpoints
(save_steps=2000, all retained: 2k/4k/6k/8k/10k at max_steps=10000).
Checkpoint selection happens post-hoc via sawti.training.eval_checkpoint
(FP16 base + attached adapter, explicit greedy decoding, batched) —
Run 1 proved live per-clip 4-bit evaluation inside on_save costs hours
per checkpoint. compute_selection stays here as the shared regime both
sides agree on.
"""
from __future__ import annotations

from pathlib import Path

from transformers import TrainerCallback

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
        # Run 2: sparse checkpoints, ALL retained (None — save_total_limit
        # would delete early ones). Selection is post-hoc; training past the
        # eventual best checkpoint is fine as long as the adapters survive.
        save_steps=2000,
        save_total_limit=None,
        eval_strategy="no",
        optim=optim,
        gradient_checkpointing=True,
        per_device_eval_batch_size=8,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        save_safetensors=True,
    )


class SetEpochCallback(TrainerCallback):
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

    # FAIL CLOSED: every core dialect MUST have a baseline. An empty or
    # partial baseline map silently disabling the regression guards is a
    # wiring bug, not a valid configuration.
    missing = [d for d in CORE_DIALECTS if d not in baselines]
    if missing:
        raise ValueError(
            f"compute_selection: missing baselines for {missing}; "
            f"import sawti.training.baselines.VALIDATION_BASELINES "
            f"(all three core dialects required)"
        )
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
        if wer > baselines[d] + dialect_tolerance_pp:
            guard_fail.append({"dialect": d, "wer": wer,
                               "baseline": baselines[d],
                               "exceeds_by_pp": wer - baselines[d]})
    eligible = loop_ok and not guard_fail and len(dialect_wers) == len(CORE_DIALECTS)
    score = sum(dialect_wers) / len(dialect_wers) if dialect_wers else float("inf")
    return {"eligible": eligible, "selection_score": score,
            "guard_fail": guard_fail, "loop_ok": loop_ok}


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
    p.add_argument("--out", required=True)
    p.add_argument("--flavor", default="qlora", choices=["qlora", "lora"])
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--base", default="openai/whisper-large-v3")
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

    # Training reads the CORE manifest view (Najdi/Hijazi/Khaliji only).
    # Shamali/Janubi live in the diagnostic view and are NEVER seen
    # during optimization or checkpoint selection. Missing core view =
    # wiring error -> fail loudly, never fall back to full manifest.
    core_manifest = Path(a.train) / "manifest_core.jsonl"
    if not core_manifest.exists():
        raise FileNotFoundError(
            f"{core_manifest} not found — run "
            f"sawti.training.data_prep.carve_manifests first; refusing "
            f"to silently train on the full (uncarved) manifest")
    train_ds = SadaDataset(a.train, augment_enabled=True, seed=42,
                           manifest_name="manifest_core.jsonl")
    targs = build_training_args(a.out, flavor=a.flavor,
                                max_steps=a.max_steps)

    # Run 2: training ONLY. No evaluation callback — checkpoint selection
    # is post-hoc via sawti.training.eval_checkpoint on the saved sparse
    # adapters (FP16 + explicit greedy + batched).
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds,
        data_collator=WhisperCollator(
            processor, decoder_start_token_id=decoder_start_token_id),
        callbacks=[SetEpochCallback(train_ds)],
    )
    trainer.train()
    model.save_pretrained(str(Path(a.out) / "last_adapter"))
    processor.save_pretrained(str(Path(a.out) / "last_adapter"))


if __name__ == "__main__":
    main()
