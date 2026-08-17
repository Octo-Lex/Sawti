# Sawti — Saudi ASR Milestone (SA) Design Specification

**Status:** Presented for user review
**Date:** 2026-08-17
**Position:** Parallel milestone, executed before M2 (live mic)
**Trajectory:** Research / non-commercial (unlocks SADA, CC BY-NC-SA)
**Evidence base:** `docs/superpowers/research/2026-08-17-sada-whisper-baseline.md` (baseline + two addenda: 4 external models + turbo base evaluated; none beat stock large-v3)

---

## 1. Goal and end state

Train a Saudi-dialect Whisper-large-v3 QLoRA fine-tune on SADA, evaluate it
against the measured 43.3% zero-shot baseline, and integrate it into Sawti as:

1. the real `AsrMtProvider` implementation behind the quality gate
   (Arabic speech → dialectal Arabic text → MT → target language), and
2. an explicit `--engine sawti-sa` primary mode in the CLI.

No language-ID routing is added anywhere (spec §1.4 rule 1 unchanged).

## 2. Training pipeline

### 2.1 Base model — validated by measurement

**Stock `openai/whisper-large-v3`.** Empirically the strongest available
starting point for Saudi: 43.3% clean WER zero-shot vs turbo 50.8%, and four
external fine-tunes all measured worse than stock large-v3 (Addendum 2).

### 2.2 Recipe — adopted from the strongest published config

QLoRA (from dev-ahmedhany's published large-v3-Arabic config): r=8, α=16,
dropout 0.05, targets q/v/k/out_proj + fc1/fc2, paged_adamw_8bit, lr 1e-4,
warmup 0.1, effective batch 16 (adjust grad-accum for 12GB VRAM; reduce
batch before touching the recipe). bf16 compute where supported.

### 2.3 Data

- SADA train split, streamed, filtered to Saudi-dialect labels.
  `data_prep.py` must first run a **dialect-label census** (log counts
  per label across splits) and select the Saudi label set from the
  census — empirically, not by assumption. Najdi, Hijazi, Khaliji are
  confirmed present; any additional Saudi-specific labels the census
  reveals join the filter. The census is logged in the training report.
- Manual bytes-decoding via soundfile (torchcodec is Windows-hostile).
- Materialized to `data/sada_training/` (gitignored), expected ~20–30GB.
- Validation: SADA validation split, same filter, capped for eval speed.
- Transcripts: `cleaned_text` (fallback `text`), normalized with
  `normalize_arabic_for_match` at eval time only — training targets stay
  as-is (undiacritized, matching SADA convention).

### 2.4 Checkpoint selection — mandatory per-dialect

**Checkpoint selection on held-out Saudi dev WER + loop-rate, not train
loss.** Rationale: Addendum 2 showed fine-tuning on the wrong dialect mix
actively hurts the target dialect (oddadmix worse than its own base);
only per-dialect held-out measurement protects against this.

- Dev set: the existing 75-clip Saudi sample (fast, per-dialect).
- Every N checkpoints (N tunable, start 500): dev WER (overall + per
  dialect) and loop-rate; log; keep best-on-dev.
- Early stop if dev WER regresses for 3 consecutive evals.

### 2.5 Budget

Estimate: 1–3 GPU-days on the RTX 3080 Ti; ~10–12GB VRAM at effective
batch 16 with QLoRA. **These are paper-derived, unverified** — first run
may require batch/accum adjustment; the OOM fallback is batch 8 × accum 2.

## 3. Success criteria

**Overall clean WER ≤ 20%** on the Saudi-filtered SADA test slice
(≈ halving 43.3%), **loop-rate < 5%** of clips, **no dialect worse than
zero-shot**. Measured on the full Saudi test slice at milestone end
(the 75-clip sample remains the fast dev metric).

> This bar is the author's recommendation, not yet user-confirmed.
> Missing it triggers iteration (more steps / more data), and will be
> surfaced honestly rather than reframed.

## 4. Integration into Sawti

### 4.1 New components

- `sawti/asr_whisper_sa.py` — `SaudiWhisperAsr implements AsrMtProvider`:
  loads base + LoRA (merged at export time), transcribes chunk audio →
  dialectal Arabic text. Injectable model (fakes for unit tests).
- `sawti/mt_m4t.py` — SeamlessM4T T2TT wrapper (text → target language).
  Reuses the processor already in the stack; NLLB-200 documented as the
  alternative if T2TT quality on dialectal transcripts disappoints.
- `sawti/training/` — `data_prep.py`, `train_qlora.py`, `export_merge.py`
  (§5). Training code is not imported by the runtime package.

### 4.2 Wiring

- `FallbackHandler` unchanged; at last receives a real `asr_mt` provider
  (its graceful-degradation stub retires).
- CLI: `--engine sawti-sa` → segmenter → `SaudiWhisperAsr` → `mt_m4t`
  → `BalancedQualityGate` → `RealPostProcessor`. Existing engines
  (`stub`, `m4t`) unchanged.

### 4.3 Contracts

Frozen M0 contracts (`types.py`, `config.py`, `pipeline.py`) untouched.
All new components follow the Protocol + injectable-model pattern (unit
tests hermetic; real-model tests `@pytest.mark.integration`, opt-in via
`SAWTI_RUN_INTEGRATION=1`). All work via feature branch → PR →
squash-merge (branch protection active).

## 5. Repo layout

```
sawti/
├── asr_whisper_sa.py        # SaudiWhisperAsr (AsrMtProvider impl)
├── mt_m4t.py                # SeamlessM4T T2TT wrapper
├── training/
│   ├── data_prep.py         # stream SADA → filter → materialize train/val
│   ├── train_qlora.py       # QLoRA loop, checkpointing, dev WER eval
│   └── export_merge.py      # merge adapter → shippable model dir
├── cli.py                   # + sawti-sa engine mode (modify)
tests/                       # hermetic unit tests for both components
data/sada_training/          # gitignored training data
```

Training dependencies (`peft`, `bitsandbytes`/paged-optimizer stack,
`evaluate`-free — reuse jiwer) added to the dev group only, GPU overlay
stays local (uv-local pattern, unstaged cu126 block).

## 6. Evaluation protocol & artifacts

- Final eval: full Saudi-filtered SADA test slice, same normalization as
  all spike evals; report per-dialect clean WER + loop-rate, side-by-side
  with zero-shot large-v3.
- Artifacts: merged model + LoRA adapter on HuggingFace under
  CC BY-NC-SA-compatible terms (SADA-derived), model card stating data
  lineage; research report continuing the spike document.

## 7. Risks and honest unknowns

1. **VRAM/time estimates unverified** (paper-derived).
2. **T2TT on dialectal Arabic transcripts unmeasured** — ASR quality may
   outrun translation quality in the fallback lane; measured at
   integration, NLLB swap is the fallback.
3. **SADA `cleaned_text` quality at scale** verified only by 75-clip
   eyeball; data_prep includes a transcript sanity pass.
4. **Success bar unconfirmed by user** (§3) and ambitious vs published
   gains (−7 to −15pp typical; ours needs −23pp).
5. 12GB VRAM is the binding constraint throughout; QLoRA is what makes
   large-v3 feasible at all on this GPU.
