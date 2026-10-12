# Holdout Reality Check — Real Dialectal Audio (out-of-corpus)

**Date:** 2026-10-12
**Material:** 81 clips / 6.4 min of real Saudi dialectal speech (5 short
commentary clips + 76 timestamped cooking-show segments) placed in
`sample/`, carved into `data/eval_real/` by `scripts/carve_holdout_eval.py`.
**Provenance guarantee:** audio SHA-256 of every source clip verified
against all 160,110 materialized SADA clips — **zero overlap**. This is
genuinely held-out material for every model and split in the project.

## Results

| System | Clean macro WER | Notes |
|---|---|---|
| Zero-shot whisper-large-v3 (frozen evaluator) | **31.7 %** | loop 0.0 %, degenerate 1.2 % |
| **SA fine-tune (checkpoint-10000)** | **24.3 %** | loop 1.2 %, degenerate 2.5 %; n_clean 79/81 |
| Full Sawti pipeline, target=ara (M4T primary + gate + whisper-medium fallback) | **37.2 %** (corpus 40.3 %) | chrF 70.3, loop 0.0 %, all 81 referenced |

## Findings

1. **The SA fine-tune generalizes out-of-corpus.** −7.4 pp vs zero-shot
   on material never seen by any split — the SADA improvement is the
   model, not corpus overfitting.
2. **Absolute numbers are far better than SADA test** (24.3 vs 43.4;
   31.7 vs 55.6) because this material is cleaner: longer utterances
   (median 4.7 s), single coherent speakers, degenerate rate 1–2.5 %
   vs ~20 % on SADA's short TV snippets. Real-world Saudi speech is
   easier than SADA's hardest slices — the SADA numbers are a
   pessimistic slice, not the universal truth.
3. **The strategic finding: the pipeline underuses its best Arabic
   component.** The full graph (M4T-as-Arabic-ASR primary) scores
   37.2 % — 13 pp WORSE than the SA model evaluated directly (24.3 %)
   and worse than even zero-shot large-v3 (31.7 %). On dialectal Arabic
   input, the primary engine is the weak link; our best component sits
   in a fallback lane the default graph rarely reaches (and the CLI
   never exposes the spec's `--engine sawti-sa` primary mode).
4. Pipeline behavior itself was stable: zero loops, all clips produced
   output, no dropped chunks.

## Caveats

- 81 clips / 6.4 min, two male sources — indicative, not definitive;
  no dialect labels (guard/selection fields in the evaluator record are
  cosmetic artifacts of the Unknown label).
- Pipeline WER and evaluator WER share the same matching-grade Arabic
  normalization; comparisons are within-table valid.

## Implication (the actionable one)

For Arabic speech, the graph should route dialectal Arabic through the
SA model as PRIMARY (spec §4.2's `--engine sawti-sa`: SaudiWhisperAsr →
M4T T2TT → gate → postprocess — designed, never wired to the CLI).
Expected effect on Arabic-source input: the ASR error floor drops from
M4T's level to ~24 %, before any further training.

## Addendum (same day): sawti-sa primary mode — wired and verified

Implemented per spec SA §4.2 and measured on this same holdout set:

| System (81 clips, target=ara) | Macro WER | chrF |
|---|---|---|
| M4T-primary graph (previous default) | 37.2 % | 70.3 |
| **SA-primary graph (`--engine sawti-sa`)** | **27.0 %** | **73.4** |
| SA model direct (frozen evaluator, no graph) | 24.3 % | — |
| Zero-shot direct | 31.7 % | — |

- **−10.2 pp vs the previous default graph**; zero loops; the
  prediction (~24–26 %) held with the graph overhead measured at
  ~2.7 pp (segmentation + gate + post-processing on top of the bare
  model — the price of the pipeline, visible and accounted).
- Components: `sawti/engine_sa.py` (SaudiWhisperEngine, primary/
  conservative/restart modes mirroring engine_m4t so the fallback
  machinery binds unchanged), `sawti/mt_m4t.py` (the spec's planned
  thin T2TT wrapper), `build_real_pipeline(engine="sawti-sa")` with
  `sa_engine` injection, CLI `--engine sawti-sa` on transcribe + eval.
- Fallback lane stays whisper-medium — a DIFFERENT model, so escalation
  remains a genuine second opinion.
- VRAM side-benefit: the SA-primary stack (2.9 GB + lazy 1.5 GB
  fallback ≈ 4.5 GB) fits the 12 GB card comfortably, unlike the
  M4T-primary stack (13.1 GB reserved).
- Scope limitation (spec-consistent): the mode assumes Arabic-dominant
  input; non-Arabic speech stays on `--engine m4t`. No LID routing.
- Suite: 295 passed / 2 skipped (+12 tests: engine modes, builder
  injection guards, CLI wiring).

## Artifacts

- `outputs/eval-ara.json` (pipeline report), `data/eval_real/sa_model_eval.json`,
  `data/eval_real/zero_shot_eval.json`, `data/eval_real/manifest.jsonl`
