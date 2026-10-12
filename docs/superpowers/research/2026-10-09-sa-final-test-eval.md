# SA Fine-tune — Formal Full-Test-Slice Evaluation (milestone-final)

**Date:** 2026-10-09 (run 01:10–03:41, unattended after GPU-idle confirmation)
**Protocol:** spec §6 final eval — full Saudi-filtered SADA test slice
(3,639 clips), frozen evaluator regime (FP16 base + attached adapter,
explicit greedy, batch 4, attention-mask plumbed), identical regime for
the zero-shot side by construction. Artifacts carry full per-clip rows
and regime provenance.

## Results (clean macro WER, %)

| Dialect | Zero-shot | Fine-tune | Δ | n_clean |
|---|---|---|---|---|
| Najdi | 56.1 | **43.9** | −12.1 | 1,304 |
| Hijazi | 53.0 | **41.0** | −12.0 | 661 |
| Khaliji | 56.8 | **44.2** | −12.6 | 943 |
| **Pooled clean macro** | **55.6** | **43.4** | **−12.3** | 3,639 |

Robustness views: all-valid macro 86.9 → **51.9**; corpus WER 62.0 →
**38.0**. Loop rate 2.4 → **2.7 %**; degenerate ~20 % (data property).
Selection score 43.0, eligible, no dialect-regression guard failures.

## Verdicts against spec §3 criteria

| Criterion | Result |
|---|---|
| Clean macro WER ≤ 20 % | ❌ **Miss** — 43.4 %, ~23 pp over |
| Loop rate < 5 % | ✅ **Pass** — 2.7 %, comfortable margin |
| No dialect worse than zero-shot | ✅ **Pass** — every dialect −12 pp |

## Notes

- Prediction check (preview-based, 360 clips): predicted 42.5 ± 2 pp —
  actual 43.4 %, inside the interval; every qualitative call held.
- The preview's Khaliji/Hijazi near-tie resolved: Hijazi is the
  fine-tune's best test dialect (41.0), Khaliji the worst (44.2),
  consistent with validation ordering.
- All-valid-macro divergence (86.9 vs 51.9) shows the fine-tune's
  largest effect is on the *hard/degenerate tail*, exactly where the
  zero-shot model hallucinated.
- Iterate-or-park: the winner was the last checkpoint at max-steps
  (curve not saturated). Realistic next moves, in cost order: more
  QLoRA steps; pseudo-labeling Saudi audio (semi-supervised); the
  self-supervised from-scratch encoder path (license-clean IP).

## Artifacts

- `checkpoints/sa_qlora_run2/eval/test_full_ft.json` (fine-tune)
- `data/sada_training/test/zero_shot_full.json` (zero-shot)
- Preview slice + results: `data/sada_training/test/manifest_preview.jsonl`,
  `checkpoints/sa_qlora_run2/eval/test_preview_ft.json`,
  `data/sada_training/test/test_preview_zeroshot.json`
