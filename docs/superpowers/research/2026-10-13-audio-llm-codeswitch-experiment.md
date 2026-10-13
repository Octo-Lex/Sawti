# Audio-LLM vs S2TT on Code-Switched Speech — Experiment Results

**Date:** 2026-10-12/13
**Question (from the failure analysis, Mode 1):** does an instruction-
steered audio-native LLM translate Arabic-mixed speech more faithfully
than the dedicated S2TT engine? This is the first intervention tested
against the dominant failure mode (clean English, wrong content —
invisible to the gate).

## Setup

- Condition: google/gemma-4-E2B-it (Apache-2.0, audio-native
  multimodal, ~10.3 GB bf16), prompted translate-to-English with an
  explicit code-switch instruction, greedy decoding. Isolated env
  (transformers 5.x; the project env is pinned <5 for the frozen eval
  regimes). Ran on the 28 code-switched clips of the synthetic set,
  same references, same chrF/WER as the baseline graph.
- Runtime: 1,927 s for 28 clips ≈ **69 s per ~10 s clip (RTF ≈ 7)** on
  the 3080 Ti — offline-feasible, live-prohibitive.

## Results (chrF, higher is better)

| Slice | Audio-LLM | M4T graph (baseline) | Δ |
|---|---|---|---|
| en-ar | **49.0** | 44.3 | +4.7 |
| en-fr | **78.3** | 68.1 | +10.2 |
| mixed | **45.9** | 36.5 | +9.4 |
| **CS overall** | **55.7** | 48.6 | **+7.1** |

Every slice improves; both failing slices cross the ≥45 bar for the
first time; the three-language slice gains the most.

## Exhibits

- **Turnaround (enar_06, 27.1 → 80.5):** the clip whose baseline output
  leaked diacritized Arabic now translates the Arabic span correctly
  and idiomatically ("…just let me know when you'll visit us for Eid").
- **Regression (mix_05, 31.7 → 23.6):** complete confabulation — an
  invented diving story for a car-repair reference. The LLM failure
  mode is *fluent hallucination on misheard audio*, not graceful
  degradation. The hard tail (mix_05 23.6, enar_03 26.6, enar_15 29.5)
  persists.

## Conclusions

1. **The hypothesis is confirmed directionally**: instruction steering
   beats the dedicated S2TT pipeline on exactly the failure mode the
   gate cannot see. First measured movement on Mode 1.
2. **Not a drop-in replacement**: RTF ≈ 7 (no M3 path), untuned single
   prompt, hallucination tail persists, and the eval is synthetic-clean
   (optimistic). A production comparison needs the mic-recorded set.
3. **Licensing fact worth recording:** the audio-LLM is Apache-2.0 —
   commercially usable, unlike the CC-BY-NC engines currently doing the
   core translation. Under the project's from-scratch direction this
   remains an experimental condition, not a product dependency, but it
   changes the license calculus if that direction ever revisits.
4. **Design input for the from-scratch model (now with our own data):**
   instruction steerable at inference, audio-native, strong semantic
   grounding — and it needs an anti-confabulation check (the failure
   analysis's cross-check-agreement idea applies to LLM outputs even
   more than to S2TT).

## Artifacts

- `outputs/gemma4_cs_results.json` (per-clip hypotheses, refs, chrF/WER)
- `outputs/gemma4_experiment_log.txt`
- Baseline comparison data: `outputs/eval-eng-tts-baseline.json`,
  `outputs/cs_failure_rows.json`
