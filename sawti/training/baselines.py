"""Validation-specific core-dialect baselines for checkpoint selection.

Established 2026-08-19 by running stock openai/whisper-large-v3 (zero-shot,
fp16, CUDA, greedy, chunked pipeline) on the EXACT materialized validation
set (data/sada_training/val: 3,423 clips) with the standard eval_utils
recipe. These are the ONLY baselines DevEvalCallback may use — never the
test-derived spike numbers (Addendum 4), which would reintroduce the
contamination the experimental structure removed.

Full per-clip results: data/sada_training/val/zero_shot_baseline.json
(gitignored, local operator artifact — numbers pinned here in code).
"""

VALIDATION_BASELINES = {
    "Najdi": 46.685,
    "Hijazi": 49.928,
    "Khaliji": 56.399,
}

# Context (reported, not used for selection):
#   overall clean macro: 49.0%
#   all-valid macro: 154.5% | corpus: 65.5%
#   loop rate: 2.8% | degenerate: 18.0%
#   per-dialect n_clean: Najdi 1861, Hijazi 408, Khaliji 538
