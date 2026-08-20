"""Validation-specific core-dialect baselines for checkpoint selection.

Established 2026-08-19 by running stock openai/whisper-large-v3 (zero-shot,
fp16, CUDA) on the EXACT materialized validation set
(data/sada_training/val: 3,423 clips) with the eval_utils metric recipe.

REGIME CAVEAT (found 2026-08-20): the commit and the original docstring
claimed "greedy", but no script or per-clip artifact survives to prove it
— data/sada_training/val/zero_shot_baseline.json contains ONLY aggregates,
and the surviving spike recipe omitted num_beams, which the HF ASR
pipeline defaults to 5 (transformers 4.57.6). The v1 numbers below are
therefore regime-AMBIGUOUS. They are superseded by a v2 recompute with
sawti.training.eval_checkpoint (explicit greedy, batched FP16, full
per-clip records in zero_shot_baseline_v2.json) so baseline and candidate
evaluation share the decoding regime by construction.

These remain the ONLY baselines selection may use — never the test-derived
spike numbers (Addendum 4), which would reintroduce the contamination the
experimental structure removed.
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
