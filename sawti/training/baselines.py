"""Validation-specific core-dialect baselines for checkpoint selection.

V2 (AUTHORITATIVE, established 2026-08-20, reviewer-approved recompute):
stock openai/whisper-large-v3 on the EXACT materialized validation set
(3,423 clips) under the FROZEN evaluator (sawti.training.eval_checkpoint
@ 9ef600f): FP16, greedy (num_beams=1, do_sample=False), language=arabic,
task=transcribe, batch_size=4, attention_mask passed to generate. Full
regime record + per-clip rows: data/sada_training/val/
zero_shot_baseline_v2.json (gitignored local artifact; regime is
reconstructible from config.manifest.sha256 + evaluator_commit).

V1 (HISTORICAL, regime-ambiguous, NOT eligible for guards): the original
2026-08-19 numbers (Najdi 46.685 / Hijazi 49.928 / Khaliji 56.399; overall
clean 49.0, all-valid macro 154.5, loop 2.8) were computed via the HF ASR
pipeline with num_beams omitted — which transformers 4.57.6 defaults to 5
— and no per-clip artifact or script survives to establish the actual
regime. The v2 recompute shifted per-dialect baselines by +2.8 to +4.5pp
and all-valid macro by -79.5pp: the ambiguity was material.

Selection consumes ONLY these v2 numbers — never the test-derived spike
numbers (Addendum 4), which would reintroduce the contamination the
experimental structure removed.
"""

VALIDATION_BASELINES = {
    "Najdi": 51.157,
    "Hijazi": 52.764,
    "Khaliji": 59.872,
}

# V2 context (reported, not used for selection):
#   overall clean macro: 53.08% (v1: 49.02)
#   all-valid macro: 75.00% (v1: 154.54) | corpus: 51.86% (v1: 65.52)
#   loop rate: 1.26% (v1: 2.83) | degenerate: 16.97% (v1: 18.00)
#   per-dialect n_clean: Najdi 1883, Hijazi 408, Khaliji 551
