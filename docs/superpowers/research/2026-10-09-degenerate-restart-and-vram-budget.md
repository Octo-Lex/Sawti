# In-stream Degenerate Recovery + VRAM Budget (implementation note)

**Date:** 2026-10-09
**Approved by:** project owner
**Inputs:** [Task 7 evidence](2026-10-09-m2-task7-first-smoke-evidence.md) §5–6, [perf spike](2026-10-09-m2-perf-spike.md)

## What changed

1. **Restart stage (fallback.py stage 1.5).** A degenerate primary —
   `empty_output` / `garbage_output` / `repetition_loop` (producer
   malfunction, per `DEGENERATE_CHECKS`) — gets a cheap greedy re-decode
   with `no_repeat_ngram_size=8` (aligned with `loop_detect`'s 1..8-token
   span class) BEFORE any escalation. Primary cost, not beam cost.
   Low-confidence failures skip it (that is the beam retry's job).
   Config: `quality_gate.retries.max_engine_restarts` (default 1; 0
   restores the pre-restart ladder).
2. **Engine restart mode (engine_m4t.py).** `translate(..., restart=True)`
   passes `no_repeat_ngram_size=RESTART_NO_REPEAT_NGRAM`; timing_ms gains
   a `mode` key (`primary` / `conservative` / `restart`).
3. **Degenerate-transcript collapse at the ASR+MT seam (providers.py,
   providers_sa.py).** Whisper phrase-loop transcripts collapse (spec §4.5
   n-gram collapse) BEFORE translation and BEFORE the gate: T2TT decodes
   ~11 tokens instead of ~250 (measured 0.8 s vs 2.8 s), and the gate
   judges content instead of the loop.
4. **Fallback pre-warm (S2TTConfig.prewarm_fallback).** Eagerly loads the
   ASR+MT lane at BUILD time — a mid-session fallback never pays the
   disk→GPU load stall that killed re-smoke #2 (input-overflow via
   callback starvation). Enabled in `config/task7smoke.yaml`.
5. **VRAM budget guard (S2TTConfig.vram_budget_gb).** Build fails loudly
   when torch reserves more than the budget (release-rule pattern: the
   stack must FIT before a session, not thrash during one). CPU-only
   hosts skip the check.

## Validation

- Hermetic suite: **283 passed, 2 skipped** (was 268). New:
  `tests/test_restart_stage.py` (6), `tests/test_build_hardening.py` (6),
  `tests/test_provider_collapse.py` (2), engine restart-kwargs test (1).
- Real graph on `outputs/mic_recording.wav` (the chunk-3 recording,
  default 0.40 threshold): chunk c3 went from **exhausted → asr_mt
  accepted**, emitting the collapsed 11-token transcript instead of the
  40× loop; wall 42.65 s → 39.02 s in a COLD run where c3's 24.5 s
  `asr_mt` is dominated by lazy model loads (prewarm removes those).
- Open observation from that run: c3's accepted output is Arabic-script
  for an eng target (verbatim-passthrough path — most likely no detected
  language from the ASR lane; `script_mismatch` is soft for eng so
  nothing flags it). This is the known Q3/Q5 family; owned by M4 gate
  calibration once the §7.1 eval set exists.

## Explicitly not changed

The escalation ladder's stages, semantics, and decision-log format are
unchanged (restart is additive and first). `config/default.yaml`
untouched: `max_engine_restarts` defaults via the schema, prewarm/budget
default OFF for M1/CPU-portable behavior. The restart stage is a behavior
change on by default — flagged for the reviewer in the PR description.
