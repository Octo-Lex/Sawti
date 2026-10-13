# Code-Switch Failure Analysis — Where Quality Dies

**Date:** 2026-10-12 (analysis of existing artifacts — no new runs)
**Input:** the 28 code-switched clips of the synthetic eval set
(en-ar 15, en-fr 7, mixed 6) across three runs: baseline (m4t graph),
SA-primary, eng-strict gate. Joined with the kit's source scripts
(switch positions known by construction). Per-clip rows:
`outputs/cs_failure_rows.json`.

## The taxonomy (baseline numbers, three runs agree)

### Mode 1 — semantic divergence (DOMINANT: 14/28 baseline; 18/21 SA-mode)

The engine emits clean English with the WRONG content — dropped
clauses, merged meanings, invented continuations (chrF 24-58 on clips
whose references are exact). No wrong script, no loops, nothing the
gate can see: **zero low-confidence flags across all 28 clips in every
run.** The confidence signal is structurally blind to this mode.

- Not ASR-bound (experiment A: swapping in the best Arabic ASR changed
  nothing).
- Not switch-bound in general: the en-fr slice (same switching, no
  Arabic) averages 68 chrF with no cluster below 47. **The damage is
  Arabic-switching specifically** — consistent with the M4T-Arabic
  weakness documented in the holdout study.
- SA-mode is the purest exhibit: its 0.8 heuristic confidence rejects
  nothing (0 fallbacks), so 18/21 clips are pure primary divergence.

### Mode 2 — wrong-script leak at exhaustion (SECONDARY: 4-5/28)

Untranslated Arabic spans in English output. Every leak traces to the
`asr_mt` fallback lane: whisper transcribes the Arabic span (0.8
confidence), and when the chunk exhausts, **best-effort selection picks
the highest-confidence candidate regardless of script** — the Arabic
transcript beats the English candidates on confidence.

- The eng-strict experiment PROVES the mechanism: strict rejection
  pushes chunks INTO exhaustion, and the leak count went UP (5 vs 4)
  — the flagged best-effort is still the Arabic transcript. Gate
  strictness cannot fix a selection rule that is script-blind.

### Mode 3 — segmentation splits at switches (MINOR: 5 suspects)

Some clips segment at or near switch points (TTS pauses at commas
coincide with switches), but with 1-3 segments on 8-14 s clips this is
not the primary damage path.

## What follows

1. **Fixable now (small, testable):** script-aware best-effort
   selection in `FallbackHandler` — at exhaustion, prefer candidates
   in the target script over higher-confidence wrong-script ones. Kills
   Mode 2 outright.
2. **Not fixable by policy (Mode 1):** the dominant mode needs better
   translation of Arabic-mixed audio — the options are a fine-tuned or
   from-scratch S2TT/MT stack (the synthetic code-switched-conversation
   data recipe is the data path), or a cross-check quality signal
   (two independent translations agreeing) which costs a second decode.
   Confidence-gating cannot see this failure class; no threshold will.
3. **SA-mode gate blindness (operational hazard):** the 0.8 heuristic
   means no SA primary is ever rejected. Acceptable while the mode is
   explicitly Arabic-primary; worth a real signal before it carries
   production weight.

## Verdict

The core-promise gap decomposes into one dominant model-quality problem
(Arabic-mixed translation divergence, invisible to the gate) and one
selection-rule bug (script-blind best-effort). The second is an
afternoon; the first is the actual product work the from-scratch
direction exists for.
