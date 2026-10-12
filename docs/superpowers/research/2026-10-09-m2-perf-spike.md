# M2 Perf Spike — Threshold & Precision Matrix (Task 7 unblock study)

**Date:** 2026-10-09
**Script:** `spikes/m2_perf_spike.py` (matrix runner; one combo per process)
**Data:** `outputs/mic_recording.wav` (24 s real BT-mic speech, EN+AR mix,
recorded in today's Task 7 session) + `outputs/test01.wav` cross-check
**Raw results:** `outputs/m2_perf_spike_results.jsonl`
**Question:** can the unchanged production graph run live at RTF < 1 on the
RTX 3080 Ti, and what knob gets it there?

## Matrix (5 chunks; wall = pipeline processing time, load excluded)

| dtype | threshold | RTF | primary pass | retry | exhausted | engine Σms | Whisper Σms |
|---|---|---|---|---|---|---|---|
| fp32 | 0.40 (current) | 1.58 | 2/5 | 2 | 1 | 6 588 | 22 212 |
| fp32 | 0.35 | 1.46 | 3/5 | 1 | 1 | 5 881 | 21 292 |
| fp32 | **0.30** | **0.19** | **5/5** | 0 | 0 | 4 035 | 0 |
| fp16 | 0.40 | 1.49 | 2/5 | 2 | 1 | 6 039 | 20 527 |
| fp16 | 0.35 | 1.47 | 3/5 | 1 | 1 | 6 478 | 20 622 |
| fp16 | 0.30 | 0.25 | 5/5 | 0 | 0 | 5 344 | 0 |

## Findings

1. **The confidence threshold is the whole story.** At 0.30 every chunk
   passes primary — no beam retries, no fallback storms, no 20 s Whisper
   cliff. RTF 0.19 (fp32). At 0.40 the same audio rejects 3/5 primaries;
   one chunk exhausts the stack and its `asr_mt` stage alone costs ~21 s,
   which alone exceeds the audio's total duration.
2. **fp16 buys nothing here** — engine Σ is equal or slightly WORSE than
   fp32 at every threshold (no optimized kernels for this architecture on
   torch 2.13/3080 Ti). Recommendation: do not change load dtype; the
   proposal is dropped. (VRAM relief is real but not the binding
   constraint: fp32 M4T + lazy Whisper fits in 12 GB.)
3. **M4T score-confidence is poorly calibrated for gating.** Low scores
   accompany fine outputs (c0 conf 0.37 → "What is the information?" ≈
   the 0.40-retry output), while the beam-5 retry can produce a
   HIGH-confidence hallucination (c1: conf 0.693 Europarl doubled-clause
   accepted at 0.40). Confidence ranks decoding stability, not truth.
4. **Quality caveat on 0.30 (Q4):** the Arabic chunk c3 — where 0.40
   exhausted and emitted a flagged loop — passes primary at 0.30 with a
   plausible-but-wrong English sentence ("The problem is, it's not just
   the brain itself.", conf 0.334), **unflagged**. Lowering the
   threshold converts visible failure into silent mistranslation on
   chunks like this one. This is the central honest tradeoff of the
   unblock.
5. **One exhaustion is survivable live:** the asr_mt cliff costs ~21 s of
   backlog; the capture queue caps at 30 s. A single exhaustion followed
   by speech/silence drains; repeated exhaustion (the 0.40 behavior) is
   what killed the live session.

## Recommendation

- **Task 7 re-smoke on `confidence_threshold: 0.30`** via a session
  config file (no committed default changed) — `config/task7smoke.yaml`.
  Expected live RTF ≈ 0.2–0.5 with headroom for one fallback event.
- **Do not adopt fp16.**
- **The 0.40→0.30 production default is an M4 decision**, not an M2 one:
  it needs the §7.1 self-recorded eval set (which does not exist yet) to
  measure chrF impact, and it interacts with Q2/Q3 from the Task 7
  evidence doc. Until then, 0.30 is a smoke-session configuration.
- Follow-up for M4: make the `asr_mt` cliff cheaper (smaller Whisper,
  batching) and/or re-rank by a better-calibrated signal than raw score
  confidence.

## Addendum (same day): live re-smokes & the real bottleneck

The 0.30 re-smoke still died (input overflow via a mid-session lazy
fallback load), and the pre-warmed re-smoke #3 survived to 219 s then
died by queue-full (124 s emitted, ~65 s with a full queue). Full
isolation experiments in the Task 7 evidence doc §6 — every single
hypothesis was excluded (fp16/priority/thermal/VRAM/duration/pacing;
the paced replay runs with ZERO overhead beyond real time). Root cause:
**fallback cost clusters** (retry → rechunk → whisper+T2TT at 10–30×
happy-path cost each) filling the 30 s queue during speech-dense
marginal regions. T2TT on a 40×-loop transcript costs 2.8 s vs 0.8 s
collapsed — the Q1 n-gram collapse applied at the asr_mt seam is a
measured, principled mitigation, but the structural fix is cheaper
fallback models plus a load-aware fallback budget (M4 scope).
