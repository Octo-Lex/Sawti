# M2 Task 7 — First Operator Smoke: Evidence & Findings

**Date:** 2026-10-09
**Operator session:** live mic → production graph (`sawti listen --target eng`)
**Verdict:** **TRANSPORT PASS / PERFORMANCE FAIL** — M2 release blocked
pending the performance decision in §5. Surfaced honestly per the plan
("a failure because downstream inference cannot catch up is an M2
performance failure, not permission to drop input").

## 1. Environment (operator evidence)

| Item | Value |
|---|---|
| OS | Windows 11 (NT 10.0.26200) |
| GPU | NVIDIA RTX 3080 Ti, 12 GB (11.8 GB free at session start) |
| Python / stack | 3.11.14, torch 2.13.0+cu126, transformers 4.57.6 |
| Capture | sounddevice 0.5.6, PortAudio V19.7.0-devel |
| Mic (default) | `Headset (soundcore Space One)` — Bluetooth, selected as backend default |
| Models | seamless-m4t-v2-large (cached, fp32), whisper-medium (cached, fp16) |
| Config | `config/default.yaml` unchanged (pause 350 ms, max chunk 12 s, threshold 0.40) |

## 2. Device check (capture-only) — PASS

- Preflight (release rule): default mic opens **1 ch / float32 / 16 kHz** ✅
- Of 14 input endpoints, 6 satisfy the rule: 0, 1, 6, 7, 15 (*soundcore
  Space One*) and 25 (BT hands-free 16 kHz). Realtek onboard mic (19),
  Stereo Mix, Line In, and all 8 kHz BT hands-free endpoints FAIL
  (PaErrorCode -9997) — do not select them.
- Live level check through `MicSource`: peak 0.84 / RMS 0.072 on speech;
  digital-silence noise floor when not speaking; captured == emitted
  exactly; timestamps exact-cursor; 0 overflows.

## 3. Live conversational session — FAIL (performance)

Session ended after **16 s** by the fail-loud contract:

```
[0.03-1.23] Ah, the economy.
[0.93-4.20] If there's no communication,
[3.90-5.60] the whole project.
listen: PortAudio input status: input overflow
[capture] 16.2s captured / 6.0s emitted | queue high-water 103 | input overflows 1 | queue overflows 0
[exit code 1]
```

- Multiple non-empty target-language segments: **yes** (3) — quality of
  those is suspect; all came from the fallback lane.
- Silent audio loss: **none** (fail-loud fired; queue never overflowed).
- Clean shutdown with stats line: **yes** (no traceback).
- RTF: **≈ 2.7** (16.2 s captured, 6.0 s processed) — far above the 1.0 bar.

## 4. Offline replay — root causes (24 s mic recording, same graph)

Same mic audio through `FileSource` (no queue pressure): 5 chunks,
42.65 s wall for 24.0 s audio — **RTF 1.78**. Full decision trace in
`outputs/offline_trace_eng.txt`; recording in `outputs/mic_recording.wav`.

| # | Cause | Evidence |
|---|---|---|
| 1 | **Confidence calibration is marginal on BT-mic speech.** Score-based M4T confidence hovers at the 0.40 gate threshold → primary rejections → every rejection re-decodes with beam-5 (2× cost) | 3/5 chunks rejected `low_confidence` on primary; retry rescued at conf 0.439 / 0.693 |
| 2 | **Fallback storms are expensive.** One exhausted chunk = primary + beam retry + rechunk + Whisper(+T2TT) | chunk c3: `asr_mt` alone **25.9 s** for a 2.9 s chunk |
| 3 | **12 s max-duration chunks × beam-5** | c1: 4.2 s single engine call |
| 4 | **PortAudio input overflow under load** (BT headset; callback starvation during heavy inference) — first observed instance | overflow at t≈16 s with queue at 103/300 |

### Quality bugs surfaced (beyond performance)

- **Q1 — Phrase-loop collapse is unimplemented (spec gap, §4.5).** Spec
  requires collapsing repeated *unigram/n-gram* loops ≥ 3;
  `sawti/text_normalize.py:26 collapse_repeated_loops` handles unigrams
  only. The exhausted chunk c3 emitted a **~40× repeated Arabic phrase**
  verbatim. (The gate's `is_loop` n-gram detector *did* catch the loop;
  only the emitting post-processor cannot collapse it.)
- **Q2 — ×2 repeated-phrase hallucination passes the gate.** Chunk c1
  (12 s force-close) emitted a doubled Europarl-style clause at conf
  0.693; the loop detector requires ≥ 3 repeats by design.
- **Q3 — ara→eng fallback returned Arabic-script text for an eng
  target.** Whisper transcribed Arabic (correctly), M4T T2TT returned
  Arabic-script text on the looped input; `script_mismatch` is `soft`
  for eng so nothing hard-flagged the script. Emitted `low_confidence`
  (correct per "never drop a chunk", but violates the product promise
  for that chunk).

## 5. Decision needed (blocks M2 release)

The M2 contract freezes pipeline semantics; gate/engine tuning is M4
scope. Options:

- **A. Record & defer:** mark Task 7 performance-blocked; M4 owns
  calibration/tuning. Most spec-faithful; M2 release waits on M4.
- **B. Spike the two cheap unblockers** (no semantics change): fp16
  inference for M4T (~2× engine speedup, load-time `dtype` only) +
  threshold-vs-confidence calibration study on recorded audio. If RTF
  < 1 holds, propose the minimal change through review and re-smoke.
- Independent of A/B: **fix Q1** (spec-mandated n-gram collapse —
  small, hermetic, testable).

**Update (same day):** B executed — see
`docs/superpowers/research/2026-10-09-m2-perf-spike.md`. Outcome: the
threshold, not precision, is the lever — fp16 buys nothing; at
`confidence_threshold: 0.30` the same audio runs RTF 0.19 with all
primaries passing. Q1 fixed (n-gram collapse implemented; suite 268
green). Re-smoke scheduled on `config/task7smoke.yaml` (0.30) with the
committed default untouched; the 0.40→0.30 production change remains an
M4 decision (spike finding 4: 0.30 trades flagged failure for silent
mistranslation on Arabic→eng chunks — needs the §7.1 eval set to judge).

## 6. Re-smokes #2 and #3 + isolation experiments (same day)

**Re-smoke #2** (`listen`, 0.30 config): died at 19.4 s — PortAudio
input overflow again; Whisper lane loaded mid-session (a chunk hit a
HARD check, threshold-independent) → ~20 s lazy-load stall → callback
starvation. New quality finding **Q5**: M4T emitted **Greek-script**
text for an eng target (script_mismatch is soft for eng) — passed
unflagged.

**Re-smoke #3** (`spikes/m2_live_prewarm.py`: fallback lane pre-warmed
before capture + audio tee): survived to **219.4 s** (13× longer), zero
input overflows, then died by **queue-full** fail-loud: 124.0 s emitted,
654 queue-overflow events ≈ ~65 s with a full queue. Session audio
preserved: `outputs/live_session2.wav` (124 s — everything the consumer
pulled; the final 65.4 s of captured audio was never pulled and is lost
to the overflow, by design).

**Isolation experiments on the tee'd audio** (all pre-warmed 0.30
config unless noted):

| Experiment | Result | Rules out |
|---|---|---|
| Cold replay (default 0.40) | RTF 0.39 (8 retries + 1 asr_mt included) | "graph is slow" |
| Background-job priority | RTF 0.38, same as foreground | harness priority |
| Pre-warmed replay (VRAM 13.1/12.9 GB!) | RTF 0.35 | VRAM residency/pressure |
| 8-min sustained loop (496 s audio) | RTF 0.33, per-chunk cost flat start→end; GPU 48–67 °C, 1980 MHz | thermal, fragmentation, duration |
| **Real-time PACED replay** (100 ms frames, exact MicSource pacing) | **zero overhead beyond real time**, 2 retries, no exhaustion | frame pacing / arrival interplay |
| T2TT on 40×-loop transcript | 2.8 s (vs 0.8 s collapsed) | single-stage decode stall |

**Unified root cause (evidence-backed):** the happy path has ~3×
headroom (processing ratio ≈ 0.3). Live death is caused by **fallback
cost clusters**: marginal/degenerate chunks escalate (beam retry ~1.5 s
→ rechunk ~2 engines → asr_mt: whisper 5–15 s + T2TT), 10–30× happy-path
cost each. A speech-dense cluster fills the 30 s queue (a single 21–26 s
asr_mt nearly fills it; two in a 30 s window guarantee death), and the
backlog itself extends the cluster — a death spiral that ends in the
fail-loud queue-full. The lazy-load stall (re-smoke #2) is the same
cliff minus pre-warming. NOT caused by: precision, priority, thermal,
VRAM, duration, or pacing — all excluded above.

**Consequences for M2/M4:**
- M2 transport verdict stands: capture edge + fail-loud contract PASS.
- M2 release remains performance-blocked. The unblock is NOT a knob:
  it requires M4-scope work on fallback economics — (a) cheapen the
  asr_mt lane (whisper-small/tiny or faster-whisper INT8), (b) collapse
  loops in asr_mt transcripts BEFORE T2TT (measured 3.5× on that stage;
  the Q1 collapse is the tool), (c) a wall-clock/load-aware fallback
  budget (skip escalation when queue depth is critical — audio is never
  dropped, but escalation effort can be), and (d) threshold calibration
  via the §7.1 eval set.
- Q5 (Greek-script output unflagged for eng) joins Q2–Q4 as M4 gate
  findings.

Q2/Q3 belong to M4 tuning and need the §7.1 self-recorded eval set
(which does not exist yet) for evidence-based decisions.

## Artifacts

- `outputs/mic_recording.wav` — 24 s mic sample (16 kHz mono PCM16)
- `outputs/offline_trace_eng.txt` — full per-chunk decision trace
- Diagnostic scripts (session-temp, not committed): offline trace runner
  (`build_real_pipeline` + `on_decision` → `format_trace`), mic recorder
