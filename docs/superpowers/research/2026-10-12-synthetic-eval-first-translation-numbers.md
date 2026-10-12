# First System-Level Translation Numbers (synthetic eval set)

**Date:** 2026-10-12
**Set:** the 62-clip read-aloud kit (eval/clips_v1.json) rendered by a
local offline neural TTS engine (data/eval_tts_v1/, 9.1 min; dominant-
language voice per slice; provenance in _provenance.json). References
exact by construction. **This is an optimistic bound**: clean studio
acoustics, single voice, accented cross-language rendering on the
code-switched slices. The microphone recording (spec 7.1) remains the
authoritative measure; this baseline is deterministic and repeatable.

## Results (production graph, target=eng)

| Slice | n | Macro WER | mean chrF | fallbacks fired |
|---|---|---|---|---|
| en (verbatim) | 12 | 6.7 % | **91.2** | 0 |
| fr → en | 10 | 21.7 % | **79.1** | 0 |
| ar → en | 12 | 46.9 % | **62.2** | 0 |
| en-ar → en | 15 | 66.7 % | **44.3** | 4 |
| en-fr → en | 7 | 37.6 % | **68.1** | 0 |
| mixed → en | 6 | 83.6 % | **36.5** | 3 |
| **overall** | 62 | 42.3 % | **64.4** | 7 |

Zero loops; every clip produced output (coverage 100 %).

## Findings

1. **The happy path works**: English verbatim at chrF 91, French→English
   at 79 — comfortably above the spec's ≥45 per-language bar.
2. **Arabic→English is the weak monolingual lane** (62.2), consistent
   with the documented M4T-on-Arabic weakness (the sawti-sa mode exists
   for exactly this, unmeasured here in the translate direction).
3. **Code-switched slices fall below the bar** — en-ar 44.3 and mixed
   36.5. First direct quantification of the core-promise gap. Observed
   failure modes: untranslated Arabic spans surviving into the output
   (soft script-mismatch for eng lets them pass — the Q3/Q5 family) and
   occasional hallucinated tails.
4. The gate is active exactly where it should be: all 7 fallback events
   are on code-switched slices.

## Caveats

Synthetic acoustics (optimistic); TTS code-switch rendering is accented
single-voice (stresses segmentation differently than human switching);
single-reference chrF penalizes valid paraphrase; WER is harsh on
translation output — chrF is the primary metric per spec §7.2.

## Companion benchmark (same day): fallback-lane ASR size curve

Zero-shot clean macro WER on the 81-clip real holdout set:

| Model | WER | clips/s |
|---|---|---|
| whisper-large-v3 (SA fine-tune: 24.3) | 31.7 (stock) | 1.8 |
| whisper-medium (current fallback lane) | 46.4 | 2.5 |
| whisper-small | 52.4 | 4.8 |

The current fallback lane (medium) is ~22 pp worse than the SA primary
on dialectal Arabic — escalation from the SA engine to the current
fallback usually degrades quality. M4 implication: the Arabic-product
fallback lane should be SA-based (provider="sa") or larger, and the
size choice trades ~6 pp per size step.

## Artifacts

- `outputs/eval-eng.json` (per-clip rows + traces)
- `data/eval_tts_v1/` (wavs + refs + provenance)
- `data/eval_real/zero_shot_medium.json`, `zero_shot_small.json`

## Addendum (late 2026-10-12): two config experiments + the night's forensics

**Experiment A — sawti-sa in the translate direction** (33-clip ar/enar/
mix subset, SA engine + T2TT): ar 61.2 / enar 42.1 / mix 39.3 chrF vs
the m4t graph's 62.2 / 44.3 / 36.5. A wash on this set: the synthetic
voice is MSA-clean, so the SA model's dialect advantage has nothing to
bite on, and both paths share the same T2TT bottleneck. The mode's
proven value remains real dialectal speech (holdout: 24.3 vs 31.7 WER
verbatim). ar→en translation quality is T2TT-bound, not ASR-bound.

**Experiment B — eng script-mismatch strict** (full 62 clips): overall
64.3 vs 64.4 chrF; enar 44.9 vs 44.3, mix 36.8 vs 36.5. The one-line
config flip does NOT rescue the code-switched slices — surviving
wrong-script output is not dominant-script-mismatched enough to trip
the gate, and the fallback lane is not better on this material. The
code-switch gap is real M4 work (segmentation-level or MT-level), not a
gate-policy toggle.

**Forensics — every failed run this night, root-caused:** the SA-mode
evals crashed natively (torch_cpu.dll 0xC0000005, identical fault
offset twice) with no Python traceback. Windows System log: *low
virtual memory condition — python.exe consuming 44.7 GB commit*.
Machine: 32 GB RAM + 32 GB pagefile = 64 GB ceiling. Mechanism: the
harness's per-clip pipeline rebuild leaked host commit (~44 GB after
~30 model reload cycles), compounded once by my own error — running a
diagnostic probe CONCURRENTLY with an eval (two model-holding
processes). Fixes, all committed:
1. `_shared_component_factory` (cli.py): eval builds engine + provider
   ONCE and injects them per clip — per-clip state semantics intact,
   models resident, ~10x faster evals. Guarded by test.
2. `low_cpu_mem_usage=True` on all five model loaders (mmap weights —
   roughly halves per-process commit).
3. engine_sa no longer passes `return_dict_in_generate`/`output_scores`
   to Whisper's generate (invalid flags for it; heuristic 0.8 confidence
   as the provider lane already did).
4. Process discipline recorded: never two model-holding processes on
   this box without enlarging the pagefile.
Both experiments then completed cleanly. artifacts:
`outputs/eval-eng-tts-sa.json`, `outputs/eval-eng-tts-strict.json`
(baseline preserved at `outputs/eval-eng-tts-baseline.json`).
