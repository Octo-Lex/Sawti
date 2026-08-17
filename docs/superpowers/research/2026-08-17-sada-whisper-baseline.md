# Spike Report: Zero-shot Whisper-large-v3 on Saudi Dialects (SADA)

**Date:** 2026-08-17
**Purpose:** Baseline measurement to decide whether a Saudi-dialect
fine-tuning milestone (Path A) is justified.
**Scripts:** `spikes/sada_sample.py`, `spikes/sada_whisper_eval.py`,
`spikes/sada_reanalyze.py` (reproducible, seeded).

## Method

- Sample: 75 clips (~5.7 min) stratified from the SADA **test** split
  (streamed, not full-downloaded), 25 each Najdi / Hijazi / Khaliji.
- ASR: `openai/whisper-large-v3` zero-shot via transformers pipeline
  (`language=arabic`, `task=transcribe`), fp16 on RTX 3080 Ti.
- WER: jiwer, computed after matching-grade normalization on BOTH sides
  using Sawti's own `normalize_arabic_for_match` (alef unification,
  diacritics/tatweel removal) plus punctuation stripping.
- Degenerate-case separation: clips <1s, empty refs, or hallucination-loop
  hypotheses (unique-token ratio <0.25 or top-token >60%) are reported
  separately, not averaged into the clean WER.

## Results

| Dialect | Clean WER | Degenerate rate | Degenerate causes |
|---|---|---|---|
| Najdi | **29.4%** (n=17) | 32% (8/25) | 7 short, 1 loop |
| Hijazi | **44.2%** (n=21) | 16% (4/25) | 4 short |
| Khaliji | **53.7%** (n=21) | 16% (4/25) | 3 short, 1 loop |
| **All clean** | **43.3%** (n=59) | 21% (16/75) | 14 short, 2 loops |

Observed failure modes:
1. **Hallucination loops on short clips** — e.g. ref "يلا" (one word) →
   hyp "نحن نحن نحن…" ×35. Known Whisper failure mode on short/non-speech
   audio; SADA's TV-snippet format (median 3s) triggers it often.
2. **Genuine dialect error** — on real speech content, ~30–55% of words
   are wrong. Zero-shot Whisper is not production-usable for Saudi
   dialect. (For scale: Whisper large-v3 English ≈ 5–10% WER, MSA ≈
   10–15%.)
3. **Occasional unrelated-content hallucination** (e.g. "ترجمة نانسي
   قنقر" for ref "بقوم").

## Caveats (stated plainly)

- **Small n:** 59 clean clips / ~5 min audio. Numbers are indicative, not
  definitive; per-dialect CIs are wide. The full test split (6.19k rows,
  ~10h) would firm them up if needed.
- One display line in the reanalysis script (WER-band histogram) had a
  fraction-vs-percent bug; per-dialect and overall numbers above are
  correct.
- `chunk_length_s=30` pipeline warning: max clip was 25s, so chunking did
  not affect results materially.
- SADA transcripts are TV-derived; some short-clip "references" may
  themselves be marginal.

## Findings relevant to Sawti's architecture

1. **The fine-tuning case is supported by data.** 43.3% clean zero-shot
   WER is far above usable; published dialect fine-tuning results
   (ArabicNLP-2025 shared task; Interspeech 2025 continual-pretraining
   work) report large reductions from exactly this recipe
   (Whisper + dialectal data). Fine-tuning on SADA (or a
   commercially-licensed equivalent) is the proven path.
2. **Short-clip hallucination is an architectural threat to Sawti.**
   Sawti's segmenter emits short chunks (min 0.6s by config). Zero-shot
   Whisper as the ASR+MT fallback would loop-hallucinate on exactly the
   clip lengths Sawti produces. Mitigations already in the design: the
   BalancedQualityGate `repetition_loop` check (validated by this spike —
   it would catch the observed loops) and `length_ratio_anomaly`. The
   gate is not optional hardening; it is load-bearing.
3. **Najdi (the primary Saudi dialect) zero-shots best** — 29% vs 44–54%
   for Hijazi/Khaliji — but still 3–5× worse than English. No Saudi
   dialect is "already fine" zero-shot.

## License constraint (blocking for commercial fine-tuning)

SADA is **CC BY-NC-SA 4.0 (NonCommercial-ShareAlike)**. Fine for this
research spike and for internal experimentation; conservatively, a model
fine-tuned on it inherits NC restrictions and could not ship in a
commercial Sawti. Before a production fine-tune: either Sawti remains
non-commercial, or the training corpus must be commercially licensable
(self-recorded data — already planned in spec §7.1 — or other-licensed
corpora, each requiring its own license check).

## Recommendation

Proceed to designing the Saudi-dialect fine-tuning milestone, with two
preconditions resolved in the design:
1. Data licensing decision (commercial vs research trajectory).
2. Optionally firm up baselines on the full test split first (~10h audio,
   sub-hour GPU) if tighter numbers would change the decision — they
   likely will not, given the gap (43% vs the ~10–15% MSA reference).

Raw artifacts: `data/sada_spike/` (gitignored): manifest.jsonl,
eval_results.json, reanalysis.json, 75 wav clips.

## Addendum (same day): existing Arabic adapter evaluated — rejected

`dev-ahmedhany/whisper-large-v3-arabic-ft-v3-lora` (Apache-2.0, ~38h
Egyptian/Levantine/Gulf/MSA, no Saudi data) evaluated on our 75-clip Saudi
sample at its best-reported revision (`7923fe7bc9b7`), merged fp16, greedy
decoding — same harness as the baseline:

| Dialect | Zero-shot | Adapter | Δ |
|---|---|---|---|
| Najdi | 29.4% | 48.5% | +19.1pp |
| Hijazi | 44.2% | 56.0% | +11.8pp |
| Khaliji | 53.7% | 64.2% | +10.6pp |
| **All** | **43.3%** | **56.8%** | **+13.4pp** |

Worse across the board; regressions dominated by YouTube-style repetition
hallucinations ("اشتركوا في القناه" loops) — consistent with its CV/MGB
training mix. Decision: do NOT build on this adapter; train from stock
large-v3 on SADA, but ADOPT its QLoRA recipe (r=8, α=16, q/v/k/out_proj+fc,
paged_adamw_8bit, lr 1e-4, eff. batch 16) as starting hyperparameters. Loop
rate joins WER as a first-class fine-tune metric.

Caveat: greedy fp16 vs authors' beam=2 int8 — fair for drop-in use in our
pipeline, not a reproduction of their best decode.
