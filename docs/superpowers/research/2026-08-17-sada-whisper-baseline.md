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

## Addendum 2: four external models + turbo base evaluated — none beat stock large-v3

Same 75-clip harness, greedy fp16, identical normalization. Turbo's stock
processor used for oddadmix (repo tokenizer config malformed:
extra_special_tokens list-vs-dict).

| Model | Najdi | Hijazi | Khaliji | All | vs large-v3 |
|---|---|---|---|---|---|
| **Stock large-v3 (baseline)** | 29.4 | 44.2 | 53.7 | **43.3** | — |
| Turbo zero-shot | 40.5 | 49.2 | 61.2 | 50.8 | +7.5 |
| Bruno7 turbo-saudi-phase2 (LoRA) | 46.4 | 48.6 | 50.9 | 48.7 | +5.4 |
| oddadmix turbo-dialectal (full FT) | 53.1 | 55.0 | 57.6 | 55.4 | +12.1 |
| dev-ahmedhany arabic-ft (Addendum 1) | 48.5 | 56.0 | 64.2 | 56.8 | +13.4 |

Findings:
1. Turbo is a weaker base for Saudi (+7.5pp zero-shot) — none of the three
   turbo-based candidates recovered the deficit. large-v3 base validated.
2. oddadmix is worse than its own zero-shot base on our sample, despite its
   card claiming large gains on its own test set — their Gulf/Saudi mix does
   not match SADA's Saudi distribution. Fine-tuning on the wrong dialect mix
   actively hurts the target dialect; per-dialect checkpoint selection on
   held-out Saudi data is therefore mandatory in our training loop.
3. Degenerate/loop rate is ~constant (14–15/75) across all models —
   decoder-level failure mode, independent of fine-tune; the quality gate's
   repetition_loop check stays load-bearing for any model choice.

Saudi TTS catalog (for the future TTS milestone, unevaluated):
NAMAA-Space/NAMAA-Saudi-TTS (Chatterbox config/prompting, MIT, unverified
training), AhmedEladl/saudi-tts (XTTS lineage, Apache-2.0, no provenance
disclosed, base-license inheritance to check), vadimbelsky/qwen3-TTS-KSA
(Qwen3-TTS-12Hz full FT on ~13k KSA clips, Apache-2.0, single speaker, no
metrics). All require listening tests before any adoption.

Generalized harness: spikes/sada_model_eval.py (full or adapter, arbitrary
repo, optional processor override).

## Addendum 3 (2026-08-18): CORRECTION — paired methodology + two retractions

**Methodology flaw fixed:** prior comparisons computed each model's clean
subset independently (unpaired) — headline deltas averaged over different
clips. All numbers below are PAIRED, recomputed from stored per-clip
hypotheses (no ASR re-runs; `spikes/sada_paired_reanalysis.py`).

### Paired results (base = stock large-v3 zero-shot)

| Model | Paired clean WER (Δ, n) | All-valid WER (Δ, n) | Loop% | Short-clip mean WER |
|---|---|---|---|---|
| turbo zero-shot | 42.3→49.1 (+6.8, n=58) | 287.4→212.2 | 2.7 | 702 |
| Bruno7 saudi-phase2 | 42.3→43.5 (**+1.2**, n=58) | 287.4→183.9 | 2.7 | 607 |
| oddadmix dialectal | 43.3→53.9 (+10.6, n=59) | 287.4→**59.7** | **0.0** | 79 |
| dev-ahmedhany arabic-ft | 43.3→53.6 (+10.3, n=59) | 287.4→274.4 | 2.7 | 1223 |
| **base degeneracy** | — | 287.4 (n=75) | 2.7 | 134 |

### Retraction 1 — repetition_loop check claim

The claim that the spike "validates the BalancedQualityGate repetition_loop
check as load-bearing" is RETRACTED. The check is unigram-only
(`len(set(tokens)) == 1`); it misses the phrase-level loops
("اشتركوا في القناه" ×3) that dominated observed failures. A deterministic
n-gram repetition detector (1–8-token spans, ≥3 repeats) is required before
any Whisper joins the fallback lane. (External review finding, verified.)

### Retraction 2 — "none beat stock large-v3" is view-dependent

On PAIRED CLEAN speech the conclusion holds (every candidate worse; Bruno7
closer than previously reported, +1.2pp). On ALL-VALID-REFERENCE WER
(hallucination cost included — the product-relevant view for Sawti's short
chunks), stock large-v3 scores 287.4% while oddadmix scores 59.7% with ZERO
loops: substantially more hallucination-robust than the base. The prior
headline was an artifact of clean-only reporting.

Also corrected: "degenerate rate ~constant across models" conflated
degenerates (short-clip dominated, 19–21% everywhere) with loops (0–2.7%
overall; 0–14.3% on short clips).

### Design consequences (superseding prior addenda where they conflict)

1. SA training decision unchanged for accuracy: stock large-v3 + SADA QLoRA.
2. **Training must add audio augmentation** (noise/music/speed/reverb —
   oddadmix's recipe) — the plausible cause of its zero-loop behavior.
   Hallucination robustness is a training objective, not just a gate
   concern.
3. Metric set for checkpoint selection/final eval expands: clean WER,
   **all-valid WER**, loop-rate (all first-class).
4. oddadmix (Apache-2.0 full FT) is a robustness REFERENCE, not a base:
   worse paired-clean accuracy (+10.6) but the benchmark for loop-free
   decoding behavior.
5. Gate n-gram repetition detection: still required before Whisper
   fallback (phrase loops observed in 3 of 5 models).

## Addendum 4 (2026-08-18): CORRECTION v2 — n-gram loop detector; supersedes Addendum 3 numbers

Addendum 3 fixed the pairing but kept the legacy loop detector (unigram
dominance), which classifies the x3 phrase loop ("اشتركوا في القناه" ×3:
uniq 0.33, most 0.33) as NOT a loop — the exact failure mode it claimed to
exclude. Recomputed with the deterministic n-gram detector (1–8-token spans,
≥3 consecutive repeats + legacy dominance; self-tested against the known
examples). All numbers below supersede Addendum 3's.

Base = stock large-v3. Paired common-clean; all-valid views; n-gram loops.

| Model | Clean macro (Δ, n) | All-valid macro | All-valid corpus | Loop% |
|---|---|---|---|---|
| turbo zero-shot | 43.1→48.2 (+5.1, n=57) | 212.2 | 72.6 | 4.0 |
| Bruno7 saudi-phase2 | 43.1→42.5 (**−0.6**, n=57) | 183.9 | 57.8 | 2.7 |
| oddadmix dialectal | 44.1→53.1 (+9.0, n=58) | **59.7** | **47.0** | **0.0** |
| dev-ahmedhany arabic-ft | 44.1→52.8 (+8.7, n=58) | 274.4 | 70.5 | 4.0 |
| **base** | — | 287.4 | 76.2 | 4.0 |

Metric definitions (named precisely): **macro** = mean per-clip WER
(short-reference hallucinations dominate it — hence 287.4%); **corpus** =
word errors weighted by reference length (base 76.2%).

### Corrected conclusions (superseding Addendum 3 §"Design consequences")

1. **SA training from stock large-v3 stands** — on provenance and clean
   accuracy. Bruno7 reaches clean parity (−0.6pp) but discloses no training
   data ("None dataset"); it cannot be responsibly built upon or cited as a
   base. No candidate with known provenance beats the base on clean speech.
2. **oddadmix's zero-loop behavior SURVIVES the n-gram detector** (0.0%
   loops) and it leads every robustness view (corpus 47.0% vs base 76.2%).
   The augmentation hypothesis is now detector-robust: audio augmentation
   (noise/speed/reverb-class) is a required SA training component, and
   oddadmix remains the robustness reference.
3. **First-class metric set (final): clean macro WER, all-valid macro WER,
   all-valid corpus WER, n-gram loop-rate.** Checkpoint selection gates on
   clean macro subject to a loop-rate constraint; macro and corpus are both
   reported so neither view can hide behind the other.
4. Bruno7's quiet improvement (−0.6) is noted but within noise (n=57);
   no action.

Short-clip view (0.5–1.0s, n=14, floor now enforced): base macro 134.5%/0
loops; oddadmix 78.6%/0 loops; dev-ahmedhany 1222.6%/14.3% loops —
short-clip hallucination remains the product-risk differentiator.

## Addendum 5 (2026-08-18): historical metrics recomputed under the AUTHORITATIVE detector — unchanged

SA preflight reconciliation switched the paired-reanalysis harness
(spikes/sada_paired_reanalysis.py) from its local n-gram+dominance
detector to the SHARED production detector (sawti.loop_detect.is_loop —
pure consecutive-block semantics) and recomputed every stored
hypothesis. Result: **all values are identical to Addendum 4** —
paired clean macros (+5.1 / −0.6 / +9.0 / +8.7), loop rates (4.0 / 2.7
/ 0.0 / 4.0), degeneracy membership, and both all-valid views. No
stored hypothesis was a dominance-only catch, so removing the heuristic
reclassified nothing. Training selection, the evaluator, the runtime
gate, and these historical baselines now share one metric definition.
(The harness retains its paired/common-clean methodology; only the
detector import changed.)

## Addendum 6 (2026-08-21): FINAL ACCEPTANCE — one-shot SADA test evaluation (Run 2 merged model)

The SA milestone's pre-registered final experiment, executed once per the
Task 12 protocol (paired frozen-evaluator, commit `b638e85`; consumption
rule in force from this point).

**Setup.** Fresh test materialization: 3,639 clips / 4.12 h (Najdi 1,684 /
Khaliji 1,150 / Hijazi 805). Two passes over the identical manifest —
stock `openai/whisper-large-v3` and merged `models/sa_whisper_v1`
(checkpoint-10000, selection_score 42.26 validation) — through the frozen
evaluator: FP16, greedy (num_beams=1, do_sample=False), language=arabic,
task=transcribe, batch 4, attention_mask into generate. Cross-artifact
assertions passed BEFORE interpretation: identical manifest SHA-256
(`7030ef78…`), batch_size 4 both, identical generate_kwargs,
attention_mask true both, identical 3,639 clip IDs, same evaluator
commit. Artifacts: `data/sada_training/test/{stock_final,sa_final}.json`
(local, gitignored; full per-clip records + regime records).

**Results.**

| Metric | Stock | Merged (SA) |
|---|---|---|
| clean_macro_wer | 55.61 | **43.33** |
| all_valid_macro_wer | 86.95 | **51.88** |
| all_valid_corpus_wer | 62.02 | **37.94** |
| loop_pct | 2.42 | 2.72 |
| degenerate_rate | 19.73 | 20.09 |

Paired common-clean dialect deltas (merged − stock; a clip comparable only
when BOTH passes classify it non-degenerate):

| Dialect | n | Stock | Merged | Delta |
|---|---|---|---|---|
| Najdi | 1,296 | 55.91 | 43.71 | **−12.20pp** |
| Hijazi | 653 | 52.82 | 40.51 | **−12.32pp** |
| Khaliji | 930 | 56.64 | 43.75 | **−12.89pp** |

**Verdict per pre-registered criteria:**
1. merged clean_macro_wer <= 20%: **FAIL** (43.33)
2. merged loop_pct < 5%: **PASS** (2.72)
3. no dialect regression vs full-test stock on paired common-clean: **PASS**
   (−12.2 to −12.9pp on all three core dialects; consistent with the
   validation result, so this is a genuine generalization improvement, not
   a split artifact)

**Honest reading.** The absolute 20% target was not reached. The high
degenerate/short-clip rate and large stock-model error indicate
substantial dataset and task difficulty, but do not establish a
mathematical WER floor (clean_macro_wer excludes degenerate rows
entirely; corpus WER includes valid-reference rows regardless of
degeneracy — neither metric turns the degenerate rate into a bound).
Further progress plausibly requires cleaner and more representative
Saudi dialectal data rather than simply extending optimization on the
current SADA training set. The held-out test result nevertheless
demonstrates generalization: the merged model improves paired
common-clean WER by 12.2-12.9pp on each core dialect while keeping loop
rate below 5%. (Note: the validation selection score 42.26 is a
dialect-balanced mean; the test 43.33 is clip-weighted clean macro —
different metrics, not directly comparable deltas.)

**Consumption rule (now ACTIVE).** This SADA test split is spent as final
acceptance evidence. Any training/data/hyperparameter iteration triggered
by the FAIL on criterion 1 requires a NEW untouched test set; this split
must not become checkpoint-development data.

Historical context only (75-clip spike, Addendum 4 regime): Najdi 29.4 /
Hijazi 44.2 / Khaliji 53.7 — not the full population, not decode-regime
matched, superseded by the full-test paired numbers above.
