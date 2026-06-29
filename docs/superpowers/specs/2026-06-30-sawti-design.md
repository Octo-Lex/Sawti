# Sawti — Design Specification

**Status:** Approved (Sections 1–8 locked)
**Date:** 2026-06-30
**Repo:** [Octo-Lex/Sawti](https://github.com/Octo-Lex/Sawti)

---

## 1. Overview

### 1.1 Product promise

> Speak in any supported language, or a mid-utterance code-switched mix, and
> receive a continuous text stream in a **single, pre-selected target language**.

**Target languages (initial scope):** English, Arabic, French (`eng`, `ara`, `fra`).
The target language is **session-fixed** — chosen before speaking, constant for the session.

### 1.2 Task definition

**Transcribe + translate mix.** If the spoken language matches the target, transcribe verbatim; if it differs, translate into the target. One unified output stream in the target language. This is *not* language-gated ASR (which would ignore non-target speech) and *not* pure S2TT-only routing.

### 1.3 Architecture (top level)

```
audio stream
  │
  ▼
SEGMENTATION          VAD · pause detection · max-duration · overlap
  │   (└─ side channel: LID / confidence / diagnostics — non-blocking)
  ▼
S2TT ENGINE           SeamlessM4T, target_lang = pre-selected
  │
  ▼
QUALITY / FALLBACK    retry · re-chunk · ASR+MT fallback behind the gate
  │
  ▼
POST-PROCESSING       merge · dedupe · punctuate · normalize per target language
  │
  ▼
target-language text stream
```

### 1.4 Design principles (locked)

1. **No hard LID routing.** The S2TT engine infers the source language internally; LID runs only as a non-blocking side channel for diagnostics.
2. **Chunk coherence over language purity.** A multi-language chunk representing one coherent phrase stays together; splitting around a language switch hurts translation.
3. **Conservative post-processing.** Merge/dedupe/punctuate/normalize only. No paraphrasing, summarizing, or inference.
4. **Fallback behind the quality gate.** Recovery is triggered by measurable quality conditions, not ad hoc handling.
5. **Chunked, not token-level streaming, for MVP.** Technically it is `audio stream → utterance/chunk stream → target-text chunk stream`. Token-level real-time S2TT is a later milestone.

---

## 2. Segmentation layer

The segmentation layer is the highest-risk component. Its job is to feed the engine coherent chunks.

### 2.1 Core principle (rule 2)

Chunk on **acoustic/semantic coherence, not language identity.** A chunk should represent one speakable phrase, bounded by natural pauses, even if it contains multiple languages.

### 2.2 Algorithm (four layered signals)

1. **VAD** — Silero VAD per frame → marks speech / non-speech regions.
2. **Pause detection** — a non-speech run ≥ `pause_threshold_ms` → candidate cut.
3. **Max-duration ceiling** — chunk ≥ `max_chunk_duration_s` → forced cut at nearest pause.
4. **Overlap** — each chunk carries `overlap_ms` of the prior chunk's tail → enables edge dedup later.

### 2.3 Parameters (MVP defaults, runtime-configurable)

| Parameter | Symbol | Default | Range | Role |
|---|---|---|---|---|
| Pause threshold | `pause_threshold_ms` | **350** | 200–600 | Silence to close a chunk |
| Max chunk duration | `max_chunk_duration_s` | **12** | 8–12 | Force-close ceiling |
| Min chunk duration | `min_chunk_duration_ms` | **600** | 500–700 | Don't close until reached |
| Overlap | `overlap_ms` | **300** | 200–400 | Tail carried for edge dedup |
| Min speech to open | `min_speech_ms` | 300 | — | Below this is not a phrase |
| Min inter-chunk gap | `min_gap_ms` | 100 | — | Prevents chatter of tiny chunks |

All values are **runtime config**, not hardcoded constants. UX modes map to pause-threshold values:

| Mode | `pause_threshold_ms` | Use case |
|---|---|---|
| Balanced (MVP default) | 350 | General conversational speech |
| Coherence-first | 500–600 | Longer utterances, slower output |
| Responsiveness-first | 200–250 | Fast captions, more fragmentation |

### 2.4 Close-decision policy

```
close chunk when:
    silence ≥ pause_threshold_ms
    AND chunk_duration ≥ min_chunk_duration_ms

force close when:
    chunk_duration ≥ max_chunk_duration_s

extend (don't close) when:
    pause occurs but chunk_duration < min_chunk_duration_ms
    → keep buffering until more speech arrives
      or a longer timeout confirms end-of-utterance
```

### 2.5 What segmentation does NOT do (MVP)

- No acoustic language-boundary detection inside a chunk.
- No LID-driven cutting.
- No sub-chunk re-segmentation on language change.

A mid-chunk switch without a pause *may* confuse SeamlessM4T. MVP accepts this and relies on the quality gate (§3) to catch it via re-chunking or ASR+MT fallback.

### 2.6 Future evolution (not MVP)

The segmentation interface emits a uniform `AudioChunk`, so the decision logic upgrades transparently:

- **M2-tier:** Add LID-based *soft* hints (tag chunks likely to contain switches for closer quality scrutiny).
- **M3-tier:** Add a small acoustic language-change detector for true mid-chunk boundaries.

---

## 3. S2TT engine + quality/fallback gate

### 3.1 Engine interface

```
Input:   AudioChunk + target_lang ∈ {"eng", "ara", "fra"}  (session-fixed)
Output:  EngineResult
           .raw_text              # exactly what the model emitted
           .confidence            # scalar 0..1
           .source_lang_guess     # from side-channel LID, diagnostic only
           .timing_ms             # wall-clock per stage
           .target_lang
```

The engine is a thin wrapper around SeamlessM4T. The wrapper isolates the model choice — swapping the engine leaves segmentation and post-processing untouched.

### 3.2 Model selection

- **Primary:** `facebook/seamless-m4t-v2-large` (speech → target-language text).
- **Call mode:** batched per chunk (chunked interaction, not streaming).
- **SeamlessStreaming** is reserved for the streaming milestone (M3).

### 3.3 Model loading policy

```yaml
s2tt:
  engine: seamless_m4t
  target_lang: eng
  load_policy: resident         # resident | lazy | idle_unload
  idle_unload_seconds: 300
  device: cuda
```

`EngineManager` supports all three policies. **MVP operates in resident mode:** SeamlessM4T loads once at startup and stays in GPU memory for low, predictable per-chunk latency. Lazy/idle-unload exist as deployment policies for serverless later, not the default interactive path.

### 3.4 The quality/fallback gate

Sits between engine and post-processing. Decides whether an `EngineResult` is good enough to emit, or whether to trigger recovery.

```
          EngineResult
                │
                ▼
        ┌───────────────────┐
        │  QUALITY CHECKS   │  (cheap, deterministic)
        └─────────┬─────────┘
            ┌─────┴─────┐
            ▼           ▼
         PASS       FAIL/WEAK
            │           │
            │           ▼
            │     ┌──────────────────┐
            │     │ FALLBACK ACTIONS │ (escalating order)
            │     └────────┬─────────┘
            │   ┌──────────┼──────────┐
            │   ▼          ▼          ▼
            │ retry      re-chunk    ASR+MT
            │ (beam/         (tighter  (Whisper +
            │  low temp)      bounds)   NLLB/M2M)
            │   │          │          │
            ▼   └──────────┴──────────┘
       emit            best wins
```

### 3.5 Quality signals

| Signal | Source | FAIL condition |
|---|---|---|
| Engine confidence | engine `.confidence` | < `confidence_threshold` (0.40) |
| Output length ratio | `len(text)` vs chunk duration | too short/long vs speech |
| Empty/garbage output | regex heuristics | empty, all-punctuation, repeated token |
| Script mismatch | text vs `target_lang` | target=ara but output is Latin chars |
| Side-channel LID agreement | diagnostics vs output plausibility | extreme disagreement (**soft flag only**) |

The first four are **hard gates** (fail → fallback). LID agreement is a **soft flag** (logged; may bias toward re-chunking but never fails on its own).

### 3.6 Fallback action priority (escalating)

1. **Retry same engine** with conservative decoding (beam search, lower temperature, length penalty). Catches greedy-decode noise.
2. **Re-chunk** with tighter boundaries (smaller `max_chunk_duration_s`, split at any micro-pause) and re-run. Catches segmentation-induced errors.
3. **ASR + MT fallback:** `faster-whisper` (multilingual ASR + LID) → translate non-target segments via NLLB-200 / M2M-100. The debugging oracle and safety net.

The gate emits whichever path produces the highest-confidence valid output, and **logs which path was taken**. If all paths fail, it emits the best-effort result with a `low_confidence` flag rather than dropping the chunk.

### 3.7 What the gate does NOT do

- No LLM-based quality judging in the happy path (cost/latency). Reserved for V2.
- No blocking on side-channel LID.
- No silent dropping — a chunk always produces *some* output.

### 3.8 Gate policy (configurable, default Balanced)

```yaml
quality_gate:
  policy: balanced                # conservative | balanced | aggressive
  confidence_threshold: 0.40
  retry_once: true
  rechunk_on_failure: true
  fallback_to_asr_mt: true

  checks:
    empty_output: true
    garbage_output: true
    script_mismatch: true
    length_ratio_anomaly: true
    repetition_loop: true

  length_ratio:
    min_chars_per_audio_second: 0.5
    max_chars_per_audio_second: 35

  retries:
    max_s2tt_retries: 1
    max_rechunk_attempts: 1

  script_mismatch_strictness:
    eng: soft
    fra: soft
    ara: strict                   # mostly-Latin output → fallback unless
                                  # chunk has expected entities/numbers/URLs
```

> The quality gate defaults to **Balanced**: hard failures, low confidence, script mismatch, repetition, and length-ratio anomalies trigger one retry, then re-chunking, then ASR+MT fallback. Policy and thresholds are runtime-configurable. Aggressive parallel ASR+MT is out of scope for MVP.

**Arabic-specific:** script mismatch is **strict** — mostly-Latin output triggers fallback unless the chunk contains legitimate entities, numbers, URLs, or brand names. For EN/FR it is **soft**.

---

## 4. Post-processing

> Post-processing is **deterministic and conservative**. It operates on gate results in time order, maintaining separate **raw**, **matching-normalized**, and **display** forms. It removes duplicated overlap, collapses obvious decoder loops, repairs whitespace and punctuation, applies non-destructive per-language normalization, emits timestamp-aligned target-language text, and **logs every modification**. It does not paraphrase, summarize, infer, dialect-normalize, or re-translate.

### 4.1 Internal text representations (three forms)

```
raw_text       = original gate output (never destructively modified)
match_text     = normalized copy used for dedupe/repeat detection
display_text   = cleaned emitted text
```

This makes rules like "remove diacritics for matching but preserve in output" unambiguous.

### 4.2 Pipeline (six steps, all deterministic)

```
chunk results (in time order)
        │
        ▼
1. DEDUPE OVERLAP
   remove text duplicated at chunk edges (overlap_ms tail)
        │
        ▼
2. REPEATED FRAGMENT CLEANUP
   collapse decoder-style loops ("the the the")
        │
        ▼
3. WHITESPACE / SCRIPT NORMALIZATION
   per target language (see 4.4)
        │
        ▼
4. PUNCTUATION REPAIR
   fix spacing, capitalization, sentence boundaries
        │
        ▼
5. EMIT
   append cleaned text to output stream
   (with timestamps for UI alignment)
        │
        ▼
6. LOG / METRICS
   record normalization actions, dedupe span,
   low-confidence flag, fallback path (auditable)
```

### 4.3 Dedupe algorithm (token-aware)

Compares **normalized token** sequences; emits **raw** tokens (preserves casing, diacritics, punctuation).

```python
def dedupe_adjacent(prev_raw, curr_raw, min_overlap=2):
    """Compare normalized token sequences; emit raw tokens.
    prev_raw, curr_raw: raw token lists (casing/diacritics/punct preserved).
    Returns the suffix of curr_raw that was not duplicated.
    """
    # Build normalized copies for comparison only (non-destructive).
    prev_norm = [normalize(t) for t in prev_raw]
    curr_norm = [normalize(t) for t in curr_raw]
    max_k = min(len(prev_norm), len(curr_norm))
    for k in range(max_k, min_overlap - 1, -1):
        if prev_norm[-k:] == curr_norm[:k]:   # normalized comparison
            return curr_raw[k:]               # raw tokens for emit
    return curr_raw
```

**MVP:** exact-token overlap only. **Fuzzy overlap matching** is deferred (can delete real content when S2TT translates overlapped audio slightly differently across chunks) and gated behind a strict config flag.

### 4.4 Per-target-language normalization (non-destructive)

**English / French (Latin script):**
- Collapse multiple spaces, trim whitespace.
- Fix spacing around punctuation (`word ,` → `word,`).
- Capitalize first letter of sentence starts.
- Lowercase all-caps spans longer than N words (decoder artifact) — *configurable, off by default* (protects proper nouns).
- French-specific: normalize spacing before `: ; ! ?` (French typography requires a non-breaking space).

**Arabic (RTL, richer normalization):**
- Remove tatweel (`ـ`, U+0640).
- Normalize alef forms `أ إ آ ٱ` → `ا` — **matching only**, default `arabic_alef_normalization: matching_only`.
- Normalize alef-maqsura `ى` → `ي` — **matching only**, default `arabic_yeh_maqsura_normalization: matching_only`.
- Remove diacritics (harakat) for matching — **preserve in output** (`preserve_arabic_diacritics: true`).
- Normalize Arabic punctuation spacing (`، ؛ ؟`).
- **RTL is an emit/UI concern, not stored text:** store plain Unicode; apply `dir="rtl"`/`dir="auto"` at the UI boundary. Force-inserting direction marks into stored text causes copy/paste and matching issues.

### 4.5 Repeat collapse (narrow)

```
Collapse exact repeated unigram/n-gram loops only when repetition count ≥ 3,
unless the phrase is on an allowlist or punctuation suggests intentional repetition.
```

- Collapse: `"the the the the"` → `"the"`; `"مرحبا مرحبا مرحبا مرحبا"` → `"مرحبا"`.
- Preserve: `"no no, wait"`; `"very very important"`; `"لا لا"`.

### 4.6 Config (all steps individually toggleable)

```yaml
postprocess:
  dedup_overlap: true
  collapse_repeats: true
  repeat_min_count: 3
  normalize_script: true
  repair_punctuation: true
  preserve_arabic_diacritics: true
  arabic_alef_normalization: matching_only        # off | matching_only | output
  arabic_yeh_maqsura_normalization: matching_only  # off | matching_only | output
  fuzzy_overlap: false                            # deferred; strict flag
```

### 4.7 Explicitly out of scope

Paraphrasing, summarization, gap-filling/inference, dialect → MSA conversion, aggressive caps normalization, back-translation/re-translation.

---

## 5. Component interfaces & isolation

### 5.1 Core data types (the shared vocabulary)

```python
@dataclass
class AudioChunk:
    """A segment of audio emitted by the segmentation layer."""
    id: str                          # unique, monotonic per session
    audio: np.ndarray                # float32 mono PCM, 16 kHz
    sample_rate: int                 # 16000
    start_time: float                # seconds from session start
    end_time: float
    overlap_from_prev_s: float       # seconds carried from prior chunk's tail
    meta: dict                       # optional VAD scores, pause boundaries

@dataclass
class EngineResult:
    """Output of the S2TT engine for one chunk."""
    chunk_id: str
    raw_text: str
    confidence: float                # engine-provided, 0..1
    source_lang_guess: str | None    # from side-channel LID, diagnostic only
    timing_ms: dict
    target_lang: str                 # "eng" | "ara" | "fra"

@dataclass
class GateDecision:
    """The quality gate's verdict on an EngineResult."""
    chunk_id: str
    accepted: bool
    result: EngineResult             # the chosen result (possibly from fallback)
    checks: dict                     # {"empty": False, "script_mismatch": False, ...}
    fallback_path: str | None        # None | "retry" | "rechunk" | "asr_mt"
    low_confidence: bool
    needs_retry: bool
    log: list[dict]                  # every action taken, in order

@dataclass
class OutputSegment:
    """Emitted unit: timestamp-aligned target-language text."""
    chunk_id: str
    text: str                        # display text
    start_time: float
    end_time: float
    low_confidence: bool
```

### 5.2 Component contracts

| Component | In | Out | Config (key params) | Knows nothing about |
|---|---|---|---|---|
| **Segmenter** | audio stream (frames + timestamps) | `Iterable[AudioChunk]` | `pause_threshold_ms`, `max_chunk_duration_s`, `min_chunk_duration_ms`, `overlap_ms` | downstream models, target language |
| **EngineManager** | `AudioChunk`, `target_lang` | `EngineResult` | `engine`, `load_policy`, `device` | segmentation params, post-processing |
| **QualityGate** | `EngineResult`, `AudioChunk`, `target_lang` | `GateDecision` | `policy`, `confidence_threshold`, `checks{}`, `script_mismatch_strictness{}` | segmentation internals, post-processing |
| **PostProcessor** | `Iterable[GateDecision]`, `target_lang` | `Iterable[OutputSegment]` | dedupe/normalize/repeat flags (§4.6) | which model produced the text |

### 5.3 Orchestrator (locked: sequential generator)

> The MVP orchestrator is a **sequential generator**: each chunk is processed fully through S2TT, quality gate, fallback, and post-processing before the next chunk is emitted. Component interfaces are stream-compatible so the orchestrator can later be upgraded to a bounded pipeline, but concurrency and reordering are out of scope for MVP.

```text
AudioSource → Segmenter yields AudioChunk
           → Engine returns EngineResult
           → Gate returns GateDecision / Result
           → PostProcessor returns OutputSegment
           → Orchestrator yields OutputSegment
```

```python
for chunk in segmenter(audio_stream):
    result = s2tt_engine.translate(chunk)
    gated = quality_gate.evaluate(result, chunk)
    if gated.needs_retry:
        gated = fallback_handler.retry_or_fallback(chunk, result)
    cleaned = postprocessor.process(gated)
    yield cleaned
```

Guarantees: correct output order by construction; debuggable fallback; single-threaded access to the resident SeamlessM4T model; simple backpressure; easy-to-reason-about timestamps.

### 5.4 Isolation rationale

| Boundary exists so that… | …this can change independently |
|---|---|
| Segmentation emits `AudioChunk` only | Swap VAD engine (Silero → WebRTC) or add LID-aware cutting later (M2/M3) |
| `EngineManager` abstracts the S2TT engine | Replace SeamlessM4T with a fine-tuned/custom model in V2 |
| `QualityGate` owns fallback logic | Fallback strategies evolve without entangling engine/post-processing |
| `PostProcessor` is stateful but single-purpose | Per-language normalization rules grow without affecting other components |

### 5.5 Testability

Each component is testable as a pure-ish function over the shared data types. The real SeamlessM4T model is only needed for **integration tests** and end-to-end evaluation — never for unit tests.

---

## 6. Milestones

### 6.1 MVP scope (locked)

> The first usable release is **M1: offline file-to-text translation**. It must accept an audio file and pre-selected target language, process the file through the complete sequential pipeline, and emit a timestamped target-language transcript. Live mic input is explicitly deferred to M2.

### 6.2 Milestone ladder

| Milestone | Scope | Release status |
|---|---|---|
| **M0** | Foundation: skeleton, data types, config schema, stub components, eval harness scaffold | Pre-MVP |
| **M1** | Offline audio file → target-language text | **MVP release** |
| **M2** | Live mic → chunked target-language text | Next interactive demo |
| **M3** | Near-real-time streaming behavior | Later product milestone |
| **M4** | Optimize: tune segmentation/gate/post-processing | After evaluation baseline |
| **M5** | Fine-tune components/languages (V2) | Conditional on M4 findings |

### 6.3 M1 detail (the key milestone)

**Build:** real Segmenter (Silero VAD + close-decision policy §2.4); real EngineManager (SeamlessM4T resident); real QualityGate (Balanced + ASR+MT fallback); real PostProcessor (all six steps); FileSource (WAV/MP3) as `AudioSource`.

**Demo:** `sawti transcribe recording.wav --target eng` → target-language text with timestamps.

**Exit criteria:** on the held-out code-switched set, produces readable target-language text with no dropped chunks; ASR+MT fallback fires on deliberate garbage and recovers; Arabic script-mismatch strictness works.

**Why M1 retires 90% of architectural risk:** segmentation on code-switched speech, the quality gate, and Arabic normalization are the novel, uncertain pieces — all exercisable on a recorded file with no real-time/streaming complexity. Every component built here survives into M2/M3.

### 6.4 Deferred (not in MVP)

| Item | When |
|---|---|
| True token-level streaming | M3 |
| LID-based code-switch boundary cutting | M4 (soft hint) |
| Fuzzy overlap dedup | M4 |
| Dialect → MSA normalization | M5 or beyond |
| Custom end-to-end model | Only if M5 shows it's needed |
| Multi-target language in one session | Possibly never (product decision) |

---

## 7. Evaluation & observability

### 7.1 Evaluation dataset (locked)

> M1 evaluation uses a **self-recorded code-switched dataset** as the primary benchmark, with hand-written target-language references. Public monolingual corpora may be added later as secondary regression tests, but they do not define MVP success.

**Scope (~50–75 clips, 5–20s each):**

| Dataset slice | Count | Purpose |
|---|---|---|
| EN-only → target | 10–15 | Basic sanity |
| AR-only → target | 10–15 | Arabic source coverage |
| FR-only → target | 10–15 | French source coverage |
| EN/AR code-switch | 10–20 | Core hard case |
| EN/FR code-switch | 10–15 | Secondary code-switch |
| AR/FR/EN mixed | 5–10 | Stress cases |

Each clip: single speaker, clean-ish audio, natural pauses, named entities/numbers, code-switched phrases.

### 7.2 Quality metrics

| Metric | What | Target |
|---|---|---|
| **chrF / chrF++** (primary) | Translation quality vs. reference, character-level | ≥ 45 per language |
| **COMET** (optional) | Neural translation quality | positive, baseline vs. SeamlessM4T-direct |
| **Chunk coverage** | `% chunks with non-empty output` | ≥ 98% |
| **Script accuracy** | `% target-script tokens` | ≥ 95% (esp. Arabic) |
| **Dropped-chunk rate** | chunks producing nothing | 0% |
| **Fallback rate** | `% chunks triggering fallback` | logged (diagnostic) |
| **Manual adequacy** | human adequacy score | spot-checked |
| **Hallucination/omission flags** | manual | spot-checked |

> **Primary automated metric is chrF (target-language), not WER.** WER assumes output is the same language as reference; chrF is script-friendly and standard for S2TT.

### 7.3 Pipeline/latency metrics

| Metric | What | Why |
|---|---|---|
| RTF (real-time factor) | processing_time / audio_duration | < 1.0 needed for M2/M3 |
| Per-stage timing | ms in segmentation/engine/gate/post | M4 bottleneck ID |
| P95 chunk latency | 95th-percentile wall time | catches fallback outliers |

### 7.4 Segmentation-specific metrics

| Metric | Target |
|---|---|
| Boundary F1 (cuts vs. hand-marked phrase boundaries) | high precision/recall |
| Over-fragmentation rate (`% chunks < min_chunk_duration`) | → 0 |
| Oversized-chunk rate (`% chunks > max_chunk_duration`) | → 0 |
| Code-switch containment (`% switches kept inside one chunk`) | ≥ 80% |

### 7.5 Runtime observability (structured per-chunk logs)

Newline-delimited JSON (`jsonl`):

```json
{
  "chunk_id": "...",
  "timestamps": {"start": 12.34, "end": 15.67, "duration_s": 3.33},
  "segmentation": {"vad": true, "pause_ms": 410, "force_closed": false},
  "engine": {
    "raw_text": "...",
    "confidence": 0.71,
    "source_lang_guess": "arb",
    "timing_ms": {"engine": 820, "path": "primary"}
  },
  "gate": {
    "checks": {"empty": false, "script_mismatch": false,
               "length_ratio_anomaly": false, "repetition_loop": false},
    "fallback_path": null,
    "low_confidence": false
  },
  "postprocess": {
    "dedup_removed": ["the"],
    "repeats_collapsed": 0,
    "normalization_actions": ["collapse_whitespace"]
  }
}
```

Every gate decision and every post-processing modification is auditable.

### 7.6 Evaluation harness (runnable tool, built in M0)

```bash
sawti eval <eval_set_dir> --target eng --report report.json
```

Outputs: per-clip metrics (chrF, coverage, script accuracy, RTF); per-stage timing aggregates; segmentation boundary analysis; fallback path distribution; pass/fail verdict against M1 targets.

### 7.7 Out of scope for M1

- No human MOS (chrF + spot-checks suffice; MOS is for TTS).
- No A/B against production systems.
- No hard live-latency targets (measured, not gated).

---

## 8. Tech stack

### 8.1 Locked stack

| Component | Choice |
|---|---|
| Core runtime | Python 3.11+ |
| Model execution | PyTorch |
| Happy-path S2TT | `seamless-m4t-v2-large` |
| Fallback ASR | `faster-whisper` |
| Fallback MT | NLLB-200 |
| Segmentation / VAD | Silero VAD |
| Audio I/O | `soundfile` / `librosa` |
| Evaluation | `sacrebleu` (chrF), `unbabel-comet` (optional) |
| Config / schema | `pydantic` + YAML |
| Logging / observability | `structlog` (→ JSONL) |
| Packaging / env | `uv` |

### 8.2 Architectural constraint (locked)

> The core pipeline remains **deployment-agnostic**. CLI, local app/server, and web service wrappers must depend on the core package, not the other way around.

```
┌─────────────────────────────────────────────────────┐
│  DEPLOYMENT-SPECIFIC EDGES (swappable)              │
│  ─ M1: FileSource (soundfile) → stdout/file sink    │
│  ─ M2: MicSource (sounddevice) → terminal UI        │
│  ─ M3: StreamingSource → streaming sink             │
└───────────────────────┬─────────────────────────────┘
                        │  (AudioChunk in / OutputSegment out)
                        ▼
┌─────────────────────────────────────────────────────┐
│  AGNOSTIC CORE (shared across all milestones)       │
│  Segmenter · EngineManager · QualityGate ·          │
│  PostProcessor · Pipeline · Config · Logging · Eval │
└─────────────────────────────────────────────────────┘
```

The core never imports from the edges. `Pipeline.run(audio_source, target_lang) -> Iterable[OutputSegment]` is identical across milestones.

### 8.3 Project layout

```
sawti/
├── pyproject.toml              # uv, deps, lockfile
├── sawti/
│   ├── __init__.py
│   ├── types.py                # AudioChunk, EngineResult, GateDecision, OutputSegment
│   ├── config.py               # pydantic config schema + YAML loader
│   ├── pipeline.py             # the sequential orchestrator
│   ├── segmentation/           # VAD + close-decision policy
│   ├── engine/                 # EngineManager + SeamlessM4T wrapper
│   ├── quality_gate/           # checks + fallback handler
│   ├── postprocess/            # 6-step deterministic pipeline + per-lang rules
│   └── sources/                # FileSource (M1), MicSource (M2), ...
├── eval/                       # eval harness (M0), metrics, report gen
├── tests/                      # unit (stubs) + integration (real models)
├── docs/superpowers/specs/     # this design doc
└── .env.example                # committed template (real .env gitignored)
```

### 8.4 Hardware

- **Development & M1:** NVIDIA GPU (CUDA). SeamlessM4T-v2-large is heavy; CPU inference is impractical for interactive use but acceptable for batch eval.
- **No minimum spec locked** — Section 7's RTF measurement determines the actual floor.

### 8.5 Deferred

Web framework, frontend, Docker, CI/CD pipeline, model-serving (Triton) — all deferred to the milestones that need them.

---

## Open items / future work

- Data collection: the ~50–75 self-recorded code-switched clips with hand-written target-language references (the bulk of M1 prep labor).
- M2/M3: live audio capture, session state, streaming engine, bounded-pipeline orchestrator.
- M4: parameter tuning driven by §7 evaluation; optional LID soft hints; optional fuzzy dedup.
- M5: component/language fine-tuning; possible custom end-to-end model (the DSM path studied earlier) only if warranted.

## References (background research)

- *Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling* (Kyutai, arXiv:2509.08753) — the streaming-S2TT architecture that motivates the long-term custom-model path.
- *A Non-autoregressive Model for Joint STT and TTS* (IBM/OSU, arXiv:2501.09104) — alternative NAR architecture; relevant to data-efficient joint training.
- *DASB — Discrete Audio and Speech Benchmark* (arXiv:2403.02249) — codec/architecture decision framework.
