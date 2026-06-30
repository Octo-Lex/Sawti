# Sawti

> Multilingual speech-to-text translation: speak in any supported language or a
> code-switched mix, and receive text in a single, pre-selected target language.

Sawti takes spoken input (including mid-utterance code-switching across supported
languages) and emits a continuous text stream in a target language chosen
beforehand. The initial focus is **English, Arabic, and French**.

## Approach

Sawti is built as a composable pipeline of existing state-of-the-art models,
designed so individual components can later be fine-tuned or replaced:

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

### Design principles

- **No hard LID routing.** The S2TT engine infers the source language; LID runs
  only as a non-blocking side channel for diagnostics.
- **Chunk coherence over language purity.** A multi-language phrase stays one
  chunk — splitting around a language switch hurts translation.
- **Conservative post-processing.** Merge/dedupe/punctuate/normalize only. No
  paraphrasing, summarizing, or inference.
- **Fallback behind the quality gate.** Recovery is triggered by measurable
  quality conditions, not ad hoc handling.
- **Chunked, not token-level streaming, for MVP.** True streaming S2TT is a
  later milestone.

## Status

🚧 Early design phase. See the design specification under
`docs/superpowers/specs/` for the full architecture.

## Running (M0)

```bash
uv sync
uv run pytest                       # run the test suite
uv run sawti transcribe --target eng   # stub pipeline demo
uv run sawti eval tests/fixtures --target eng  # eval harness skeleton
```

## Running (M1)

```bash
uv sync
uv run pytest                                   # unit suite (fast, hermetic)
SAWTI_RUN_INTEGRATION=1 uv run pytest -m integration   # real-model tests (GPU, ~2.3GB download)
uv run sawti transcribe sample.wav --target eng --engine m4t   # real pipeline
```

`--engine m4t` loads SeamlessM4T-v2-large (CUDA) and requires a real audio
file. Without a file, or with `--engine stub`, the stub pipeline runs
(hermetic, no model download).

**M1 acceptance note:** the unit suite (hermetic) is the CI bar. The real
SeamlessM4T integration test and the `--engine m4t` smoke are implemented
but deferred pending first-time model download and a real speech sample:

```bash
SAWTI_RUN_INTEGRATION=1 uv run pytest -m integration
uv run sawti transcribe path/to/speech.wav --target eng --engine m4t
```

## License

To be determined.
