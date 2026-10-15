# M3 Wire Contract — Draft (sawti.speech.v1)

**Status:** DRAFT for review — not locked. Written 2026-10-15 from the
platform-study convergence (partial/final + `final_kind`, versioned
envelopes, capture-at-edge) and everything M0–M2 built. This document
defines the contract M3's streaming engine should land on, so the
engine is built *to* the edge, not retrofitted.

**Sources:** VoiceStudio platform study (voicestudio.speech.v1 shape),
FluidAudio/VoiceStudio platform docs (WS + versioning patterns),
DuplexCascade/MicroTurn studies (micro-turn cadence), Sawti failure
analysis (low-confidence/streaming semantics), M2 listen (existing
edges). No third-party code; contract names are our own.

---

## 1. Design rules (from the studies, now requirements)

1. **Capture stays at the interface edge.** The model service never
   owns a microphone; any client (terminal, editor, browser, remote
   over tunnel) streams audio in. Matches M2's MicSource discipline.
2. **Versioning is in-band from byte one.** Every session carries a
   `protocol` field; unknown-version clients fail loudly at hello, not
   mid-utterance.
3. **Partial vs final is a UI contract, not a courtesy.** Partials are
   *unstable* (client may rewrite); finals are *stable* (append-only).
   Mistranslating this makes every caption UI flicker.
4. **Control is explicit, never parasitic on the text stream.** Our
   Mode-1 lesson plus the `<LISTEN>` probability-mass lesson: quality
   state, endpoint decisions, and language tags ride as typed fields,
   never as text tokens.
5. **Timestamps on everything.** Source-time anchors on every segment
   (M2 invariant), so clients can align, replay, and repair.

## 2. Session lifecycle

```
client                                   sawti speech service
  |-- WS /v1/speech (token?ws_ticket) -->|
  |<-- session.started ------------------|
  |    {protocol:"sawti.speech.v1",      |
  |     session_id, target_lang,         |
  |     audio:{sr:16000,ch:1,fmt:"s16le"}}|
  |-- binary audio frames -------------->|   (100 ms nominal)
  |<-- partial / final events -----------|
  |-- {"type":"input_audio.end"} ------->|   (or half-close)
  |<-- session.ended --------------------|
```

- **Target language is session-fixed** (spec §1.1) — set at hello,
  constant for the session, echoed in `session.started`.
- Audio input: raw signed 16-bit mono PCM at 16 kHz by default
  (`?pcm=1&sr=16000`); Opus/WebM frames permitted with explicit
  `audio.fmt` in a future revision. **One audio format per session.**

## 3. Event grammar (server → client)

### 3.1 `partial` — unstable, rewriteable
```json
{"type":"partial","session_id":"...","seq":41,
 "chunk_id":"c12",
 "text":"So we need to talk about",
 "start_time":12.40,"end_time":14.10}
```
Client MAY replace the text of the trailing `chunk_id`. Never appended
to a transcript of record.

### 3.2 `final` — stable, append-only
```json
{"type":"final","session_id":"...","seq":42,
 "final_kind":"utterance",
 "chunk_id":"c12",
 "text":"So we need to talk about this problem.",
 "start_time":12.40,"end_time":15.10,
 "low_confidence":false,
 "fallback_path":null}
```
- **`final_kind`**:
  - `utterance` — an endpoint decision closed this chunk (the normal
    M2-style close). Client appends.
  - `summary` — authoritative whole-session or whole-file result,
    emitted at `session.ended`; supersedes utterance finals for
    of-record use. (Mirrors the two-tier convention the platform
    study documented; lets batch and streaming share one grammar.)
- **`low_confidence`** / **`fallback_path`**: Sawti's observability
  passthrough — the gate's verdict travels with the text (rule 4).
- `seq` is monotonic per session; gaps mean dropped events (client may
  reconnect or request gap report in v1.1).

### 3.3 `error`
```json
{"type":"error","code":"...","message":"...","fatal":true|false}
```
Non-fatal errors carry a session; fatal errors precede socket close.
The M2 fail-loud vocabulary maps 1:1: `capture_overflow`,
`queue_overflow`, `vram_budget` etc.

## 4. Endpointing (M3's core addition)

- v1 ships **server-side endpointing** (the segmenter's decision), with
  cadence and thresholds server-configured. The events above are all a
  compliant client needs.
- v1.1 (deferred): `endpoint_hint` client→server advisories
  (e.g. UI mute states), and server→server `commit`/`hold` telemetry
  exposing the learned policy's decisions for debugging — aligned with
  the micro-turn commit/hold vocabulary already specced for the
  learned segmenter.

## 5. What deliberately does NOT ship in v1

- No token-level streaming of finals (chunk-granular only; spec §1.4
  rule 5 survives into M3).
- No bidirectional sessions (single target; the conversation-mediation
  product, if ever, is a v2 contract with two mirrored streams).
- No diarization, no speaker fields (single-speaker through M3).
- No compression of the event channel (JSON lines; binary framing is
  a v2 consideration if profiling demands it).

## 6. Client obligations (minimal)

1. Render partials by replacement, finals by append.
2. Treat unknown event types as ignorable (forward compat), unknown
   `final_kind` as `utterance`.
3. Never assume audio-out; v1 is a transcription/translation stream.

## 7. Relationship to existing edges

- `sawti listen` (M2) becomes the first v1 client: stdout rendering of
  finals, partials suppressed (terminal).
- `sawti transcribe` (M1) maps to the `summary`-only mode.
- The eval harness consumes the same events (factory seam unchanged).

---

**Review asks:** (1) is `final_kind: summary` worth carrying in v1, or
defer until batch/streaming actually share the service? (2) is
`chunk_id` the right stability anchor for partial replacement, or do we
need utterance-ids? (3) anything to add for replay/repair (stored
sessions) before the fields freeze?
