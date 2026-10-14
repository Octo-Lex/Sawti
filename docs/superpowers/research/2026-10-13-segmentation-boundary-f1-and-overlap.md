# Segmentation Boundary F1 + Overlap-Carry Verification (first measurements)

**Date:** 2026-10-13 · CPU-only (GPU busy with unrelated training)
**Data:** `sample/000519_4.wav` — 7.8 min continuous Saudi speech, 76
human-timestamped utterance boundaries (ground truth by construction).
**Script:** `scripts/segment_boundary_eval.py` (reproducible).

## Item 1 — Boundary F1 (spec §7.4, measured for the first time)

| pause_threshold_ms | chunks | P | R | F1 | coverage | >12 s | <1 s |
|---|---|---|---|---|---|---|---|
| 250 | 126 | 0.488 | **0.813** | **0.610** | 76/76 | 1 | 0 |
| **350 (default)** | 118 | 0.487 | 0.760 | 0.594 | 76/76 | 2 | 0 |
| 500 | 92 | 0.462 | 0.560 | 0.506 | 76/76 | 5 | 0 |
| 600 | 81 | 0.487 | 0.520 | 0.503 | 76/76 | 8 | 0 |

Readings:
- **Zero dropped utterances at every threshold** — the never-drop
  guarantee holds even at 600 ms (spec's dropped-chunk rate = 0 %).
- **Over-fragmentation vs human marks is structural**: the segmenter
  produces ~1.5x the human boundary count; recall >> precision
  everywhere. Humans mark semantic utterances; VAD cuts at acoustic
  pauses — part philosophy mismatch, part real splitting.
- **Precision is pinned at ~0.49 across the whole sweep** — threshold
  tuning cannot buy boundary accuracy; raising it only loses recall and
  creates oversized chunks (8 > 12 s at 600 ms). The fixed-threshold
  policy's measured ceiling on this material is **F1 ≈ 0.6** — the
  number the M4 learned/text-side segmenter must beat.
- The min-duration gate works (zero sub-1 s chunks at all settings).

Caveats: single file, single speaker, clean broadcast audio; human
timestamps reflect annotation philosophy. Indicative, not universal.

## Item 2 — Overlap-carry verification (deterministic, no ML)

The 300 ms overlap carry is a verbatim waveform copy by the segmenter's
strong invariant. Verified empirically with sample-level
cross-correlation (head of each chunk vs tail of its predecessor):

| | xcorr peak |
|---|---|
| Carried overlap pairs (n=117) | **mean 1.000, min 1.000** |
| Non-adjacent control (n≈30) | mean 0.143, max 0.375 |

Perfect separation. Implications:
1. The segmenter's literal-waveform invariant holds exactly — 117/117.
2. **The deferred fuzzy-overlap dedupe problem (spec §4.3) has a
   deterministic solution shape**: cross-correlate head-vs-prev-tail;
   when the peak is ~1.0 the duplicated region is identified at sample
   resolution — the post-processor can then dedupe/trimm with zero text
   risk, regardless of how differently two chunks' TRANSLATIONS render
   the same audio. Token-based dedupe remains the fallback for
   non-verbatim overlaps.

## Design note (honest correction)

The originally filed CAM++ speaker-embedding approach is the WRONG
instrument for this dedupe: speaker embeddings are content-invariant by
design, so in a single-speaker stream they cannot separate duplicated
audio from novel audio (always "same speaker"). Speaker embeddings
remain filed for genuinely multi-speaker problems (diarization,
cross-talk separation). The cross-correlation method replaces the idea
and is strictly better here: deterministic, zero model, sample-exact.

## Artifacts

- `outputs/segmentation_eval.json` (full sweep + overlap detail)
