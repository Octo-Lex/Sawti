"""Segmentation boundary-F1 + overlap-carry verification (2026-10-13).

Item 1 (spec 7.4, never measured): run the production segmenter over
sample/000519_4.wav (7.8 min continuous, 76 human-timestamped utterance
boundaries) across pause_threshold_ms settings; report boundary P/R/F1,
over-fragmentation, oversized rate, and reference coverage.

Item 2 (overlap-dedupe feasibility, deterministic): the segmenter's
overlap carry is a VERBATIM waveform copy by construction - verify with
sample-level cross-correlation between each chunk's head and the
previous chunk's tail. Carried pairs should peak ~1.0 at lag 0;
dropped/fresh opens should not. No ML involved.

CPU-only by design (Silero VAD on CPU): coexists with GPU training.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

REF = json.loads(json.dumps([
    json.loads(l) for l in
    Path("data/eval_real/manifest.jsonl").read_text(encoding="utf-8").splitlines()
    if json.loads(l)["source_file"] == "000519_4.wav"]))
WAV = "sample/000519_4.wav"
TOLERANCE_S = 0.30
THRESHOLDS = [250, 350, 500, 600]


def load_audio():
    import soundfile as sf

    audio, sr = sf.read(WAV, dtype="float32")
    assert sr == 16000
    return audio, sr


def frames(audio, sr, frame_samples=16000):
    from sawti.sources import AudioFrame

    for i in range(0, len(audio), frame_samples):
        block = audio[i:i + frame_samples]
        yield AudioFrame(audio=np.ascontiguousarray(block), sample_rate=sr,
                         timestamp_s=i / sr)


def segment(audio, sr, pause_ms):
    from sawti.config import SegmentationConfig
    from sawti.segmenter_silero import RealSegmenter
    from sawti.vad import SileroVad

    seg = RealSegmenter(vad=SileroVad(),
                        config=SegmentationConfig(pause_threshold_ms=pause_ms))
    return list(seg.process(frames(audio, sr)))


def boundary_prf(chunk_ends, ref_ends, tol=TOLERANCE_S):
    """Boundary found if a predicted end lies within +-tol of a ref end."""
    used = set()
    hits = 0
    for r in ref_ends:
        for j, p in enumerate(chunk_ends):
            if j not in used and abs(p - r) <= tol:
                used.add(j)
                hits += 1
                break
    precision = hits / len(chunk_ends) if chunk_ends else 0.0
    recall = hits / len(ref_ends) if ref_ends else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return precision, recall, f1


def main():
    audio, sr = load_audio()
    ref_starts = [r["start_s"] for r in REF]
    ref_ends = [r["end_s"] for r in REF]
    n_ref = len(REF)
    results = []
    chunks_350 = None

    for thr in THRESHOLDS:
        t0 = time.perf_counter()
        chunks = segment(audio, sr, thr)
        wall = time.perf_counter() - t0
        ends = [c.end_time for c in chunks[:-1]]  # internal boundaries
        p, r, f1 = boundary_prf(ends, ref_ends[1:])  # ref internal too
        covered = sum(
            1 for rs, re_ in zip(ref_starts, ref_ends)
            if any(c.start_time < re_ and c.end_time > rs for c in chunks))
        durs = [c.end_time - c.start_time for c in chunks]
        results.append({
            "pause_threshold_ms": thr, "n_chunks": len(chunks),
            "boundary_precision": round(p, 3),
            "boundary_recall": round(r, 3),
            "boundary_f1": round(f1, 3),
            "ref_coverage": f"{covered}/{n_ref}",
            "oversized_gt12s": sum(1 for d in durs if d > 12.0),
            "tiny_lt1s": sum(1 for d in durs if d < 1.0),
            "wall_s": round(wall, 1),
        })
        if thr == 350:
            chunks_350 = chunks
        print(f"[thr={thr}] {results[-1]}", flush=True)

    # ---- overlap-carry verification on the 350ms run ---------------------
    def xcorr_peak(a, b):
        n = min(len(a), len(b))
        if n < 1600:  # need >=100ms
            return None
        a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
        a = (a - a.mean()) / (a.std() + 1e-9)
        b = (b - b.mean()) / (b.std() + 1e-9)
        best = -1.0
        for lag in range(-800, 801, 8):  # +-50ms
            if lag >= 0:
                x, y = a[lag:], b[:len(a) - lag]
            else:
                x, y = a[:len(a) + lag], b[-lag:]
            if len(x) < 800:
                continue
            c = float(np.dot(x, y) / len(x))
            best = max(best, c)
        return best

    overlap_stats = {"carried": [], "dropped": []}
    for prev, cur in zip(chunks_350, chunks_350[1:]):
        ov = cur.overlap_from_prev_s
        if ov <= 0:
            continue
        k = int(ov * sr)
        peak = xcorr_peak(cur.audio[:k], prev.audio[-k:])
        key = "carried" if ov > 0.25 else "carried_partial"
        overlap_stats["carried" if ov > 0.25 else "dropped"].append(
            {"overlap_s": round(ov, 2), "xcorr_peak": round(peak, 3)
             if peak is not None else None})
    # control: NON-adjacent same-position windows (random pairs, no carry)
    import random

    rng = random.Random(0)
    control = []
    for _ in range(30):
        i, j = rng.sample(range(len(chunks_350)), 2)
        if abs(i - j) < 2:
            continue
        k = 4800  # 300ms windows
        a = chunks_350[i].audio[-k:]
        b = chunks_350[j].audio[:k]
        peak = xcorr_peak(b, a)
        if peak is not None:
            control.append(round(peak, 3))

    carried = [x["xcorr_peak"] for x in overlap_stats["carried"]
               if x["xcorr_peak"] is not None]
    out = {
        "boundary_sweep": results,
        "overlap_verification": {
            "n_carried": len(carried),
            "carried_mean_peak": round(float(np.mean(carried)), 3) if carried else None,
            "carried_min_peak": round(float(np.min(carried)), 3) if carried else None,
            "control_mean_peak": round(float(np.mean(control)), 3) if control else None,
            "control_max": round(float(np.max(control)), 3) if control else None,
            "detail": overlap_stats,
        },
    }
    Path("outputs/segmentation_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["overlap_verification"], indent=1)[:600])


if __name__ == "__main__":
    main()
