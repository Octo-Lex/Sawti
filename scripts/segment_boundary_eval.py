"""Segmentation boundary-F1 + overlap-carry verification (2026-10-13).

Research mode (default): sweep pause thresholds over the committed
continuous-speech fixture, report boundary P/R/F1, coverage,
fragmentation stats, and deterministic overlap-carry verification.

Gate mode (--gate): ONE segmentation pass at the default threshold;
fails (exit 1) if boundary F1 falls below the floor or ANY reference
boundary is uncovered (the never-drop invariant). Wired into CI so
segmentation changes are quality-gated by construction, not review
diligence. CPU-only: Silero ships inside the silero-vad wheel, no
network, no GPU.

Fixture: tests/fixtures/segmentation/ - 7.8 min continuous Saudi
Arabic with 76 human-timestamped utterance boundaries.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

FIXTURE_DIR = Path("tests/fixtures/segmentation")
DEFAULT_AUDIO = FIXTURE_DIR / "continuous_ar_sa.wav"
DEFAULT_BOUNDARIES = FIXTURE_DIR / "boundaries.json"
TOLERANCE_S = 0.30

# CI floors (2026-10-13 measurement: F1 0.594 at thr=350; precision is
# pinned ~0.49 across sweeps so the floor sits below observed noise).
GATE_THRESHOLD_MS = 350
GATE_F1_FLOOR = 0.55


def evaluate_gate(f1: float, covered: int, n_ref: int,
                  floor: float = GATE_F1_FLOOR) -> tuple[bool, list[str]]:
    """Pure gate verdict (unit-tested): F1 floor + never-drop coverage."""
    problems = []
    if f1 < floor:
        problems.append(f"boundary F1 {f1:.3f} below floor {floor}")
    if covered < n_ref:
        problems.append(f"coverage {covered}/{n_ref} - reference "
                        "utterance(s) with no chunk (never-drop violated)")
    return (not problems), problems


def load_boundaries(path: Path = DEFAULT_BOUNDARIES) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["segments"]


def load_audio(path: Path = DEFAULT_AUDIO):
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
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


def run_pass(chunks, ref_starts, ref_ends):
    ends = [c.end_time for c in chunks[:-1]]
    p, r, f1 = boundary_prf(ends, ref_ends[1:])
    covered = sum(
        1 for rs, re_ in zip(ref_starts, ref_ends)
        if any(c.start_time < re_ and c.end_time > rs for c in chunks))
    return p, r, f1, covered


def xcorr_peak(a, b):
    n = min(len(a), len(b))
    if n < 1600:
        return None
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    best = -1.0
    for lag in range(-800, 801, 8):
        if lag >= 0:
            x, y = a[lag:], b[:len(a) - lag]
        else:
            x, y = a[:len(a) + lag], b[-lag:]
        if len(x) < 800:
            continue
        best = max(best, float(np.dot(x, y) / len(x)))
    return best


def research_mode(thresholds):
    audio, sr = load_audio()
    segs = load_boundaries()
    ref_starts = [s["start_s"] for s in segs]
    ref_ends = [s["end_s"] for s in segs]
    n_ref = len(segs)
    results = []
    chunks_gate = None
    for thr in thresholds:
        t0 = time.perf_counter()
        chunks = segment(audio, sr, thr)
        wall = time.perf_counter() - t0
        p, r, f1, covered = run_pass(chunks, ref_starts, ref_ends)
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
        if thr == GATE_THRESHOLD_MS:
            chunks_gate = chunks
        print(f"[thr={thr}] {results[-1]}", flush=True)

    overlap_stats = {"carried": [], "dropped": []}
    for prev, cur in zip(chunks_gate, chunks_gate[1:]):
        ov = cur.overlap_from_prev_s
        if ov <= 0:
            continue
        k = int(ov * sr)
        peak = xcorr_peak(cur.audio[:k], prev.audio[-k:])
        overlap_stats["carried" if ov > 0.25 else "dropped"].append(
            {"overlap_s": round(ov, 2),
             "xcorr_peak": round(peak, 3) if peak is not None else None})
    import random

    rng = random.Random(0)
    control = []
    for _ in range(30):
        i, j = rng.sample(range(len(chunks_gate)), 2)
        if abs(i - j) < 2:
            continue
        k = 4800
        peak = xcorr_peak(chunks_gate[j].audio[:k], chunks_gate[i].audio[-k:])
        if peak is not None:
            control.append(round(peak, 3))
    carried = [x["xcorr_peak"] for x in overlap_stats["carried"]
               if x["xcorr_peak"] is not None]
    out = {"boundary_sweep": results,
           "overlap_verification": {
               "n_carried": len(carried),
               "carried_mean_peak": round(float(np.mean(carried)), 3) if carried else None,
               "carried_min_peak": round(float(np.min(carried)), 3) if carried else None,
               "control_mean_peak": round(float(np.mean(control)), 3) if control else None,
               "control_max": round(float(np.max(control)), 3) if control else None,
               "detail": overlap_stats}}
    Path("outputs/segmentation_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def gate_mode():
    audio, sr = load_audio()
    segs = load_boundaries()
    ref_starts = [s["start_s"] for s in segs]
    ref_ends = [s["end_s"] for s in segs]
    chunks = segment(audio, sr, GATE_THRESHOLD_MS)
    p, r, f1, covered = run_pass(chunks, ref_starts, ref_ends)
    ok, problems = evaluate_gate(f1, covered, len(segs))
    print(f"[segmentation gate] thr={GATE_THRESHOLD_MS} "
          f"F1={f1:.3f} (floor {GATE_F1_FLOOR}) "
          f"coverage={covered}/{len(segs)} "
          f"chunks={len(chunks)}")
    if ok:
        print("segmentation gate: OK")
        return 0
    print("SEGMENTATION GATE FAILED:")
    for pr in problems:
        print(f"  - {pr}")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true",
                    help="single-threshold CI gate mode")
    ap.add_argument("--thresholds", default="250,350,500,600",
                    help="comma list (research mode)")
    args = ap.parse_args()
    if args.gate:
        sys.exit(gate_mode())
    research_mode([int(t) for t in args.thresholds.split(",")])


if __name__ == "__main__":
    main()
