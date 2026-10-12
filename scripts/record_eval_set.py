"""Interactive recorder for the self-recorded eval set (spec 7.1).

Records read-aloud clips from eval/clips_v1.json through the PRODUCTION
MicSource (16 kHz / mono / float32, fail-loud contract) into
data/eval_v1/, writing per clip:

    data/eval_v1/<id>.wav   16-bit PCM (production capture path)
    data/eval_v1/<id>.txt   the reference_en text (harness contract)

Read-aloud design: reference text is exact by construction — the txt is
written from the manifest, never transcribed.

Run (operator):

    .venv\\Scripts\\python.exe scripts\\record_eval_set.py
    .venv\\Scripts\\python.exe scripts\\record_eval_set.py --only en,ar
    .venv\\Scripts\\python.exe scripts\\record_eval_set.py --re-record ar_03

Keys per clip: Enter = record now, s = skip, r = re-record previous,
q = quit. Already-recorded clips are skipped automatically (use
--re-record to overwrite). A silent take (peak < 0.02) is flagged for
re-recording.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # Arabic/French text on Windows

import numpy as np
import soundfile as sf

from sawti.mic_source import MicSource, TARGET_SR

SILENCE_PEAK = 0.02


def est_seconds(text: str) -> int:
    """Read-aloud duration estimate: ~2.3 words/s + pad, clamped 6..20."""
    words = max(1, len(text.split()))
    return max(6, min(20, round(words / 2.3 + 4)))


def record_clip(clip: dict, out_dir: Path, device=None) -> dict:
    """Record one clip; returns {'ok', 'peak', 'seconds', 'why'}."""
    dur = est_seconds(clip["source_text"])
    for step in ("3", "2", "1"):
        print(f"   {step}...", flush=True)
        time.sleep(0.8)
    print("   >>> SPEAK NOW", flush=True)

    src = MicSource(device=device)
    frames = []
    t0 = time.perf_counter()
    try:
        for frame in src.iter_frames():
            frames.append(frame.audio)
            if time.perf_counter() - t0 >= dur:
                break
    finally:
        src.close()
    s = src.stats()
    audio = np.concatenate(frames) if frames else np.zeros(0)
    if not s["captured_samples"] or s["input_overflow_count"]:
        return {"ok": False, "peak": 0.0, "seconds": 0.0,
                "why": f"capture problem (overflows="
                       f"{s['input_overflow_count']})"}

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    wav = out_dir / f"{clip['id']}.wav"
    txt = out_dir / f"{clip['id']}.txt"
    sf.write(str(wav), audio, TARGET_SR, subtype="PCM_16")
    txt.write_text(clip["reference_en"].strip(), encoding="utf-8")
    ok = peak >= SILENCE_PEAK
    return {"ok": ok, "peak": peak,
            "seconds": len(audio) / TARGET_SR,
            "why": None if ok else "silent take (mic muted / not worn?)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="eval/clips_v1.json")
    ap.add_argument("--out", default="data/eval_v1")
    ap.add_argument("--only", help="comma-separated slices, e.g. en,ar")
    ap.add_argument("--re-record", help="clip id to force re-recording")
    ap.add_argument("--device", default=None)
    ap.add_argument("--yes", action="store_true",
                    help="no input prompts — record every pending clip "
                         "immediately (for single-clip re-records)")
    ap.add_argument("--list", action="store_true",
                    help="show the plan and exit (no mic)")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    clips = manifest["clips"]
    out_dir = Path(args.out)

    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        clips = [c for c in clips if c["slice"] in keep]
    if args.re_record:
        clips = [c for c in clips if c["id"] == args.re_record]
        if not clips:
            sys.exit(f"no clip with id {args.re_record!r}")
    if args.list:
        for c in clips:
            status = "DONE" if (out_dir / f"{c['id']}.wav").exists() else "-"
            print(f"{status:4} {c['id']:<9} {c['slice']:<6} "
                  f"~{est_seconds(c['source_text'])}s  {c['source_text'][:60]}")
        print(f"\n{len(clips)} clips | output dir: {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    done_before = {p.stem for p in out_dir.glob("*.wav")}
    todo = [c for c in clips
            if c["id"] not in done_before or c["id"] == args.re_record]
    print(f"{len(clips)} in plan | {len(done_before & {c['id'] for c in clips})} "
          f"already recorded | {len(todo)} to record now")
    if not todo:
        print("Nothing to do. Evaluate with:")
        print("  uv run sawti eval " + str(out_dir) + " --target eng --engine real")
        return

    flagged: list[str] = []
    recorded = 0
    for i, clip in enumerate(todo):
        print("\n" + "=" * 62)
        print(f"[{i + 1}/{len(todo)}] {clip['id']} "
              f"(slice {clip['slice']}, ~{est_seconds(clip['source_text'])}s)")
        print("-" * 62)
        print(clip["source_text"])
        print("-" * 62)
        while True:
            if args.yes:
                break
            key = input("Enter=record  s=skip  q=quit > ").strip().lower()
            if key in ("", "r"):
                break
            if key == "s":
                print("   skipped")
                clip = None
                break
            if key == "q":
                print("bye — progress is saved; rerun to resume")
                _summary(recorded, flagged, out_dir)
                return
        if clip is None:
            continue
        res = record_clip(clip, out_dir, args.device)
        recorded += 1
        if res["ok"]:
            print(f"   saved {clip['id']}.wav  {res['seconds']:.1f}s  "
                  f"peak {res['peak']:.2f}")
        else:
            print(f"   !! {res['why']} (peak {res['peak']:.2f}) — "
                  f"re-record with: --re-record {clip['id']}")
            flagged.append(clip["id"])
    _summary(recorded, flagged, out_dir)


def _summary(recorded: int, flagged: list[str], out_dir: Path) -> None:
    n = len(list(out_dir.glob("*.wav")))
    print("\n" + "=" * 62)
    print(f"recorded this run: {recorded} | total in set: {n} "
          f"| flagged: {flagged or 'none'}")
    print("Evaluate with:")
    print(f"  uv run sawti eval {out_dir} --target eng --engine real")


if __name__ == "__main__":
    main()
