"""Import recorded eval-set audio into the harness contract.

Pairs audio recordings with the reading-sheet clips (en/ar/en-ar slices
of eval/clips_v1.json) and writes data/eval_v1/<id>.wav + <id>.txt -
the exact contract the eval harness consumes. References come from the
kit (exact by construction), never from transcription.

Hand-over modes:

1. FOLDER (default): one audio file per clip.
   - A filename containing the clip id (e.g. "take_en_07.m4a") matches
     that clip directly.
   - Remaining files are matched in sort order to pending clips.

2. LONG TAKE (--long-take FILE): one continuous recording; clips are
   split at silences >= --gap seconds and paired sequentially to the
   pending clips in order.

Any format librosa reads is accepted; everything is resampled to
16 kHz mono PCM_16. Already-imported clips are skipped unless forced.

Usage:
  uv run python scripts/import_eval_recordings.py --from-folder D:/rec
  uv run python scripts/import_eval_recordings.py --long-take take1.m4a
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

KIT = Path("eval/clips_v1.json")
OUT = Path("data/eval_v1")
TARGET_SR = 16000
AUDIO_EXTS = (".wav", ".m4a", ".mp3", ".ogg", ".flac", ".webm", ".aac")


def recordable_clips() -> list[dict]:
    clips = json.loads(KIT.read_text(encoding="utf-8"))["clips"]
    order = {"en": 0, "ar": 1, "en-ar": 2}
    rec = [c for c in clips if c["slice"] in order]
    rec.sort(key=lambda c: (order[c["slice"]], c["id"]))
    return rec


def clip_id_from_name(path: Path) -> str | None:
    m = re.search(r"(en|ar|enar)[_ -]?(\d{1,2})", path.stem, re.IGNORECASE)
    if not m:
        return None
    cand = f"{m.group(1).lower()}_{int(m.group(2)):02d}"
    return cand if any(c["id"] == cand for c in recordable_clips()) else None


def split_on_silence(audio: np.ndarray, gap_s: float) -> list[np.ndarray]:
    win = TARGET_SR // 10
    if len(audio) < win:
        return [audio]
    rms = np.array([np.sqrt(np.mean(audio[i:i + win] ** 2))
                    for i in range(0, len(audio) - win + 1, win)])
    thr = max(0.004, float(np.median(rms[rms > 0])) * 0.12)
    quiet = rms < thr
    bounds = [0]
    i = 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * 0.1 >= gap_s:
                bounds.extend((i * win, j * win))
            i = j
        else:
            i += 1
    bounds.append(len(audio))
    return [audio[a:b] for a, b in zip(bounds[::2], bounds[1::2])
            if b - a > TARGET_SR // 2]


def write_clip(cid: str, clip: dict, audio: np.ndarray,
               peak: float) -> str:
    import soundfile as sf

    sf.write(str(OUT / f"{cid}.wav"), audio, TARGET_SR, subtype="PCM_16")
    (OUT / f"{cid}.txt").write_text(clip["reference_en"].strip(),
                                    encoding="utf-8")
    return "OK" if peak >= 0.02 else "SILENT?"


def main() -> None:
    import librosa
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--from-folder", default=None,
                    help="folder of per-clip recordings (any names)")
    ap.add_argument("--long-take", default=None,
                    help="one continuous recording file")
    ap.add_argument("--gap", type=float, default=2.0,
                    help="silence seconds separating clips in a long take")
    ap.add_argument("--force", action="store_true",
                    help="re-import clips that already exist")
    args = ap.parse_args()

    rec = recordable_clips()
    out = Path(args.out) if hasattr(args, "out") else OUT
    done_ids = {p.stem for p in out.glob("*.wav")} if out.exists() else set()
    pending = [c for c in rec if args.force or c["id"] not in done_ids]
    print(f"{len(rec)} recordable | {len(done_ids)} imported | "
          f"{len(pending)} pending", flush=True)
    if not pending:
        print("nothing pending; evaluate with:")
        print(f"  uv run sawti eval {out} --target eng --engine real")
        return

    pieces: list[tuple[str, np.ndarray]] = []
    if args.long_take:
        audio, _ = librosa.load(args.long_take, sr=TARGET_SR, mono=True)
        for i, piece in enumerate(split_on_silence(audio, args.gap)):
            pieces.append((f"take_{i:03d}", piece))
    elif args.from_folder:
        d = Path(args.from_folder)
        files = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in AUDIO_EXTS)
        for f in files:
            audio, _ = librosa.load(str(f), sr=TARGET_SR, mono=True)
            pieces.append((f.stem, audio))
    else:
        print("give --from-folder or --long-take")
        sys.exit(1)

    # pairing: explicit id in filename wins; otherwise sort order.
    id_match = {}
    for tag, audio in pieces:
        cid = clip_id_from_name(Path(tag))
        if cid and cid in {c["id"] for c in pending}:
            id_match[cid] = (tag, audio)
    pairs: list[tuple[str, np.ndarray]] = []
    unmatched = [(t, a) for t, a in pieces
                 if clip_id_from_name(Path(t)) is None
                 or clip_id_from_name(Path(t)) not in id_match]
    for cid, c in [(c["id"], c) for c in pending
                   if c["id"] not in id_match]:
        if unmatched:
            pairs.append((cid, unmatched.pop(0)[1]))
    for cid, (tag, audio) in id_match.items():
        pairs.append((cid, audio))

    done, flagged = [], []
    for cid, audio in pairs:
        clip = next(c for c in rec if c["id"] == cid)
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        state = write_clip(cid, clip, audio, peak)
        if state == "OK":
            done.append(cid)
            print(f"  {cid}: OK  peak {peak:.3f}", flush=True)
        else:
            flagged.append((cid, peak))
            print(f"  {cid}: SILENT? peak {peak:.4f} - re-record",
                  flush=True)

    still = [c["id"] for c in pending
             if not (out / f"{c['id']}.wav").exists()]
    print(f"\nimported {len(done)} | flagged: {flagged or 'none'} | "
          f"still missing: {len(still)} ({', '.join(still) or '-'})")
    print("evaluate with:")
    print(f"  uv run sawti eval {out} --target eng --engine real")


if __name__ == "__main__":
    main()
