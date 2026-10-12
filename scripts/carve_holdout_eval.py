"""Carve the holdout eval set from sample/ into harness-ready form.

Inputs (real Saudi dialectal material, verified non-overlapping with all
160,110 materialized SADA clips by audio SHA-256, 2026-10-12):

  sample/000098_2-0010..0014.wav + .txt   5 short clips, plain refs
  sample/000519_4.wav + .txt              7.8 min, 76 timestamped
                                         segments (start TAB end TAB
                                         speaker TAB text)

Outputs to data/eval_real/:
  holdout_NNN.wav / .txt   the harness contract (sawti eval / SA evaluator)
  manifest.jsonl           the frozen-evaluator contract (clip_id, text,
                           cleaned_text, dialect "Unknown", duration_s,
                           provenance fields)

Deterministic: shorts first in numeric order, then long-file segments in
time order. Segments < 0.30 s are skipped (unusable slivers).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import soundfile as sf

SRC = Path("sample")
OUT = Path("data/eval_real")
MIN_SEG_S = 0.30


def parse_segments(txt: Path) -> list[dict]:
    segs = []
    for i, line in enumerate(txt.read_text(encoding="utf-8").splitlines()):
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        start, end, speaker, text = (parts[0].strip(), parts[1].strip(),
                                     parts[2].strip(), "\t".join(parts[3:]).strip())
        try:
            s, e = float(start), float(end)
        except ValueError:
            continue
        if e - s >= MIN_SEG_S and text:
            segs.append({"start_s": s, "end_s": e, "speaker": speaker,
                         "text": text, "source_index": i})
    return segs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    n = 0

    # 1. the five short clips (copied verbatim)
    for wav in sorted(SRC.glob("000098_2-*.wav")):
        n += 1
        cid = f"holdout_{n:03d}"
        shutil.copy(wav, OUT / f"{cid}.wav")
        text = wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
        (OUT / f"{cid}.txt").write_text(text, encoding="utf-8")
        info = sf.info(str(wav))
        rows.append({"clip_id": cid, "source_file": wav.name,
                     "dialect": "Unknown", "text": text,
                     "cleaned_text": text, "duration_s": round(info.duration, 2)})

    # 2. the long recording, carved at its timestamps
    long_wav = SRC / "000519_4.wav"
    audio, sr = sf.read(str(long_wav), dtype="float32")
    for seg in parse_segments(long_wav.with_suffix(".txt")):
        n += 1
        cid = f"holdout_{n:03d}"
        clip = audio[int(seg["start_s"] * sr):int(seg["end_s"] * sr)]
        sf.write(str(OUT / f"{cid}.wav"), clip, sr, subtype="PCM_16")
        (OUT / f"{cid}.txt").write_text(seg["text"], encoding="utf-8")
        rows.append({"clip_id": cid, "source_file": long_wav.name,
                     "seg_index": seg["source_index"],
                     "start_s": round(seg["start_s"], 2),
                     "end_s": round(seg["end_s"], 2),
                     "dialect": "Unknown", "text": seg["text"],
                     "cleaned_text": seg["text"],
                     "duration_s": round(len(clip) / sr, 2)})

    with (OUT / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    durs = sorted(r["duration_s"] for r in rows)
    total = sum(durs)
    print(f"carved {len(rows)} clips into {OUT}")
    print(f"duration: min {durs[0]:.2f}s | median {durs[len(durs)//2]:.2f}s | "
          f"max {durs[-1]:.2f}s | total {total/60:.1f} min")
    print(f"skipped segments < {MIN_SEG_S}s: "
          f"{76 - sum(1 for r in rows if r['source_file'] == '000519_4.wav')}")


if __name__ == "__main__":
    main()
