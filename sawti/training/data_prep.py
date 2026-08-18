"""SADA data prep: dialect census, Saudi filter, materialization (SA Task 2).

Usage (OPERATOR):
  uv run python -m sawti.training.data_prep --split validation --out data/sada_training/val --cap 200
  uv run python -m sawti.training.data_prep --split train --out data/sada_training/train

Streams the split (no full 50GB download), decodes audio bytes with
soundfile (torchcodec is Windows-hostile), writes wav + manifest.jsonl +
census.json. Saudi labels: spike-confirmed core + any census labels the
operator adds via --extra-label after inspecting census.json.
"""
from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from sawti.env import load_env

CORE = ["Najdi", "Hijazi", "Khaliji"]
MAX_S = 30.0   # Whisper native window; longer clips are dropped (logged)
MIN_S = 0.5


def census_labels(rows) -> dict[str, int]:
    c: Counter = Counter()
    for r in rows:
        c[r.get("speaker_dialect") or "unknown"] += 1
    return dict(c)


def saudi_label_set(census: dict[str, int], extra: list[str]) -> set[str]:
    return set(CORE) | set(extra)


def keep_row(row: dict, labels: set[str]) -> bool:
    if row.get("speaker_dialect") not in labels:
        return False
    if not (MIN_S <= float(row.get("duration_s", 0)) <= MAX_S):
        return False
    return bool((row.get("cleaned_text") or row.get("text") or "").strip())


def _decode(audio_entry: dict):
    arr, sr = sf.read(io.BytesIO(audio_entry["bytes"]), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, sr


def materialize(split: str, out: str, extra_labels: list[str],
                cap: int | None) -> dict:
    from datasets import Audio, load_dataset

    ds = load_dataset("MohamedRashad/SADA22", split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    dropped = Counter()
    manifest, kept_hours, seen = [], 0.0, 0
    labels = saudi_label_set({}, extra_labels)
    for row in ds:
        seen += 1
        arr, sr = _decode(row["audio"])
        rec = {
            "speaker_dialect": row.get("speaker_dialect"),
            "cleaned_text": row.get("cleaned_text") or "",
            "text": row.get("text") or "",
            "duration_s": round(len(arr) / sr, 2),
        }
        if not keep_row(rec, labels):
            if rec["speaker_dialect"] in labels:
                dropped["duration_or_text"] += 1
            else:
                dropped["dialect"] += 1
            continue
        clip_id = f"{split}_{len(manifest):06d}"
        sf.write(out_p / f"{clip_id}.wav", arr, sr)
        manifest.append(
            {"clip_id": clip_id, "dialect": rec["speaker_dialect"],
             "cleaned_text": rec["cleaned_text"], "text": rec["text"],
             "duration_s": rec["duration_s"],
             "speaker_gender": row.get("speaker_gender", ""),
             "speaker_age": row.get("speaker_age", "")})
        kept_hours += rec["duration_s"] / 3600
        if cap and len(manifest) >= cap:
            break
    stats = {
        "split": split, "scanned": seen, "kept": len(manifest),
        "kept_hours": round(kept_hours, 2), "dropped": dict(dropped),
        "labels": sorted(labels),
        "dialect_counts": dict(Counter(m["dialect"] for m in manifest)),
    }
    (out_p / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest),
        encoding="utf-8")
    (out_p / "census.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def main() -> None:
    load_env()  # operator entry edge; OS wins by default
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["train", "validation", "test"])
    p.add_argument("--out", required=True)
    p.add_argument("--extra-label", action="append", default=[])
    p.add_argument("--cap", type=int, default=None)
    a = p.parse_args()
    stats = materialize(a.split, a.out, a.extra_label, a.cap)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
