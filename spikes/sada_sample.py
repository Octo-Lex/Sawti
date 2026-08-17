"""Spike: materialize a stratified Saudi-dialect sample from SADA's test split.

Streams the test split (avoids the 50GB full download), filters to Saudi
dialects, takes a seeded stratified sample, and writes audio + metadata to
data/sada_spike/ for inspection and zero-shot Whisper eval.

Research use only — SADA is CC BY-NC-SA 4.0.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

import sawti.env  # noqa: F401  corrected HF cache path before HF imports

OUT = Path("data/sada_spike")
OUT.mkdir(parents=True, exist_ok=True)

# Saudi core dialects per SADA's 11-value label set; Khaliji is Gulf-broad
# (kept, but reported separately).
SAUDI_DIALECTS = {"Najdi", "Hijazi"}
GULF_BROAD = {"Khaliji"}
TARGET_PER_DIALECT = 25  # ~25 clips each -> ~50-60 total after availability
SEED = 42

def main() -> None:
    from datasets import Audio, load_dataset

    ds = load_dataset(
        "MohamedRashad/SADA22", split="test", streaming=True
    )
    # Decode audio bytes ourselves with soundfile (torchcodec — the datasets
    # lib's new audio decoder — is not Windows-friendly).
    ds = ds.cast_column("audio", Audio(decode=False))
    rng = np.random.default_rng(SEED)

    # Collect candidate indices per dialect (reservoir-free: we just take a
    # seeded random subset per dialect from the first N candidates seen).
    candidates: dict[str, list] = defaultdict(list)
    seen = 0
    CAP_PER_DIALECT = 400  # stop early once we have plenty of candidates

    for row in ds:
        seen += 1
        d = row.get("speaker_dialect", "")
        if d in SAUDI_DIALECTS or d in GULF_BROAD:
            candidates[d].append(row)
        if all(len(v) >= CAP_PER_DIALECT for v in candidates.values()) and \
                len(candidates) >= len(SAUDI_DIALECTS | GULF_BROAD):
            break

    print(f"streamed rows scanned: {seen}")
    for d, rows in candidates.items():
        print(f"  candidates[{d}] = {len(rows)}")

    manifest = []
    for dialect, rows in candidates.items():
        idx = rng.permutation(len(rows))[:TARGET_PER_DIALECT]
        for j, i in enumerate(idx):
            row = rows[i]
            audio = row["audio"]  # undecoded: {"bytes": ..., "path": ...}
            import io

            arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            if arr.ndim > 1:  # stereo -> mono
                arr = arr.mean(axis=1)
            clip_id = f"{dialect}_{j:03d}"
            wav_path = OUT / f"{clip_id}.wav"
            sf.write(wav_path, arr, sr)
            manifest.append(
                {
                    "clip_id": clip_id,
                    "dialect": dialect,
                    "text": row.get("text", ""),
                    "cleaned_text": row.get("cleaned_text", ""),
                    "duration_s": round(len(arr) / sr, 2),
                    "speaker_gender": row.get("speaker_gender", ""),
                    "speaker_age": row.get("speaker_age", ""),
                }
            )

    (OUT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest),
        encoding="utf-8",
    )
    total_min = sum(m["duration_s"] for m in manifest) / 60
    print(f"\nwrote {len(manifest)} clips ({total_min:.1f} min audio) to {OUT}")
    by = defaultdict(float)
    for m in manifest:
        by[m["dialect"]] += m["duration_s"]
    for d, s in by.items():
        print(f"  {d}: {s/60:.1f} min")


if __name__ == "__main__":
    main()
