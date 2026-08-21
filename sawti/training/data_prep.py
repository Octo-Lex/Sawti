"""SADA data prep: genuine label census, Saudi filter, provenance-rich
materialization, and split-disjointness verification (SA Task 2, corrective).

Usage (OPERATOR):
  uv run python -m sawti.training.data_prep --split validation --census-only
      # Full-split label inventory; no WAVs, no audio decode.
  uv run python -m sawti.training.data_prep --split validation --out data/sada_training/val
  uv run python -m sawti.training.data_prep --split train --out data/sada_training/train

Experimental structure (locked after review):
  train       -> training
  validation  -> checkpoint selection / dev metrics (Task 5 --dev)
  test        -> final SA acceptance only
  data/sada_spike (test-derived 75 clips) -> frozen historical baseline,
      NEVER selection data.

Provenance: every manifest row records source_split, source_ordinal
(row position in the streamed split), and audio_sha256 (SHA-256 of the
ORIGINAL audio bytes, pre-decode) so train/validation disjointness is
verifiable: `assert_no_overlap(train_dir, val_dir)`. SADA22 exposes no
native clip identifier (columns: audio/text/cleaned_text/speaker_*),
so the byte hash is the fingerprint.
"""
from __future__ import annotations

import argparse
import hashlib
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


def _stream(split: str):
    from datasets import Audio, load_dataset

    ds = load_dataset("MohamedRashad/SADA22", split=split, streaming=True)
    return ds.cast_column("audio", Audio(decode=False)), split


def census_only(split: str, out: str | None = None) -> dict:
    """Full-split label inventory WITHOUT decoding audio or writing WAVs.

    Records every speaker_dialect label with its count, plus an
    empty-transcript count (metadata-only; duration filtering requires
    decode and is deliberately excluded from census semantics)."""
    ds, _ = _stream(split)
    label_counts: Counter = Counter()
    empty_text = 0
    seen = 0
    for row in ds:
        seen += 1
        label_counts[row.get("speaker_dialect") or "unknown"] += 1
        if not ((row.get("cleaned_text") or row.get("text") or "").strip()):
            empty_text += 1
    stats = {
        "split": split, "total_rows": seen,
        "label_inventory": dict(sorted(label_counts.items(),
                                       key=lambda kv: -kv[1])),
        "empty_transcript_rows": empty_text,
    }
    if out:
        out_p = Path(out)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "census.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def materialize(split: str, out: str, extra_labels: list[str],
                cap: int | None) -> dict:
    ds, _ = _stream(split)
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    label_counts: Counter = Counter()      # genuine census over ALL rows
    kept_label_counts: Counter = Counter()
    dropped_by_label: Counter = Counter()  # dropped, by label NAME
    dropped_by_reason: Counter = Counter()
    manifest, kept_hours = [], 0.0
    labels = saudi_label_set({}, extra_labels)
    for ordinal, row in enumerate(ds):
        label = row.get("speaker_dialect") or "unknown"
        label_counts[label] += 1
        raw_bytes = row["audio"]["bytes"]
        arr, sr = _decode(row["audio"])
        rec = {
            "speaker_dialect": label,
            "cleaned_text": row.get("cleaned_text") or "",
            "text": row.get("text") or "",
            "duration_s": round(len(arr) / sr, 2),
        }
        if not keep_row(rec, labels):
            dropped_by_label[label] += 1
            reason = ("duration_or_text" if label in labels else "dialect")
            dropped_by_reason[reason] += 1
            continue
        clip_id = f"{split}_{len(manifest):06d}"
        sf.write(out_p / f"{clip_id}.wav", arr, sr)
        manifest.append(
            {"clip_id": clip_id, "dialect": label,
             "cleaned_text": rec["cleaned_text"], "text": rec["text"],
             "duration_s": rec["duration_s"],
             "speaker_gender": row.get("speaker_gender", ""),
             "speaker_age": row.get("speaker_age", ""),
             # Provenance: split identity, stream ordinal, and a
             # deterministic fingerprint of the ORIGINAL audio bytes.
             "source_split": split,
             "source_ordinal": ordinal,
             "audio_sha256": hashlib.sha256(raw_bytes).hexdigest()})
        kept_label_counts[label] += 1
        kept_hours += rec["duration_s"] / 3600
        if cap and len(manifest) >= cap:   # cap = KEPT rows
            break
    stats = {
        "split": split, "scanned": len(label_counts.values()) and sum(
            label_counts.values()),
        "kept": len(manifest), "kept_hours": round(kept_hours, 2),
        "labels": sorted(labels),
        # Genuine census: every label seen, kept, and dropped by name.
        "label_inventory": dict(sorted(label_counts.items(),
                                       key=lambda kv: -kv[1])),
        "kept_dialect_counts": dict(kept_label_counts),
        "dropped_by_label": dict(sorted(dropped_by_label.items(),
                                        key=lambda kv: -kv[1])),
        "dropped_by_reason": dict(dropped_by_reason),
        "dialect_counts": dict(kept_label_counts),  # backward-compat alias
    }
    (out_p / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest),
        encoding="utf-8")
    (out_p / "census.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def carve_manifests(source_dir: str | Path, out_dir: str | Path,
                    labels_to_carve: list[str],
                    source_name: str = "manifest.jsonl",
                    core_name: str = "manifest_core.jsonl",
                    carved_name: str = "manifest_diagnostic.jsonl") -> dict:
    """Deterministically split a materialized manifest into VIEWS by label.

    The source-faithful manifest is NOT mutated: this writes two new
    manifest files in out_dir referencing the SAME WAVs (paths resolved
    against source_dir). The core view EXCLUDES the carved labels; the
    diagnostic view contains ONLY them. This is a manifest-selection
    seam, not dialect policy — the caller decides which labels to carve.
    """
    src = Path(source_dir)
    rows = [json.loads(l) for l in (src / source_name).read_text(
        encoding="utf-8").splitlines()]
    carve = set(labels_to_carve)
    core, diag = [], []
    for r in rows:
        (diag if r.get("dialect") in carve else core).append(r)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    for name, rows_ in ((core_name, core), (carved_name, diag)):
        (out_p / name).write_text(
            chr(10).join(json.dumps(r, ensure_ascii=False) for r in rows_),
            encoding="utf-8")
    return {"source_rows": len(rows), "core_rows": len(core),
            "carved_rows": len(diag), "carved_labels": sorted(carve)}


def manifest_audio_hashes(data_dir: str | Path,
                          manifest_name: str = "manifest.jsonl") -> set[str]:
    """All audio_sha256 values in a materialized manifest."""
    hashes = set()
    for line in (Path(data_dir) / manifest_name).read_text(
            encoding="utf-8").splitlines():
        hashes.add(json.loads(line)["audio_sha256"])
    return hashes


def assert_no_overlap(dir_a: str | Path, dir_b: str | Path) -> None:
    """Raise if any ORIGINAL audio appears in both manifests (split
    leakage guard, run before training)."""
    shared = manifest_audio_hashes(dir_a) & manifest_audio_hashes(dir_b)
    if shared:
        raise AssertionError(
            f"{len(shared)} audio clip(s) present in BOTH manifests "
            f"({dir_a} and {dir_b}) — split leakage; refusing to train. "
            f"First offenders: {sorted(shared)[:5]}")


def main() -> None:
    load_env()  # operator entry edge; OS wins by default
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True,
                   choices=["train", "validation", "test"])
    p.add_argument("--out", default=None)
    p.add_argument("--extra-label", action="append", default=[])
    p.add_argument("--cap", type=int, default=None,
                   help="stop after N KEPT rows (smoke runs)")
    p.add_argument("--census-only", action="store_true",
                   help="full-split label inventory; no WAVs, no decode")
    a = p.parse_args()
    if a.census_only:
        stats = census_only(a.split, a.out)
    else:
        if not a.out:
            p.error("--out is required without --census-only")
        stats = materialize(a.split, a.out, a.extra_label, a.cap)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
