"""Shared Saudi-eval logic: one authoritative metric recipe (SA plan Task 1,
reconciled against M1).

Detector: the SHARED production implementation (sawti.loop_detect) —
training-time selection, the evaluator, and the runtime gate agree by
construction; there is no SA-specific loop fork.

Metric recipe (the selection regime): clean macro WER (checkpoint gate)
+ n-gram loop-rate (eligibility constraint) + all-valid macro AND corpus
WER (robustness views, both reported) + per-dialect clean metrics.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from sawti.loop_detect import is_loop  # shared production detector
from sawti.text_normalize import normalize_arabic_for_match


def norm(text: str) -> str:
    t = normalize_arabic_for_match(text)
    t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t)
    return " ".join(t.split())


def wer_clean(ref: str, hyp: str) -> float | None:
    """WER after normalization; None when the reference normalizes to
    zero words (empty/invalid). None — never NaN — so aggregation and
    checkpoint comparison can never be poisoned by a non-finite value
    (matches the M1 evaluator convention)."""
    import jiwer

    n_ref, n_hyp = norm(ref), norm(hyp)
    if not n_ref:
        return None
    return jiwer.wer(n_ref, n_hyp)


def load_manifest(data_dir: str | Path, name: str = "manifest.jsonl") -> list[dict]:
    """Manifest loader with a view seam: name selects manifest.jsonl
    (official) or a carved view (e.g. manifest_diagnostic.jsonl)."""
    return [
        json.loads(line)
        for line in (Path(data_dir) / name).read_text(encoding="utf-8").splitlines()
    ]


def annotate_degenerate(rows: list[dict]) -> list[dict]:
    """Sets every field aggregate() needs: loop flag (shared detector),
    degenerate, valid_ref, and reference word count (for corpus WER)."""
    for r in rows:
        ref = (r.get("cleaned_text") or r.get("text") or "").strip()
        n_ref = norm(ref)
        r["valid_ref"] = bool(n_ref)
        r["n_ref_words"] = len(n_ref.split())
        r["loop"] = is_loop(r.get("hyp", ""))
        r["degenerate"] = (
            r.get("duration_s", 99) < 1.0
            or r["loop"]
            or r.get("wer") is None
            or not r["valid_ref"]
        )
    return rows


def _macro(rows: list[dict]) -> float:
    w = [r["wer"] for r in rows if r.get("wer") is not None]
    return 100 * float(np.mean(w)) if w else float("nan")


def _corpus(rows: list[dict]) -> float:
    """Corpus WER over valid-reference rows: per-clip WER weighted by
    reference word count (exact from stored values)."""
    err = total = 0.0
    for r in rows:
        if r.get("wer") is None:
            continue
        err += r["wer"] * r["n_ref_words"]
        total += r["n_ref_words"]
    return 100 * err / total if total else float("nan")


def aggregate(rows: list[dict]) -> dict:
    """The four-metric recipe + per-dialect clean metrics."""
    by_d: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_d[r["dialect"]].append(r)
    per = {}
    for d, rs in by_d.items():
        clean = [r for r in rs if not r["degenerate"]]
        per[d] = {"clean_macro_wer": _macro(clean), "n_clean": len(clean)}
    valid = [r for r in rows if r.get("valid_ref")]
    clean_all = [r for r in rows if not r["degenerate"]]
    return {
        "clean_macro_wer": _macro(clean_all),
        "all_valid_macro_wer": _macro(valid),
        "all_valid_corpus_wer": _corpus(valid),
        "loop_pct": 100 * sum(bool(r.get("loop")) for r in rows) / max(1, len(rows)),
        "degenerate_rate": 100 * sum(bool(r["degenerate"]) for r in rows) / max(1, len(rows)),
        "per_dialect": per,
        "n": len(rows),
    }


def run_eval(asr_fn, data_dir: str | Path) -> list[dict]:
    """asr_fn(wav_path) -> str. Returns annotated rows (writes nothing)."""
    rows = []
    for m in load_manifest(data_dir):
        hyp = asr_fn(str(Path(data_dir) / f"{m['clip_id']}.wav"))
        ref = (m.get("cleaned_text") or m.get("text") or "").strip()
        rows.append({**m, "hyp": hyp, "wer": wer_clean(ref, hyp)})
    return annotate_degenerate(rows)
