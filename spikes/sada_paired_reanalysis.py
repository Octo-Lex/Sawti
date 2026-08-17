"""Spike correction: PAIRED model comparison from stored hypotheses.

Fixes the methodology flaw in sada_model_eval.py: clean subsets were computed
per-model (unpaired), so headline deltas averaged over different clips. This
script recomputes everything from stored per-clip hypotheses (no ASI re-runs)
with four views per reviewer spec:

  1. common-clean paired WER  — clips non-degenerate under BOTH models
  2. all-valid-reference WER  — every clip with a non-empty reference
                                (hallucination cost lands in the WER)
  3. degeneracy rates         — reported separately per model
  4. short-clip view          — 0.5–1.0s clips: loop rate + WER (product risk)

Degeneracy rule (uniform, recomputed from stored fields for every model):
duration < 1.0s OR is_loop(hyp) OR wer is None.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sawti.text_normalize import normalize_arabic_for_match

DATA = Path("data/sada_spike")

MODELS = {
    "large-v3 zero-shot": "eval_results.json",
    "turbo zero-shot": "eval_whisper-large-v3-turbo.json",
    "Bruno7 saudi-phase2": "eval_whisper-large-v3-turbo-arabic-saudi-phase2.json",
    "oddadmix dialectal": "eval_whisper-large-v3-turbo-arabic-dialectal_main.json",
    "dev-ahmedhany arabic-ft": "adapter_eval_results.json",
}
BASE = "large-v3 zero-shot"


def norm(text: str) -> str:
    import re

    t = normalize_arabic_for_match(text)
    t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t)
    return " ".join(t.split())


def is_loop(hyp: str) -> bool:
    toks = hyp.split()
    if len(toks) < 6:
        return False
    uniq = len(set(toks)) / len(toks)
    most = Counter(toks).most_common(1)[0][1] / len(toks)
    return uniq < 0.25 or most > 0.6


def load_all() -> dict[str, dict[str, dict]]:
    out = {}
    for name, fn in MODELS.items():
        rows = json.loads((DATA / fn).read_text(encoding="utf-8"))
        out[name] = {}
        for r in rows:
            loop = is_loop(r.get("hyp", ""))
            d = r["duration_s"] < 1.0 or loop or r.get("wer") is None
            valid_ref = bool(norm(r.get("cleaned_text") or r.get("text") or ""))
            out[name][r["clip_id"]] = {
                "dialect": r["dialect"], "duration_s": r["duration_s"],
                "wer": r.get("wer"), "degenerate": d, "loop": loop,
                "valid_ref": valid_ref,
            }
    return out


def wer_view(a: dict, b: dict, ids, key_a, key_b) -> tuple[float, int]:
    """Mean WERs for models a and b over `ids` (paired)."""
    wa = [a[i]["wer"] for i in ids if a[i]["wer"] is not None]
    wb = [b[i]["wer"] for i in ids if b[i]["wer"] is not None]
    return (100 * float(np.mean(wa)) if wa else float("nan"),
            100 * float(np.mean(wb)) if wb else float("nan"),
            len(ids))


def main() -> None:
    models = load_all()
    base = models[BASE]
    all_ids = sorted(base.keys())
    report = {}

    print(f"BASE = {BASE} (n={len(all_ids)} clips)")
    hdr = f"{'model':26s} | {'paired-clean':>21s} | {'all-valid':>21s} | {'degen%':>6s} {'loop%':>6s}"
    print(hdr)
    print("-" * len(hdr))

    for name in MODELS:
        if name == BASE:
            continue
        m = models[name]
        common_clean = [i for i in all_ids
                        if not base[i]["degenerate"] and not m[i]["degenerate"]]
        all_valid = [i for i in all_ids
                     if base[i]["valid_ref"] and m[i]["valid_ref"]]
        ba, ma, n1 = wer_view(base, m, common_clean, "wer", "wer")
        bb, mb, n2 = wer_view(base, m, all_valid, "wer", "wer")
        deg = 100 * sum(m[i]["degenerate"] for i in all_ids) / len(all_ids)
        loop = 100 * sum(m[i]["loop"] for i in all_ids) / len(all_ids)
        print(f"{name:26s} | {ba:5.1f} -> {ma:5.1f} ({ma-ba:+5.1f} n={n1:2d}) "
              f"| {bb:5.1f} -> {mb:5.1f} ({mb-bb:+5.1f} n={n2:2d}) | {deg:5.1f} {loop:5.1f}")
        report[name] = {
            "paired_common_clean": {"base_wer": ba, "model_wer": ma, "delta": ma - ba, "n": n1},
            "all_valid_refs": {"base_wer": bb, "model_wer": mb, "delta": mb - bb, "n": n2},
            "degenerate_pct": deg, "loop_pct": loop,
        }

    # Base's own degeneracy for reference.
    bdeg = 100 * sum(base[i]["degenerate"] for i in all_ids) / len(all_ids)
    bloop = 100 * sum(base[i]["loop"] for i in all_ids) / len(all_ids)
    print(f"\nBASE degeneracy: {bdeg:.1f}% (loop {bloop:.1f}%)")

    # Short-clip view: 0.5-1.0s clips, product-risk context.
    short = [i for i in all_ids if base[i]["duration_s"] < 1.0]
    print(f"\n=== short-clip view (0.5-1.0s, n={len(short)}) — excluded by the "
          f"clean rule, shown for product risk ===")
    print(f"{'model':26s} | {'loop%':>6s} | {'WER where valid':>15s} | {'median WER':>10s}")
    for name, m in models.items():
        loops = 100 * sum(m[i]["loop"] for i in short) / max(1, len(short))
        wers = [m[i]["wer"] for i in short
                if m[i]["wer"] is not None and m[i]["valid_ref"]]
        med = 100 * float(np.median(wers)) if wers else float("nan")
        mean = 100 * float(np.mean(wers)) if wers else float("nan")
        print(f"{name:26s} | {loops:5.1f} | {mean:14.1f} | {med:9.1f}")
        report.setdefault(name, {})["short_clips"] = {
            "n": len(short), "loop_pct": loops,
            "mean_wer_valid": mean, "median_wer_valid": med,
        }

    (DATA / "paired_results.json").write_text(
        json.dumps({"base": BASE, "models": report}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"\nwritten -> {DATA / 'paired_results.json'}")


if __name__ == "__main__":
    main()
