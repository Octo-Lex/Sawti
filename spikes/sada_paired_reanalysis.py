"""Spike correction v2: PAIRED model comparison from stored hypotheses.

Supersedes the first paired correction (Addendum 3): that version still used
the legacy unigram/dominance loop detector, which misses the x3 phrase-loop
failure mode it was supposed to exclude. This version uses the n-gram
repetition detector (1-8-token spans, >=3 consecutive repeats, plus the
legacy dominance signal) — self-tested against the known examples.

Four views per reviewer spec:
  1. paired common-clean WER  — non-degenerate under BOTH models (n-gram rule)
  2. all-valid-reference WER  — macro mean per-clip AND corpus-level
  3. degeneracy + loop rates  — per model, separately
  4. short-clip view          — 0.5-1.0s clips (floor enforced)

Corpus WER = sum(per-clip WER * ref_words) / sum(ref_words) — exact from
stored values; weights by reference length (macro does not).
"""
from __future__ import annotations

import json
from collections import Counter
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


def _loop_run(toks, s, n) -> int:
    run = 1
    while s + n * (run + 1) <= len(toks) and toks[s:s + n] == toks[s + n * run: s + n * run + n]:
        run += 1
    return run


def is_loop(hyp: str, min_repeats: int = 3, max_n: int = 8) -> bool:
    """Any 1..8-token span repeating >=3 times consecutively, plus the legacy
    token-dominance signal. Catches the x3 phrase loop (uniq 0.33, most 0.33
    — invisible to the old rule)."""
    toks = hyp.split()
    if len(toks) < min_repeats:
        return False
    for n in range(1, max_n + 1):
        if n * min_repeats > len(toks):
            break
        for s in range(len(toks) - n * min_repeats + 1):
            if toks[s:s + n] == toks[s + n:s + 2 * n] and _loop_run(toks, s, n) >= min_repeats:
                return True
    if len(toks) >= 6:
        uniq = len(set(toks)) / len(toks)
        most = Counter(toks).most_common(1)[0][1] / len(toks)
        if uniq < 0.25 or most > 0.6:
            return True
    return False


def load_all() -> dict[str, dict[str, dict]]:
    out = {}
    for name, fn in MODELS.items():
        rows = json.loads((DATA / fn).read_text(encoding="utf-8"))
        out[name] = {}
        for r in rows:
            loop = is_loop(r.get("hyp", ""))
            d = r["duration_s"] < 1.0 or loop or r.get("wer") is None
            ref = norm(r.get("cleaned_text") or r.get("text") or "")
            out[name][r["clip_id"]] = {
                "dialect": r["dialect"], "duration_s": r["duration_s"],
                "wer": r.get("wer"), "degenerate": d, "loop": loop,
                "valid_ref": bool(ref), "n_ref_words": len(ref.split()),
            }
    return out


def _macro(m: dict, ids: list[str]) -> float:
    w = [m[i]["wer"] for i in ids if m[i]["wer"] is not None]
    return 100 * float(np.mean(w)) if w else float("nan")


def _corpus(m: dict, ids: list[str]) -> float:
    """Exact corpus WER: per-clip WERs weighted by reference word count."""
    err = total = 0.0
    for i in ids:
        r = m[i]
        if r["wer"] is None or not r["valid_ref"]:
            continue
        err += r["wer"] * r["n_ref_words"]
        total += r["n_ref_words"]
    return 100 * err / total if total else float("nan")


def main() -> None:
    models = load_all()
    base = models[BASE]
    all_ids = sorted(base.keys())
    report: dict = {"base": BASE, "detector": "ngram(1-8,>=3)+dominance", "models": {}}

    print(f"BASE = {BASE} (n={len(all_ids)} clips)")
    hdr = (f"{'model':26s} | {'clean macro':>15s} | {'allval macro':>13s} "
           f"| {'allval corpus':>14s} | {'degen%':>6s} {'loop%':>6s}")
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
        clean_ma, clean_mb = _macro(base, common_clean), _macro(m, common_clean)
        av_ma, av_mb = _macro(base, all_valid), _macro(m, all_valid)
        av_ca, av_cb = _corpus(base, all_valid), _corpus(m, all_valid)
        deg = 100 * sum(m[i]["degenerate"] for i in all_ids) / len(all_ids)
        loop = 100 * sum(m[i]["loop"] for i in all_ids) / len(all_ids)
        print(f"{name:26s} | {clean_ma:5.1f}->{clean_mb:5.1f} ({clean_mb-clean_ma:+5.1f} n={len(common_clean):2d}) "
              f"| {av_ma:6.1f}->{av_mb:6.1f} | {av_ca:6.1f}->{av_cb:6.1f} (n={len(all_valid):2d}) "
              f"| {deg:5.1f} {loop:5.1f}")
        report["models"][name] = {
            "paired_common_clean": {
                "base_macro": clean_ma, "model_macro": clean_mb,
                "delta": clean_mb - clean_ma, "n": len(common_clean)},
            "all_valid": {
                "base_macro": av_ma, "model_macro": av_mb,
                "base_corpus": av_ca, "model_corpus": av_cb, "n": len(all_valid)},
            "degenerate_pct": deg, "loop_pct": loop,
        }

    valid_all = [i for i in all_ids if base[i]["valid_ref"]]
    bdeg = 100 * sum(base[i]["degenerate"] for i in all_ids) / len(all_ids)
    bloop = 100 * sum(base[i]["loop"] for i in all_ids) / len(all_ids)
    print(f"\nBASE: degeneracy {bdeg:.1f}% | loop {bloop:.1f}% | "
          f"all-valid macro {_macro(base, valid_all):.1f}% | "
          f"corpus {_corpus(base, valid_all):.1f}%")
    report["base_stats"] = {"degenerate_pct": bdeg, "loop_pct": bloop}

    short = [i for i in all_ids if 0.5 <= base[i]["duration_s"] < 1.0]
    print(f"\n=== short-clip view (0.5-1.0s, n={len(short)}) ===")
    print(f"{'model':26s} | {'loop%':>6s} | {'macro WER':>9s} | {'median':>7s}")
    for name, m in models.items():
        loops = 100 * sum(m[i]["loop"] for i in short) / max(1, len(short))
        wers = [m[i]["wer"] for i in short
                if m[i]["wer"] is not None and m[i]["valid_ref"]]
        med = 100 * float(np.median(wers)) if wers else float("nan")
        macro = 100 * float(np.mean(wers)) if wers else float("nan")
        print(f"{name:26s} | {loops:5.1f} | {macro:8.1f} | {med:6.1f}")
        report["models"].setdefault(name, {})["short_clips"] = {
            "n": len(short), "loop_pct": loops,
            "macro_wer_valid": macro, "median_wer_valid": med}

    (DATA / "paired_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten -> {DATA / 'paired_results.json'}")


if __name__ == "__main__":
    main()
