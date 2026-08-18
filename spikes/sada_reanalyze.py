"""Spike re-analysis: separate Whisper hallucination-loop failures from
genuine dialect-recognition error. No ASR re-run — reads eval_results.json.

Degenerate = short clip (<1s) OR hypothesis is a repetition loop
(unique-token ratio < 0.25) OR reference empty after norm.
Reports: degenerate rate per dialect + clean-subset WER per dialect.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("data/sada_spike")


def is_loop(hyp: str) -> bool:
    toks = hyp.split()
    if len(toks) < 6:
        return False
    uniq = len(set(toks)) / len(toks)
    most = Counter(toks).most_common(1)[0][1] / len(toks)
    return uniq < 0.25 or most > 0.6


def main() -> None:
    results = json.loads((DATA / "eval_results.json").read_text(encoding="utf-8"))
    for r in results:
        r["degenerate"] = (
            r["duration_s"] < 1.0 or is_loop(r["hyp"]) or r["wer"] is None
        )
        r["degen_reason"] = (
            "short" if r["duration_s"] < 1.0 else
            "loop" if is_loop(r["hyp"]) else
            "nower" if r["wer"] is None else ""
        )

    by_d = defaultdict(list)
    for r in results:
        by_d[r["dialect"]].append(r)

    print("=== degenerate-rate and clean-subset WER ===")
    clean_all = []
    for d, rows in sorted(by_d.items()):
        deg = [r for r in rows if r["degenerate"]]
        clean = [r for r in rows if not r["degenerate"]]
        clean_all.extend(clean)
        wers = [r["wer"] for r in clean]
        w = 100 * sum(wers) / len(wers) if wers else float("nan")
        reasons = Counter(r["degen_reason"] for r in deg)
        print(
            f"{d:10s} n={len(rows)}  degenerate={len(deg)} ({100*len(deg)/len(rows):.0f}%)"
            f"  {dict(reasons)}"
        )
        print(
            f"{'':10s} clean WER = {w:5.1f}%  (n={len(clean)}, "
            f"median={100*sorted(wers)[len(wers)//2]:.1f}%)"
        )

    wers = [r["wer"] for r in clean_all]
    print(
        f"\nALL clean subset: WER {100*sum(wers)/len(wers):.1f}% (n={len(wers)})"
        f"  | degenerate overall: {sum(r['degenerate'] for r in results)}/{len(results)}"
    )
    # How bad is bad on clean subset? Distribution.
    bands = [(0, 20), (20, 50), (50, 100), (100, 1e9)]
    for lo, hi in bands:
        c = sum(1 for w in wers if lo <= w < hi)
        print(f"  WER [{lo:>3},{('%d+' % hi) if hi > 1e8 else int(hi)}): {c}")

    (DATA / "reanalysis.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nannotated results -> {DATA / 'reanalysis.json'}")


if __name__ == "__main__":
    main()
