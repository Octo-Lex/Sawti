"""Eval metrics. M0 ships a chrF stub; M1 swaps in sacrebleu chrF (spec §7.2)."""
from __future__ import annotations

from collections import Counter


def compute_chrf_stub(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """A tiny self-contained chrF-like score in [0, 100].

    This is NOT sacrebleu's chrF — it's a deterministic placeholder so the
    harness runs without the heavy dependency in M0. M1 replaces it.
    """
    def ngrams(s: str) -> Counter:
        s = " " + s.strip() + " "
        return Counter(s[i : i + n] for i in range(len(s) - n + 1))

    hg, rg = ngrams(hyp), ngrams(ref)
    if not rg:
        return 0.0
    overlap = sum((hg & rg).values())
    prec = overlap / sum(hg.values()) if hg else 0.0
    rec = overlap / sum(rg.values())
    if prec + rec == 0:
        return 0.0
    f = (1 + beta * beta) * (prec * rec) / (beta * beta * prec + rec)
    return f * 100.0
