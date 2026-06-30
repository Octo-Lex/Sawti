"""Eval metrics (spec §7.2). Real chrF via sacrebleu + the M0 stub retained."""
from __future__ import annotations

from collections import Counter


def compute_chrf_stub(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Tiny self-contained chrF-like score in [0,100] (kept for fast tests)."""
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


def compute_chrf(hyp: str, ref: str) -> float:
    """Real chrF via sacrebleu in [0, 100]."""
    import sacrebleu

    bleu = sacrebleu.sentence_chrf(hyp, [ref])
    return bleu.score
