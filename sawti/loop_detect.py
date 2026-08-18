"""Deterministic n-gram repetition (decoder-loop) detector.

Targets CONSECUTIVE repeated blocks — the decoder failure mode — not
lexical frequency, so legitimate recurrence stays valid. Any 1..8-token
span repeated >= min_repeats times back-to-back triggers; a legacy
token-dominance signal is kept as an additional catch. Validated against
the Saudi spike's observed failure modes (unigram loops AND the
phrase-level "اشتركوا في القناه" x3 loop invisible to unigram-only rules).
"""
from __future__ import annotations

from collections import Counter


def _loop_run(toks: list[str], s: int, n: int) -> int:
    run = 1
    while s + n * (run + 1) <= len(toks) and toks[s:s + n] == toks[s + n * run: s + n * run + n]:
        run += 1
    return run


def is_loop(hyp: str, min_repeats: int = 3, max_n: int = 8) -> bool:
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
