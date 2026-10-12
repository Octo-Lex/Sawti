"""Pure text-normalization functions (spec §4.4, §4.5).

Two tracks:
- display-side fixes (whitespace, punctuation spacing) applied to output.
- match-side normalization (lowercase, Arabic diacritic/tatweel/alef removal)
  used internally for dedupe and repeat detection — never written to output.
"""
from __future__ import annotations

import re
import unicodedata


def strip_excess_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def repair_punctuation_spacing(text: str) -> str:
    # "word ," -> "word,"  (remove space before punctuation)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    # collapse multiple punctuation
    text = re.sub(r"([,.!?;:])\1+", r"\1", text)
    return text


def _loop_run_len(toks: list[str], s: int, n: int) -> int:
    """Count back-to-back repeats of the n-gram at position s."""
    run = 1
    while (s + n * (run + 1) <= len(toks)
           and toks[s:s + n] == toks[s + n * run:s + n * run + n]):
        run += 1
    return run


def collapse_repeated_loops(text: str, min_count: int = 3,
                            max_n: int = 8) -> str:
    """Collapse exact unigram AND n-gram loops repeated >= min_count
    times back-to-back (spec §4.5).

    'the the the the' -> 'the'; 'a b a b a b' -> 'a b';
    'لكي نتحدث لكي نتحدث لكي نتحدث' -> 'لكي نتحدث'. The smallest repeating
    unit that reaches min_count wins, so pure unigram runs collapse to a
    single token exactly as before. Intentional shorter repetitions are
    preserved ('no no, wait', 'very very important'). max_n matches the
    gate's detector (loop_detect.is_loop, 1..8-token spans).
    """
    tokens = text.split()
    if len(tokens) < min_count:
        return text
    out: list[str] = []
    i = 0
    while i < len(tokens):
        collapsed = False
        cap = min(max_n, (len(tokens) - i) // min_count)
        for n in range(1, cap + 1):
            if _loop_run_len(tokens, i, n) >= min_count:
                run = _loop_run_len(tokens, i, n)
                out.extend(tokens[i:i + n])
                i += n * run
                collapsed = True
                break
        if not collapsed:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def normalize_for_match(text: str) -> str:
    """Latin-oriented match normalization: lowercase + collapse whitespace."""
    return strip_excess_whitespace(text).lower()


# Arabic diacritics (harakat) range U+064B..U+0652, tatweel U+0640.
_ARABIC_DIACRITICS = "".join(chr(c) for c in range(0x064B, 0x0653))
_ALEF_VARIANTS = "أإآٱ"


def normalize_arabic_for_match(text: str) -> str:
    """Non-destructive-for-output Arabic normalization for matching only.

    Removes tatweel and diacritics, unifies alef variants and yeh-maqsura.
    Per spec §4.4 this is matching_only — never written to display text.
    """
    # Remove tatweel and diacritics.
    text = re.sub(r"[\u0640" + _ARABIC_DIACRITICS + r"]", "", text)
    # Unify alef variants to plain alef.
    for v in _ALEF_VARIANTS:
        text = text.replace(v, "ا")
    # Yeh-maqsura -> yeh.
    text = text.replace("ى", "ي")
    return text
