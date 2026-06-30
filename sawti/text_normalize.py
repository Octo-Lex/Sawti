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


def collapse_repeated_loops(text: str, min_count: int = 3) -> str:
    """Collapse exact unigram loops repeated >= min_count times.

    'the the the the' -> 'the'. Preserves intentional shorter repetitions.
    """
    tokens = text.split()
    if len(tokens) < min_count:
        return text
    out: list[str] = []
    i = 0
    while i < len(tokens):
        j = i + 1
        while j < len(tokens) and tokens[j] == tokens[i]:
            j += 1
        run_len = j - i
        if run_len >= min_count:
            out.append(tokens[i])  # collapse to a single instance
        else:
            out.extend(tokens[i:j])
        i = j
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
