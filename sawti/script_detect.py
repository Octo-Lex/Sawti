"""Unicode script detection for quality-gate checks (spec §3.5).

Used to detect target-language script mismatches (e.g. target=ara but
output is mostly Latin). Pure functions, no I/O.
"""
from __future__ import annotations

import unicodedata


def _char_script(ch: str) -> str:
    """Classify a single character as latin/arabic/other."""
    if ch.isspace() or ch.isdigit() or unicodedata.category(ch).startswith("P"):
        return "other"
    cp = ord(ch)
    # Basic Latin + Latin-1 Supplement + Latin Extended
    if (0x0041 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF):
        return "latin"
    # Arabic block + Arabic Supplement/Extended
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFDFF:
        return "arabic"
    return "other"


def dominant_script(text: str) -> str:
    """Return 'latin', 'arabic', or 'other' based on letter-class counts."""
    counts = {"latin": 0, "arabic": 0, "other": 0}
    for ch in text:
        counts[_char_script(ch)] += 1
    letters = counts["latin"] + counts["arabic"]
    if letters == 0:
        return "other"
    if counts["latin"] >= counts["arabic"]:
        return "latin"
    return "arabic"


def is_mostly_arabic(text: str) -> bool:
    return dominant_script(text) == "arabic"


def is_mostly_latin(text: str) -> bool:
    return dominant_script(text) == "latin"
