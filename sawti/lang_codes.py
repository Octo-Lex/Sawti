"""Mapping between Sawti language codes and backend model codes (spec §3.1).

Sawti exposes a small stable set of codes (eng|ara|fra). SeamlessM4T uses
ISO-3 codes where Arabic is `arb` (Modern Standard Arabic). The mapping is
contained here so frozen code never sees backend-specific codes.
"""
from __future__ import annotations


# Sawti code -> SeamlessM4T target/source code.
SAWTI_TO_M4T: dict[str, str] = {
    "eng": "eng",
    "fra": "fra",
    "ara": "arb",  # Modern Standard Arabic
}


def to_m4t_lang(sawti_code: str) -> str:
    """Map a Sawti language code to the SeamlessM4T code."""
    return SAWTI_TO_M4T[sawti_code]


def validate_sawti_lang(code: str) -> None:
    """Raise ValueError if `code` is not a supported Sawti language."""
    if code not in SAWTI_TO_M4T:
        raise ValueError(
            f"unsupported Sawti language code {code!r}; "
            f"expected one of {sorted(SAWTI_TO_M4T)}"
        )
