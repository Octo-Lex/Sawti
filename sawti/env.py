"""Load a local `.env` file into os.environ at import time.

Runs BEFORE any Hugging Face / transformers imports so that HF_HOME and
HF_HUB_CACHE (set in `.env`) take effect. The system env vars are not
cleared — values in `.env` override them only if present.

This exists because the system env on this machine sets HF_HOME with
literal embedded quotes that break pathlib; `.env` carries the corrected
unquoted path. See README / .env.example.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env", override: bool = True) -> None:
    """Load key=value lines from `path` into os.environ.

    override=True (default): .env values replace existing env values. This is
    intentional so the corrected HF cache path wins over the broken system one.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip one layer of surrounding quotes if present (handles the case
        # where a value was written as KEY="value").
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


# Load on import so any module that imports sawti.env gets the corrected env.
load_env()
