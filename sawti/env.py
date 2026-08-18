"""Local ``.env`` loading with explicit precedence (Commit 6 policy).

Retired behavior: import-time loading with override=True (file values
replacing OS environment silently). Importing this module now has NO
side effects — library modules must never mutate ``os.environ``.

Policy: the OS environment wins by default; ``load_env()`` only fills
variables that are absent. Intentional file-over-OS override requires
the explicit ``override=True`` argument, used ONLY at entry edges
(CLI command bodies, the test-runner conftest) — on this workstation
the OS ``HF_HOME`` is malformed (literal quotes) and the repo ``.env``
carries the corrected cache path.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """Load key=value lines from ``path`` into ``os.environ``.

    override=False (default): only keys NOT already present in the OS
    environment are set — OS wins. override=True: file values replace
    existing ones (explicit, entry-edge-only). Returns what was set.
    """
    p = Path(path)
    if not p.exists():
        return {}
    applied: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip one layer of surrounding quotes if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied
