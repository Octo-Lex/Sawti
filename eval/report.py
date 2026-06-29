"""Report writer (spec §7.6)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_report(path: str | Path, payload: dict[str, Any]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)
