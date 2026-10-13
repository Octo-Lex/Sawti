"""Model-registry gate (2026-10-13, VoiceStudio-inspired).

Fails when:
- code references a known-provider model id that has no registry record
- a registry record is missing required fields
- commercial_use=true lacks a reviewer (no record may claim it yet)

Run standalone or via tests/test_model_registry.py (the CI gate).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "config" / "model_registry.json"
SCAN_DIRS = ["sawti", "eval", "scripts", "spikes"]

# HF-style ids from known providers (the registry's scope for now)
ID_RE = re.compile(
    r"[\"']((?:facebook|openai|google|kyutai|nvidia|mistralai|"
    r"unsloth|kyutai-labs|MohamedRashad)/[\w.\-]+)[\"']")
LOCAL_MODEL_DIRS = ["models/sa_whisper_v1"]

REQUIRED_FIELDS = ["id", "role", "license", "commercial_use", "evidence",
                   "used_by"]


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def check() -> list[str]:
    problems: list[str] = []
    reg = load_registry()
    known = {m["id"] for m in reg["models"]} | {
        d["id"] for d in reg["datasets"]}

    # 1. record completeness (key PRESENCE: commercial_use is legitimately
    # false/null — truthiness would flag honest records as incomplete)
    for section in ("models", "datasets"):
        for rec in reg[section]:
            missing = [f for f in REQUIRED_FIELDS if f not in rec]
            if missing:
                problems.append(
                    f"{rec.get('id', '?')}: missing fields {missing}")
            if rec.get("commercial_use") is True and not rec.get("reviewer"):
                problems.append(
                    f"{rec['id']}: commercial_use=true without reviewer")

    # 2. code references must be registered
    seen: dict[str, set[str]] = {}
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            text = p.read_text(encoding="utf-8", errors="replace")
            for m in ID_RE.finditer(text):
                seen.setdefault(m.group(1), set()).add(str(p.relative_to(ROOT)))
            for local in LOCAL_MODEL_DIRS:
                if local in text:
                    seen.setdefault(f"local:{local}", set()).add(
                        str(p.relative_to(ROOT)))
    for ref, files in sorted(seen.items()):
        if ref not in known:
            problems.append(
                f"unregistered model reference {ref!r} in {sorted(files)}")
    return problems


if __name__ == "__main__":
    issues = check()
    if issues:
        print("MODEL REGISTRY GATE FAILED:")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)
    print("model registry gate: OK")
