"""Guard: fail if pyproject.toml or uv.lock bind torch to a CUDA-only index.

Protects the committed portable-default policy (PR #1 review fix). The local
GPU overlay (cu126 block appended for this workstation) must never be staged.

Modes:
  python scripts/check_portable_config.py          # check files on disk
  python scripts/check_portable_config.py --staged # check git-index blobs

Comment lines are ignored in pyproject.toml — only active configuration
counts. Exit 0 = portable, exit 1 = violation (with a clear message).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MARKERS = ("pytorch-cu126", "download.pytorch.org/whl/cu", "+cu126")


def _staged(path: str) -> str:
    return subprocess.run(
        ["git", "show", f":{path}"], capture_output=True, text=True, check=True
    ).stdout


def _active_lines(text: str) -> str:
    return "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )


def check(pyproject: str, lock: str) -> list[str]:
    violations = []
    active = _active_lines(pyproject)
    for m in MARKERS:
        if m in active:
            violations.append(f"pyproject.toml (active config): {m!r}")
    for m in ("pytorch-cu126", "whl/cu126", "+cu126"):
        if m in lock:
            violations.append(f"uv.lock: {m!r}")
    return violations


def main() -> int:
    staged = "--staged" in sys.argv
    if staged:
        py = _staged("pyproject.toml")
        lock = _staged("uv.lock")
    else:
        py = Path("pyproject.toml").read_text(encoding="utf-8")
        lock = Path("uv.lock").read_text(encoding="utf-8")
    violations = check(py, lock)
    if violations:
        print("PORTABILITY GUARD FAILED — CUDA pin detected in committed config:")
        for v in violations:
            print(f"  - {v}")
        print(
            "\nThe portable default is CPU torch (works on macOS/CI/CPU-only). "
            "GPU users keep the cu126 overlay LOCAL and unstaged — see the "
            "comment block at the end of pyproject.toml."
        )
        return 1
    print("portability guard: OK (no active CUDA pins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
