"""Commit 6: environment precedence policy — OS wins by default, importing
never mutates, explicit override only."""
import os
import subprocess
import sys
from pathlib import Path

from sawti.env import load_env


def test_importing_sawti_env_has_no_side_effects(tmp_path: Path, monkeypatch):
    # A .env exists and OS var is set to the CANARY: a fresh interpreter
    # importing sawti.env must not overwrite the OS value.
    (tmp_path / ".env").write_text("SAWTI_ENV_TEST=os_file_value\n", encoding="utf-8")
    monkeypatch.setenv("SAWTI_ENV_TEST", "os_canary")
    code = (
        "import os, sys; sys.path.insert(0, r'%s'); "
        "import sawti.env; "
        "print(os.environ.get('SAWTI_ENV_TEST'))" % os.getcwd()
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=tmp_path, check=True)
    assert out.stdout.strip() == "os_canary"


def test_load_env_os_wins_by_default(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text("SAWTI_ENV_A=file_a\nSAWTI_ENV_B=file_b\n",
                                   encoding="utf-8")
    monkeypatch.setenv("SAWTI_ENV_A", "os_a")
    applied = load_env(tmp_path / ".env")            # override=False default
    assert os.environ["SAWTI_ENV_A"] == "os_a"       # OS wins
    assert os.environ["SAWTI_ENV_B"] == "file_b"     # absent -> filled
    assert applied == {"SAWTI_ENV_B": "file_b"}


def test_load_env_explicit_override_replaces(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text('SAWTI_ENV_C="quoted_value"\n', encoding="utf-8")
    monkeypatch.setenv("SAWTI_ENV_C", "os_c")
    applied = load_env(tmp_path / ".env", override=True)
    assert os.environ["SAWTI_ENV_C"] == "quoted_value"  # quotes stripped
    assert applied == {"SAWTI_ENV_C": "quoted_value"}


def test_load_env_missing_file_is_noop(tmp_path: Path):
    assert load_env(tmp_path / "absent.env") == {}
