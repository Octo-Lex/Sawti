"""CI gate: the model registry must cover every code-referenced model,
and no record may claim commercial use without review. Inspired by the
VoiceStudio license-inventory pattern, scaled to Sawti."""
import pytest

from scripts.check_model_registry import check, load_registry


def test_registry_gate_passes():
    issues = check()
    assert issues == [], "registry gate failures:\n" + "\n".join(issues)


def test_registry_covers_every_model_reference():
    """Every known-provider id in scanned code has a record (the gate
    above already enforces it; this test documents the invariant)."""
    reg = load_registry()
    assert any(m["id"] == "facebook/seamless-m4t-v2-large"
               for m in reg["models"]), "primary engine must stay registered"
    assert reg["models"], "registry must not be empty"


def test_no_unreviewed_commercial_claims():
    reg = load_registry()
    for section in ("models", "datasets"):
        for rec in reg[section]:
            if rec.get("commercial_use") is True:
                assert rec.get("reviewer"), rec["id"]
