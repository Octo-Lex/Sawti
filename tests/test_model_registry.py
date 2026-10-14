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


# ---- segmentation boundary gate logic (CI gate, 2026-10-13) -----------

def test_segmentation_gate_logic():
    from scripts.segment_boundary_eval import evaluate_gate

    ok, problems = evaluate_gate(f1=0.594, covered=76, n_ref=76)
    assert ok and problems == []
    ok, problems = evaluate_gate(f1=0.55, covered=76, n_ref=76)
    assert ok  # exactly at the floor passes
    ok, problems = evaluate_gate(f1=0.549, covered=76, n_ref=76)
    assert not ok and "below floor" in problems[0]
    ok, problems = evaluate_gate(f1=0.9, covered=75, n_ref=76)
    assert not ok and "never-drop" in problems[0]
