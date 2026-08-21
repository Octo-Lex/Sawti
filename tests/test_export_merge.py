"""Task 6 exporter contract tests.

Hermetic: provenance assembly, selection-artifact traceability, and the
fail-closed selection->checkpoint binding (reviewer correction 1) —
merge_and_export/verify_parity are operator GPU paths (reviewer-reviewed
by code + parity evidence, not unit tests).
"""
import json

import pytest

from sawti.training.export_merge import (
    build_provenance,
    verify_selection_binding,
)


def _make_ckpt(tmp_path, name="checkpoint-10000", cfg=b"{}", weights=b"\x01\x02"):
    ckpt = tmp_path / name
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_bytes(cfg)
    (ckpt / "adapter_model.safetensors").write_bytes(weights)
    return ckpt


def _sha_of(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _selection_artifact(tmp_path, ckpt, eligible=True, score=42.26,
                        name="checkpoint-10000.json"):
    art = tmp_path / name
    payload = {
        "selection": {"eligible": eligible, "selection_score": score,
                      "guard_fail": [] if eligible else [{"dialect": "Najdi"}],
                      "loop_ok": eligible},
        "config": {"adapter": {
            "path": str(ckpt),
            "adapter_config.json": _sha_of((ckpt / "adapter_config.json").read_bytes()),
            "adapter_model.safetensors": _sha_of((ckpt / "adapter_model.safetensors").read_bytes()),
        }},
    }
    art.write_text(json.dumps(payload), encoding="utf-8")
    return art


def test_provenance_records_adapter_hashes_and_contract(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    prov = build_provenance(str(ckpt), "openai/whisper-large-v3",
                            tmp_path / "out", None)
    assert prov["base_model"] == "openai/whisper-large-v3"
    # The selection-identity contract: per-file adapter SHA-256 present.
    assert len(prov["adapter"]["adapter_config.json"]) == 64
    assert len(prov["adapter"]["adapter_model.safetensors"]) == 64
    # Inference contract pinned in provenance, not just in weights:
    assert prov["generation_config"] == {"language": "arabic",
                                         "task": "transcribe",
                                         "forced_decoder_ids": None}
    assert prov["evaluator_greedy_kwargs"]["num_beams"] == 1
    assert prov["selection_batch_size"] == 4
    assert prov["selection_artifact"] is None
    assert prov["exporter_commit"]  # sha or 'unknown', never empty


def test_provenance_requires_existing_selection_artifact(tmp_path):
    """The export must be traceable to the evidence that selected the
    adapter — a dangling reference fails loudly."""
    ckpt = _make_ckpt(tmp_path)
    ghost = tmp_path / "eval" / "checkpoint-10000.json"
    with pytest.raises(FileNotFoundError, match="selection artifact"):
        build_provenance(str(ckpt), "openai/whisper-large-v3",
                         tmp_path / "out", str(ghost))


def test_provenance_hashes_selection_artifact(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    sel = _selection_artifact(tmp_path, ckpt)
    prov = build_provenance(str(ckpt), "openai/whisper-large-v3",
                            tmp_path / "out", str(sel))
    assert len(prov["selection_artifact_sha256"]) == 64
    assert prov["selection_artifact"] == str(sel)


# ---- fail-closed selection->checkpoint binding (reviewer correction 1) ----

def test_binding_succeeds_on_eligible_matching_artifact(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    sel = _selection_artifact(tmp_path, ckpt)
    binding = verify_selection_binding(str(ckpt), str(sel))
    assert binding == {"eligible": True, "adapter_hashes_match": True,
                       "selection_score": 42.26}
    prov = build_provenance(str(ckpt), "openai/whisper-large-v3",
                            tmp_path / "out", str(sel))
    assert prov["selection_binding"]["adapter_hashes_match"] is True


def test_binding_rejects_checkpoint_mismatch(tmp_path):
    """THE reviewer scenario: --checkpoint checkpoint-8000 with
    --selection eval/checkpoint-10000.json must fail closed."""
    selected = _make_ckpt(tmp_path, "checkpoint-10000", b"{}", b"\x01\x02")
    wrong = _make_ckpt(tmp_path, "checkpoint-8000", b"{}", b"\xff\xff")
    sel = _selection_artifact(tmp_path, selected)
    with pytest.raises(ValueError, match="MISMATCH"):
        verify_selection_binding(str(wrong), str(sel))
    with pytest.raises(ValueError, match="MISMATCH"):
        build_provenance(str(wrong), "openai/whisper-large-v3",
                         tmp_path / "out", str(sel))


def test_binding_rejects_ineligible_selection(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    sel = _selection_artifact(tmp_path, ckpt, eligible=False, score=99.9)
    with pytest.raises(ValueError, match="INELIGIBLE"):
        verify_selection_binding(str(ckpt), str(sel))


def test_binding_rejects_baseline_mode_artifact(tmp_path):
    """A stock-baseline artifact (selection: None) justifies no export."""
    ckpt = _make_ckpt(tmp_path)
    art = tmp_path / "zero_shot_baseline_v2.json"
    art.write_text(json.dumps({"selection": None, "config": {}}),
                   encoding="utf-8")
    with pytest.raises(ValueError, match="no selection record"):
        verify_selection_binding(str(ckpt), str(art))
