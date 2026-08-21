"""Task 6 exporter contract tests.

Hermetic: provenance assembly and selection-artifact traceability only —
merge_and_export/verify_parity are operator GPU paths (reviewer-reviewed
by code + parity evidence, not unit tests).
"""
import json

import pytest

from sawti.training.export_merge import build_provenance


def _make_ckpt(tmp_path):
    ckpt = tmp_path / "checkpoint-10000"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ckpt / "adapter_model.safetensors").write_bytes(b"\x01\x02")
    return ckpt


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
    sel = tmp_path / "checkpoint-10000.json"
    payload = {"aggregate": {"selection_score": 42.26}, "n": 3423}
    sel.write_text(json.dumps(payload), encoding="utf-8")
    prov = build_provenance(str(ckpt), "openai/whisper-large-v3",
                            tmp_path / "out", str(sel))
    assert len(prov["selection_artifact_sha256"]) == 64
    assert prov["selection_artifact"] == str(sel)
