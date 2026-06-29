from pathlib import Path

import pytest

from sawti.config import SawtiConfig, load_config


def test_defaults_match_spec():
    cfg = SawtiConfig()
    assert cfg.segmentation.pause_threshold_ms == 350
    assert cfg.segmentation.max_chunk_duration_s == 12
    assert cfg.segmentation.min_chunk_duration_ms == 600
    assert cfg.segmentation.overlap_ms == 300
    assert cfg.quality_gate.policy == "balanced"
    assert cfg.quality_gate.confidence_threshold == 0.40
    assert cfg.quality_gate.script_mismatch_strictness["ara"] == "strict"
    assert cfg.postprocess.preserve_arabic_diacritics is True
    assert cfg.postprocess.arabic_alef_normalization == "matching_only"


def test_load_default_yaml():
    cfg = load_config(Path("config/default.yaml"))
    assert cfg.target_lang in ("eng", "ara", "fra")
    assert cfg.s2tt.engine == "seamless_m4t"


def test_invalid_target_lang_rejected():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        SawtiConfig(target_lang="spanish")  # not a supported code
