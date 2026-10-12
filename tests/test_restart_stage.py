"""In-stream degenerate recovery (2026-10-09): the restart stage.

Contract under test (fallback.py stage 1.5):
- A DEGENERATE primary (empty/garbage/loop check fired) gets a cheap
  restart re-decode FIRST; acceptance there short-circuits the ladder
  with fallback_path="restart".
- A low-confidence-only primary NEVER triggers restart (that is what the
  beam retry is for).
- A failed restart falls through to the retry stage unchanged.
- retries.max_engine_restarts=0 disables the stage (pre-restart ladder).
"""
from unittest.mock import MagicMock

import numpy as np

from sawti.config import QualityGateConfig, RetriesConfig
from sawti.fallback import FallbackHandler, is_degenerate
from sawti.types import AudioChunk, EngineResult, GateDecision


def _chunk():
    return AudioChunk(id="c0", audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=1.0)


def _result(text="hi", conf=0.9):
    return EngineResult("c0", text, conf, "ara", {}, "eng")


def _primary(checks, text="bad", conf=0.2):
    return GateDecision(chunk_id="c0", accepted=False, result=_result(text, conf),
                        checks=checks, start_time=0.0, end_time=1.0,
                        low_confidence=True, needs_retry=True)


def test_is_degenerate_classifies_checks():
    assert is_degenerate(_primary({"repetition_loop": True}))
    assert is_degenerate(_primary({"empty_output": True, "low_conf": True}))
    assert is_degenerate(_primary({"garbage_output": True}))
    assert not is_degenerate(_primary({"low_confidence": True}))
    assert not is_degenerate(_primary({"script_mismatch": True}))
    assert not is_degenerate(_primary({"length_ratio_anomaly": True}))


def test_degenerate_primary_recovers_via_restart_and_skips_beam():
    engine = MagicMock()
    restart = MagicMock(return_value=_result("clean re-decode", 0.75))
    conservative = MagicMock(return_value=_result("beam result", 0.9))
    fb = FallbackHandler(engine=engine, restart=restart,
                         conservative=conservative)
    out = fb.recover(_chunk(), _primary({"repetition_loop": True}), "eng")

    assert out.accepted is True
    assert out.fallback_path == "restart"
    restart.assert_called_once()
    conservative.assert_not_called()   # beam never paid for a degenerate chunk
    engine.translate.assert_not_called()


def test_low_confidence_primary_skips_restart_entirely():
    engine = MagicMock()
    engine.translate.return_value = _result("beam recovered", 0.9)
    restart = MagicMock()
    fb = FallbackHandler(engine=engine, restart=restart)
    out = fb.recover(_chunk(), _primary({"low_confidence": True}), "eng")

    restart.assert_not_called()
    assert out.fallback_path == "retry"


def test_failed_restart_falls_through_to_retry():
    engine = MagicMock()
    engine.translate.return_value = _result("beam recovered", 0.9)
    restart = MagicMock(return_value=_result("still looping", 0.2))
    fb = FallbackHandler(engine=engine, restart=restart)
    out = fb.recover(_chunk(), _primary({"repetition_loop": True}), "eng")

    restart.assert_called_once()
    assert out.fallback_path == "retry"
    stages = [e["stage"] for e in out.log
              if isinstance(e, dict) and "stage" in e]
    assert stages[:3] == ["primary", "restart", "retry"]


def test_max_engine_restarts_zero_disables_stage():
    engine = MagicMock()
    engine.translate.return_value = _result("beam recovered", 0.9)
    restart = MagicMock()
    cfg = QualityGateConfig(retries=RetriesConfig(max_engine_restarts=0))
    fb = FallbackHandler(engine=engine, restart=restart, config=cfg)
    out = fb.recover(_chunk(), _primary({"garbage_output": True}), "eng")

    restart.assert_not_called()
    assert out.fallback_path == "retry"
    stages = [e["stage"] for e in out.log
              if isinstance(e, dict) and "stage" in e]
    assert "restart" not in stages


def test_no_restart_seam_means_no_restart_even_for_degenerate():
    """Pre-restart construction (no seam): degenerate primaries go to the
    beam retry exactly as before — backward compatible by default."""
    engine = MagicMock()
    engine.translate.return_value = _result("beam recovered", 0.9)
    fb = FallbackHandler(engine=engine)
    out = fb.recover(_chunk(), _primary({"repetition_loop": True}), "eng")
    assert out.fallback_path == "retry"
