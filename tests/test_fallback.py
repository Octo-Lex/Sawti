"""FallbackHandler behavior tests (updated to the recover() contract).

The full escalation contract lives in test_fallback_contract.py; these cover
the two original scenarios: retry-recovers, and graceful degradation when no
ASR+MT provider is configured.
"""
from unittest.mock import MagicMock

import numpy as np

from sawti.fallback import FallbackHandler
from sawti.types import AudioChunk, EngineResult, GateDecision


def _chunk():
    return AudioChunk(id="c0", audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=1.0)


def _result(text="hi", conf=0.1):
    return EngineResult("c0", text, conf, "eng", {}, "eng")


def _primary(text="hi", conf=0.1):
    """Rejected primary decision as the Pipeline hands it to recover()."""
    return GateDecision(chunk_id="c0", accepted=False, result=_result(text, conf),
                        checks={"low_confidence": True}, start_time=0.0,
                        end_time=1.0, low_confidence=True, needs_retry=True)


def test_fallback_retry_re_invokes_engine():
    engine = MagicMock()
    engine.translate.return_value = _result("recovered", 0.9)
    fb = FallbackHandler(engine=engine)
    out = fb.recover(_chunk(), _primary("weak", 0.1), "eng")
    assert out.result.raw_text == "recovered"
    assert out.fallback_path == "retry"
    assert out.accepted is True


def test_fallback_exhausted_degrades_gracefully_without_provider():
    """No ASR+MT provider: retry fails, no rechunker, escalate -> exhausted.
    The best weak result returns flagged low_confidence rather than crashing."""
    engine = MagicMock()
    engine.translate.return_value = _result("weak", 0.1)
    fb = FallbackHandler(engine=engine, asr_mt=None)
    out = fb.recover(_chunk(), _primary("weak", 0.1), "eng")
    assert out.accepted is False
    assert out.low_confidence is True
    assert out.fallback_path == "exhausted"
