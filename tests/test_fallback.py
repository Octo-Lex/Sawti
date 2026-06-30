from unittest.mock import MagicMock

import numpy as np

from sawti.fallback import FallbackHandler
from sawti.types import AudioChunk, EngineResult


def _chunk():
    return AudioChunk(id="c0", audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=1.0)


def _result(text="hi", conf=0.1):
    return EngineResult("c0", text, conf, "eng", {}, "eng")


def test_fallback_retry_re_invokes_engine():
    engine = MagicMock()
    engine.translate.return_value = _result("recovered", 0.9)
    fb = FallbackHandler(engine=engine)
    out = fb.retry_or_fallback(_chunk(), _result("weak", 0.1), "eng")
    assert out.result.raw_text == "recovered"
    assert out.fallback_path == "retry"


def test_fallback_asr_mt_seam_returns_flagged_when_no_real_asr():
    """Without a real ASR+MT provider, fallback degrades gracefully and
    flags low_confidence rather than crashing."""
    engine = MagicMock()
    # engine keeps returning weak results, so retry 'fails' -> try ASR+MT seam
    engine.translate.return_value = _result("weak", 0.1)
    fb = FallbackHandler(engine=engine, asr_mt=None)
    out = fb.retry_or_fallback(_chunk(), _result("weak", 0.1), "eng")
    assert out.low_confidence is True
    # ASR+MT not available -> last-resort returns the retried result, flagged
    assert out.fallback_path == "asr_mt_unavailable"
