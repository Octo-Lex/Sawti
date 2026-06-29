from sawti.engine import StubEngine
from sawti.quality_gate import StubQualityGate
from sawti.types import AudioChunk
import numpy as np


def _chunk(cid="c0"):
    return AudioChunk(id=cid, audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=1.0)


def test_stub_gate_accepts_high_confidence():
    eng = StubEngine("hi", confidence=0.9)
    r = eng.translate(_chunk(), "eng")
    gate = StubQualityGate()
    d = gate.evaluate(r, _chunk(), target_lang="eng")
    assert d.accepted is True
    assert d.needs_retry is False
    assert d.low_confidence is False


def test_stub_gate_flags_low_confidence():
    eng = StubEngine("hi", confidence=0.2)
    r = eng.translate(_chunk(), "eng")
    gate = StubQualityGate()
    d = gate.evaluate(r, _chunk(), target_lang="eng")
    assert d.low_confidence is True
    assert d.needs_retry is True
