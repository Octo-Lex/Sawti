from sawti.engine import StubEngine
from sawti.postprocess import StubPostProcessor
from sawti.quality_gate import StubQualityGate
from sawti.types import AudioChunk, OutputSegment
import numpy as np


def _chunk(cid="c0", start=0.0, end=1.0):
    return AudioChunk(id=cid, audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=start, end_time=end)


def _decision(cid="c0", text="hi", conf=0.9, start=0.0, end=1.0):
    eng = StubEngine(text, conf)
    r = eng.translate(_chunk(cid, start, end), "eng")
    return StubQualityGate().evaluate(r, _chunk(cid, start, end), "eng")


def test_stub_postprocessor_emits_output_segment():
    pp = StubPostProcessor()
    out = list(pp.process([_decision()], target_lang="eng"))
    assert len(out) == 1
    assert isinstance(out[0], OutputSegment)
    assert out[0].text == "hi"


def test_stub_postprocessor_is_stateful_across_calls():
    """process() takes a list; the stub holds no cross-call state in M0,
    but the signature supports streaming via repeated calls in M1."""
    pp = StubPostProcessor()
    out = list(pp.process([_decision("c0"), _decision("c1", "bye", start=1.0, end=2.0)],
                          target_lang="eng"))
    assert [o.text for o in out] == ["hi", "bye"]
