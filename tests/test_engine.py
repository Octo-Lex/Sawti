from sawti.engine import EngineManager, StubEngine
from sawti.types import AudioChunk
import numpy as np


def _chunk(cid: str) -> AudioChunk:
    return AudioChunk(id=cid, audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=1.0)


def test_stub_engine_returns_target_text():
    eng = StubEngine(canned_text="hello world", confidence=0.9)
    r = eng.translate(_chunk("c0"), target_lang="eng")
    assert r.chunk_id == "c0"
    assert r.raw_text == "hello world"
    assert r.confidence == 0.9
    assert r.target_lang == "eng"


def test_engine_manager_translates_delegates():
    mgr = EngineManager(engine=StubEngine("hi", 0.8))
    r = mgr.translate(_chunk("c1"), target_lang="ara")
    assert r.raw_text == "hi"
    assert r.target_lang == "ara"
