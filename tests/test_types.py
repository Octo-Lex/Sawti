import numpy as np
from sawti.types import AudioChunk, EngineResult, GateDecision, OutputSegment


def test_audio_chunk_construction():
    audio = np.zeros(16000, dtype=np.float32)  # 1s of silence
    chunk = AudioChunk(
        id="c0",
        audio=audio,
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
        overlap_from_prev_s=0.0,
        meta={},
    )
    assert chunk.id == "c0"
    assert chunk.audio.dtype == np.float32
    assert chunk.duration_s == 1.0


def test_engine_result_construction():
    r = EngineResult(
        chunk_id="c0",
        raw_text="hello",
        confidence=0.9,
        source_lang_guess="eng",
        timing_ms={"engine": 12},
        target_lang="eng",
    )
    assert r.raw_text == "hello"
    assert 0.0 <= r.confidence <= 1.0


def test_gate_decision_defaults():
    r = EngineResult("c0", "hi", 0.8, "eng", {}, "eng")
    d = GateDecision(chunk_id="c0", accepted=True, result=r, checks={})
    assert d.accepted is True
    assert d.fallback_path is None
    assert d.low_confidence is False
    assert d.needs_retry is False
    assert d.log == []


def test_output_segment_construction():
    seg = OutputSegment(
        chunk_id="c0", text="Hello", start_time=0.0, end_time=1.0, low_confidence=False
    )
    assert seg.text == "Hello"
