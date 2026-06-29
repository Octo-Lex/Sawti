import numpy as np

from sawti.engine import EngineManager, StubEngine
from sawti.pipeline import Pipeline
from sawti.postprocess import StubPostProcessor
from sawti.quality_gate import StubQualityGate
from sawti.segmenter import StubSegmenter
from sawti.sources import StubAudioSource
from sawti.types import AudioChunk, EngineResult, OutputSegment


def test_pipeline_end_to_end_on_stubs():
    src = StubAudioSource(n_frames=4, samples_per_frame=16000)
    pipe = Pipeline(
        segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
        engine=EngineManager(engine=StubEngine("hello", 0.9)),
        gate=StubQualityGate(),
        postprocessor=StubPostProcessor(),
    )
    out = list(pipe.run(src, target_lang="eng"))
    assert len(out) == 2
    assert all(isinstance(o, OutputSegment) for o in out)
    assert all(o.text == "hello" for o in out)
    assert all(o.low_confidence is False for o in out)


def test_pipeline_carries_chunk_timestamps_to_output():
    """OutputSegment timestamps must reflect the source AudioChunk timing,
    not be hardcoded to 0.0 (regression guard for the GateDecision contract)."""
    src = StubAudioSource(n_frames=4, samples_per_frame=16000)  # 2s per chunk
    pipe = Pipeline(
        segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
        engine=EngineManager(engine=StubEngine("hello", 0.9)),
        gate=StubQualityGate(),
        postprocessor=StubPostProcessor(),
    )
    out = list(pipe.run(src, target_lang="eng"))
    assert out[0].start_time == 0.0
    assert out[0].end_time > 0.0          # non-zero: real chunk timing carried through
    assert out[1].start_time == out[0].end_time  # contiguous chunks


class _CountingEngine:
    """Engine that records how many times translate() is called.

    First call returns low confidence (triggers retry); subsequent calls
    return high confidence so the orchestrator's second pass can succeed.
    Used to verify the retry path actually re-invokes the engine.
    """

    def __init__(self) -> None:
        self.calls = 0

    def translate(self, chunk: AudioChunk, target_lang: str) -> EngineResult:
        self.calls += 1
        conf = 0.1 if self.calls == 1 else 0.9
        return EngineResult(
            chunk_id=chunk.id,
            raw_text="recovered" if self.calls > 1 else "weak",
            confidence=conf,
            source_lang_guess="und",
            timing_ms={"engine": 0},
            target_lang=target_lang,
        )


def test_pipeline_retry_actually_re_invokes_engine():
    """The retry path must call translate() twice on a needs_retry chunk.
    This test fails if the orchestrator's retry branch is deleted."""
    src = StubAudioSource(n_frames=2, samples_per_frame=16000)  # 1 chunk
    engine = _CountingEngine()
    pipe = Pipeline(
        segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
        engine=EngineManager(engine=engine),
        gate=StubQualityGate(),
        postprocessor=StubPostProcessor(),
    )
    out = list(pipe.run(src, target_lang="eng"))
    assert engine.calls == 2          # original + retry
    assert len(out) == 1
    assert out[0].text == "recovered"  # the retry result, not the original
    assert out[0].low_confidence is False


def test_pipeline_no_retry_when_confident():
    """When the gate does NOT flag needs_retry, translate() is called once."""
    src = StubAudioSource(n_frames=2, samples_per_frame=16000)
    engine = StubEngine("hi", 0.9)  # high confidence, no retry
    pipe = Pipeline(
        segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
        engine=EngineManager(engine=engine),
        gate=StubQualityGate(),
        postprocessor=StubPostProcessor(),
    )
    out = list(pipe.run(src, target_lang="eng"))
    assert len(out) == 1
    assert out[0].text == "hi"
