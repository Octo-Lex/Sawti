import numpy as np

from sawti.engine import EngineManager, StubEngine
from sawti.pipeline import Pipeline
from sawti.postprocess import StubPostProcessor
from sawti.quality_gate import StubQualityGate
from sawti.segmenter import StubSegmenter
from sawti.sources import StubAudioSource
from sawti.types import OutputSegment


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


def test_pipeline_retry_path_when_low_confidence():
    """When the gate says needs_retry, the orchestrator must call the
    fallback handler once and emit a (still-stub) result."""
    src = StubAudioSource(n_frames=2, samples_per_frame=16000)
    pipe = Pipeline(
        segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
        engine=EngineManager(engine=StubEngine("x", 0.1)),  # low conf
        gate=StubQualityGate(),
        postprocessor=StubPostProcessor(),
    )
    out = list(pipe.run(src, target_lang="eng"))
    assert len(out) == 1
    assert out[0].low_confidence is True  # stub gate can't recover, but flow is intact
