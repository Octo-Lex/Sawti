"""Commit 2 acceptance: Pipeline delegates recovery to FallbackHandler.

Boundary under test:

    segment -> engine.translate (exactly once) -> gate.evaluate (exactly
    once) -> PASS: postprocess | FAIL: fallback.recover(chunk, decision,
    target) exactly once -> postprocess (accepted OR exhausted decision).

Pipeline performs no second engine.translate itself and holds no recovery
policy of its own.
"""
from __future__ import annotations

import numpy as np

from sawti.engine import EngineManager, StubEngine
from sawti.fallback import FallbackHandler
from sawti.pipeline import Pipeline
from sawti.postprocess import StubPostProcessor
from sawti.quality_gate import StubQualityGate
from sawti.segmenter import StubSegmenter
from sawti.sources import StubAudioSource
from sawti.types import AudioChunk, EngineResult, GateDecision, OutputSegment


def _chunk(cid: str, dur: float = 1.0) -> AudioChunk:
    return AudioChunk(id=cid, audio=np.zeros(int(16000 * dur), np.float32),
                      sample_rate=16000, start_time=0.0, end_time=dur)


class _CountingEngine:
    """conf 0.1 on first call, 0.9 afterwards — primary fails, retry works."""

    def __init__(self) -> None:
        self.calls = 0

    def translate(self, chunk, target_lang):
        self.calls += 1
        conf = 0.1 if self.calls == 1 else 0.9
        return EngineResult(chunk_id=chunk.id,
                            raw_text="weak" if self.calls == 1 else "recovered",
                            confidence=conf, source_lang_guess="ara",
                            timing_ms={}, target_lang=target_lang)


class _SpyFallback:
    """Records recover() invocations; returns a crafted decision."""

    def __init__(self, decision_factory) -> None:
        self.factory = decision_factory
        self.calls: list[tuple[str, GateDecision]] = []

    def recover(self, chunk, primary, target_lang):
        self.calls.append((chunk.id, primary))
        return self.factory(chunk, primary)


def _make(engine, gate=None, fallback=None) -> Pipeline:
    return Pipeline(
        segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
        engine=EngineManager(engine=engine),
        gate=gate or StubQualityGate(),
        postprocessor=StubPostProcessor(),
        fallback=fallback,
    )


def _src(n_frames: int = 2):
    return StubAudioSource(n_frames=n_frames, samples_per_frame=16000)


def test_happy_path_never_calls_recover():
    class ExplodingFallback:
        def recover(self, *a, **k):  # pragma: no cover
            raise AssertionError("recover() must not be called on PASS")

    pipe = _make(StubEngine("hello", 0.9), fallback=ExplodingFallback())
    out = list(pipe.run(_src(), target_lang="eng"))
    assert [o.text for o in out] == ["hello"]
    assert all(o.low_confidence is False for o in out)


def test_failed_primary_calls_recover_exactly_once_with_that_decision():
    spy = _SpyFallback(
        lambda chunk, primary: GateDecision(
            chunk_id=chunk.id, accepted=True,
            result=EngineResult(chunk.id, "from-fallback", 0.9, "ara", {}, "eng"),
            checks={}, start_time=chunk.start_time, end_time=chunk.end_time))
    engine = _CountingEngine()
    pipe = _make(engine, fallback=spy)
    out = list(pipe.run(_src(), target_lang="eng"))

    assert len(spy.calls) == 1                     # exactly once
    chunk_id, primary = spy.calls[0]
    assert chunk_id == "c0"
    assert primary.result.raw_text == "weak"       # the exact primary decision
    assert primary.needs_retry is True
    assert engine.calls == 1                       # Pipeline translated ONCE
    assert out[0].text == "from-fallback"          # fallback's decision emitted


def test_pipeline_without_fallback_does_not_retry_itself():
    engine = _CountingEngine()
    pipe = _make(engine)                           # no fallback configured
    out = list(pipe.run(_src(), target_lang="eng"))
    assert engine.calls == 1                       # no second translate, ever
    assert out[0].text == "weak"                   # flagged primary forwarded
    assert out[0].low_confidence is True


def test_exhausted_decision_reaches_postprocessing():
    def exhausted(chunk, primary):
        return GateDecision(
            chunk_id=chunk.id, accepted=False,
            result=EngineResult(chunk.id, "best-effort", 0.2, "ara", {}, "eng"),
            checks={}, start_time=chunk.start_time, end_time=chunk.end_time,
            fallback_path="exhausted", low_confidence=True, needs_retry=True)

    engine = _CountingEngine()
    pipe = _make(engine, fallback=_SpyFallback(exhausted))
    out = list(pipe.run(_src(), target_lang="eng"))
    assert out[0].text == "best-effort"
    assert out[0].low_confidence is True           # flagged, not dropped


def test_sequential_ordering_preserved_with_mixed_fallback():
    class SeqEngine:
        def __init__(self): self.n = 0
        def translate(self, chunk, target):
            self.n += 1
            ok = self.n != 2                       # chunk 2 fails primary
            return EngineResult(chunk.id, f"t{self.n}", 0.9 if ok else 0.1,
                                "ara", {}, target)

    def fix(chunk, primary):
        return GateDecision(chunk.id, True,
                            EngineResult(chunk.id, "fixed", 0.9, "ara", {}, "eng"),
                            {}, chunk.start_time, chunk.end_time)

    src = StubAudioSource(n_frames=6, samples_per_frame=16000)   # 3 chunks
    pipe = _make(SeqEngine(), fallback=_SpyFallback(fix))
    out = list(pipe.run(src, target_lang="eng"))
    assert [o.chunk_id for o in out] == ["c0", "c1", "c2"]       # in order
    assert [o.text for o in out] == ["t1", "fixed", "t3"]        # c1 recovered


def test_real_fallback_handler_end_to_end_through_pipeline():
    """Full stack: Pipeline + FallbackHandler (identity conservative retry):
    primary weak -> handler retry recovers. Total engine calls = 2
    (one Pipeline primary + one handler retry — never two from Pipeline)."""
    engine = _CountingEngine()
    pipe = _make(engine, fallback=FallbackHandler(engine=engine))
    out = list(pipe.run(_src(), target_lang="eng"))
    assert engine.calls == 2
    assert out[0].text == "recovered"
    assert out[0].low_confidence is False
