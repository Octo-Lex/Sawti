"""Commit 6: the production builder — full recovery stack, injectable,
refusing weaker-than-configured construction."""
import numpy as np
import pytest

from sawti.build import build_real_pipeline, make_conservative_retry
from sawti.config import SawtiConfig
from sawti.fallback import FallbackHandler
from sawti.pipeline import Pipeline
from sawti.types import AudioChunk, EngineResult


class FakeM4T:
    """SeamlessM4TEngine-compatible fake recording conservative flags."""

    def __init__(self):
        self.calls: list[tuple[str, bool]] = []

    def translate(self, chunk, target_lang, conservative=False):
        self.calls.append((chunk.id, conservative))
        return EngineResult(chunk.id, f"eng:{chunk.id}", 0.9, "ara", {},
                            target_lang)


def _chunk(cid="c0", dur=1.0):
    return AudioChunk(id=cid, audio=np.zeros(int(16000 * dur), np.float32),
                      sample_rate=16000, start_time=0.0, end_time=dur)


class VowelGate:
    """Rejects any result whose text lacks the letter 'e' (scriptable)."""

    def evaluate(self, result, chunk, target_lang):
        ok = "e" in result.raw_text
        from sawti.types import GateDecision
        return GateDecision(chunk_id=chunk.id, accepted=ok, result=result,
                            checks={} if ok else {"garbage_output": True},
                            start_time=chunk.start_time,
                            end_time=chunk.end_time, needs_retry=not ok,
                            log=[{"action": "evaluate"}])


def test_builder_assembles_the_full_recovery_stack():
    engine = FakeM4T()
    pipe = build_real_pipeline(
        SawtiConfig(), m4t_engine=engine, provider="fake-provider-marker",
        gate=None,
    )
    assert isinstance(pipe, Pipeline)
    assert isinstance(pipe.fallback, FallbackHandler)
    # Conservative seam bound to the SAME engine instance.
    assert pipe.fallback.conservative is not None
    assert pipe.fallback.engine.engine is not None
    # Rechunker wired.
    assert pipe.fallback.rechunker is not None


def test_builder_rejects_asr_mt_disabled_when_config_promises_it():
    with pytest.raises(ValueError, match="weaker stack"):
        build_real_pipeline(SawtiConfig(), m4t_engine=FakeM4T(), provider=None)


def test_builder_asr_mt_off_in_config_allows_no_provider():
    from sawti.config import QualityGateConfig

    cfg = SawtiConfig(quality_gate=QualityGateConfig(fallback_to_asr_mt=False))
    pipe = build_real_pipeline(cfg, m4t_engine=FakeM4T())
    assert pipe.fallback.asr_mt is None


def test_conservative_seam_uses_same_engine_with_conservative_flag():
    from sawti.engine import EngineManager

    engine = FakeM4T()
    mgr = EngineManager(engine=engine)          # resident adoption
    seam = make_conservative_retry(mgr)
    seam(_chunk(), "eng")
    assert engine.calls == [("c0", True)]      # conservative=True, same engine


def test_full_stack_happy_path_through_builder():
    from sawti.segmenter import StubSegmenter
    from sawti.sources import StubAudioSource

    engine = FakeM4T()
    pipe = build_real_pipeline(
        SawtiConfig(), m4t_engine=engine, provider="fake", gate=None,
        segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
    )
    src = StubAudioSource(n_frames=2, samples_per_frame=16000)
    out = list(pipe.run(src, target_lang="eng"))
    assert [o.text for o in out] == ["eng:c0"]
    assert engine.calls == [("c0", False)]          # primary only, no retry


# --- corrective pass: production lifecycle is builder-level real ---

def _stub_segmenter():
    from sawti.segmenter import StubSegmenter
    return StubSegmenter(chunk_frames=2, sample_rate=16000)


def test_builder_lazy_does_not_load_at_construction():
    from sawti.config import S2TTConfig

    builds = {"n": 0}
    def factory():
        builds["n"] += 1
        return FakeM4T()
    pipe = build_real_pipeline(
        SawtiConfig(s2tt=S2TTConfig(load_policy="lazy")),
        m4t_factory=factory, provider="fake", segmenter=_stub_segmenter())
    assert builds["n"] == 0                    # NOTHING loaded at build
    out = list(pipe.run(_stub_src(), target_lang="eng"))
    assert builds["n"] == 1 and out            # loaded on first translate


def _stub_src():
    from sawti.sources import StubAudioSource
    return StubAudioSource(n_frames=2, samples_per_frame=16000)


def test_builder_idle_unload_defers_and_rebuilds():
    from sawti.config import S2TTConfig
    from tests.test_engine_manager import FakeClock

    builds = {"n": 0}
    def factory():
        builds["n"] += 1
        return FakeM4T()
    clock = FakeClock()
    pipe = build_real_pipeline(
        SawtiConfig(s2tt=S2TTConfig(load_policy="idle_unload",
                                    idle_unload_seconds=300)),
        m4t_factory=factory, provider="fake", clock=clock,
        segmenter=_stub_segmenter())
    assert builds["n"] == 0                    # deferred at construction
    list(pipe.run(_stub_src(), target_lang="eng"))
    assert builds["n"] == 1
    clock.advance(500)                          # past the idle window
    list(pipe.run(_stub_src(), target_lang="eng"))
    assert builds["n"] == 2                    # released and REBUILT


def test_builder_resident_loads_at_construction():
    from sawti.config import S2TTConfig

    builds = {"n": 0}
    def factory():
        builds["n"] += 1
        return FakeM4T()
    build_real_pipeline(SawtiConfig(s2tt=S2TTConfig(load_policy="resident")),
                        m4t_factory=factory, provider="fake")
    assert builds["n"] == 1                    # eager


def test_builder_m4t_engine_injection_requires_resident():
    from sawti.config import S2TTConfig

    with pytest.raises(ValueError, match="resident"):
        build_real_pipeline(
            SawtiConfig(s2tt=S2TTConfig(load_policy="lazy")),
            m4t_engine=FakeM4T(), provider="fake")


def test_builder_rejects_unsupported_gate_policy():
    from sawti.config import QualityGateConfig

    with pytest.raises(ValueError, match="quality_gate.policy"):
        build_real_pipeline(
            SawtiConfig(quality_gate=QualityGateConfig(policy="aggressive")),
            m4t_engine=FakeM4T(), provider="fake")
