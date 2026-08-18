"""Commit 6: EngineManager lifecycle is observable, not aspirational."""
import numpy as np
import pytest

from sawti.config import S2TTConfig
from sawti.engine import EngineManager, StubEngine
from sawti.types import AudioChunk


def _chunk(cid="c0"):
    return AudioChunk(id=cid, audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=1.0)


class FakeClock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def _factory_counter():
    builds = {"n": 0}
    def factory():
        builds["n"] += 1
        return StubEngine("lazy-text", 0.9)
    return factory, builds


def test_resident_builds_eagerly():
    factory, builds = _factory_counter()
    mgr = EngineManager(config=S2TTConfig(load_policy="resident"),
                        engine_factory=factory)
    assert builds["n"] == 1                    # built at construction
    mgr.translate(_chunk(), "eng")
    assert builds["n"] == 1                    # no rebuild on use


def test_lazy_defers_to_first_translate():
    factory, builds = _factory_counter()
    mgr = EngineManager(config=S2TTConfig(load_policy="lazy"),
                        engine_factory=factory)
    assert builds["n"] == 0                    # nothing built yet
    r = mgr.translate(_chunk(), "eng")
    assert builds["n"] == 1 and r.raw_text == "lazy-text"
    mgr.translate(_chunk("c1"), "eng")
    assert builds["n"] == 1                    # stays resident while used


def test_idle_unload_rebuilds_after_timeout():
    factory, builds = _factory_counter()
    clock = FakeClock()
    mgr = EngineManager(
        config=S2TTConfig(load_policy="idle_unload", idle_unload_seconds=300),
        engine_factory=factory, clock=clock)
    assert builds["n"] == 0                    # deferred
    mgr.translate(_chunk(), "eng")
    assert builds["n"] == 1
    clock.advance(100)
    mgr.translate(_chunk("c1"), "eng")
    assert builds["n"] == 1                    # within idle window: kept
    clock.advance(500)                          # past idle_unload_seconds
    mgr.translate(_chunk("c2"), "eng")
    assert builds["n"] == 2                    # rebuilt


def test_explicit_instance_keeps_working_and_never_rebuilds():
    engine = StubEngine("direct", 0.9)
    mgr = EngineManager(engine=engine,
                        config=S2TTConfig(load_policy="idle_unload"))
    mgr.translate(_chunk(), "eng")
    assert mgr.translate(_chunk("c1"), "eng").raw_text == "direct"
    assert mgr.builds == 0                     # instance adopted, not built


def test_construction_requires_engine_or_factory():
    with pytest.raises(ValueError):
        EngineManager()
    factory, _ = _factory_counter()
    with pytest.raises(ValueError):
        EngineManager(engine=StubEngine(), engine_factory=factory)
