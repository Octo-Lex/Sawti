"""Builder hardening (2026-10-09): restart seam binding, fallback
pre-warm, and the VRAM budget guard."""
import numpy as np
import pytest

from sawti.build import build_real_pipeline
from sawti.config import S2TTConfig, SawtiConfig
from sawti.pipeline import Pipeline
from sawti.types import AudioChunk, EngineResult


class RestartAwareFakeM4T:
    """Records every translate call's decode flags."""

    def __init__(self):
        self.calls: list[dict] = []

    def translate(self, chunk, target_lang, conservative=False,
                  restart=False):
        self.calls.append({"id": chunk.id, "conservative": conservative,
                           "restart": restart})
        return EngineResult(chunk.id, f"eng:{chunk.id}", 0.9, "ara", {},
                            target_lang)


class CountingProvider:
    """AsrMtProvider fake with counting pre-warm seams."""

    def __init__(self):
        self.asr_warmed = 0
        self.mt_warmed = 0

    def _ensure_asr(self):
        self.asr_warmed += 1

    def _ensure_mt(self):
        self.mt_warmed += 1

    def asr_mt(self, chunk, target_lang):  # pragma: no cover
        raise AssertionError("not used in these tests")


def _chunk():
    return AudioChunk(id="c0", audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=1.0)


def test_builder_binds_restart_seam_to_engine():
    engine = RestartAwareFakeM4T()
    pipe = build_real_pipeline(SawtiConfig(), m4t_engine=engine,
                               provider=CountingProvider())
    assert pipe.fallback.restart is not None
    pipe.fallback.restart(_chunk(), "eng")
    assert engine.calls == [{"id": "c0", "conservative": False,
                             "restart": True}]


def test_builder_prewarms_fallback_lane_when_configured():
    provider = CountingProvider()
    cfg = SawtiConfig(s2tt=S2TTConfig(prewarm_fallback=True))
    build_real_pipeline(cfg, m4t_engine=RestartAwareFakeM4T(),
                        provider=provider)
    assert provider.asr_warmed == 1
    assert provider.mt_warmed == 1


def test_builder_default_does_not_prewarm():
    provider = CountingProvider()
    build_real_pipeline(SawtiConfig(), m4t_engine=RestartAwareFakeM4T(),
                        provider=provider)
    assert provider.asr_warmed == 0
    assert provider.mt_warmed == 0


def test_vram_budget_fails_loud_when_exceeded(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 13.2e9)
    cfg = SawtiConfig(s2tt=S2TTConfig(vram_budget_gb=12.0))
    with pytest.raises(ValueError, match="VRAM budget exceeded"):
        build_real_pipeline(cfg, m4t_engine=RestartAwareFakeM4T(),
                            provider=CountingProvider())


def test_vram_budget_passes_when_within(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 10.0e9)
    cfg = SawtiConfig(s2tt=S2TTConfig(vram_budget_gb=12.0))
    pipe = build_real_pipeline(cfg, m4t_engine=RestartAwareFakeM4T(),
                               provider=CountingProvider())
    assert pipe is not None


def test_vram_budget_skipped_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = SawtiConfig(s2tt=S2TTConfig(vram_budget_gb=0.001))
    pipe = build_real_pipeline(cfg, m4t_engine=RestartAwareFakeM4T(),
                               provider=CountingProvider())
    assert pipe is not None


# --- SA primary mode (2026-10-12, spec SA 4.2) ---------------------------

class FakeSAEngine:
    """SaudiWhisperEngine-compatible fake recording decode flags."""

    def __init__(self):
        self.calls: list[dict] = []

    def translate(self, chunk, target_lang, conservative=False,
                  restart=False):
        self.calls.append({"id": chunk.id, "conservative": conservative,
                           "restart": restart})
        return EngineResult(chunk.id, f"ara:{chunk.id}", 0.9, "ara", {},
                            target_lang)


def test_builder_sa_primary_adopts_injected_engine():
    sa = FakeSAEngine()
    pipe = build_real_pipeline(SawtiConfig(), engine="sawti-sa",
                               sa_engine=sa, provider=CountingProvider())
    assert isinstance(pipe, Pipeline)
    # the primary engine IS the SA engine (resident adoption)
    assert pipe.engine.engine is sa
    # the fallback seams bind to the SAME engine instance
    pipe.fallback.restart(_chunk(), "ara")
    pipe.fallback.conservative(_chunk(), "ara")
    assert sa.calls == [
        {"id": "c0", "conservative": False, "restart": True},
        {"id": "c0", "conservative": True, "restart": False},
    ]


def test_builder_rejects_cross_engine_injections():
    with pytest.raises(ValueError, match="m4t_engine injection"):
        build_real_pipeline(SawtiConfig(), engine="sawti-sa",
                            m4t_engine=RestartAwareFakeM4T(),
                            provider=CountingProvider())
    with pytest.raises(ValueError, match="sa_engine injection"):
        build_real_pipeline(SawtiConfig(), m4t_engine=RestartAwareFakeM4T(),
                            sa_engine=FakeSAEngine(),
                            provider=CountingProvider())


def test_builder_rejects_unknown_engine_kind():
    with pytest.raises(ValueError, match="unsupported engine kind"):
        build_real_pipeline(SawtiConfig(), engine="whisper",
                            provider=CountingProvider())


def test_builder_sa_engine_injection_requires_resident():
    from sawti.config import S2TTConfig

    with pytest.raises(ValueError, match="resident"):
        build_real_pipeline(
            SawtiConfig(s2tt=S2TTConfig(load_policy="lazy")),
            engine="sawti-sa", sa_engine=FakeSAEngine(),
            provider=CountingProvider())
