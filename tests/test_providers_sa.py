"""SA integration contract tests: SaudiWhisperM4TProvider + builder wiring.

Hermetic: the SA model and M4T are injected fakes; nothing loads.
"""
import json

import numpy as np
import pytest

from sawti.types import AudioChunk

ASR_TEXT = "كلام سعودي واضح"


def _chunk(dur=2.0):
    return AudioChunk(id="c0", audio=np.zeros(int(16000 * dur), np.float32),
                      sample_rate=16000, start_time=0.0, end_time=dur)


class _FakeTok:
    def batch_decode(self, ids, skip_special_tokens=None):
        return [ASR_TEXT]


class _FakeFE:
    def __call__(self, audios, sampling_rate=None, return_tensors=None,
                 return_attention_mask=None):
        from types import SimpleNamespace

        import torch

        return SimpleNamespace(
            input_features=torch.zeros((len(audios), 1, 8), dtype=torch.float32),
            attention_mask=torch.ones((len(audios), 1), dtype=torch.long))


class _FakeSA:
    def __init__(self):
        self.calls = []

    def generate(self, input_features=None, attention_mask=None, **kwargs):
        self.calls.append(kwargs)
        return [[1, 2, 3]]


class _FakeProc:
    tokenizer = _FakeTok()
    feature_extractor = _FakeFE()


class _FakeMtTok:
    def decode(self, ids, skip_special_tokens=None):
        return "translated:" + "".join(map(str, ids))


class _FakeMtModel:
    def generate(self, **kw):
        return [[7, 7]]


class _FakeMtProc:
    tokenizer = _FakeMtTok()

    def __call__(self, text, src_lang, tgt_lang, return_tensors=None):
        assert src_lang == "arb", f"SA source must be arb, got {src_lang}"
        self.seen = (src_lang, tgt_lang)

        class _In:
            @staticmethod
            def to(device):
                return {}

        return _In()


def _provider(tmp_path=None, mt=True):
    from sawti.providers_sa import SaudiWhisperM4TProvider

    fake_mt = (_FakeMtProc(), _FakeMtModel()) if mt else None
    return (SaudiWhisperM4TProvider(_sa=(_FakeProc(), _FakeSA()), _mt=fake_mt),
            fake_mt)


def test_sa_greedy_kwargs_match_frozen_evaluator_regime():
    """The runtime path must decode EXACTLY like the evaluator that
    produced the selection evidence — no silent pipeline defaults."""
    from sawti.training.eval_checkpoint import GREEDY_KWARGS

    from sawti.providers_sa import SA_GREEDY_KWARGS
    assert SA_GREEDY_KWARGS == GREEDY_KWARGS


def test_asr_arabic_target_returns_transcript_verbatim():
    prov, _ = _provider()
    res = prov.asr_mt(_chunk(), "ara")
    assert res.raw_text == ASR_TEXT           # no MT hop for ara->ara
    assert res.source_lang_guess == "ara"
    assert res.confidence == 0.8
    assert res.timing_ms["path"] == "sa_whisper+m4t"


def test_asr_other_target_translates_via_m4t():
    prov, mt = _provider()
    res = prov.asr_mt(_chunk(), "eng")
    assert res.raw_text.startswith("translated:")
    assert mt[0].seen == ("arb", "eng")       # M4T codes, src fixed arb
    assert res.target_lang == "eng"


def test_attention_mask_and_greedy_forwarded_to_generate():
    prov, _ = _provider()
    prov.asr_mt(_chunk(), "ara")
    (proc, model) = prov._sa
    assert model.calls == [{"language": "arabic", "task": "transcribe",
                            "num_beams": 1, "do_sample": False}]
    # mask plumbed: generate received attention_mask kwarg
    assert len(model.calls) == 1


def test_builder_sa_provider_wiring():
    """build_real_pipeline(provider='sa') puts the SA provider in the
    fallback lane's ASR+MT seat (construction is lazy — no loads)."""
    from sawti.build import build_real_pipeline
    from sawti.config import SawtiConfig
    from sawti.providers_sa import SaudiWhisperM4TProvider

    class FakeM4T:
        def translate(self, chunk, target_lang, conservative=False):
            from sawti.types import EngineResult
            return EngineResult(chunk.id, "x", 0.9, "ara", {}, target_lang)

    pipe = build_real_pipeline(SawtiConfig(), m4t_engine=FakeM4T(),
                               provider="sa")
    assert isinstance(pipe.fallback.asr_mt, SaudiWhisperM4TProvider)
    # And the default remains the stock provider:
    from sawti.providers import WhisperM4TProvider
    pipe2 = build_real_pipeline(SawtiConfig(), m4t_engine=FakeM4T(),
                                provider="real")
    assert isinstance(pipe2.fallback.asr_mt, WhisperM4TProvider)
