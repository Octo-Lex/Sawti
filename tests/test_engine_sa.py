"""SaudiWhisperEngine (SA primary mode) contract tests — hermetic.

The SA model and the T2TT lane are injected fakes; nothing loads.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from sawti.engine_sa import SaudiWhisperEngine
from sawti.types import AudioChunk

ASR_TEXT = "كلام سعودي واضح"
LOOP_TEXT = "لذلك لازم لذلك لازم لذلك لازم لذلك لازم"


def _chunk(dur=2.0):
    return AudioChunk(id="c0", audio=np.zeros(int(16000 * dur), np.float32),
                      sample_rate=16000, start_time=0.0, end_time=dur)


class _FakeTok:
    def __init__(self, text=ASR_TEXT):
        self.text = text

    def batch_decode(self, ids, skip_special_tokens=None):
        return [self.text]


class _FakeFE:
    def __call__(self, audios, sampling_rate=None, return_tensors=None,
                 return_attention_mask=None):
        return SimpleNamespace(
            input_features=torch.zeros((len(audios), 1, 8),
                                       dtype=torch.float32),
            attention_mask=torch.ones((len(audios), 1), dtype=torch.long))


class _FakeSA:
    def __init__(self):
        self.calls = []

    def to(self, device):
        return self

    def generate(self, input_features=None, attention_mask=None, **kw):
        # record everything: attention_mask is a named parameter at the
        # call site (the frozen SA regime), not a **kwarg
        self.calls.append({"attention_mask": attention_mask is not None,
                           **kw})
        return [[1, 2, 3]]


def _proc(text=ASR_TEXT):
    return SimpleNamespace(tokenizer=_FakeTok(text),
                           feature_extractor=_FakeFE())


class _FakeMt:
    def __init__(self):
        self.calls = []

    def translate(self, text, src, tgt):
        self.calls.append((text, src, tgt))
        return "english translation"


def test_primary_uses_frozen_sa_regime():
    model, mt = _FakeSA(), _FakeMt()
    eng = SaudiWhisperEngine(processor=_proc(), model=model, mt=mt)
    r = eng.translate(_chunk(), "ara")
    kw = model.calls[0]
    assert kw["language"] == "arabic" and kw["task"] == "transcribe"
    assert kw["num_beams"] == 1 and kw["do_sample"] is False
    assert kw["attention_mask"] is not None
    assert r.raw_text == ASR_TEXT
    assert r.source_lang_guess == "ara"
    assert mt.calls == []                      # ara target: verbatim, no MT


def test_non_arabic_target_goes_through_mt():
    model, mt = _FakeSA(), _FakeMt()
    eng = SaudiWhisperEngine(processor=_proc(), model=model, mt=mt)
    r = eng.translate(_chunk(), "eng")
    assert mt.calls == [(ASR_TEXT, "ara", "eng")]
    assert r.raw_text == "english translation"
    assert r.timing_ms["path"] == "sa_whisper+m4t"
    assert "mt_ms" in r.timing_ms


def test_conservative_and_restart_modes_distinct():
    model = _FakeSA()
    eng = SaudiWhisperEngine(processor=_proc(), model=model, mt=_FakeMt())
    eng.translate(_chunk(), "ara", conservative=True)
    eng.translate(_chunk(), "ara", restart=True)
    beam_kw, restart_kw = model.calls[-2], model.calls[-1]
    assert beam_kw["num_beams"] == 5 and beam_kw["do_sample"] is False
    assert restart_kw["no_repeat_ngram_size"] == 8
    assert restart_kw.get("num_beams", 1) == 1               # greedy cost
    assert eng.translate(_chunk(), "ara").timing_ms["mode"] == "primary"


def test_degenerate_transcript_collapses_before_mt():
    model = _FakeSA()
    mt = _FakeMt()
    eng = SaudiWhisperEngine(
        processor=_proc(text=LOOP_TEXT), model=model, mt=mt)
    r = eng.translate(_chunk(), "eng")
    sent = mt.calls[0][0]
    assert sent == "لذلك لازم"               # collapsed, not the 5x loop
    assert r.raw_text == "english translation"


def test_confidence_is_heuristic_and_scores_never_requested():
    """Whisper's generate() must never receive return_dict_in_generate/
    output_scores (invalid flags for it; carried by every native crash
    in the 2026-10-12 SA eval runs). Confidence is the provider-lane
    heuristic (0.8), matching providers_sa."""
    model = _FakeSA()
    eng = SaudiWhisperEngine(processor=_proc(), model=model, mt=_FakeMt())
    r = eng.translate(_chunk(), "ara")
    assert r.confidence == 0.8
    for call in model.calls:
        assert "return_dict_in_generate" not in call
        assert "output_scores" not in call
