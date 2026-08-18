"""Commit 6 corrective pass: ASR+MT provider contract, hermetic via the
_asr_pipeline/_mt injection seams."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from sawti.providers import WhisperM4TProvider
from sawti.types import AudioChunk


def _chunk(cid="c0", dur=1.0):
    return AudioChunk(id=cid, audio=np.zeros(int(16000 * dur), np.float32),
                      sample_rate=16000, start_time=0.0, end_time=dur)


def _asr(text, language):
    m = MagicMock()
    m.return_value = {"text": text, "language": language}
    return m


def test_same_language_returns_transcript_verbatim_mt_unused():
    mt = MagicMock()
    p = WhisperM4TProvider(_asr_pipeline=_asr("مرحبا", "arabic"), _mt=mt)
    r = p.asr_mt(_chunk(), "ara")
    assert r.raw_text == "مرحبا"
    assert r.source_lang_guess == "ara"
    assert r.confidence == 0.8
    mt.assert_not_called()                     # MT never invoked


def test_different_language_routes_through_mt_with_correct_codes():
    from sawti.types import EngineResult  # noqa: F401
    processor, model = MagicMock(), MagicMock()
    model.generate.return_value = [[7, 8, 9]]
    processor.tokenizer.decode.return_value = "hello"
    p = WhisperM4TProvider(_asr_pipeline=_asr("مرحبا", "arabic"),
                           _mt=(processor, model))
    r = p.asr_mt(_chunk(), "eng")
    kwargs = processor.call_args.kwargs
    assert kwargs["src_lang"] == "arb" and kwargs["tgt_lang"] == "eng"
    assert model.generate.call_args.kwargs["tgt_lang"] == "eng"
    assert r.raw_text == "hello"
    assert r.target_lang == "eng" and r.source_lang_guess == "ara"


def test_unmapped_language_is_explicitly_untrusted():
    mt = MagicMock()
    p = WhisperM4TProvider(_asr_pipeline=_asr("guten tag", "german"), _mt=mt)
    r = p.asr_mt(_chunk(), "eng")
    # Transcript preserved, but marked: zero confidence + explicit marker
    # — the gate flags it rather than passing German off as English.
    assert r.raw_text == "guten tag"
    assert r.confidence == 0.0
    assert r.timing_ms["unmapped_language"] == "german"
    assert r.source_lang_guess == "german"
    mt.assert_not_called()
