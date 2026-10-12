"""Degenerate-transcript collapse at the ASR+MT seam (2026-10-09).

A Whisper phrase-loop transcript must collapse BEFORE translation (cost)
and BEFORE the gate sees it (the gate then judges content, not the loop).
"""
import numpy as np

from sawti.providers import WhisperM4TProvider
from sawti.types import AudioChunk

PHRASE = "لكي نتحدث عن هذه المشكلة"
LOOP_TEXT = "لذلك يجب أن نتحدث عن هذه المشكلة " + " ".join([PHRASE] * 40)
COLLAPSED = "لذلك يجب أن نتحدث عن هذه المشكلة لكي نتحدث عن هذه المشكلة"


def _chunk():
    return AudioChunk(id="c0", audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=3.0)


def _provider_with_loop_transcript(seen):
    """Provider whose fake Whisper lane returns the looped Arabic
    transcript; _translate records what it was handed."""

    def fake_asr(audio, return_language=True):
        return {"text": LOOP_TEXT, "language": "arabic"}

    provider = WhisperM4TProvider(_asr_pipeline=fake_asr, _mt=("p", "m"))

    def capture_translate(text, src, tgt):
        seen.append(text)
        return "so we need to talk about this problem"

    provider._translate = capture_translate
    return provider


def test_looped_asr_transcript_collapses_before_translation():
    seen: list[str] = []
    provider = _provider_with_loop_transcript(seen)
    result = provider.asr_mt(_chunk(), "eng")

    assert seen == [COLLAPSED]        # T2TT got the collapsed text only
    assert result.raw_text == "so we need to talk about this problem"
    assert result.source_lang_guess == "ara"
    assert result.confidence == 0.8   # mapped language -> trusted lane


def test_verbatim_same_language_transcript_also_collapses():
    """detected == target: the transcript is returned verbatim — but a
    degenerate loop still collapses (the gate should judge content)."""
    seen: list[str] = []

    def fake_asr(audio, return_language=True):
        return {"text": LOOP_TEXT, "language": "arabic"}

    provider = WhisperM4TProvider(_asr_pipeline=fake_asr, _mt=("p", "m"))
    provider._translate = lambda text, src, tgt: seen.append(text) or "x"
    result = provider.asr_mt(_chunk(), "ara")

    assert result.raw_text == COLLAPSED
    assert seen == []                 # no translation for ara->ara
