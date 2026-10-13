"""Script-aware best-effort selection + provider script honesty (2026-10-13).

Pins the Mode 2 fix from the 2026-10-12 failure analysis: a wrong-script
candidate must never beat a target-script candidate on confidence alone,
and the ASR+MT providers must report dominant-script mismatches as
UNTRUSTED (conf 0.0 + marker) instead of confident successes.
"""
from unittest.mock import MagicMock

import numpy as np

from sawti.fallback import FallbackHandler, _better
from sawti.providers import WhisperM4TProvider
from sawti.types import AudioChunk, EngineResult, GateDecision

EN_TEXT = "So we need to talk about this problem."
AR_TEXT = "لذلك يجب أن نتحدث عن هذه المشكلة"


def _chunk():
    return AudioChunk(id="c0", audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=2.0)


def _decision(text, conf, chunk_id="c0"):
    return GateDecision(
        chunk_id=chunk_id, accepted=False,
        result=EngineResult(chunk_id, text, conf, "ara", {}, "eng"),
        checks={}, start_time=0.0, end_time=2.0, low_confidence=True,
        needs_retry=True)


# ---- fallback selection rules -------------------------------------------

def test_target_script_beats_higher_confidence_wrong_script():
    """The enar_06 scenario verbatim: the asr_mt Arabic transcript at 0.8
    must NOT outcompete the English candidate at 0.45."""
    wrong = _decision(AR_TEXT, 0.8)
    right = _decision(EN_TEXT, 0.45)
    assert _better(wrong, right, "eng") is right
    assert _better(right, wrong, "eng") is right


def test_symmetric_for_ara_target():
    right = _decision(AR_TEXT, 0.3)
    wrong = _decision(EN_TEXT, 0.9)
    assert _better(wrong, right, "ara") is right


def test_tie_within_tier_falls_back_to_confidence():
    a = _decision("first candidate", 0.5)
    b = _decision("second candidate", 0.7)
    assert _better(a, b, "eng") is b


def test_no_target_lang_is_pure_confidence():
    wrong = _decision(AR_TEXT, 0.8)
    right = _decision(EN_TEXT, 0.45)
    assert _better(wrong, right, None) is wrong


def test_all_wrong_script_still_picks_highest_confidence():
    """Never drop a chunk: when every candidate is wrong-script, the
    confidence rule still selects one (emitted flagged, as today)."""
    a = _decision(AR_TEXT + " واحد", 0.4)
    b = _decision(AR_TEXT + " اثنان", 0.6)
    assert _better(a, b, "eng") is b


def test_empty_text_never_reaches_top_tier():
    empty = _decision("", 0.9)
    right = _decision(EN_TEXT, 0.2)
    assert _better(empty, right, "eng") is right


def test_exhausted_emission_prefers_target_script_end_to_end():
    """Full recover(): every stage fails; the exhausted best-effort must
    be the English candidate even though the asr_mt Arabic transcript
    scores higher on confidence."""
    engine = MagicMock()
    engine.translate.return_value = EngineResult(
        "c0", "english attempt", 0.3, "ara", {}, "eng")

    asr_mt = MagicMock()
    # Post-fix provider behavior: the wrong-script result arrives
    # UNTRUSTED (conf 0.0 + marker) — the provider honesty of seam 1.
    asr_mt.asr_mt.return_value = EngineResult(
        "c0", AR_TEXT, 0.0, "ara", {"script_mismatch": True}, "eng")

    rechunker = MagicMock()
    rechunker.rechunk.return_value = []

    fb = FallbackHandler(engine=engine, asr_mt=asr_mt, rechunker=rechunker)
    out = fb.recover(_chunk(), _decision("bad primary", 0.2), "eng")
    assert out.fallback_path == "exhausted"
    # Seam 2: among the exhausted candidates (primary/retry at 0.3
    # English vs asr_mt at tier 0), the target-script one is emitted.
    assert out.result.raw_text.startswith("english attempt")


# ---- provider script honesty --------------------------------------------

def _provider(fake_asr, translate_result):
    p = WhisperM4TProvider(_asr_pipeline=fake_asr, _mt=("p", "m"))
    p._translate = lambda text, src, tgt: translate_result
    return p


def test_provider_dominant_mismatch_is_untrusted():
    """T2TT returns mostly-Arabic for eng: conf 0.0 + script_mismatch."""
    def fake_asr(audio, return_language=True):
        return {"text": "نص عربي واضح جداً هنا", "language": "arabic"}

    r = _provider(fake_asr, translate_result="نص عربي لم يُترجم").asr_mt(
        _chunk(), "eng")
    assert r.confidence == 0.0
    assert r.timing_ms["script_mismatch"] is True


def test_provider_target_script_keeps_confidence():
    def fake_asr(audio, return_language=True):
        return {"text": "نص عربي واضح جداً هنا", "language": "arabic"}

    r = _provider(fake_asr, translate_result=EN_TEXT).asr_mt(_chunk(), "eng")
    assert r.confidence == 0.8
    assert r.timing_ms["script_mismatch"] is None


def test_provider_entities_do_not_trigger_mismatch():
    """Spec soft-for-eng intent: Latin text containing Arabic entities
    (names, numbers) stays trusted."""
    def fake_asr(audio, return_language=True):
        return {"text": "mixed", "language": "arabic"}

    mixed = "He met Abu Khalid in Riyadh near the Al Malqa district."
    r = _provider(fake_asr, translate_result=mixed).asr_mt(_chunk(), "eng")
    assert r.confidence == 0.8
