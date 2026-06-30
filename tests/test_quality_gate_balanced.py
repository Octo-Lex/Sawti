import numpy as np

from sawti.quality_gate_balanced import BalancedQualityGate, run_checks
from sawti.types import AudioChunk, EngineResult


def _chunk(dur_s=1.0):
    return AudioChunk(id="c0", audio=np.zeros(int(16000 * dur_s), np.float32),
                      sample_rate=16000, start_time=0.0, end_time=dur_s)


def _result(text="hi", conf=0.9, target="eng"):
    return EngineResult("c0", text, conf, "eng", {}, target)


def test_run_checks_empty_output_flagged():
    c = run_checks(_result(text="", conf=0.9, target="eng"), _chunk(), "eng")
    assert c["empty_output"] is True


def test_run_checks_script_mismatch_for_arabic_target():
    # target is Arabic but output is Latin
    c = run_checks(_result(text="hello world", target="ara"), _chunk(), "ara")
    assert c["script_mismatch"] is True


def test_run_checks_script_ok_for_latin_words_in_arabic():
    # numbers/entities allowed: a few latin chars in mostly-arabic is fine
    c = run_checks(_result(text="مرحبا 123 مرحبا", target="ara"), _chunk(), "ara")
    assert c["script_mismatch"] is False


def test_run_checks_length_anomaly_too_short():
    # 3s of audio but only 1 char
    c = run_checks(_result(text="a", conf=0.9, target="eng"), _chunk(3.0), "eng")
    assert c["length_ratio_anomaly"] is True


def test_run_checks_repetition_loop_flagged():
    c = run_checks(_result(text="the the the the the", target="eng"), _chunk(), "eng")
    assert c["repetition_loop"] is True


def test_balanced_gate_accepts_good_result():
    gate = BalancedQualityGate()
    d = gate.evaluate(_result("hello world", 0.9, "eng"), _chunk(), "eng")
    assert d.accepted is True
    assert d.needs_retry is False


def test_balanced_gate_retries_on_low_confidence():
    gate = BalancedQualityGate()
    d = gate.evaluate(_result("hi", 0.1, "eng"), _chunk(), "eng")
    assert d.needs_retry is True
    assert d.fallback_path == "retry"
