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


# --- Commit 3: n-gram loops, load-bearing toggles, strictness, invariant ---

from sawti.config import ChecksConfig, QualityGateConfig as _QGC
from sawti.quality_gate_balanced import soft_script_mismatch


def test_gate_catches_phrase_loop():
    # The Saudi-spike failure mode ("اشتركوا في القناه" x3) invisible to the
    # old unigram-only check.
    r = _result(text="اشتركوا في القناه اشتركوا في القناه اشتركوا في القناه", target="eng")
    c = run_checks(r, _chunk(), "eng")
    assert c["repetition_loop"] is True


def test_gate_phrase_loop_drives_needs_retry():
    gate = BalancedQualityGate()
    d = gate.evaluate(_result(text="اشتركوا في القناه " * 3, conf=0.9), _chunk(), "eng")
    assert d.needs_retry is True and d.accepted is False


def test_every_check_toggle_is_load_bearing():
    cases = [
        # (toggle name, a result/chunk that triggers that check when enabled)
        ("empty_output", _result(text="", conf=0.9), _chunk()),
        ("garbage_output", _result(text="!!! ... ???", conf=0.9), _chunk()),
        ("script_mismatch", _result(text="hello world", target="ara"), _chunk()),
        ("length_ratio_anomaly", _result(text="a", conf=0.9), _chunk(3.0)),
        # 8s chunk: the phrase loop alone triggers; length-ratio stays in
        # bounds (~60 chars / 8s) so ONLY the repetition toggle is under test.
        ("repetition_loop", _result(text="اشتركوا في القناه " * 3, conf=0.9), _chunk(8.0)),
    ]
    for name, res, chk in cases:
        on = run_checks(res, chk, "ara" if name == "script_mismatch" else "eng")
        assert on[name] is True, f"{name} did not trigger when enabled"
        cfg = _QGC(checks=ChecksConfig(**{name: False}))
        off = run_checks(res, chk, "ara" if name == "script_mismatch" else "eng", cfg)
        assert off[name] is False, f"disabling {name} did not clear it"
        # And the disabled check no longer drives the decision:
        gate = BalancedQualityGate(config=cfg)
        d = gate.evaluate(res, chk, "ara" if name == "script_mismatch" else "eng")
        assert d.needs_retry is False, f"disabled {name} still escalated"


def test_script_strictness_strict_hard_fails_soft_does_not():
    # ara default = strict: Latin-dominant output hard-fails.
    r_ara = _result(text="hello world", target="ara")
    assert run_checks(r_ara, _chunk(), "ara")["script_mismatch"] is True

    # eng with Arabic-dominant output: signal present, strictness soft ->
    # NOT a hard check failure, but observable via soft_script_mismatch.
    r_eng = _result(text="مرحبا كيف حالك اليوم هنا", target="eng")
    assert run_checks(r_eng, _chunk(), "eng")["script_mismatch"] is False
    assert soft_script_mismatch(r_eng, "eng") is True
    gate = BalancedQualityGate()
    d = gate.evaluate(r_eng, _chunk(), "eng")
    assert d.needs_retry is False  # soft never escalates
    assert d.log[0]["soft_script_mismatch"] is True  # but is observable

    # Explicit strictness flip makes eng strict too.
    cfg = _QGC(script_mismatch_strictness={"eng": "strict", "fra": "soft", "ara": "strict"})
    assert run_checks(r_eng, _chunk(), "eng", cfg)["script_mismatch"] is True


def test_needs_retry_invariant_only_hard_failures_and_confidence():
    # High confidence + soft signal only + all hard checks untriggered.
    r = _result(text="مرحبا كيف حالك اليوم هنا", conf=0.9, target="eng")
    gate = BalancedQualityGate()
    d = gate.evaluate(r, _chunk(), "eng")
    assert d.needs_retry is False and d.accepted is True

    # Low confidence alone escalates; every check untriggered.
    d2 = gate.evaluate(_result(text="نص جيد هنا", conf=0.1), _chunk(), "eng")
    assert d2.needs_retry is True
    assert d2.low_confidence is True
    assert not any(d2.checks.values())


def test_script_mismatch_toggle_disables_soft_signal_too():
    # A fully disabled check is off for BOTH modes: no hard failure AND
    # no soft diagnostic in the decision log.
    cfg = _QGC(checks=ChecksConfig(script_mismatch=False))
    r = _result(text="مرحبا كيف حالك اليوم هنا", conf=0.9, target="eng")
    assert run_checks(r, _chunk(), "eng", cfg)["script_mismatch"] is False
    assert soft_script_mismatch(r, "eng", cfg) is False
    d = BalancedQualityGate(config=cfg).evaluate(r, _chunk(), "eng")
    assert d.log[0]["soft_script_mismatch"] is False
    assert d.needs_retry is False
