"""Acceptance tests: the fallback escalation contract (spec §3.6, §5.3).

Proves the full traversal with fakes:

    primary -> conservative retry -> rechunk -> ASR+MT -> best-effort

every stage traversed in exactly that order, the final GateDecision's
fallback_path naming the stage that won, and decision.log carrying a
complete trace. Config flags must demonstrably change the trace.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sawti.config import QualityGateConfig
from sawti.fallback import FallbackHandler, format_trace
from sawti.types import AudioChunk, EngineResult, GateDecision


def _chunk(cid: str = "c0", dur_s: float = 4.0) -> AudioChunk:
    n = int(16000 * dur_s)
    return AudioChunk(id=cid, audio=np.zeros(n, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=dur_s)


def _result(cid: str, text: str, conf: float, target: str = "eng") -> EngineResult:
    return EngineResult(chunk_id=cid, raw_text=text, confidence=conf,
                        source_lang_guess="ara", timing_ms={}, target_lang=target)


class ScriptedGate:
    """Gate whose evaluate() pops scripted verdicts in call order."""

    def __init__(self, verdicts: list[dict]) -> None:
        # verdict: {"accepted": bool, "checks": {...}, "low_conf": bool}
        self._verdicts = list(verdicts)
        self.calls: list[str] = []

    def evaluate(self, result, chunk, target_lang):
        self.calls.append(chunk.id)
        v = self._verdicts.pop(0)
        return GateDecision(
            chunk_id=chunk.id, accepted=v["accepted"], result=result,
            checks=v.get("checks", {}),
            start_time=chunk.start_time, end_time=chunk.end_time,
            low_confidence=v.get("low_conf", False),
            needs_retry=not v["accepted"],
            log=[{"action": "evaluate"}],
        )


class RecordingEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def translate(self, chunk, target_lang):
        self.calls.append(chunk.id)
        return _result(chunk.id, f"eng:{chunk.id}", 0.5)


class FakeRechunker:
    """Always splits into two sub-chunks: <cid>.r0 / <cid>.r1."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def rechunk(self, chunk):
        self.calls.append(chunk.id)
        half = len(chunk.audio) // 2
        mid = chunk.start_time + (chunk.end_time - chunk.start_time) / 2
        out = []
        for i, (a, b, s, e) in enumerate([
            (0, half, chunk.start_time, mid),
            (half, len(chunk.audio), mid, chunk.end_time),
        ]):
            out.append(AudioChunk(id=f"{chunk.id}.r{i}",
                                  audio=chunk.audio[a:b].copy(),
                                  sample_rate=chunk.sample_rate,
                                  start_time=s, end_time=e))
        return out


class FakeAsrMt:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def asr_mt(self, chunk, target_lang):
        self.calls.append(chunk.id)
        return _result(chunk.id, "asr_mt:recovered", 0.9)


def _primary(chunk_id: str, checks: dict) -> GateDecision:
    """A rejected primary decision, as the Pipeline will hand to recover()."""
    return GateDecision(
        chunk_id=chunk_id, accepted=False,
        result=_result(chunk_id, "eng:primary-loop", 0.3),
        checks=checks, start_time=0.0, end_time=4.0,
        low_confidence=False, needs_retry=True,
        log=[{"action": "evaluate"}],
    )


FULL_SCRIPT = [
    # retry verdict
    {"accepted": False, "checks": {"low_confidence": True}, "low_conf": True},
    # rechunk[0] verdict
    {"accepted": False, "checks": {"empty_output": True}},
    # rechunk[1] verdict
    {"accepted": False, "checks": {"low_confidence": True}, "low_conf": True},
    # asr_mt verdict
    {"accepted": True, "checks": {}},
]


def test_full_escalation_path_order_and_trace():
    chunk = _chunk()
    engine, gate, rechunker, provider = RecordingEngine(), ScriptedGate(FULL_SCRIPT), FakeRechunker(), FakeAsrMt()
    fb = FallbackHandler(engine=engine, gate=gate, asr_mt=provider,
                         rechunker=rechunker)
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")

    # Final decision: accepted via the ASR+MT stage.
    assert out.accepted is True
    assert out.fallback_path == "asr_mt"
    assert out.result.raw_text == "asr_mt:recovered"
    assert out.low_confidence is False

    # Engine called for retry + both rechunk sub-chunks (NOT primary — the
    # Pipeline owns the primary attempt) in exactly this order.
    assert engine.calls == ["c0", "c0.r0", "c0.r1"]
    # Gate evaluated those same three plus the provider's result.
    assert gate.calls == ["c0", "c0.r0", "c0.r1", "c0"]
    # Provider saw the ORIGINAL chunk.
    assert provider.calls == ["c0"]
    # Rechunker invoked once, on the original chunk.
    assert rechunker.calls == ["c0"]

    # The human-readable trace matches the locked control flow.
    trace = format_trace(out)
    assert trace.splitlines() == [
        "chunk c0",
        "  primary       -> rejected: repetition_loop",
        "  retry         -> rejected: low_confidence",
        "  rechunk[0]    -> rejected: empty_output",
        "  rechunk[1]    -> rejected: low_confidence",
        "  asr_mt        -> accepted",
    ]
    # decision.log carries the same traversal as structured entries.
    stages = [e["stage"] for e in out.log if isinstance(e, dict) and "stage" in e]
    assert stages == ["primary", "retry", "rechunk[0]", "rechunk[1]", "asr_mt"]


def test_retry_recovery_short_circuits_before_rechunk():
    chunk = _chunk()
    engine, gate, rechunker, provider = RecordingEngine(), ScriptedGate(
        [{"accepted": True, "checks": {}}]), FakeRechunker(), FakeAsrMt()
    fb = FallbackHandler(engine=engine, gate=gate, asr_mt=provider,
                         rechunker=rechunker)
    out = fb.recover(chunk, _primary("c0", {"low_confidence": True}), "eng")
    assert out.accepted is True
    assert out.fallback_path == "retry"
    assert engine.calls == ["c0"]           # retry only
    assert rechunker.calls == []            # never reached
    assert provider.calls == []             # never reached


def test_rechunk_acceptance_beats_escalation():
    chunk = _chunk()
    engine = RecordingEngine()
    gate = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}},   # retry
        {"accepted": False, "checks": {"empty_output": True}},     # rechunk[0]
        {"accepted": True, "checks": {}},                          # rechunk[1]
    ])
    rechunker, provider = FakeRechunker(), FakeAsrMt()
    fb = FallbackHandler(engine=engine, gate=gate, asr_mt=provider,
                         rechunker=rechunker)
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")
    assert out.accepted is True
    assert out.fallback_path == "rechunk"
    assert out.chunk_id == "c0.r1"          # the accepted sub-chunk won
    assert provider.calls == []             # escalation never reached


def test_exhausted_path_returns_flagged_best_effort():
    chunk = _chunk()
    engine = RecordingEngine()
    gate = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}, "low_conf": True},
        {"accepted": False, "checks": {"empty_output": True}},
        {"accepted": False, "checks": {"empty_output": True}},
        {"accepted": False, "checks": {"repetition_loop": True}},
    ])
    fb = FallbackHandler(engine=engine, gate=gate, asr_mt=FakeAsrMt(),
                         rechunker=FakeRechunker())
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")
    assert out.accepted is False
    assert out.fallback_path == "exhausted"
    assert out.low_confidence is True
    trace = format_trace(out)
    assert "  asr_mt        -> rejected: repetition_loop" in trace
    assert "  exhausted     -> best-effort flagged output" in trace


# --- config switches must demonstrably change the trace ---

def test_retry_once_false_removes_retry_stage():
    chunk = _chunk()
    engine = RecordingEngine()
    gate = ScriptedGate([
        {"accepted": True, "checks": {}},    # rechunk[0]
        {"accepted": True, "checks": {}},    # rechunk[1] (all subs evaluated)
    ])
    rechunker, provider = FakeRechunker(), FakeAsrMt()
    fb = FallbackHandler(engine=engine, gate=gate, asr_mt=provider,
                         rechunker=rechunker,
                         config=QualityGateConfig(retry_once=False))
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")
    # No retry on the full chunk; both sub-chunks were tried and evaluated.
    assert engine.calls == ["c0.r0", "c0.r1"]
    assert gate.calls == ["c0.r0", "c0.r1"]
    assert out.fallback_path == "rechunk"
    stages = [e["stage"] for e in out.log if isinstance(e, dict) and "stage" in e]
    assert stages == ["primary", "rechunk[0]", "rechunk[1]"]


def test_rechunk_on_failure_false_removes_rechunk_stage():
    chunk = _chunk()
    engine = RecordingEngine()
    gate = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}},
        {"accepted": True, "checks": {}},    # asr_mt accepts
    ])
    rechunker, provider = FakeRechunker(), FakeAsrMt()
    fb = FallbackHandler(engine=engine, gate=gate, asr_mt=provider,
                         rechunker=rechunker,
                         config=QualityGateConfig(rechunk_on_failure=False))
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")
    assert rechunker.calls == []
    assert out.fallback_path == "asr_mt"
    stages = [e["stage"] for e in out.log if isinstance(e, dict) and "stage" in e]
    assert stages == ["primary", "retry", "asr_mt"]


def test_no_rechunker_configured_skips_rechunk_stage():
    chunk = _chunk()
    engine = RecordingEngine()
    gate = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}},
        {"accepted": True, "checks": {}},
    ])
    fb = FallbackHandler(engine=engine, gate=gate, asr_mt=FakeAsrMt())
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")
    assert out.fallback_path == "asr_mt"
    stages = [e["stage"] for e in out.log if isinstance(e, dict) and "stage" in e]
    assert stages == ["primary", "retry", "asr_mt"]


def test_fallback_to_asr_mt_false_exhausts_instead():
    chunk = _chunk()
    engine = RecordingEngine()
    gate = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}},
    ])
    provider = FakeAsrMt()
    fb = FallbackHandler(engine=engine, gate=gate, asr_mt=provider,
                         config=QualityGateConfig(fallback_to_asr_mt=False))
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")
    assert provider.calls == []             # provider present but disabled
    assert out.fallback_path == "exhausted"
    assert out.low_confidence is True
