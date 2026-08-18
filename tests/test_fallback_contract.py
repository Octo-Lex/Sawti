"""Acceptance tests: the fallback escalation contract (spec §3.6, §5.3).

Proves the full traversal with fakes:

    primary -> conservative retry (explicit seam) -> rechunk (content-
    preserving parent composition) -> ASR+MT -> best-effort

every stage traversed in exactly that order, the final GateDecision's
fallback_path naming the stage that won, decision.log carrying a complete
trace, and config switches demonstrably changing that trace.
"""
from __future__ import annotations

import numpy as np

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
    """Ordinary primary-style translation (used for rechunk sub-chunks)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def translate(self, chunk, target_lang):
        self.calls.append(chunk.id)
        return _result(chunk.id, f"eng:{chunk.id}", 0.5)


class ConservativeFake:
    """The conservative-retry seam — must be invoked for the retry stage,
    distinct from ordinary engine.translate."""

    def __init__(self, conf: float = 0.4) -> None:
        self.conf = conf
        self.calls: list[str] = []

    def __call__(self, chunk, target_lang):
        self.calls.append(chunk.id)
        return _result(chunk.id, f"conservative:{chunk.id}", self.conf)


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
    return GateDecision(
        chunk_id=chunk_id, accepted=False,
        result=_result(chunk_id, "eng:primary-loop", 0.3),
        checks=checks, start_time=0.0, end_time=4.0,
        low_confidence=False, needs_retry=True,
        log=[{"action": "evaluate"}],
    )


FULL_SCRIPT = [
    {"accepted": False, "checks": {"low_confidence": True}, "low_conf": True},
    {"accepted": False, "checks": {"empty_output": True}},
    {"accepted": False, "checks": {"low_confidence": True}, "low_conf": True},
    {"accepted": True, "checks": {}},
]


def _handler(gate, *, engine=None, provider=None, rechunker=None,
             conservative=None, config=None) -> FallbackHandler:
    return FallbackHandler(engine=engine or RecordingEngine(), gate=gate,
                           asr_mt=provider, rechunker=rechunker,
                           conservative=conservative, config=config)


def test_full_escalation_path_order_and_trace():
    chunk = _chunk()
    engine = RecordingEngine()
    conservative = ConservativeFake()
    gate = ScriptedGate(FULL_SCRIPT)
    rechunker, provider = FakeRechunker(), FakeAsrMt()
    fb = _handler(gate, engine=engine, provider=provider, rechunker=rechunker,
                  conservative=conservative)
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")

    assert out.accepted is True
    assert out.fallback_path == "asr_mt"
    assert out.result.raw_text == "asr_mt:recovered"
    assert out.low_confidence is False

    # Retry went through the CONSERVATIVE seam (not ordinary translation);
    # ordinary translation served only the rechunk sub-chunks, in order.
    assert conservative.calls == ["c0"]
    assert engine.calls == ["c0.r0", "c0.r1"]
    assert gate.calls == ["c0", "c0.r0", "c0.r1", "c0"]
    assert provider.calls == ["c0"]           # original chunk
    assert rechunker.calls == ["c0"]

    trace = format_trace(out)
    assert trace.splitlines() == [
        "chunk c0",
        "  primary       -> rejected: repetition_loop",
        "  retry         -> rejected: low_confidence",
        "  rechunk[0]    -> rejected: empty_output",
        "  rechunk[1]    -> rejected: low_confidence",
        "  asr_mt        -> accepted",
    ]
    stages = [e["stage"] for e in out.log if isinstance(e, dict) and "stage" in e]
    assert stages == ["primary", "retry", "rechunk[0]", "rechunk[1]", "asr_mt"]


def test_retry_recovery_short_circuits_before_rechunk():
    chunk = _chunk()
    engine = RecordingEngine()
    conservative = ConservativeFake(conf=0.9)
    gate = ScriptedGate([{"accepted": True, "checks": {}}])
    rechunker, provider = FakeRechunker(), FakeAsrMt()
    fb = _handler(gate, engine=engine, provider=provider, rechunker=rechunker,
                  conservative=conservative)
    out = fb.recover(chunk, _primary("c0", {"low_confidence": True}), "eng")
    assert out.accepted is True
    assert out.fallback_path == "retry"
    assert out.result.raw_text == "conservative:c0"  # the seam produced it
    assert conservative.calls == ["c0"]
    assert engine.calls == []
    assert rechunker.calls == []
    assert provider.calls == []


class ScriptedSubtextEngine:
    """Returns scripted text/conf per sub-chunk id suffix."""

    def __init__(self, script: dict[str, tuple[str, float]]) -> None:
        self.script = script
        self.calls: list[str] = []

    def translate(self, chunk, target_lang):
        self.calls.append(chunk.id)
        text, conf = self.script[chunk.id.rsplit(".", 1)[-1]]
        return _result(chunk.id, text, conf)


def test_successful_rechunk_composes_parent_result():
    """The required regression:

    parent: "A B C D"; r0 -> "A B" accepted; r1 -> "C D" accepted;
    final: chunk_id == parent, text == "A B C D", path == "rechunk"."""
    chunk = _chunk()
    engine = ScriptedSubtextEngine({"r0": ("A B", 0.8), "r1": ("C D", 0.7)})
    gate = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}},  # retry
        {"accepted": True, "checks": {}},                         # r0
        {"accepted": True, "checks": {}},                         # r1
        {"accepted": True, "checks": {}},                         # composed
    ])
    conservative, provider = ConservativeFake(), FakeAsrMt()
    fb = _handler(gate, engine=engine, provider=provider,
                  rechunker=FakeRechunker(), conservative=conservative)
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")

    assert out.accepted is True
    assert out.fallback_path == "rechunk"
    assert out.chunk_id == "c0"                    # parent-level result
    assert out.result.raw_text == "A B C D"        # temporal composition
    assert out.result.confidence == 0.7            # conservative min()
    assert provider.calls == []                    # escalation never reached
    trace = format_trace(out)
    assert "  rechunk[0]    -> accepted" in trace
    assert "  rechunk[1]    -> accepted" in trace
    assert "  rechunk       -> accepted" in trace


def test_partial_subchunk_never_terminal_and_composition_blocked():
    """r0 accepted, r1 rejected: no composition, escalation continues, and
    exhaustion can never emit the half-utterance r0 result."""
    chunk = _chunk()
    engine = ScriptedSubtextEngine({"r0": ("A B", 0.8), "r1": ("C D", 0.2)})
    gate = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}},  # retry
        {"accepted": True, "checks": {}},                         # r0
        {"accepted": False, "checks": {"empty_output": True}},    # r1
        {"accepted": True, "checks": {}},                         # asr_mt
    ])
    fb = _handler(gate, engine=engine, provider=FakeAsrMt(),
                  rechunker=FakeRechunker(), conservative=ConservativeFake())
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")
    assert out.fallback_path == "asr_mt"
    assert out.chunk_id == "c0"

    # And with no provider, exhaustion emits a parent-level candidate —
    # never the accepted-but-partial r0.
    gate2 = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}},
        {"accepted": True, "checks": {}},
        {"accepted": False, "checks": {"empty_output": True}},
    ])
    fb2 = _handler(gate2, engine=ScriptedSubtextEngine(
        {"r0": ("A B", 0.8), "r1": ("C D", 0.2)}),
        rechunker=FakeRechunker(), conservative=ConservativeFake())
    out2 = fb2.recover(_chunk(), _primary("c0", {"repetition_loop": True}), "eng")
    assert out2.fallback_path == "exhausted"
    assert out2.chunk_id == "c0"
    assert out2.result.raw_text != "A B"   # no half-utterance terminal output


def test_composed_rejected_escalates():
    """All subs valid but the composed parent fails the gate -> asr_mt."""
    chunk = _chunk()
    engine = ScriptedSubtextEngine({"r0": ("A B", 0.8), "r1": ("C D", 0.7)})
    gate = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}},   # retry
        {"accepted": True, "checks": {}},                          # r0
        {"accepted": True, "checks": {}},                          # r1
        {"accepted": False, "checks": {"length_ratio_anomaly": True}},  # composed
        {"accepted": True, "checks": {}},                          # asr_mt
    ])
    fb = _handler(gate, engine=engine, provider=FakeAsrMt(),
                  rechunker=FakeRechunker(), conservative=ConservativeFake())
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")
    assert out.fallback_path == "asr_mt"
    trace = format_trace(out)
    assert "  rechunk       -> rejected: length_ratio_anomaly" in trace


def test_exhausted_path_returns_flagged_best_effort():
    chunk = _chunk()
    gate = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}, "low_conf": True},
        {"accepted": False, "checks": {"empty_output": True}},
        {"accepted": False, "checks": {"empty_output": True}},
        {"accepted": False, "checks": {"repetition_loop": True}},
    ])
    fb = _handler(gate, provider=FakeAsrMt(), rechunker=FakeRechunker(),
                  conservative=ConservativeFake())
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
    engine = ScriptedSubtextEngine({"r0": ("A B", 0.8), "r1": ("C D", 0.7)})
    gate = ScriptedGate([
        {"accepted": True, "checks": {}},    # r0
        {"accepted": True, "checks": {}},    # r1
        {"accepted": True, "checks": {}},    # composed
    ])
    conservative = ConservativeFake()
    fb = _handler(gate, engine=engine, provider=FakeAsrMt(),
                  rechunker=FakeRechunker(), conservative=conservative,
                  config=QualityGateConfig(retry_once=False))
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")
    assert conservative.calls == []           # retry stage fully removed
    assert out.fallback_path == "rechunk"
    assert out.chunk_id == "c0"
    stages = [e["stage"] for e in out.log if isinstance(e, dict) and "stage" in e]
    assert stages == ["primary", "rechunk[0]", "rechunk[1]", "rechunk"]


def test_rechunk_on_failure_false_removes_rechunk_stage():
    chunk = _chunk()
    gate = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}},  # retry
        {"accepted": True, "checks": {}},                         # asr_mt
    ])
    rechunker, conservative = FakeRechunker(), ConservativeFake()
    fb = _handler(gate, provider=FakeAsrMt(), rechunker=rechunker,
                  conservative=conservative,
                  config=QualityGateConfig(rechunk_on_failure=False))
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")
    assert rechunker.calls == []
    assert out.fallback_path == "asr_mt"
    stages = [e["stage"] for e in out.log if isinstance(e, dict) and "stage" in e]
    assert stages == ["primary", "retry", "asr_mt"]


def test_no_rechunker_configured_skips_rechunk_stage():
    chunk = _chunk()
    gate = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}},
        {"accepted": True, "checks": {}},
    ])
    fb = _handler(gate, provider=FakeAsrMt(), conservative=ConservativeFake())
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")
    assert out.fallback_path == "asr_mt"
    stages = [e["stage"] for e in out.log if isinstance(e, dict) and "stage" in e]
    assert stages == ["primary", "retry", "asr_mt"]


def test_fallback_to_asr_mt_false_exhausts_instead():
    chunk = _chunk()
    gate = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}},
    ])
    provider = FakeAsrMt()
    fb = _handler(gate, provider=provider, conservative=ConservativeFake(),
                  config=QualityGateConfig(fallback_to_asr_mt=False))
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")
    assert provider.calls == []             # provider present but disabled
    assert out.fallback_path == "exhausted"
    assert out.low_confidence is True


# --- Commit 3: retry/rechunk limits are load-bearing execution policy ---

from sawti.config import RetriesConfig, QualityGateConfig as _QGC


def test_max_s2tt_retries_controls_retry_count():
    chunk = _chunk()
    engine = RecordingEngine()
    conservative = ConservativeFake(conf=0.2)          # retries always fail
    gate = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}},  # retry 1
        {"accepted": False, "checks": {"low_confidence": True}},  # retry 2
        {"accepted": False, "checks": {"empty_output": True}},    # r0
        {"accepted": False, "checks": {"empty_output": True}},    # r1
        {"accepted": True, "checks": {}},                         # asr_mt
    ])
    fb = _handler(gate, engine=engine, provider=FakeAsrMt(),
                  rechunker=FakeRechunker(), conservative=conservative,
                  config=_QGC(retries=RetriesConfig(max_s2tt_retries=2,
                                                    max_rechunk_attempts=1)))
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")
    assert conservative.calls == ["c0", "c0"]          # exactly two retries
    assert out.fallback_path == "asr_mt"
    retry_lines = [l for l in format_trace(out).splitlines()
                   if l.strip().startswith("retry")]
    assert len(retry_lines) == 2
    entries = [e for e in out.log if isinstance(e, dict) and e.get("stage") == "retry"]
    assert [e["attempt"] for e in entries] == [1, 2]


def test_retry_once_false_beats_nonzero_max_count():
    chunk = _chunk()
    conservative = ConservativeFake()
    gate = ScriptedGate([{"accepted": True, "checks": {}}])   # asr_mt
    fb = _handler(gate, provider=FakeAsrMt(), conservative=conservative,
                  config=_QGC(
                      retry_once=False,
                      retries=RetriesConfig(max_s2tt_retries=2,
                                            max_rechunk_attempts=0)))
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")
    assert conservative.calls == []                    # count overridden off
    assert out.fallback_path == "asr_mt"


class TightenableFake:
    """2 sub-chunks in round 1, 4 after one with_tighter() call."""

    def __init__(self) -> None:
        self.rnd = 0
        self.rechunk_calls = 0
        self.tighten_calls = 0

    def rechunk(self, chunk):
        self.rechunk_calls += 1
        k = 2 if self.rnd == 0 else 4
        n = len(chunk.audio)
        span = chunk.end_time - chunk.start_time
        out = []
        for i in range(k):
            a, b = i * n // k, (i + 1) * n // k
            out.append(AudioChunk(
                id=f"{chunk.id}.r{i}", audio=chunk.audio[a:b].copy(),
                sample_rate=16000,
                start_time=chunk.start_time + (a / n) * span,
                end_time=chunk.start_time + (b / n) * span))
        return out

    def with_tighter(self, factor):
        self.tighten_calls += 1
        self.rnd += 1
        return self


def test_max_rechunk_attempts_controls_rounds_and_tightens():
    chunk = _chunk()
    engine = RecordingEngine()
    rechunker = TightenableFake()
    gate = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}},  # retry
    ] + [{"accepted": False, "checks": {"empty_output": True}}] * 2   # round 1
    + [{"accepted": False, "checks": {"empty_output": True}}] * 4     # round 2
    + [{"accepted": True, "checks": {}}])                             # asr_mt
    fb = _handler(gate, engine=engine, provider=FakeAsrMt(),
                  rechunker=rechunker, conservative=ConservativeFake(),
                  config=_QGC(retries=RetriesConfig(max_s2tt_retries=1,
                                                    max_rechunk_attempts=2)))
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")

    assert rechunker.rechunk_calls == 2
    assert rechunker.tighten_calls == 1               # tightened between rounds
    assert len(engine.calls) == 6                     # 2 + 4 sub translations
    assert out.fallback_path == "asr_mt"
    rounds = [e.get("round") for e in out.log
              if isinstance(e, dict) and str(e.get("stage", "")).startswith("rechunk")]
    assert rounds == [1, 1, 2, 2, 2, 2]               # structured round fields


def test_max_rechunk_attempts_zero_skips_rechunk_stage():
    chunk = _chunk()
    rechunker = FakeRechunker()
    gate = ScriptedGate([
        {"accepted": False, "checks": {"low_confidence": True}},  # retry
        {"accepted": True, "checks": {}},                         # asr_mt
    ])
    fb = _handler(gate, provider=FakeAsrMt(), rechunker=rechunker,
                  conservative=ConservativeFake(),
                  config=_QGC(retries=RetriesConfig(max_s2tt_retries=1,
                                                    max_rechunk_attempts=0)))
    out = fb.recover(chunk, _primary("c0", {"repetition_loop": True}), "eng")
    assert rechunker.calls == []                      # stage fully skipped
    assert out.fallback_path == "asr_mt"
