"""M1 EXIT ACCEPTANCE: the locked architecture is executable,
configurable, testable, and measurable — proven through actual YAML
variation.

One deterministic bad-chunk scenario runs through the REAL
Pipeline + FallbackHandler machinery (observed via the evaluator's own
transcriber seam) under two different YAML files; the structured trace
must match each configuration's policy exactly, demonstrating that
changing YAML settings demonstrably changes behavior.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from sawti.config import load_config
from sawti.engine import EngineManager
from sawti.fallback import FallbackHandler
from sawti.pipeline import Pipeline
from sawti.postprocess import StubPostProcessor
from sawti.rechunk import FixedSplitRechunker
from sawti.segmenter import StubSegmenter
from sawti.types import EngineResult, GateDecision

DEFAULTS_YAML = """
target_lang: eng
quality_gate:
  policy: balanced
  confidence_threshold: 0.40
  retry_once: true
  rechunk_on_failure: true
  fallback_to_asr_mt: true
  retries:
    max_s2tt_retries: 1
    max_rechunk_attempts: 1
"""

TUNED_YAML = """
target_lang: eng
quality_gate:
  policy: balanced
  confidence_threshold: 0.40
  retry_once: false
  rechunk_on_failure: true
  fallback_to_asr_mt: true
  retries:
    max_s2tt_retries: 2
    max_rechunk_attempts: 2
"""


class ScriptedGate:
    def __init__(self, verdicts):
        self._v = list(verdicts)

    def evaluate(self, result, chunk, target):
        ok, checks = self._v.pop(0)
        return GateDecision(chunk_id=chunk.id, accepted=ok, result=result,
                            checks=checks, start_time=chunk.start_time,
                            end_time=chunk.end_time, needs_retry=not ok,
                            log=[{"action": "evaluate"}])


class WeakEngine:
    def translate(self, chunk, target, conservative=False):
        return EngineResult(chunk.id, "weak", 0.1, "ara", {}, target)


class RescueProvider:
    def __init__(self):
        self.calls = 0

    def asr_mt(self, chunk, target):
        self.calls += 1
        return EngineResult(chunk.id, "RESCUED", 0.95, "ara", {}, target)


def _render(trace: list[dict]) -> list[str]:
    lines = []
    for e in trace:
        if not isinstance(e, dict) or "stage" not in e:
            continue
        if e.get("accepted"):
            v = "accepted"
        elif e.get("note"):
            v = e["note"]
        else:
            reason = next((k for k, x in (e.get("checks") or {}).items() if x), "")
            if not reason and e.get("low_confidence"):
                reason = "low_confidence"
            v = f"rejected: {reason}" if reason else "rejected"
        lines.append(f"{e['stage']} -> {v}")
    return lines


def _run(dir_path: Path, yaml_text: str) -> tuple[list[str], dict]:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "cfg.yaml").write_text(yaml_text, encoding="utf-8")
    cfg = load_config(dir_path / "cfg.yaml")
    wav = dir_path / "clip.wav"
    sf.write(wav, np.zeros(16000, np.float32), 16000)

    # Verdict script is built FOR THIS CONFIG (evaluation order: primary,
    # retries, rechunk round 1 subs, round 2 subs, asr_mt accept).
    n_retries = cfg.quality_gate.retries.max_s2tt_retries         if cfg.quality_gate.retry_once else 0
    rounds = cfg.quality_gate.retries.max_rechunk_attempts
    verdicts = [(False, {"low_confidence": True})]          # primary
    verdicts += [(False, {"low_confidence": True})] * n_retries
    verdicts += [(False, {"empty_output": True})] * 2        # round 1: 2 subs
    if rounds >= 2:
        verdicts += [(False, {"empty_output": True})] * 4    # round 2: 4 subs
    verdicts += [(True, {})]                                 # asr_mt

    gate = ScriptedGate(verdicts)
    engine = WeakEngine()
    provider = RescueProvider()
    fallback = FallbackHandler(
        engine=EngineManager(engine=engine), gate=gate, asr_mt=provider,
        rechunker=FixedSplitRechunker(max_sub_duration_s=0.5),
        conservative=lambda c, t: engine.translate(c, t, conservative=True),
        config=cfg.quality_gate)

    from eval.transcribers import make_pipeline_transcriber
    tr = make_pipeline_transcriber(
        lambda on_decision=None: Pipeline(
            segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
            engine=EngineManager(engine=WeakEngine()),
            gate=gate, postprocessor=StubPostProcessor(),
            fallback=fallback, on_decision=on_decision),
        "eng", frame_samples=16000)
    t = tr(str(wav))
    return _render(t.trace), {"rescued": provider.calls, "text": t.hypothesis}


def test_milestone_exit_yaml_variation_changes_the_trace(tmp_path: Path):
    trace_d, r1 = _run(tmp_path / "defaults", DEFAULTS_YAML)
    trace_t, r2 = _run(tmp_path / "tuned", TUNED_YAML)

    # DEFAULTS YAML: primary -> ONE conservative retry -> one rechunk
    # round (2 subs at 0.3s over a 1s chunk) -> asr_mt rescue.
    assert trace_d == [
        "primary -> rejected: low_confidence",
        "retry -> rejected: low_confidence",
        "rechunk[0] -> rejected: empty_output",
        "rechunk[1] -> rejected: empty_output",
        "asr_mt -> accepted",
    ]
    # TUNED YAML: retry_once=false removes the retry stage entirely;
    # max_rechunk_attempts=2 runs a second, TIGHTER round (4 subs) -> asr_mt.
    assert trace_t == [
        "primary -> rejected: low_confidence",
        "rechunk[0] -> rejected: empty_output",
        "rechunk[1] -> rejected: empty_output",
        "rechunk[0] -> rejected: empty_output",
        "rechunk[1] -> rejected: empty_output",
        "rechunk[2] -> rejected: empty_output",
        "rechunk[3] -> rejected: empty_output",
        "asr_mt -> accepted",
    ]
    # Both configurations rescue through the real provider seam and emit
    # the accepted result.
    assert r1["rescued"] == 1 and r2["rescued"] == 1
    assert r1["text"] == "RESCUED" and r2["text"] == "RESCUED"
    # The traces are demonstrably different under different YAML.
    assert trace_d != trace_t
