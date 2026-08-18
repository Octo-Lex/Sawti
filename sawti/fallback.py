"""FallbackHandler: the single authority for escalation (spec §3.6, §5.3).

Locked M1 control flow, executed in exactly this order:

    primary (evaluated by the Pipeline) -> conservative retry (same engine)
      -> rechunk (tighter sub-chunks; each engine+gate evaluated; best
         accepted sub-chunk wins)
        -> ASR+MT provider (on the ORIGINAL chunk — the provider gets the
           fullest audio context)
          -> best-effort flagged output

Contract notes:
- ``recover(chunk, primary, target_lang)`` receives the already-evaluated
  PRIMARY GateDecision (the Pipeline owns the primary attempt) and returns
  the final GateDecision.
- Every stage appends a structured entry to ``decision.log`` with a
  ``stage`` key; ``fallback_path`` names the stage that produced the
  accepted (or best-effort) result; ``format_trace`` renders the
  human-readable traversal.
- Config switches gate each stage: ``retry_once``, ``rechunk_on_failure``
  (also requires a rechunker), ``fallback_to_asr_mt``. An unavailable or
  disabled stage is skipped and noted in the trace, never silently retried.
"""
from __future__ import annotations

from typing import Protocol

from sawti.config import QualityGateConfig
from sawti.rechunk import Rechunker
from sawti.types import AudioChunk, EngineResult, GateDecision


class AsrMtProvider(Protocol):
    def asr_mt(self, chunk: AudioChunk, target_lang: str) -> EngineResult: ...


def _reason(decision: GateDecision) -> str:
    """First failed check name, else 'low_confidence' if flagged, else ''."""
    for k, v in (decision.checks or {}).items():
        if v:
            return k
    if decision.low_confidence:
        return "low_confidence"
    return ""


def _better(a: GateDecision, b: GateDecision) -> GateDecision:
    return a if a.result.confidence >= b.result.confidence else b


class FallbackHandler:
    def __init__(
        self,
        engine,  # S2TTEngine-compatible (has .translate)
        gate=None,  # QualityGate-compatible (has .evaluate)
        asr_mt: AsrMtProvider | None = None,
        rechunker: Rechunker | None = None,
        config: QualityGateConfig | None = None,
    ) -> None:
        self.engine = engine
        self.gate = gate
        self.asr_mt = asr_mt
        self.rechunker = rechunker
        self.config = config or QualityGateConfig()

    # ---- internal helpers -------------------------------------------------

    def _evaluate(self, result: EngineResult, chunk: AudioChunk,
                  target_lang: str) -> GateDecision:
        if self.gate is not None:
            return self.gate.evaluate(result, chunk, target_lang)
        ok = result.confidence >= self.config.confidence_threshold
        return GateDecision(
            chunk_id=chunk.id, accepted=ok, result=result,
            checks={"confidence": not ok},
            start_time=chunk.start_time, end_time=chunk.end_time,
            low_confidence=not ok, needs_retry=not ok,
            log=[{"action": "evaluate"}],
        )

    def _entry(self, stage: str, chunk_id: str, decision: GateDecision) -> dict:
        return {
            "stage": stage,
            "chunk_id": chunk_id,
            "accepted": decision.accepted,
            "checks": decision.checks,
            "low_confidence": decision.low_confidence,
        }

    def _finalize(self, decision: GateDecision, trace: list[dict],
                  chunk_id: str) -> GateDecision:
        decision.log = trace + list(decision.log)
        return decision

    # ---- the state machine ------------------------------------------------

    def recover(self, chunk: AudioChunk, primary: GateDecision,
                target_lang: str) -> GateDecision:
        cfg = self.config
        trace: list[dict] = [self._entry("primary", chunk.id, primary)]
        best = primary

        # Stage 2: conservative retry (same engine, full chunk).
        if cfg.retry_once:
            retried = self.engine.translate(chunk, target_lang)
            d = self._evaluate(retried, chunk, target_lang)
            if not d.needs_retry:
                d.fallback_path = "retry"
                return self._finalize(d, trace + [self._entry("retry", chunk.id, d)], chunk.id)
            trace.append(self._entry("retry", chunk.id, d))
            best = _better(best, d)

        # Stage 3: rechunk into tighter sub-chunks; best accepted wins.
        if self.rechunker is not None and cfg.rechunk_on_failure:
            subs = self.rechunker.rechunk(chunk)
            accepted: list[GateDecision] = []
            for i, sub in enumerate(subs):
                r = self.engine.translate(sub, target_lang)
                ds = self._evaluate(r, sub, target_lang)
                stage = f"rechunk[{i}]"
                trace.append(self._entry(stage, sub.id, ds))
                if not ds.needs_retry:
                    accepted.append(ds)
                else:
                    best = _better(best, ds)
            if accepted:
                win = max(accepted, key=lambda d: d.result.confidence)
                win.fallback_path = "rechunk"
                return self._finalize(win, trace, chunk.id)

        # Stage 4: ASR+MT on the ORIGINAL chunk.
        if self.asr_mt is not None and cfg.fallback_to_asr_mt:
            mt = self.asr_mt.asr_mt(chunk, target_lang)
            dm = self._evaluate(mt, chunk, target_lang)
            if not dm.needs_retry:
                dm.fallback_path = "asr_mt"
                return self._finalize(dm, trace + [self._entry("asr_mt", chunk.id, dm)], chunk.id)
            trace.append(self._entry("asr_mt", chunk.id, dm))
            best = _better(best, dm)
        elif self.asr_mt is None:
            trace.append({"stage": "asr_mt", "chunk_id": chunk.id,
                          "accepted": False, "checks": {},
                          "note": "provider not configured"})

        # Stage 5: exhausted -> best-effort flagged output.
        best.accepted = False
        best.low_confidence = True
        best.fallback_path = "exhausted"
        trace.append({"stage": "exhausted", "chunk_id": chunk.id,
                      "accepted": False, "note": "best-effort flagged output"})
        return self._finalize(best, trace, chunk.id)


def format_trace(decision: GateDecision) -> str:
    """Render the escalation traversal, e.g.:

    chunk c0
      primary       -> rejected: repetition_loop
      retry         -> rejected: low_confidence
      rechunk[0]    -> rejected
      asr_mt        -> accepted
    """
    lines = [f"chunk {decision.chunk_id.rsplit('.', 1)[0]}"]
    for e in decision.log:
        if not isinstance(e, dict) or "stage" not in e:
            continue
        stage = e["stage"]
        if e.get("accepted"):
            verdict = "accepted"
        elif e.get("note") in ("provider not configured",
                               "best-effort flagged output"):
            verdict = e["note"]
        else:
            reason = next((k for k, v in (e.get("checks") or {}).items() if v), "")
            if not reason and e.get("low_confidence"):
                reason = "low_confidence"
            verdict = f"rejected: {reason}" if reason else "rejected"
        lines.append(f"  {stage:<13} -> {verdict}")
    return "\n".join(lines)
