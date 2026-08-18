"""FallbackHandler: the single authority for escalation (spec §3.6, §5.3).

Locked M1 control flow, executed in exactly this order:

    primary (evaluated by the Pipeline) -> conservative retry -> rechunk
      -> ASR+MT provider (on the ORIGINAL chunk) -> best-effort flagged output

Contract notes:
- ``recover(chunk, primary, target_lang)`` receives the already-evaluated
  PRIMARY GateDecision (the Pipeline owns the primary attempt) and returns
  the final GateDecision.
- **Conservative retry is an explicit seam** (``ConservativeRetry``). When
  none is provided the retry stage falls back to ordinary
  ``engine.translate`` (identity "conservative" mode); binding the seam to
  real conservative generation parameters is a later implementation step.
- **Rechunk is content-preserving.** All sub-chunks are translated in
  temporal order; their texts are concatenated into an EngineResult for
  the PARENT chunk id with conservative aggregate confidence
  ``min(sub.confidence)``; that composed result is gated against the
  original parent chunk. ``fallback_path="rechunk"`` requires EVERY
  sub-result to be valid AND the composed parent result to pass. Partial
  sub-chunk results never compete for the terminal best-effort candidate —
  exhaustion must not emit half an utterance.
- Every stage appends a structured entry to ``decision.log`` with a
  ``stage`` key; ``fallback_path`` names the stage that produced the
  accepted (or best-effort) result; ``format_trace`` renders the
  human-readable traversal.
- Config switches gate each stage: ``retry_once``, ``rechunk_on_failure``
  (also requires a rechunker), ``fallback_to_asr_mt``. An unavailable or
  disabled stage is skipped, never silently retried.
"""
from __future__ import annotations

from typing import Protocol

from sawti.config import QualityGateConfig
from sawti.rechunk import Rechunker
from sawti.types import AudioChunk, EngineResult, GateDecision


class AsrMtProvider(Protocol):
    def asr_mt(self, chunk: AudioChunk, target_lang: str) -> EngineResult: ...


class ConservativeRetry(Protocol):
    """Explicit seam for conservative decoding on the retry stage.

    A later step binds this to the M4T engine with conservative generation
    parameters (beam/temperature/length-penalty); the handler only knows
    the callable.
    """

    def __call__(self, chunk: AudioChunk, target_lang: str) -> EngineResult: ...


def _better(a: GateDecision, b: GateDecision) -> GateDecision:
    return a if a.result.confidence >= b.result.confidence else b


class FallbackHandler:
    def __init__(
        self,
        engine,  # S2TTEngine-compatible (has .translate)
        gate=None,  # QualityGate-compatible (has .evaluate)
        asr_mt: AsrMtProvider | None = None,
        rechunker: Rechunker | None = None,
        conservative: ConservativeRetry | None = None,
        config: QualityGateConfig | None = None,
    ) -> None:
        self.engine = engine
        self.gate = gate
        self.asr_mt = asr_mt
        self.rechunker = rechunker
        self.conservative = conservative
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

        # Stage 2: conservative retries (explicit seam; identity mode when
        # no seam is bound — ordinary engine.translate). Count is
        # config-owned execution policy: max_s2tt_retries when retry_once
        # is set, else zero. Default (retry_once + max=1) = one retry.
        retry_attempts = cfg.retries.max_s2tt_retries if cfg.retry_once else 0
        if retry_attempts > 0:
            retry_translate = (
                self.conservative if self.conservative is not None
                else self.engine.translate
            )
            for attempt in range(retry_attempts):
                retried = retry_translate(chunk, target_lang)
                d = self._evaluate(retried, chunk, target_lang)
                entry = self._entry("retry", chunk.id, d)
                entry["attempt"] = attempt + 1
                if not d.needs_retry:
                    d.fallback_path = "retry"
                    return self._finalize(d, trace + [entry], chunk.id)
                trace.append(entry)
                best = _better(best, d)

        # Stage 3: rechunk -> COMPOSE a parent-level candidate. Rounds are
        # config-owned execution policy (retries.max_rechunk_attempts):
        # each round re-splits (tighter via the optional with_tighter seam
        # between rounds) and must compose; round 0's stage labels match
        # the single-round contract exactly.
        if self.rechunker is not None and cfg.rechunk_on_failure:
            rounds = max(0, cfg.retries.max_rechunk_attempts)
            rc = self.rechunker
            for rnd in range(rounds):
                subs = rc.rechunk(chunk)
                sub_decisions: list[GateDecision] = []
                all_valid = True
                for i, sub in enumerate(subs):
                    r = self.engine.translate(sub, target_lang)
                    ds = self._evaluate(r, sub, target_lang)
                    entry = self._entry(f"rechunk[{i}]", sub.id, ds)
                    entry["round"] = rnd + 1
                    trace.append(entry)
                    if ds.needs_retry:
                        all_valid = False
                    else:
                        sub_decisions.append(ds)
                    # Partial sub-chunks never compete for `best`.
                if all_valid and sub_decisions:
                    texts = [d.result.raw_text.strip() for d in sub_decisions]
                    composed = EngineResult(
                        chunk_id=chunk.id,
                        raw_text=" ".join(t for t in texts if t),
                        confidence=min(d.result.confidence for d in sub_decisions),
                        source_lang_guess=sub_decisions[0].result.source_lang_guess,
                        timing_ms={"rechunk_subs": [d.chunk_id for d in sub_decisions]},
                        target_lang=target_lang,
                    )
                    dc = self._evaluate(composed, chunk, target_lang)
                    entry = self._entry("rechunk", chunk.id, dc)
                    entry["round"] = rnd + 1
                    if not dc.needs_retry:
                        dc.fallback_path = "rechunk"
                        return self._finalize(dc, trace + [entry], chunk.id)
                    trace.append(entry)
                    best = _better(best, dc)  # parent-level candidate may compete
                # Tighten for the next round when the rechunker supports it.
                if rnd + 1 < rounds and hasattr(rc, "with_tighter"):
                    rc = rc.with_tighter(2)

        # Stage 4: ASR+MT on the ORIGINAL chunk.
        if self.asr_mt is not None and cfg.fallback_to_asr_mt:
            mt = self.asr_mt.asr_mt(chunk, target_lang)
            dm = self._evaluate(mt, chunk, target_lang)
            if not dm.needs_retry:
                dm.fallback_path = "asr_mt"
                return self._finalize(dm, trace + [self._entry("asr_mt", chunk.id, dm)], chunk.id)
            trace.append(self._entry("asr_mt", chunk.id, dm))
            best = _better(best, dm)

        # Stage 5: exhausted -> best-effort flagged output (parent-level
        # candidates only, by construction).
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
        elif e.get("note"):
            verdict = e["note"]
        else:
            reason = next((k for k, v in (e.get("checks") or {}).items() if v), "")
            if not reason and e.get("low_confidence"):
                reason = "low_confidence"
            verdict = f"rejected: {reason}" if reason else "rejected"
        lines.append(f"  {stage:<13} -> {verdict}")
    return "\n".join(lines)
