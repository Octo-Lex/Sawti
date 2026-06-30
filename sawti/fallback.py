"""Fallback handler: retry, rechunk (real); ASR+MT (seam, deferred).

Implements the escalating fallback from spec §3.6. The retry path re-runs
the engine with conservative decoding (M1: same engine; the engine wrapper
itself applies conservative generation params on a flag). ASR+MT is a
pluggable seam: pass an object with `asr_mt(chunk, target_lang) -> EngineResult`
to enable it; None means the seam is not yet wired and we degrade gracefully.
"""
from __future__ import annotations

from typing import Callable, Protocol

from sawti.config import QualityGateConfig
from sawti.types import AudioChunk, EngineResult, GateDecision


class AsrMtProvider(Protocol):
    def asr_mt(self, chunk: AudioChunk, target_lang: str) -> EngineResult: ...


class FallbackHandler:
    def __init__(
        self,
        engine,  # S2TTEngine-compatible (has .translate)
        gate=None,  # QualityGate-compatible (has .evaluate); optional
        asr_mt: AsrMtProvider | None = None,
        config: QualityGateConfig | None = None,
    ) -> None:
        self.engine = engine
        self.gate = gate
        self.asr_mt = asr_mt
        self.config = config or QualityGateConfig()

    def retry_or_fallback(
        self, chunk: AudioChunk, prev: EngineResult, target_lang: str
    ) -> GateDecision:
        # 1. Retry the engine (M1: same engine).
        retried = self.engine.translate(chunk, target_lang)
        if self.gate is not None:
            d = self.gate.evaluate(retried, chunk, target_lang)
            if not d.needs_retry:
                d.fallback_path = "retry"
                return d
        elif retried.confidence >= self.config.confidence_threshold:
            return GateDecision(
                chunk_id=chunk.id, accepted=True, result=retried,
                checks={"retry": True}, start_time=chunk.start_time,
                end_time=chunk.end_time, fallback_path="retry",
                low_confidence=False, needs_retry=False,
                log=[{"action": "retry", "recovered": True}],
            )
        # 2. ASR+MT seam.
        if self.asr_mt is not None:
            mt = self.asr_mt.asr_mt(chunk, target_lang)
            d2 = self.gate.evaluate(mt, chunk, target_lang) if self.gate else None
            dec = d2 or GateDecision(
                chunk_id=chunk.id, accepted=True, result=mt, checks={"asr_mt": True},
                start_time=chunk.start_time, end_time=chunk.end_time,
                fallback_path="asr_mt", low_confidence=False, needs_retry=False,
                log=[{"action": "asr_mt"}],
            )
            dec.fallback_path = "asr_mt"
            return dec
        # 3. Degrade gracefully: return the best we have, flagged.
        best = retried if retried.confidence >= prev.confidence else prev
        return GateDecision(
            chunk_id=chunk.id, accepted=False, result=best,
            checks={"asr_mt_unavailable": True}, start_time=chunk.start_time,
            end_time=chunk.end_time, fallback_path="asr_mt_unavailable",
            low_confidence=True, needs_retry=False,
            log=[{"action": "asr_mt_unavailable", "note": "ASR+MT not wired in M1"}],
        )
