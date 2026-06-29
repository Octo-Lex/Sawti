"""Quality gate protocol + stub (spec §3.4–§3.8).

M1 replaces StubQualityGate with the Balanced policy + ASR+MT fallback.
The protocol and orchestrator are unchanged.
"""
from __future__ import annotations

from typing import Protocol

from sawti.config import QualityGateConfig
from sawti.types import AudioChunk, EngineResult, GateDecision


class QualityGate(Protocol):
    def evaluate(
        self, result: EngineResult, chunk: AudioChunk, target_lang: str
    ) -> GateDecision: ...


class StubQualityGate:
    """Cheap confidence-only gate (M0 only).

    Real checks (empty/garbage/script-mismatch/length-ratio/repetition)
    are deferred to M1. This gate flags results below the configured
    confidence_threshold so the orchestrator's retry path is exercisable.
    """

    def __init__(self, config: QualityGateConfig | None = None) -> None:
        self.config = config or QualityGateConfig()

    def evaluate(
        self, result: EngineResult, chunk: AudioChunk, target_lang: str
    ) -> GateDecision:
        low = result.confidence < self.config.confidence_threshold
        return GateDecision(
            chunk_id=chunk.id,
            accepted=not low,
            result=result,
            checks={"confidence": low},
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            fallback_path="retry" if low else None,
            low_confidence=low,
            needs_retry=low,
            log=[{"action": "evaluate", "confidence": result.confidence, "low": low}],
        )
