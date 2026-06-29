"""S2TT engine protocol, manager, and stub (spec §3, §5.2).

M1 replaces StubEngine with a SeamlessM4T-v2-large-backed engine resident
in GPU memory. EngineManager stays the same; load_policy is honored there.
"""
from __future__ import annotations

from typing import Protocol

from sawti.config import S2TTConfig
from sawti.types import AudioChunk, EngineResult


class S2TTEngine(Protocol):
    """Translates one AudioChunk into target-language text."""

    def translate(self, chunk: AudioChunk, target_lang: str) -> EngineResult: ...


class StubEngine:
    """Returns canned text regardless of input (M0 only)."""

    def __init__(self, canned_text: str = "[stub]", confidence: float = 0.5) -> None:
        self.canned_text = canned_text
        self.confidence = confidence

    def translate(self, chunk: AudioChunk, target_lang: str) -> EngineResult:
        return EngineResult(
            chunk_id=chunk.id,
            raw_text=self.canned_text,
            confidence=self.confidence,
            source_lang_guess="und",  # undetermined
            timing_ms={"engine": 0, "path": "stub"},
            target_lang=target_lang,
        )


class EngineManager:
    """Owns the engine lifecycle (spec §3.3).

    M0: holds any S2TTEngine and delegates. M1 adds load_policy handling
    (resident/lazy/idle_unload) and real model loading.
    """

    def __init__(self, engine: S2TTEngine, config: S2TTConfig | None = None) -> None:
        self.engine = engine
        self.config = config or S2TTConfig()

    def translate(self, chunk: AudioChunk, target_lang: str) -> EngineResult:
        return self.engine.translate(chunk, target_lang)
