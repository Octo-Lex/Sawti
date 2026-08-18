"""S2TT engine protocol, manager, and stub (spec §3, §5.2).

M1 replaces StubEngine with a SeamlessM4T-v2-large-backed engine resident
in GPU memory. EngineManager stays the same; load_policy is honored there.
"""
from __future__ import annotations

import time
from typing import Callable, Protocol

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
    """Owns the engine lifecycle (spec §3.3) — load_policy is REAL:

    - ``resident``: the engine is built eagerly at construction.
    - ``lazy``: the engine factory runs on first translate.
    - ``idle_unload``: like lazy, plus the engine is released after
      ``idle_unload_seconds`` of inactivity and rebuilt on the next
      translate (wall clock via ``time.monotonic``).

    Construction: pass an ``engine`` instance (already-built; lifecycle
    then only matters for idle_unload, which rebuilds via ``engine_factory``
    when provided, else never unloads an explicitly-given instance), or an
    ``engine_factory`` callable producing engines (required for
    lazy/idle_unload semantics). Exactly one of the two.
    """

    def __init__(
        self,
        engine: S2TTEngine | None = None,
        config: S2TTConfig | None = None,
        engine_factory=None,
        clock=None,
    ) -> None:
        if engine is None and engine_factory is None:
            raise ValueError("EngineManager requires an engine or an engine_factory")
        if engine is not None and engine_factory is not None:
            raise ValueError("pass either engine or engine_factory, not both")
        self.config = config or S2TTConfig()
        self._factory = engine_factory if engine_factory is not None else (lambda: engine)
        self._owns_instance = engine_factory is not None
        self._engine: S2TTEngine | None = None
        self._last_used: float | None = None
        self._clock = clock if clock is not None else time.monotonic
        self._builds = 0
        if self.config.load_policy == "resident":
            self._ensure_loaded()

    @property
    def engine(self) -> S2TTEngine:
        """Current engine instance (loads it if a policy defers loading)."""
        self._ensure_loaded()
        return self._engine

    @property
    def builds(self) -> int:
        """How many times the factory produced an engine (observable
        lifecycle behavior for tests and diagnostics)."""
        return self._builds

    def _ensure_loaded(self) -> S2TTEngine:
        if self._engine is None:
            self._engine = self._factory()
            if self._owns_instance:  # count factory productions, not adoptions
                self._builds += 1
        return self._engine

    def _maybe_unload_idle(self) -> None:
        if (
            self.config.load_policy == "idle_unload"
            and self._owns_instance
            and self._engine is not None
            and self._last_used is not None
            and (self._clock() - self._last_used) >= self.config.idle_unload_seconds
        ):
            self._engine = None  # release; rebuilt on next translate

    def translate(self, chunk: AudioChunk, target_lang: str) -> EngineResult:
        self._maybe_unload_idle()
        engine = self._ensure_loaded()
        self._last_used = self._clock()
        return engine.translate(chunk, target_lang)
