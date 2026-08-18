"""Sequential-generator orchestrator (spec §5.3) — Commit 2 wiring.

Locked control flow per chunk:

    engine.translate    # exactly one primary call — never retried here
    gate.evaluate       # exactly one primary evaluation
      PASS    -> postprocessor
      FAIL    -> fallback.recover(chunk, primary_decision, target_lang)
                 # exactly once; its returned decision — accepted OR
                 # exhausted — flows to postprocessor unchanged

FallbackHandler is the sole recovery authority: the Pipeline holds no
retry/rechunk/escalation policy of its own. A Pipeline constructed without
a fallback forwards the rejected primary decision as-is (flagged) rather
than inventing recovery, and never re-invokes recover() on an exhausted
decision (no loop). Sequential ordering is preserved by construction.
"""
from __future__ import annotations

from typing import Iterable

from sawti.engine import EngineManager
from sawti.postprocess import PostProcessor
from sawti.quality_gate import QualityGate
from sawti.segmenter import Segmenter
from sawti.sources import AudioSource
from sawti.types import OutputSegment


class Pipeline:
    def __init__(
        self,
        segmenter: Segmenter,
        engine: EngineManager,
        gate: QualityGate,
        postprocessor: PostProcessor,
        fallback=None,  # FallbackHandler-compatible (has .recover)
    ) -> None:
        self.segmenter = segmenter
        self.engine = engine
        self.gate = gate
        self.postprocessor = postprocessor
        self.fallback = fallback

    def run(self, source: AudioSource, target_lang: str) -> Iterable[OutputSegment]:
        for chunk in self.segmenter.process(source.iter_frames()):
            result = self.engine.translate(chunk, target_lang)
            gated = self.gate.evaluate(result, chunk, target_lang)
            if gated.needs_retry and self.fallback is not None:
                gated = self.fallback.recover(chunk, gated, target_lang)
            cleaned = list(self.postprocessor.process([gated], target_lang))
            yield from cleaned
