"""Sequential-generator orchestrator (spec §5.3).

Lock rule: each chunk is processed fully through engine → gate → (retry) →
post-processing before the next chunk is emitted. No concurrency, no
reordering for MVP.
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
    ) -> None:
        self.segmenter = segmenter
        self.engine = engine
        self.gate = gate
        self.postprocessor = postprocessor

    def run(self, source: AudioSource, target_lang: str) -> Iterable[OutputSegment]:
        for chunk in self.segmenter.process(source.iter_frames()):
            result = self.engine.translate(chunk, target_lang)
            gated = self.gate.evaluate(result, chunk, target_lang)
            if gated.needs_retry:
                # M0: re-translate via the same (stub) engine; M1 swaps in
                # the real fallback handler (retry → rechunk → ASR+MT).
                retried = self.engine.translate(chunk, target_lang)
                gated = self.gate.evaluate(retried, chunk, target_lang)
            cleaned = list(self.postprocessor.process([gated], target_lang))
            yield from cleaned
