"""Real 6-step deterministic post-processor (spec §4.2).

Stateful: holds the previous chunk's tail tokens for overlap dedupe.
Maintains raw/match/display distinction: dedupe compares normalized tokens
but emits raw tokens, preserving casing/diacritics in display text.
"""
from __future__ import annotations

from typing import Iterable

from sawti.config import PostprocessConfig
from sawti.text_normalize import (
    collapse_repeated_loops,
    normalize_arabic_for_match,
    normalize_for_match,
    repair_punctuation_spacing,
    strip_excess_whitespace,
)
from sawti.types import GateDecision, OutputSegment


class RealPostProcessor:
    def __init__(self, config: PostprocessConfig | None = None) -> None:
        self.config = config or PostprocessConfig()
        self._prev_tokens: list[str] = []

    def _match_tokens(self, text: str, target_lang: str) -> list[str]:
        m = normalize_for_match(text)
        if target_lang == "ara":
            m = normalize_arabic_for_match(m)
        return m.split()

    def process(
        self, decisions: Iterable[GateDecision], target_lang: str
    ) -> Iterable[OutputSegment]:
        for d in decisions:
            text = d.result.raw_text
            # Step 2: collapse decoder loops (on raw text, before dedupe).
            if self.config.collapse_repeats:
                text = collapse_repeated_loops(text, self.config.repeat_min_count)
            # Step 1: dedupe overlap with previous chunk (normalized compare).
            curr_raw = text.split()
            if self.config.dedup_overlap and self._prev_tokens:
                curr_norm = self._match_tokens(text, target_lang)
                prev_norm = self._match_tokens(" ".join(self._prev_tokens), target_lang)
                max_k = min(len(prev_norm), len(curr_norm))
                cut = 0
                for k in range(max_k, 1, -1):
                    if prev_norm[-k:] == curr_norm[:k]:
                        cut = k
                        break
                if cut:
                    curr_raw = curr_raw[cut:]
                    text = " ".join(curr_raw)
            # Step 3: whitespace collapse (display side).
            # NOTE: the config flag is named `normalize_script` for historical
            # (M0) reasons but currently gates whitespace stripping, not script
            # normalization. Arabic script normalization is matching-only and
            # runs in _match_tokens, never on display text (spec §4.4).
            if self.config.normalize_script:
                text = strip_excess_whitespace(text)
            # Step 4: punctuation repair (display side).
            if self.config.repair_punctuation:
                text = repair_punctuation_spacing(text)
            # Update prev-token state from this chunk's full raw tokens.
            self._prev_tokens = d.result.raw_text.split()
            # Step 5: emit.
            yield OutputSegment(
                chunk_id=d.chunk_id,
                text=text,
                start_time=d.start_time,
                end_time=d.end_time,
                low_confidence=d.low_confidence,
            )
