"""Balanced quality gate with real checks (spec §3.5–§3.8).

Pure check functions + a gate that applies them and decides retry/fallback.
"""
from __future__ import annotations

import re

from sawti.config import QualityGateConfig
from sawti.script_detect import dominant_script
from sawti.types import AudioChunk, EngineResult, GateDecision


def run_checks(
    result: EngineResult, chunk: AudioChunk, target_lang: str
) -> dict[str, bool]:
    cfg = QualityGateConfig()
    text = result.raw_text
    dur_s = max(chunk.duration_s, 0.001)

    empty = len(text.strip()) == 0
    garbage = bool(re.fullmatch(r"[\s\W_]+", text)) and not empty
    # script mismatch: target Arabic but output not Arabic
    script_mismatch = False
    if target_lang == "ara":
        ds = dominant_script(text)
        script_mismatch = ds == "latin"  # mostly-Latin output for Arabic target
    # length ratio: chars per audio second
    cps = len(text) / dur_s
    lr = QualityGateConfig().length_ratio
    length_anom = cps < lr.min_chars_per_audio_second or cps > lr.max_chars_per_audio_second
    if empty:
        length_anom = False  # don't double-flag
    # repetition loop: a single token repeated >= 3 times
    toks = text.split()
    rep = (
        len(toks) >= 3
        and len(set(toks)) == 1
    )
    return {
        "empty_output": empty,
        "garbage_output": garbage,
        "script_mismatch": script_mismatch,
        "length_ratio_anomaly": length_anom,
        "repetition_loop": rep,
    }


class BalancedQualityGate:
    def __init__(self, config: QualityGateConfig | None = None) -> None:
        self.config = config or QualityGateConfig()

    def evaluate(
        self, result: EngineResult, chunk: AudioChunk, target_lang: str
    ) -> GateDecision:
        checks = run_checks(result, chunk, target_lang)
        low_conf = result.confidence < self.config.confidence_threshold
        any_fail = any(checks.values())
        needs_retry = any_fail or low_conf
        path = "retry" if needs_retry else None
        return GateDecision(
            chunk_id=chunk.id,
            accepted=not needs_retry,
            result=result,
            checks=checks,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            fallback_path=path,
            low_confidence=low_conf,
            needs_retry=needs_retry,
            log=[{"action": "evaluate", "checks": checks, "low_conf": low_conf}],
        )
