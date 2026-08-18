"""Balanced quality gate with real checks (spec §3.5–§3.8) — Commit 3.

- Every ``config.checks.*`` toggle is load-bearing: a disabled check never
  contributes a failure.
- ``repetition_loop`` uses the deterministic n-gram consecutive-block
  detector (``sawti.loop_detect``) — catches unigram AND phrase loops
  (the Saudi spike's documented failure mode).
- ``script_mismatch_strictness`` is observable policy:
    * ``strict`` (default for ara): a dominant-script mismatch is a hard
      failure and sets needs_retry.
    * ``soft`` (default for eng/fra): the mismatch is recorded as a
      ``soft_script_mismatch`` flag in the decision log — observable,
      never a hard gate.
- Invariant: ``needs_retry`` is derived ONLY from enabled hard failures
  plus the configured confidence threshold. Soft signals never escalate.
"""
from __future__ import annotations

import re

from sawti.config import QualityGateConfig
from sawti.loop_detect import is_loop
from sawti.script_detect import dominant_script
from sawti.types import AudioChunk, EngineResult, GateDecision

# Target-language -> expected dominant output script.
_EXPECTED_SCRIPT = {"ara": "arabic", "eng": "latin", "fra": "latin"}


def run_checks(
    result: EngineResult,
    chunk: AudioChunk,
    target_lang: str,
    config: QualityGateConfig | None = None,
) -> dict[str, bool]:
    """Evaluate all ENABLED checks; returns per-check triggered-hard booleans.

    Disabled checks report False. script_mismatch reports True only under
    ``strict`` strictness for the target language (soft mismatches are the
    gate's soft-flag path, not a check failure).
    """
    cfg = config or QualityGateConfig()
    text = result.raw_text
    dur_s = max(chunk.duration_s, 0.001)

    empty = len(text.strip()) == 0
    garbage = bool(re.fullmatch(r"[\s\W_]+", text)) and not empty

    script_mismatch = False
    expected = _EXPECTED_SCRIPT.get(target_lang)
    if expected is not None:
        mismatch = dominant_script(text) not in (expected, "other")
        strict = cfg.script_mismatch_strictness.get(target_lang, "soft") == "strict"
        script_mismatch = bool(mismatch and strict)

    cps = len(text) / dur_s
    lr = cfg.length_ratio
    length_anom = cps < lr.min_chars_per_audio_second or cps > lr.max_chars_per_audio_second
    if empty:
        length_anom = False  # don't double-flag

    rep = is_loop(text)

    return {
        "empty_output": empty if cfg.checks.empty_output else False,
        "garbage_output": garbage if cfg.checks.garbage_output else False,
        "script_mismatch": script_mismatch if cfg.checks.script_mismatch else False,
        "length_ratio_anomaly": length_anom if cfg.checks.length_ratio_anomaly else False,
        "repetition_loop": rep if cfg.checks.repetition_loop else False,
    }


def soft_script_mismatch(
    result: EngineResult, target_lang: str,
    config: QualityGateConfig | None = None,
) -> bool:
    """True when the dominant script disagrees with the target AND the
    configured strictness for that language is soft (observability flag —
    never a hard gate)."""
    cfg = config or QualityGateConfig()
    expected = _EXPECTED_SCRIPT.get(target_lang)
    if expected is None:
        return False
    mismatch = dominant_script(result.raw_text) not in (expected, "other")
    strict = cfg.script_mismatch_strictness.get(target_lang, "soft") == "strict"
    return bool(mismatch and not strict)


class BalancedQualityGate:
    def __init__(self, config: QualityGateConfig | None = None) -> None:
        self.config = config or QualityGateConfig()

    def evaluate(
        self, result: EngineResult, chunk: AudioChunk, target_lang: str
    ) -> GateDecision:
        checks = run_checks(result, chunk, target_lang, self.config)
        low_conf = result.confidence < self.config.confidence_threshold
        needs_retry = any(checks.values()) or low_conf
        path = "retry" if needs_retry else None
        soft = soft_script_mismatch(result, target_lang, self.config)
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
            log=[{"action": "evaluate", "checks": checks, "low_conf": low_conf,
                  "soft_script_mismatch": soft}],
        )
