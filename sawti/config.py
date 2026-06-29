"""Typed configuration schema and loader (spec §2.3, §3.3, §3.8, §4.6)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


SUPPORTED_LANGS = ("eng", "ara", "fra")
LoadPolicy = Literal["resident", "lazy", "idle_unload"]
GatePolicy = Literal["conservative", "balanced", "aggressive"]
NormMode = Literal["off", "matching_only", "output"]


class SegmentationConfig(BaseModel):
    pause_threshold_ms: int = 350
    max_chunk_duration_s: int = 12
    min_chunk_duration_ms: int = 600
    overlap_ms: int = 300
    min_speech_ms: int = 300
    min_gap_ms: int = 100


class S2TTConfig(BaseModel):
    engine: str = "seamless_m4t"
    target_lang: str = "eng"
    load_policy: LoadPolicy = "resident"
    idle_unload_seconds: int = 300
    device: str = "cuda"


class LengthRatioConfig(BaseModel):
    min_chars_per_audio_second: float = 0.5
    max_chars_per_audio_second: float = 35.0


class RetriesConfig(BaseModel):
    max_s2tt_retries: int = 1
    max_rechunk_attempts: int = 1


class ChecksConfig(BaseModel):
    empty_output: bool = True
    garbage_output: bool = True
    script_mismatch: bool = True
    length_ratio_anomaly: bool = True
    repetition_loop: bool = True


class QualityGateConfig(BaseModel):
    policy: GatePolicy = "balanced"
    confidence_threshold: float = 0.40
    retry_once: bool = True
    rechunk_on_failure: bool = True
    fallback_to_asr_mt: bool = True
    checks: ChecksConfig = Field(default_factory=ChecksConfig)
    length_ratio: LengthRatioConfig = Field(default_factory=LengthRatioConfig)
    retries: RetriesConfig = Field(default_factory=RetriesConfig)
    script_mismatch_strictness: dict[str, str] = Field(
        default_factory=lambda: {"eng": "soft", "fra": "soft", "ara": "strict"}
    )


class PostprocessConfig(BaseModel):
    dedup_overlap: bool = True
    collapse_repeats: bool = True
    repeat_min_count: int = 3
    normalize_script: bool = True
    repair_punctuation: bool = True
    preserve_arabic_diacritics: bool = True
    arabic_alef_normalization: NormMode = "matching_only"
    arabic_yeh_maqsura_normalization: NormMode = "matching_only"
    fuzzy_overlap: bool = False


class SawtiConfig(BaseModel):
    target_lang: str = "eng"
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    s2tt: S2TTConfig = Field(default_factory=S2TTConfig)
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)

    def model_post_init(self, __context) -> None:
        if self.target_lang not in SUPPORTED_LANGS:
            raise ValueError(
                f"target_lang must be one of {SUPPORTED_LANGS}, got {self.target_lang!r}"
            )
        if self.s2tt.target_lang not in SUPPORTED_LANGS:
            raise ValueError(
                f"s2tt.target_lang must be one of {SUPPORTED_LANGS}, got {self.s2tt.target_lang!r}"
            )


def load_config(path: str | Path) -> SawtiConfig:
    """Load a YAML config file into a validated SawtiConfig."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return SawtiConfig(**raw)
