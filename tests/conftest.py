"""Shared pytest fixtures."""
from __future__ import annotations

# Load .env into os.environ FIRST, before any HF/transformers import tries to
# resolve the HF cache path. The system env sets HF_HOME with literal quotes
# that break pathlib; sawti.env loads the corrected unquoted path from .env.
import sawti.env  # noqa: F401

import numpy as np
import pytest

from sawti.types import AudioChunk, EngineResult


@pytest.fixture
def silence_chunk() -> AudioChunk:
    return AudioChunk(
        id="c_test",
        audio=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
        overlap_from_prev_s=0.0,
        meta={},
    )


@pytest.fixture
def good_result(silence_chunk: AudioChunk) -> EngineResult:
    return EngineResult(
        chunk_id=silence_chunk.id,
        raw_text="hello world",
        confidence=0.9,
        source_lang_guess="eng",
        timing_ms={"engine": 5},
        target_lang="eng",
    )


@pytest.fixture
def weak_result(silence_chunk: AudioChunk) -> EngineResult:
    return EngineResult(
        chunk_id=silence_chunk.id,
        raw_text="",
        confidence=0.1,
        source_lang_guess="und",
        timing_ms={"engine": 5},
        target_lang="eng",
    )


import os

import pytest


def pytest_collection_modifyitems(config, items):
    skip_integration = pytest.mark.skip(
        reason="set SAWTI_RUN_INTEGRATION=1 to run heavy model tests"
    )
    if not os.environ.get("SAWTI_RUN_INTEGRATION"):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
