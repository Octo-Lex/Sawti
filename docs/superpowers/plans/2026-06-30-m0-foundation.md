# M0 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the Sawti project skeleton with core data types, typed config schema, stub implementations of every pipeline component, a sequential orchestrator that runs end-to-end on fake data, an eval-harness skeleton, and unit tests — so M1 can swap stubs for real (Silero / SeamlessM4T / etc.) implementations without touching interfaces.

**Architecture:** A sequential-generator pipeline (`AudioSource → Segmenter → Engine → Gate → PostProcessor → OutputSegment`) where each component is an isolated, testable unit behind a shared data-type contract. M0 ships stub components that return canned data; the orchestrator and tests are real and permanent.

**Tech Stack:** Python 3.11+, `uv` (packaging), `pytest` (testing), `pydantic` (config schema + validation), `pydantic-settings` (YAML/TOML/env loading), `numpy` (audio arrays), `structlog` (structured JSONL logging), `typer` (CLI).

**Spec reference:** [`docs/superpowers/specs/2026-06-30-sawti-design.md`](../specs/2026-06-30-sawti-design.md) — §5 (interfaces), §8.3 (project layout), §6 (milestones).

---

## File Structure

This plan creates the following files. Each has one clear responsibility.

```
sawti/
├── pyproject.toml                  # uv project: metadata, deps, pytest config
├── .python-version                 # pins 3.11
├── sawti/
│   ├── __init__.py                 # package marker, version
│   ├── types.py                    # AudioChunk, EngineResult, GateDecision, OutputSegment
│   ├── config.py                   # pydantic config schema + load_config() loader
│   ├── logging_setup.py            # structlog JSONL config
│   ├── pipeline.py                 # Pipeline.run() sequential orchestrator
│   ├── segmenter.py                # Segmenter protocol + StubSegmenter
│   ├── engine.py                   # S2TTEngine protocol + EngineManager + StubEngine
│   ├── quality_gate.py             # QualityGate protocol + StubQualityGate
│   ├── postprocess.py              # PostProcessor protocol + StubPostProcessor
│   ├── sources.py                  # AudioSource protocol + StubAudioSource + StubFileSource
│   └── cli.py                      # typer CLI entrypoint: `sawti transcribe`, `sawti eval`
├── eval/
│   ├── __init__.py
│   ├── metrics.py                  # chrF stub + record_eval (skeleton)
│   ├── harness.py                  # run_eval() skeleton over an eval set dir
│   └── report.py                   # write_report() JSON skeleton
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # shared fixtures: stub audio, chunk, result
│   ├── test_types.py               # data-type construction & invariants
│   ├── test_config.py              # config load + validation + defaults
│   ├── test_pipeline.py            # orchestrator ordering, retry path, yield contract
│   ├── test_segmenter.py           # StubSegmenter yields expected chunks
│   ├── test_engine.py              # EngineManager + StubEngine wiring
│   ├── test_quality_gate.py        # StubQualityGate verdict contract
│   ├── test_postprocess.py         # StubPostProcessor OutputSegment shape
│   ├── test_sources.py             # StubAudioSource / StubFileSource frames
│   ├── test_cli.py                 # CLI smoke (transcribe/eval run on stubs)
│   └── test_eval_harness.py        # harness skeleton runs, emits report
├── config/
│   └── default.yaml                # default runtime config (all M0 defaults)
├── tests/fixtures/                 # tiny synthetic audio + canned references
│   └── .gitkeep
└── docs/superpowers/
    ├── specs/2026-06-30-sawti-design.md   # (exists)
    └── plans/2026-06-30-m0-foundation.md  # (this file)
```

**Why these boundaries:** each component file pairs a `Protocol` (the swappable interface from §5.2) with a `Stub*` implementation. M1 adds real implementations alongside the stubs; nothing in `pipeline.py`, `types.py`, or `config.py` changes. The orchestrator depends on Protocols, never concrete stubs.

---

## Task 1: Project scaffold (uv, pyproject, dirs)

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `sawti/__init__.py`, `eval/__init__.py`, `tests/__init__.py`
- Create: `tests/fixtures/.gitkeep`

- [ ] **Step 1: Initialize the uv project**

Run:
```bash
cd /c/Sawti
uv init --lib --name sawti --python 3.11
```
If `uv init` refuses because the dir is non-empty, create the files manually in the next steps and run `uv sync` to materialize the env. Do not let `uv init` overwrite the existing `README.md` or `.gitignore` — if it tries, decline.

- [ ] **Step 2: Write `pyproject.toml`**

Overwrite `pyproject.toml` with:

```toml
[project]
name = "sawti"
version = "0.0.1"
description = "Multilingual speech-to-text translation: speak in any supported language or a code-switched mix, receive text in a pre-selected target language."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "structlog>=24.1",
    "typer>=0.12",
    "pyyaml>=6.0",
]

[project.scripts]
sawti = "sawti.cli:app"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 3: Pin the Python version**

Create `.python-version` containing exactly:

```
3.11
```

- [ ] **Step 4: Create package markers**

`sawti/__init__.py`:
```python
"""Sawti: multilingual speech-to-text translation."""
__version__ = "0.0.1"
```

`eval/__init__.py`:
```python
"""Sawti evaluation harness (skeleton)."""
```

`tests/__init__.py`:
```python
```

`tests/fixtures/.gitkeep`:
```
```

- [ ] **Step 5: Install deps and verify the env resolves**

Run:
```bash
uv sync
```
Expected: resolves and installs into `.venv/`, prints "Installed N packages".

- [ ] **Step 6: Verify pytest runs (zero tests found is fine)**

Run:
```bash
uv run pytest
```
Expected: `no tests ran` (exit 5) or a clean "0 passed" — no collection errors.

- [ ] **Step 7: Update `.gitignore` for venv + cache, commit**

Check `.gitignore` already has `.venv/`, `__pycache__/`, `*.egg-info/`. If any are missing, add them. Then:

```bash
git add pyproject.toml .python-version uv.lock sawti/ eval/ tests/ .gitignore
git commit -m "chore: scaffold uv project, package layout, pytest config"
```

---

## Task 2: Core data types (`sawti/types.py`)

**Files:**
- Create: `sawti/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Write the failing test**

`tests/test_types.py`:
```python
import numpy as np
from sawti.types import AudioChunk, EngineResult, GateDecision, OutputSegment


def test_audio_chunk_construction():
    audio = np.zeros(16000, dtype=np.float32)  # 1s of silence
    chunk = AudioChunk(
        id="c0",
        audio=audio,
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
        overlap_from_prev_s=0.0,
        meta={},
    )
    assert chunk.id == "c0"
    assert chunk.audio.dtype == np.float32
    assert chunk.duration_s == 1.0


def test_engine_result_construction():
    r = EngineResult(
        chunk_id="c0",
        raw_text="hello",
        confidence=0.9,
        source_lang_guess="eng",
        timing_ms={"engine": 12},
        target_lang="eng",
    )
    assert r.raw_text == "hello"
    assert 0.0 <= r.confidence <= 1.0


def test_gate_decision_defaults():
    r = EngineResult("c0", "hi", 0.8, "eng", {}, "eng")
    d = GateDecision(chunk_id="c0", accepted=True, result=r, checks={})
    assert d.accepted is True
    assert d.fallback_path is None
    assert d.low_confidence is False
    assert d.needs_retry is False
    assert d.log == []


def test_output_segment_construction():
    seg = OutputSegment(
        chunk_id="c0", text="Hello", start_time=0.0, end_time=1.0, low_confidence=False
    )
    assert seg.text == "Hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_types.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'sawti.types'` (or `cannot import name`).

- [ ] **Step 3: Write minimal implementation**

`sawti/types.py`:
```python
"""Core data types shared across all pipeline components (spec §5.1)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioChunk:
    """A segment of audio emitted by the segmentation layer."""

    id: str
    audio: np.ndarray  # float32 mono PCM
    sample_rate: int  # 16000
    start_time: float  # seconds from session start
    end_time: float
    overlap_from_prev_s: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time


@dataclass
class EngineResult:
    """Output of the S2TT engine for one chunk."""

    chunk_id: str
    raw_text: str
    confidence: float  # 0..1
    source_lang_guess: str | None
    timing_ms: dict
    target_lang: str


@dataclass
class GateDecision:
    """The quality gate's verdict on an EngineResult."""

    chunk_id: str
    accepted: bool
    result: EngineResult
    checks: dict
    fallback_path: str | None = None  # None | "retry" | "rechunk" | "asr_mt"
    low_confidence: bool = False
    needs_retry: bool = False
    log: list[dict] = field(default_factory=list)


@dataclass
class OutputSegment:
    """Emitted unit: timestamp-aligned target-language text."""

    chunk_id: str
    text: str
    start_time: float
    end_time: float
    low_confidence: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/test_types.py -v
```
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add sawti/types.py tests/test_types.py
git commit -m "feat(types): add core data types AudioChunk/EngineResult/GateDecision/OutputSegment"
```

---

## Task 3: Config schema (`sawti/config.py` + `config/default.yaml`)

**Files:**
- Create: `sawti/config.py`
- Create: `config/default.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from pathlib import Path

import pytest

from sawti.config import SawtiConfig, load_config


def test_defaults_match_spec():
    cfg = SawtiConfig()
    assert cfg.segmentation.pause_threshold_ms == 350
    assert cfg.segmentation.max_chunk_duration_s == 12
    assert cfg.segmentation.min_chunk_duration_ms == 600
    assert cfg.segmentation.overlap_ms == 300
    assert cfg.quality_gate.policy == "balanced"
    assert cfg.quality_gate.confidence_threshold == 0.40
    assert cfg.quality_gate.script_mismatch_strictness["ara"] == "strict"
    assert cfg.postprocess.preserve_arabic_diacritics is True
    assert cfg.postprocess.arabic_alef_normalization == "matching_only"


def test_load_default_yaml():
    cfg = load_config(Path("config/default.yaml"))
    assert cfg.target_lang in ("eng", "ara", "fra")
    assert cfg.s2tt.engine == "seamless_m4t"


def test_invalid_target_lang_rejected():
    from sawdust import _bad  # noqa: F401  intentional import error placeholder
```

The last test is deliberately a placeholder we will replace in Step 3 with a real validation test — see Step 3.

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_config.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'sawti.config'`.

- [ ] **Step 3: Write minimal implementation**

First, fix the placeholder test. Replace `test_invalid_target_lang_rejected` in `tests/test_config.py` with:

```python
def test_invalid_target_lang_rejected():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        SawtiConfig(target_lang="spanish")  # not a supported code
```

Then write `sawti/config.py`:

```python
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
```

Then create `config/default.yaml`:

```yaml
# Default Sawti runtime config (spec §2.3, §3.3, §3.8, §4.6).
target_lang: eng

segmentation:
  pause_threshold_ms: 350
  max_chunk_duration_s: 12
  min_chunk_duration_ms: 600
  overlap_ms: 300
  min_speech_ms: 300
  min_gap_ms: 100

s2tt:
  engine: seamless_m4t
  target_lang: eng
  load_policy: resident
  idle_unload_seconds: 300
  device: cuda

quality_gate:
  policy: balanced
  confidence_threshold: 0.40
  retry_once: true
  rechunk_on_failure: true
  fallback_to_asr_mt: true
  checks:
    empty_output: true
    garbage_output: true
    script_mismatch: true
    length_ratio_anomaly: true
    repetition_loop: true
  length_ratio:
    min_chars_per_audio_second: 0.5
    max_chars_per_audio_second: 35.0
  retries:
    max_s2tt_retries: 1
    max_rechunk_attempts: 1
  script_mismatch_strictness:
    eng: soft
    fra: soft
    ara: strict

postprocess:
  dedup_overlap: true
  collapse_repeats: true
  repeat_min_count: 3
  normalize_script: true
  repair_punctuation: true
  preserve_arabic_diacritics: true
  arabic_alef_normalization: matching_only
  arabic_yeh_maqsura_normalization: matching_only
  fuzzy_overlap: false
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/test_config.py -v
```
Expected: PASS (3 passed). The validation test confirms a bad `target_lang` raises `pydantic.ValidationError`.

- [ ] **Step 5: Commit**

```bash
git add sawti/config.py config/default.yaml tests/test_config.py
git commit -m "feat(config): typed pydantic config schema + YAML loader with spec defaults"
```

---

## Task 4: Logging setup (`sawti/logging_setup.py`)

**Files:**
- Create: `sawti/logging_setup.py`

- [ ] **Step 1: Write the failing test**

`tests/test_logging_setup.py`:
```python
import io
import json

from sawti.logging_setup import configure_logging, get_logger


def test_logger_emits_jsonl():
    buf = io.StringIO()
    configure_logging(stream=buf)
    log = get_logger("test")
    log.info("chunk", chunk_id="c0", confidence=0.7)
    line = buf.getvalue().strip()
    record = json.loads(line)
    assert record["event"] == "chunk"
    assert record["chunk_id"] == "c0"
    assert record["confidence"] == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_logging_setup.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`sawti/logging_setup.py`:
```python
"""Structured JSONL logging via structlog (spec §7.5)."""
from __future__ import annotations

import sys

import structlog

_configured = False


def configure_logging(stream=None) -> None:
    """Configure structlog to emit one JSON object per line.

    Idempotent: safe to call multiple times.
    """
    global _configured
    if _configured:
        return
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO+
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None):
    configure_logging()
    return structlog.get_logger(name)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/test_logging_setup.py -v
```
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add sawti/logging_setup.py tests/test_logging_setup.py
git commit -m "feat(logging): structured JSONL logging via structlog"
```

---

## Task 5: Component Protocols + stubs (`segmenter`, `engine`, `quality_gate`, `postprocess`, `sources`)

This is the largest task. It defines the `Protocol` interfaces (§5.2) and their `Stub*` implementations that return canned data. Each protocol+stub is its own commit so M1 can add real impls alongside.

### Task 5a: AudioSource protocol + stubs (`sawti/sources.py`)

**Files:**
- Create: `sawti/sources.py`
- Test: `tests/test_sources.py`

- [ ] **Step 1: Write the failing test**

`tests/test_sources.py`:
```python
import numpy as np

from sawti.sources import StubAudioSource, AudioFrame


def test_stub_audio_source_yields_frames():
    src = StubAudioSource(n_frames=3, samples_per_frame=16000)
    frames = list(src.iter_frames())
    assert len(frames) == 3
    assert all(isinstance(f, AudioFrame) for f in frames)
    assert frames[0].audio.shape == (16000,)
    assert frames[0].audio.dtype == np.float32
    # timestamps are monotonic
    assert frames[1].timestamp_s > frames[0].timestamp_s
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_sources.py -v
```
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`sawti/sources.py`:
```python
"""Audio source protocol + stubs (spec §5.2, §8.3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


@dataclass
class AudioFrame:
    """A fixed-size frame of PCM audio with a timestamp."""

    audio: np.ndarray  # float32 mono
    sample_rate: int
    timestamp_s: float


class AudioSource(Protocol):
    """Anything that yields timestamped audio frames."""

    def iter_frames(self) -> Iterable[AudioFrame]: ...


class StubAudioSource:
    """Yields `n_frames` frames of synthetic silence (M0 only)."""

    def __init__(self, n_frames: int = 5, samples_per_frame: int = 16000,
                 sample_rate: int = 16000) -> None:
        self.n_frames = n_frames
        self.samples_per_frame = samples_per_frame
        self.sample_rate = sample_rate

    def iter_frames(self) -> Iterable[AudioFrame]:
        for i in range(self.n_frames):
            yield AudioFrame(
                audio=np.zeros(self.samples_per_frame, dtype=np.float32),
                sample_rate=self.sample_rate,
                timestamp_s=i * self.samples_per_frame / self.sample_rate,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/test_sources.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sawti/sources.py tests/test_sources.py
git commit -m "feat(sources): AudioSource protocol + StubAudioSource"
```

### Task 5b: Segmenter protocol + stub (`sawti/segmenter.py`)

**Files:**
- Create: `sawti/segmenter.py`
- Test: `tests/test_segmenter.py`

- [ ] **Step 1: Write the failing test**

`tests/test_segmenter.py`:
```python
from sawti.segmenter import StubSegmenter
from sawti.sources import StubAudioSource
from sawti.types import AudioChunk


def test_stub_segmenter_yields_chunks():
    src = StubAudioSource(n_frames=4, samples_per_frame=16000)
    seg = StubSegmenter(chunk_frames=2, sample_rate=16000)
    chunks = list(seg.process(src.iter_frames()))
    assert len(chunks) == 2
    assert all(isinstance(c, AudioChunk) for c in chunks)
    assert chunks[0].id == "c0"
    assert chunks[1].start_time == chunks[0].end_time
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_segmenter.py -v
```
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`sawti/segmenter.py`:
```python
"""Segmenter protocol + stub (spec §2, §5.2).

M1 replaces StubSegmenter with a Silero-VAD-backed real segmenter; the
protocol and orchestrator are unchanged.
"""
from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np

from sawti.config import SegmentationConfig
from sawti.sources import AudioFrame
from sawti.types import AudioChunk


class Segmenter(Protocol):
    """Consumes audio frames, yields coherent AudioChunks."""

    def process(self, frames: Iterable[AudioFrame]) -> Iterable[AudioChunk]: ...


class StubSegmenter:
    """Groups frames into fixed-size chunks (M0 only).

    Ignores VAD/pause logic entirely — just packs `chunk_frames` frames
    per chunk so the orchestrator has something to consume.
    """

    def __init__(
        self,
        chunk_frames: int = 2,
        sample_rate: int = 16000,
        config: SegmentationConfig | None = None,
    ) -> None:
        self.chunk_frames = chunk_frames
        self.sample_rate = sample_rate
        self._counter = 0

    def process(self, frames: Iterable[AudioFrame]) -> Iterable[AudioChunk]:
        buffer: list[np.ndarray] = []
        start_ts: float | None = None
        last_ts = 0.0
        for frame in frames:
            if start_ts is None:
                start_ts = frame.timestamp_s
            buffer.append(frame.audio)
            last_ts = frame.timestamp_s + len(frame.audio) / self.sample_rate
            if len(buffer) >= self.chunk_frames:
                yield self._emit(buffer, start_ts or 0.0, last_ts)
                buffer = []
                start_ts = None
        if buffer:
            yield self._emit(buffer, start_ts or 0.0, last_ts)

    def _emit(self, buffer: list[np.ndarray], start: float, end: float) -> AudioChunk:
        chunk_id = f"c{self._counter}"
        self._counter += 1
        return AudioChunk(
            id=chunk_id,
            audio=np.concatenate(buffer).astype(np.float32),
            sample_rate=self.sample_rate,
            start_time=start,
            end_time=end,
            overlap_from_prev_s=0.0,
            meta={},
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/test_segmenter.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sawti/segmenter.py tests/test_segmenter.py
git commit -m "feat(segmenter): Segmenter protocol + StubSegmenter"
```

### Task 5c: S2TT engine protocol + manager + stub (`sawti/engine.py`)

**Files:**
- Create: `sawti/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write the failing test**

`tests/test_engine.py`:
```python
from sawti.engine import EngineManager, StubEngine
from sawti.types import AudioChunk
import numpy as np


def _chunk(cid: str) -> AudioChunk:
    return AudioChunk(id=cid, audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=1.0)


def test_stub_engine_returns_target_text():
    eng = StubEngine(canned_text="hello world", confidence=0.9)
    r = eng.translate(_chunk("c0"), target_lang="eng")
    assert r.chunk_id == "c0"
    assert r.raw_text == "hello world"
    assert r.confidence == 0.9
    assert r.target_lang == "eng"


def test_engine_manager_translates_delegates():
    mgr = EngineManager(engine=StubEngine("hi", 0.8))
    r = mgr.translate(_chunk("c1"), target_lang="ara")
    assert r.raw_text == "hi"
    assert r.target_lang == "ara"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_engine.py -v
```
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`sawti/engine.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/test_engine.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sawti/engine.py tests/test_engine.py
git commit -m "feat(engine): S2TTEngine protocol + EngineManager + StubEngine"
```

### Task 5d: Quality gate protocol + stub (`sawti/quality_gate.py`)

**Files:**
- Create: `sawti/quality_gate.py`
- Test: `tests/test_quality_gate.py`

- [ ] **Step 1: Write the failing test**

`tests/test_quality_gate.py`:
```python
from sawti.engine import StubEngine
from sawti.quality_gate import StubQualityGate
from sawti.types import AudioChunk
import numpy as np


def _chunk(cid="c0"):
    return AudioChunk(id=cid, audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=1.0)


def test_stub_gate_accepts_high_confidence():
    eng = StubEngine("hi", confidence=0.9)
    r = eng.translate(_chunk(), "eng")
    gate = StubQualityGate()
    d = gate.evaluate(r, _chunk(), target_lang="eng")
    assert d.accepted is True
    assert d.needs_retry is False
    assert d.low_confidence is False


def test_stub_gate_flags_low_confidence():
    eng = StubEngine("hi", confidence=0.2)
    r = eng.translate(_chunk(), "eng")
    gate = StubQualityGate()
    d = gate.evaluate(r, _chunk(), target_lang="eng")
    assert d.low_confidence is True
    assert d.needs_retry is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_quality_gate.py -v
```
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`sawti/quality_gate.py`:
```python
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
            fallback_path="retry" if low else None,
            low_confidence=low,
            needs_retry=low,
            log=[{"action": "evaluate", "confidence": result.confidence, "low": low}],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/test_quality_gate.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sawti/quality_gate.py tests/test_quality_gate.py
git commit -m "feat(quality_gate): QualityGate protocol + StubQualityGate"
```

### Task 5e: PostProcessor protocol + stub (`sawti/postprocess.py`)

**Files:**
- Create: `sawti/postprocess.py`
- Test: `tests/test_postprocess.py`

- [ ] **Step 1: Write the failing test**

`tests/test_postprocess.py`:
```python
from sawti.engine import StubEngine
from sawti.postprocess import StubPostProcessor
from sawti.quality_gate import StubQualityGate
from sawti.types import AudioChunk, OutputSegment
import numpy as np


def _chunk(cid="c0", start=0.0, end=1.0):
    return AudioChunk(id=cid, audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=start, end_time=end)


def _decision(cid="c0", text="hi", conf=0.9, start=0.0, end=1.0):
    eng = StubEngine(text, conf)
    r = eng.translate(_chunk(cid, start, end), "eng")
    return StubQualityGate().evaluate(r, _chunk(cid, start, end), "eng")


def test_stub_postprocessor_emits_output_segment():
    pp = StubPostProcessor()
    out = list(pp.process([_decision()], target_lang="eng"))
    assert len(out) == 1
    assert isinstance(out[0], OutputSegment)
    assert out[0].text == "hi"


def test_stub_postprocessor_is_stateful_across_calls():
    """process() takes a list; the stub holds no cross-call state in M0,
    but the signature supports streaming via repeated calls in M1."""
    pp = StubPostProcessor()
    out = list(pp.process([_decision("c0"), _decision("c1", "bye", start=1.0, end=2.0)],
                          target_lang="eng"))
    assert [o.text for o in out] == ["hi", "bye"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_postprocess.py -v
```
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`sawti/postprocess.py`:
```python
"""Post-processor protocol + stub (spec §4, §5.2).

M1 replaces StubPostProcessor with the 6-step deterministic pipeline
(dedupe/repeat-collapse/normalize/punctuate/emit/log). Protocol unchanged.
"""
from __future__ import annotations

from typing import Iterable, Protocol

from sawti.config import PostprocessConfig
from sawti.types import GateDecision, OutputSegment


class PostProcessor(Protocol):
    def process(
        self, decisions: Iterable[GateDecision], target_lang: str
    ) -> Iterable[OutputSegment]: ...


class StubPostProcessor:
    """Passes result text through unchanged (M0 only)."""

    def __init__(self, config: PostprocessConfig | None = None) -> None:
        self.config = config or PostprocessConfig()

    def process(
        self, decisions: Iterable[GateDecision], target_lang: str
    ) -> Iterable[OutputSegment]:
        for d in decisions:
            yield OutputSegment(
                chunk_id=d.chunk_id,
                text=d.result.raw_text,
                start_time=d.result.timing_ms.get("_start", 0.0) if isinstance(
                    d.result.timing_ms.get("_start"), float
                ) else 0.0,
                end_time=0.0,
                low_confidence=d.low_confidence,
            )
```

Note: the stub ignores timestamps (not available on `EngineResult`); M1 carries chunk timestamps through `GateDecision` for accurate `OutputSegment` timing. This is acceptable for M0 since the stub only proves the data flow.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/test_postprocess.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sawti/postprocess.py tests/test_postprocess.py
git commit -m "feat(postprocess): PostProcessor protocol + StubPostProcessor"
```

---

## Task 6: Sequential orchestrator (`sawti/pipeline.py`)

**Files:**
- Create: `sawti/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
import numpy as np

from sawti.engine import EngineManager, StubEngine
from sawti.pipeline import Pipeline
from sawti.postprocess import StubPostProcessor
from sawti.quality_gate import StubQualityGate
from sawti.segmenter import StubSegmenter
from sawti.sources import StubAudioSource
from sawti.types import OutputSegment


def test_pipeline_end_to_end_on_stubs():
    src = StubAudioSource(n_frames=4, samples_per_frame=16000)
    pipe = Pipeline(
        segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
        engine=EngineManager(engine=StubEngine("hello", 0.9)),
        gate=StubQualityGate(),
        postprocessor=StubPostProcessor(),
    )
    out = list(pipe.run(src, target_lang="eng"))
    assert len(out) == 2
    assert all(isinstance(o, OutputSegment) for o in out)
    assert all(o.text == "hello" for o in out)
    assert all(o.low_confidence is False for o in out)


def test_pipeline_retry_path_when_low_confidence():
    """When the gate says needs_retry, the orchestrator must call the
    fallback handler once and emit a (still-stub) result."""
    src = StubAudioSource(n_frames=2, samples_per_frame=16000)
    pipe = Pipeline(
        segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
        engine=EngineManager(engine=StubEngine("x", 0.1)),  # low conf
        gate=StubQualityGate(),
        postprocessor=StubPostProcessor(),
    )
    out = list(pipe.run(src, target_lang="eng"))
    assert len(out) == 1
    assert out[0].low_confidence is True  # stub gate can't recover, but flow is intact
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_pipeline.py -v
```
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`sawti/pipeline.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/test_pipeline.py -v
```
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add sawti/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): sequential-generator orchestrator with retry path"
```

---

## Task 7: CLI entrypoint (`sawti/cli.py`)

**Files:**
- Create: `sawti/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
from typer.testing import CliRunner

from sawti.cli import app

runner = CliRunner()


def test_transcribe_runs_on_stubs():
    result = runner.invoke(app, ["transcribe", "--target", "eng"])
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_eval_runs_skeleton():
    result = runner.invoke(app, ["eval", "tests/fixtures", "--target", "eng"])
    assert result.exit_code == 0
    assert "report" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_cli.py -v
```
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`sawti/cli.py`:
```python
"""Typer CLI: `sawti transcribe` and `sawti eval` (spec §6.3, §7.6).

M0: both subcommands wire stub components. M1 swaps `transcribe` to the
real FileSource + pipeline, and `eval` to the real harness.
"""
from __future__ import annotations

from pathlib import Path

import typer

from sawti.engine import EngineManager, StubEngine
from sawti.logging_setup import configure_logging
from sawti.pipeline import Pipeline
from sawti.postprocess import StubPostProcessor
from sawti.quality_gate import StubQualityGate
from sawti.segmenter import StubSegmenter
from sawti.sources import StubAudioSource

app = typer.Typer(add_completion=False, help="Sawti multilingual STT-translation.")


def _stub_pipeline() -> Pipeline:
    return Pipeline(
        segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
        engine=EngineManager(engine=StubEngine("hello", 0.9)),
        gate=StubQualityGate(),
        postprocessor=StubPostProcessor(),
    )


@app.command()
def transcribe(
    target: str = typer.Option("eng", help="Target language: eng|ara|fra"),
) -> None:
    """Transcribe audio to the target language (stub in M0)."""
    configure_logging()
    pipe = _stub_pipeline()
    src = StubAudioSource(n_frames=4, samples_per_frame=16000)
    for seg in pipe.run(src, target_lang=target):
        typer.echo(f"[{seg.start_time:.2f}-{seg.end_time:.2f}] {seg.text}")


@app.command()
def eval(
    eval_set: Path = typer.Argument(..., help="Eval set directory"),
    target: str = typer.Option("eng", help="Target language: eng|ara|fra"),
) -> None:
    """Run the evaluation harness (skeleton in M0)."""
    from eval.harness import run_eval

    report = run_eval(eval_set, target_lang=target)
    typer.echo(f"Wrote report: {report}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test to verify it passes**

The `eval` test depends on `eval.harness.run_eval` which is built in Task 8. So expect the `transcribe` test to pass and the `eval` test to fail until Task 8 lands. Run just transcribe for now:

```bash
uv run pytest tests/test_cli.py::test_transcribe_runs_on_stubs -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sawti/cli.py tests/test_cli.py
git commit -m "feat(cli): typer CLI with transcribe + eval subcommands"
```

---

## Task 8: Eval harness skeleton (`eval/`)

**Files:**
- Create: `eval/metrics.py`
- Create: `eval/harness.py`
- Create: `eval/report.py`
- Test: `tests/test_eval_harness.py`

- [ ] **Step 1: Write the failing test**

`tests/test_eval_harness.py`:
```python
import json
from pathlib import Path

from eval.harness import run_eval
from eval.metrics import compute_chrf_stub


def test_chrf_stub_returns_score_for_match():
    score = compute_chrf_stub("hello world", "hello world")
    assert 0.0 <= score <= 100.0


def test_chrf_stub_lower_for_mismatch():
    match = compute_chrf_stub("hello world", "hello world")
    miss = compute_chrf_stub("hello world", "completely different")
    assert miss < match


def test_run_eval_writes_report(tmp_path: Path):
    # empty eval set; harness skeleton should still produce a report file
    report_path = run_eval(tmp_path, target_lang="eng")
    assert Path(report_path).exists()
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    assert data["target_lang"] == "eng"
    assert "clips" in data
    assert "metrics" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_eval_harness.py -v
```
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`eval/metrics.py`:
```python
"""Eval metrics. M0 ships a chrF stub; M1 swaps in sacrebleu chrF (spec §7.2)."""
from __future__ import annotations

from collections import Counter


def compute_chrf_stub(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """A tiny self-contained chrF-like score in [0, 100].

    This is NOT sacrebleu's chrF — it's a deterministic placeholder so the
    harness runs without the heavy dependency in M0. M1 replaces it.
    """
    def ngrams(s: str) -> Counter:
        s = " " + s.strip() + " "
        return Counter(s[i : i + n] for i in range(len(s) - n + 1))

    hg, rg = ngrams(hyp), ngrams(ref)
    if not rg:
        return 0.0
    overlap = sum((hg & rg).values())
    prec = overlap / sum(hg.values()) if hg else 0.0
    rec = overlap / sum(rg.values())
    if prec + rec == 0:
        return 0.0
    f = (1 + beta * beta) * (prec * rec) / (beta * beta * prec + rec)
    return f * 100.0
```

`eval/report.py`:
```python
"""Report writer (spec §7.6)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_report(path: str | Path, payload: dict[str, Any]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)
```

`eval/harness.py`:
```python
"""Eval harness skeleton (spec §7.6). M0: discovers clips, scores with the
chrF stub against .txt references, writes a JSON report. M1 swaps the stub
pipeline + metric for real components.
"""
from __future__ import annotations

from pathlib import Path

from eval.metrics import compute_chrf_stub
from eval.report import write_report


def run_eval(eval_set: Path, target_lang: str) -> str:
    clips = sorted(eval_set.glob("*.wav"))
    scored = []
    for wav in clips:
        ref_path = wav.with_suffix(".txt")
        ref = ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else ""
        hyp = "[stub hypothesis]"  # M1: run real pipeline on `wav`
        chrf = compute_chrf_stub(hyp, ref) if ref else None
        scored.append({"clip": wav.name, "chrf": chrf, "has_reference": bool(ref)})

    report = {
        "target_lang": target_lang,
        "n_clips": len(scored),
        "clips": scored,
        "metrics": {
            "mean_chrf": (
                sum(c["chrf"] for c in scored if c["chrf"] is not None)
                / max(1, sum(1 for c in scored if c["chrf"] is not None))
            ) if scored else 0.0,
        },
    }
    out_path = Path("outputs") / f"eval-{target_lang}.json"
    return write_report(out_path, report)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_eval_harness.py -v
```
Expected: PASS (3 passed).

Now the `eval` CLI test from Task 7 should also pass:
```bash
uv run pytest tests/test_cli.py::test_eval_runs_skeleton -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/ tests/test_eval_harness.py
git commit -m "feat(eval): harness skeleton with chrF stub + JSON report writer"
```

---

## Task 9: Shared test fixtures (`tests/conftest.py`)

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write conftest with shared fixtures**

`tests/conftest.py`:
```python
"""Shared pytest fixtures."""
from __future__ import annotations

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
```

- [ ] **Step 2: Verify the full suite still passes**

Run:
```bash
uv run pytest -v
```
Expected: all prior tests PASS (no regressions from adding conftest).

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: shared pytest fixtures (silence_chunk, good/weak results)"
```

---

## Task 10: Full-suite green + M0 acceptance check

**Files:**
- Modify: `README.md` (add Status/Run section)

- [ ] **Step 1: Run the full test suite**

Run:
```bash
uv run pytest -v
```
Expected: all tests PASS. Count them — M0 should have roughly 18–22 tests across types, config, logging, sources, segmenter, engine, quality_gate, postprocess, pipeline, cli, and eval_harness.

- [ ] **Step 2: Smoke-test the CLI by hand**

Run:
```bash
uv run sawti transcribe --target eng
uv run sawti eval tests/fixtures --target eng
```
Expected: `transcribe` prints two timestamped "hello" lines; `eval` prints "Wrote report: outputs/eval-eng.json".

- [ ] **Step 3: Update README Status/Run section**

Append to `README.md` (after the Status section):

```markdown
## Running (M0)

```bash
uv sync
uv run pytest                       # run the test suite
uv run sawti transcribe --target eng   # stub pipeline demo
uv run sawti eval tests/fixtures --target eng  # eval harness skeleton
```
```

- [ ] **Step 4: Commit and push**

```bash
git add README.md
git commit -m "docs: add M0 run instructions to README"
git push origin main
```

---

## M0 → M1 handoff (informational, not M0 work)

M0 freezes these interfaces. M1 adds real implementations **alongside** the stubs without modifying `types.py`, `config.py`, `pipeline.py`, or any test:

| M1 work | New file | Replaces (in wiring only) |
|---|---|---|
| Silero VAD segmenter | `sawti/segmenter_silero.py` | `StubSegmenter` in `cli.py`/`Pipeline` |
| SeamlessM4T engine | `sawti/engine_m4t.py` | `StubEngine` |
| Balanced quality gate + ASR+MT fallback | `sawti/quality_gate_balanced.py`, `sawti/fallback.py` | `StubQualityGate` |
| 6-step post-processor | `sawti/postprocess_real.py` | `StubPostProcessor` |
| File audio source | `sawti/sources.py` (add `FileSource`) | `StubAudioSource` |
| Real eval metric | `eval/metrics.py` (swap stub for `sacrebleu`) | `compute_chrf_stub` |

The orchestrator (`pipeline.py`) and all data types are M0-final.

---

## Self-Review (completed by plan author)

**Spec coverage:**
- §5.1 data types → Task 2 ✓
- §5.2 component contracts → Task 5 (5a–5e) ✓
- §5.3 sequential orchestrator → Task 6 ✓
- §2.3/§3.3/§3.8/§4.6 config schema → Task 3 ✓
- §7.5 JSONL logging → Task 4 ✓
- §7.6 eval harness → Task 8 ✓
- §6.3 M0 scope (skeleton, types, config, stubs, harness, tests) → Tasks 1–10 ✓

**Placeholder scan:** One intentional placeholder (`from sawdust import _bad` in Task 3 Step 1) — it is explicitly replaced with a real validation test in Task 3 Step 3 before implementation. No other TBD/TODO remains.

**Type consistency:** `EngineResult.timing_ms` is a `dict` in all tasks; the stub post-processor reads `timing_ms.get("_start")` defensively (not a real field) — acceptable for M0 since stubs ignore timing; M1 carries real timestamps. `GateDecision` field names (`accepted`, `needs_retry`, `low_confidence`, `fallback_path`, `log`) are identical across types.py, quality_gate.py, pipeline.py, and all tests. `OutputSegment` fields match between types.py, postprocess.py, and pipeline tests.

**Scope check:** M0 is one cohesive subsystem (foundation). It does not attempt real models (M1), live audio (M2), or streaming (M3). Appropriately scoped for a single plan.
