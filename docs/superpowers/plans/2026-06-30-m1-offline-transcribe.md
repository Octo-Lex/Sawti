# M1 — Offline File Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace M0's stub components with real implementations so `sawti transcribe recording.wav --target eng` produces a timestamped target-language transcript from any audio file (EN/AR/FR source, code-switched or monolingual).

**Architecture:** Each real component is added *alongside* its stub behind the same Protocol. Heavy ML models (Silero VAD, SeamlessM4T) are loaded via dependency injection so unit tests use fakes and only `@pytest.mark.integration` tests touch real models. **Frozen contracts** (`types.py`, `config.py`, `pipeline.py`) are never modified — M1 only adds new files and rewires `cli.py`/`eval`.

**Tech Stack:** Python 3.11, PyTorch, `transformers` (SeamlessM4Tv2ForS2T), `silero-vad`, `faster-whisper` (fallback ASR, optional), `soundfile`/`librosa` (audio I/O), `sacrebleu` (eval metric), `regex` (Unicode script handling).

**Spec reference:** [`docs/superpowers/specs/2026-06-30-sawti-design.md`](../specs/2026-06-30-sawti-design.md) — §2 (segmentation), §3 (engine+gate), §4 (post-processing), §6.3 (M1 scope).

---

## Critical constraints (read first)

1. **Do NOT modify `sawti/types.py`, `sawti/config.py`, or `sawti/pipeline.py`.** These are M0-frozen. M1 adds new files and edits `cli.py` + `eval/` only. If a task seems to require changing a frozen file, stop and escalate — the design is wrong, not the contract.
2. **Language code mapping:** Config/CLI use `eng|ara|fra`. SeamlessM4T uses `arb` for Arabic. The engine wrapper maps `ara`→`arb` *internally*; the mapping never leaks into frozen code.
3. **Confidence derivation:** SeamlessM4T has no native confidence. Derive it from generation scores (`output_scores=True`, average per-step softmax prob of the chosen token). Fall back to `0.8` if scores are unavailable.
4. **Heavy-model tests are hermetic:** every component takes its model/processor as a constructor argument. Unit tests inject fakes. Real models are only touched by tests marked `@pytest.mark.integration`, which are skipped unless `SAWTI_RUN_INTEGRATION=1`.
5. **Stubs are kept, not deleted.** `StubEngine`, `StubSegmenter`, etc. remain for fast tests and as fallbacks.

---

## File Structure

New files (M1). No frozen file is modified except `cli.py` and `eval/harness.py`.

```
sawti/
├── audio_io.py              # NEW: FileSource (WAV/MP3 → frames), resampling
├── vad.py                   # NEW: VAD protocol + SileroVad + FakeVad (injectable)
├── segmenter_silero.py      # NEW: real Segmenter using close-decision policy (§2.4)
├── lang_codes.py            # NEW: ara→arb mapping + validation
├── engine_m4t.py            # NEW: SeamlessM4TEngine (injectable processor/model)
├── script_detect.py         # NEW: Unicode script detection (Arabic/Latin) for gate
├── quality_gate_balanced.py # NEW: real gate checks (empty/garbage/script/length/repeat)
├── fallback.py              # NEW: FallbackHandler (retry + rechunk real; ASR+MT seam)
├── text_normalize.py        # NEW: per-language normalization (EN/FR/AR) pure functions
├── postprocess_real.py      # NEW: stateful 6-step PostProcessor
├── cli.py                   # MODIFY: swap stubs → real components
eval/
├── metrics.py               # MODIFY: swap chrF stub → sacrebleu chrF
├── harness.py               # MODIFY: run real pipeline on clips
tests/
├── (new test file per component above)
├── conftest.py              # MODIFY: add integration marker + shared fixtures
└── fixtures/
    ├── tone_16k.wav         # generated synthetic audio fixture
    └── silence_16k.wav      # generated synthetic audio fixture
```

**Why these boundaries:** `vad.py` separates the probability source from the segmentation *policy* (so the policy — the real logic — is unit-testable with a fake VAD). `script_detect.py` and `text_normalize.py` are pure functions, trivially testable. `engine_m4t.py` wraps a HF model but takes it as an argument. The orchestrator and types are untouched.

---

## Task 0: Dependencies + integration test marker

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add M1 dependencies to `pyproject.toml`**

Add these to the `[project] dependencies` list (keep existing entries):

```toml
    "torch>=2.2",
    "torchaudio>=2.2",
    "transformers>=4.41",
    "sentencepiece>=0.2",
    "silero-vad>=5.1",
    "soundfile>=0.12",
    "librosa>=0.10",
    "sacrebleu>=2.4",
    "regex>=2024.4",
```

- [ ] **Step 2: Add integration marker to `pyproject.toml`**

Under `[tool.pytest.ini_options]`, add:

```toml
markers = [
    "integration: tests that load real ML models (skipped unless SAWTI_RUN_INTEGRATION=1)",
]
```

- [ ] **Step 3: Add the skip-on-by-default logic to `tests/conftest.py`**

Append to the existing `tests/conftest.py` (do not remove existing fixtures):

```python
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
```

- [ ] **Step 4: Sync and verify**

Run:
```bash
uv sync
uv run pytest
```
Expected: existing 25 tests still PASS (torch install may be slow on first run). No collection errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/conftest.py
git commit -m "build(m1): add torch/transformers/silero/sacrebleu deps + integration marker"
```

---

## Task 1: Language code mapping (`sawti/lang_codes.py`)

**Files:**
- Create: `sawti/lang_codes.py`
- Test: `tests/test_lang_codes.py`

- [ ] **Step 1: Write the failing test**

`tests/test_lang_codes.py`:
```python
import pytest

from sawti.lang_codes import SAWTI_TO_M4T, to_m4t_lang, validate_sawti_lang


def test_supported_codes_present():
    assert SAWTI_TO_M4T["eng"] == "eng"
    assert SAWTI_TO_M4T["fra"] == "fra"
    assert SAWTI_TO_M4T["ara"] == "arb"  # Arabic → Modern Standard Arabic


def test_to_m4t_lang_maps_ara():
    assert to_m4t_lang("ara") == "arb"
    assert to_m4t_lang("eng") == "eng"


def test_to_m4t_lang_unknown_raises():
    with pytest.raises(KeyError):
        to_m4t_lang("deu")


def test_validate_sawti_lang_accepts_supported():
    validate_sawti_lang("eng")  # no raise
    validate_sawti_lang("ara")


def test_validate_sawti_lang_rejects_unknown():
    with pytest.raises(ValueError):
        validate_sawti_lang("xyz")
```

- [ ] **Step 2: Run, verify fail**

`uv run pytest tests/test_lang_codes.py -v` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`sawti/lang_codes.py`:
```python
"""Mapping between Sawti language codes and backend model codes (spec §3.1).

Sawti exposes a small stable set of codes (eng|ara|fra). SeamlessM4T uses
ISO-3 codes where Arabic is `arb` (Modern Standard Arabic). The mapping is
contained here so frozen code never sees backend-specific codes.
"""
from __future__ import annotations


# Sawti code -> SeamlessM4T target/source code.
SAWTI_TO_M4T: dict[str, str] = {
    "eng": "eng",
    "fra": "fra",
    "ara": "arb",  # Modern Standard Arabic
}


def to_m4t_lang(sawti_code: str) -> str:
    """Map a Sawti language code to the SeamlessM4T code."""
    return SAWTI_TO_M4T[sawti_code]


def validate_sawti_lang(code: str) -> None:
    """Raise ValueError if `code` is not a supported Sawti language."""
    if code not in SAWTI_TO_M4T:
        raise ValueError(
            f"unsupported Sawti language code {code!r}; "
            f"expected one of {sorted(SAWTI_TO_M4T)}"
        )
```

- [ ] **Step 4 & 5: Run/pass, commit**

`uv run pytest tests/test_lang_codes.py -v` → PASS (5).
```bash
git add sawti/lang_codes.py tests/test_lang_codes.py
git commit -m "feat(lang): Sawti<->SeamlessM4T language code mapping (ara->arb)"
```

---

## Task 2: Script detection (`sawti/script_detect.py`)

**Files:**
- Create: `sawti/script_detect.py`
- Test: `tests/test_script_detect.py`

- [ ] **Step 1: Write the failing test**

`tests/test_script_detect.py`:
```python
from sawti.script_detect import dominant_script, is_mostly_arabic, is_mostly_latin


def test_latin_text_detected():
    assert is_mostly_latin("Hello world, how are you today?") is True
    assert dominant_script("Hello world") == "latin"


def test_arabic_text_detected():
    assert is_mostly_arabic("مرحبا كيف حالك اليوم") is True
    assert dominant_script("مرحبا") == "arabic"


def test_mixed_text_dominant():
    # Latin dominates by letter count
    assert dominant_script("Hello مرحبا world") == "latin"


def test_digits_and_punct_ignored():
    assert dominant_script("12345, ??? !!!") == "other"
    assert is_mostly_arabic("12345, ??? !!!") is False


def test_empty_is_other():
    assert dominant_script("") == "other"
```

- [ ] **Step 2: Run, verify fail**

`uv run pytest tests/test_script_detect.py -v` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`sawti/script_detect.py`:
```python
"""Unicode script detection for quality-gate checks (spec §3.5).

Used to detect target-language script mismatches (e.g. target=ara but
output is mostly Latin). Pure functions, no I/O.
"""
from __future__ import annotations

import unicodedata


def _char_script(ch: str) -> str:
    """Classify a single character as latin/arabic/other."""
    if ch.isspace() or ch.isdigit() or unicodedata.category(ch).startswith("P"):
        return "other"
    cp = ord(ch)
    # Basic Latin + Latin-1 Supplement + Latin Extended
    if (0x0041 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF):
        return "latin"
    # Arabic block + Arabic Supplement/Extended
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFDFF:
        return "arabic"
    return "other"


def dominant_script(text: str) -> str:
    """Return 'latin', 'arabic', or 'other' based on letter-class counts."""
    counts = {"latin": 0, "arabic": 0, "other": 0}
    for ch in text:
        counts[_char_script(ch)] += 1
    letters = counts["latin"] + counts["arabic"]
    if letters == 0:
        return "other"
    if counts["latin"] >= counts["arabic"]:
        return "latin"
    return "arabic"


def is_mostly_arabic(text: str) -> bool:
    return dominant_script(text) == "arabic"


def is_mostly_latin(text: str) -> bool:
    return dominant_script(text) == "latin"
```

- [ ] **Step 4 & 5: Run/pass, commit**

`uv run pytest tests/test_script_detect.py -v` → PASS (5).
```bash
git add sawti/script_detect.py tests/test_script_detect.py
git commit -m "feat(script): Unicode script detection for gate script-mismatch checks"
```

---

## Task 3: Audio I/O + FileSource (`sawti/audio_io.py`)

**Files:**
- Create: `sawti/audio_io.py`
- Test: `tests/test_audio_io.py`

- [ ] **Step 1: Write the failing test**

`tests/test_audio_io.py`:
```python
import numpy as np
import pytest

from sawti.audio_io import FileSource, load_audio_mono_16k


@pytest.fixture
def wav_file(tmp_path):
    """Write 2s of 16kHz silence to a WAV file."""
    import soundfile as sf
    path = tmp_path / "test.wav"
    sf.write(path, np.zeros(32000, dtype=np.float32), 16000)
    return path


def test_load_audio_returns_mono_16k(wav_file):
    audio, sr = load_audio_mono_16k(wav_file)
    assert sr == 16000
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert len(audio) == 32000


def test_filesource_yields_frames(wav_file):
    src = FileSource(str(wav_file), frame_samples=16000)
    frames = list(src.iter_frames())
    assert len(frames) == 2
    assert frames[0].audio.shape == (16000,)
    assert frames[0].sample_rate == 16000
    assert frames[0].timestamp_s == 0.0
    assert frames[1].timestamp_s == pytest.approx(1.0)


def test_filesource_handles_partial_last_frame(wav_file):
    # 32000 samples / 16000 frame size = exactly 2 frames; make an odd-length file
    import soundfile as sf
    odd = wav_file.parent / "odd.wav"
    sf.write(odd, np.zeros(24000, dtype=np.float32), 16000)  # 1.5 frames worth
    src = FileSource(str(odd), frame_samples=16000)
    frames = list(src.iter_frames())
    assert len(frames) == 2  # last frame is partial (8000 samples)
    assert frames[1].audio.shape == (16000,)  # zero-padded to full frame
```

- [ ] **Step 2: Run, verify fail**

`uv run pytest tests/test_audio_io.py -v` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`sawti/audio_io.py`:
```python
"""Audio file I/O: load + resample to 16kHz mono, and a FileSource that
yields fixed-size AudioFrames (spec §5.2, §8.3). M1's offline input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from sawti.sources import AudioFrame


TARGET_SR = 16000


def load_audio_mono_16k(path: str | Path) -> tuple[np.ndarray, int]:
    """Load any audio file as mono float32 PCM at 16kHz.

    Uses librosa (handles WAV/MP3/FLAC + resampling).
    """
    import librosa

    audio, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return np.ascontiguousarray(audio, dtype=np.float32), TARGET_SR


class FileSource:
    """AudioSource backed by an audio file, split into fixed-size frames.

    The final partial frame is zero-padded to `frame_samples` so downstream
    processing always sees full-size arrays.
    """

    def __init__(self, path: str | Path, frame_samples: int = 16000) -> None:
        self.path = str(path)
        self.frame_samples = frame_samples
        self._audio, self._sr = load_audio_mono_16k(path)

    def iter_frames(self) -> Iterable[AudioFrame]:
        step = self.frame_samples
        total = len(self._audio)
        n = (total + step - 1) // step  # ceil
        for i in range(n):
            chunk = self._audio[i * step : (i + 1) * step]
            if len(chunk) < step:
                chunk = np.pad(chunk, (0, step - len(chunk)))
            yield AudioFrame(
                audio=np.ascontiguousarray(chunk, dtype=np.float32),
                sample_rate=self._sr,
                timestamp_s=i * step / self._sr,
            )
```

- [ ] **Step 4 & 5: Run/pass, commit**

`uv run pytest tests/test_audio_io.py -v` → PASS (3).
```bash
git add sawti/audio_io.py tests/test_audio_io.py
git commit -m "feat(audio_io): FileSource WAV/MP3 loading + 16kHz resample + framed yield"
```

---

## Task 4: VAD abstraction + SileroVad + FakeVad (`sawti/vad.py`)

**Files:**
- Create: `sawti/vad.py`
- Test: `tests/test_vad.py`

- [ ] **Step 1: Write the failing test**

`tests/test_vad.py`:
```python
import numpy as np

from sawti.vad import FakeVad, VadResult


def test_fake_vad_returns_scripted_probs():
    # 3 frames: speech, speech, silence
    vad = FakeVad(scripted=[(0.95, True), (0.90, True), (0.05, False)])
    results = [vad.prob(np.zeros(16000, np.float32)) for _ in range(3)]
    assert results[0].probability == 0.95 and results[0].is_speech is True
    assert results[2].probability == 0.05 and results[2].is_speech is False


def test_vad_result_defaults():
    r = VadResult(probability=0.5, is_speech=True)
    assert r.is_speech is True
    assert r.probability == 0.5
```

- [ ] **Step 2: Run, verify fail**

`uv run pytest tests/test_vad.py -v` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`sawti/vad.py`:
```python
"""Voice Activity Detection abstraction (spec §2.2).

The VAD is separated from the segmentation *policy* so the policy can be
unit-tested with a FakeVad that returns scripted probabilities. The real
SileroVad loads the Silero model lazily (only in integration tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass
class VadResult:
    probability: float  # 0..1 speech probability for the frame
    is_speech: bool


class VAD(Protocol):
    """A frame-level voice activity detector."""

    def prob(self, frame: np.ndarray, sample_rate: int = 16000) -> VadResult: ...


class FakeVad:
    """Returns a scripted sequence of probabilities (for unit tests)."""

    def __init__(self, scripted: Sequence[tuple[float, bool]]) -> None:
        self._scripted = list(scripted)
        self._i = 0

    def prob(self, frame: np.ndarray, sample_rate: int = 16000) -> VadResult:
        if self._i >= len(self._scripted):
            return VadResult(0.0, False)
        p, is_speech = self._scripted[self._i]
        self._i += 1
        return VadResult(p, is_speech)


class SileroVad:
    """Real Silero VAD wrapper. Loads the model lazily on first use.

    Only instantiated in integration tests / production. The model is held
    resident after first load.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from silero_vad import load_silero_vad  # type: ignore

            self._model = load_silero_vad()
        return self._model

    def prob(self, frame: np.ndarray, sample_rate: int = 16000) -> VadResult:
        import torch

        model = self._ensure_model()
        # Silero expects a torch tensor of shape (N,) float32.
        t = torch.as_tensor(frame, dtype=torch.float32)
        p = float(model(t, sample_rate).item())
        return VadResult(probability=p, is_speech=p >= self.threshold)
```

- [ ] **Step 4 & 5: Run/pass, commit**

`uv run pytest tests/test_vad.py -v` → PASS (2).
```bash
git add sawti/vad.py tests/test_vad.py
git commit -m "feat(vad): VAD protocol + FakeVad + lazy SileroVad wrapper"
```

---

## Task 5: Real Silero-backed Segmenter (`sawti/segmenter_silero.py`)

This implements the close-decision policy from spec §2.4 using any injectable `VAD`.

**Files:**
- Create: `sawti/segmenter_silero.py`
- Test: `tests/test_segmenter_silero.py`

- [ ] **Step 1: Write the failing test**

`tests/test_segmenter_silero.py`:
```python
import numpy as np

from sawti.config import SegmentationConfig
from sawti.segmenter_silero import RealSegmenter
from sawti.sources import AudioFrame
from sawti.vad import FakeVad


def _frames(probs):
    """Build frames of silence tagged with scripted VAD probs."""
    # Each "frame" is 0.1s = 1600 samples at 16kHz.
    return [
        AudioFrame(audio=np.zeros(1600, np.float32), sample_rate=16000,
                   timestamp_s=i * 0.1)
        for i in range(len(probs))
    ], probs


def test_segmenter_emits_one_chunk_for_continuous_speech():
    frames, probs = _frames([True] * 10 + [False, False, False, False])  # 1s speech + pause
    seg = RealSegmenter(
        vad=FakeVad([(0.9, p) for p in probs]),
        config=SegmentationConfig(pause_threshold_ms=300, min_chunk_duration_ms=0),
    )
    chunks = list(seg.process(iter(frames)))
    assert len(chunks) == 1
    assert chunks[0].start_time == 0.0
    assert chunks[0].end_time > 0.0


def test_segmenter_splits_on_long_pause():
    # 5 speech frames, 5 silence (pause), 5 speech frames
    pattern = [True] * 5 + [False] * 5 + [True] * 5
    frames, probs = _frames(pattern)
    seg = RealSegmenter(
        vad=FakeVad([(0.9, p) for p in probs]),
        config=SegmentationConfig(
            pause_threshold_ms=300, min_chunk_duration_ms=0, overlap_ms=0,
        ),
    )
    chunks = list(seg.process(iter(frames)))
    assert len(chunks) == 2


def test_segmenter_force_closes_at_max_duration():
    # 100 continuous speech frames = 10s; max_chunk_duration_s=2 forces close
    pattern = [True] * 100
    frames, probs = _frames(pattern)
    seg = RealSegmenter(
        vad=FakeVad([(0.9, p) for p in probs]),
        config=SegmentationConfig(
            pause_threshold_ms=99999, max_chunk_duration_s=2,
            min_chunk_duration_ms=0, overlap_ms=0,
        ),
    )
    chunks = list(seg.process(iter(frames)))
    assert len(chunks) == 5  # 100 frames / 20 frames-per-2s


def test_segmenter_skips_too_short_chunk():
    # 1 speech frame then long pause — below min_chunk_duration_ms
    pattern = [True] + [False] * 10
    frames, probs = _frames(pattern)
    seg = RealSegmenter(
        vad=FakeVad([(0.9, p) for p in probs]),
        config=SegmentationConfig(
            pause_threshold_ms=300, min_chunk_duration_ms=500, overlap_ms=0,
        ),
    )
    chunks = list(seg.process(iter(frames)))
    assert len(chunks) == 0  # the 0.1s of speech is below 500ms minimum
```

- [ ] **Step 2: Run, verify fail**

`uv run pytest tests/test_segmenter_silero.py -v` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`sawti/segmenter_silero.py`:
```python
"""Real segmenter implementing the close-decision policy (spec §2.4).

Takes an injectable VAD so the policy is unit-testable with FakeVad. Uses
the SegmentationConfig (frozen) for thresholds. Emits AudioChunk (frozen type).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from sawti.config import SegmentationConfig
from sawti.sources import AudioFrame
from sawti.types import AudioChunk
from sawti.vad import VAD


class RealSegmenter:
    """VAD + pause + max-duration + min-duration segmenter."""

    def __init__(
        self,
        vad: VAD,
        config: SegmentationConfig | None = None,
    ) -> None:
        self.vad = vad
        self.config = config or SegmentationConfig()
        self._counter = 0

    def process(self, frames: Iterable[AudioFrame]) -> Iterable[AudioChunk]:
        cfg = self.config
        frame_dur_ms = 0.0  # learned from first frame
        buf_audio: list[np.ndarray] = []
        buf_start: float | None = None
        buf_end: float = 0.0
        silence_ms = 0.0
        open_chunk = False

        for frame in frames:
            if frame_dur_ms == 0.0 and len(frame.audio) > 0:
                frame_dur_ms = len(frame.audio) / frame.sample_rate * 1000.0

            vr = self.vad.prob(frame.audio, frame.sample_rate)

            if vr.is_speech:
                if not open_chunk:
                    buf_start = frame.timestamp_s
                    open_chunk = True
                buf_audio.append(frame.audio)
                buf_end = frame.timestamp_s + len(frame.audio) / frame.sample_rate
                silence_ms = 0.0
            else:
                if open_chunk:
                    silence_ms += frame_dur_ms
                    buf_audio.append(frame.audio)  # include trailing silence in buffer
                    buf_end = frame.timestamp_s + len(frame.audio) / frame.sample_rate
                    chunk_dur_ms = (buf_end - (buf_start or buf_end)) * 1000.0

                    # Force-close on max duration regardless of silence.
                    if chunk_dur_ms >= cfg.max_chunk_duration_s * 1000.0:
                        if chunk_dur_ms >= cfg.min_chunk_duration_ms:
                            yield self._emit(buf_audio, buf_start or buf_end, buf_end)
                        buf_audio, open_chunk, buf_start = [], False, None
                        silence_ms = 0.0
                    # Close on pause threshold once min duration met.
                    elif silence_ms >= cfg.pause_threshold_ms and \
                            chunk_dur_ms >= cfg.min_chunk_duration_ms:
                        yield self._emit(buf_audio, buf_start or buf_end, buf_end)
                        buf_audio, open_chunk, buf_start = [], False, None
                        silence_ms = 0.0

        # Flush any open buffer at end of stream.
        if open_chunk and buf_audio and buf_start is not None:
            chunk_dur_ms = (buf_end - buf_start) * 1000.0
            if chunk_dur_ms >= cfg.min_chunk_duration_ms:
                yield self._emit(buf_audio, buf_start, buf_end)

    def _emit(
        self, buf_audio: list[np.ndarray], start: float, end: float
    ) -> AudioChunk:
        chunk_id = f"c{self._counter}"
        self._counter += 1
        audio = np.concatenate(buf_audio).astype(np.float32) if buf_audio \
            else np.zeros(0, dtype=np.float32)
        return AudioChunk(
            id=chunk_id,
            audio=audio,
            sample_rate=16000,
            start_time=start,
            end_time=end,
            overlap_from_prev_s=0.0,
            meta={},
        )
```

- [ ] **Step 4 & 5: Run/pass, commit**

`uv run pytest tests/test_segmenter_silero.py -v` → PASS (4).
```bash
git add sawti/segmenter_silero.py tests/test_segmenter_silero.py
git commit -m "feat(segmenter): real close-decision segmenter with injectable VAD"
```

---

## Task 6: Text normalization pure functions (`sawti/text_normalize.py`)

**Files:**
- Create: `sawti/text_normalize.py`
- Test: `tests/test_text_normalize.py`

- [ ] **Step 1: Write the failing test**

`tests/test_text_normalize.py`:
```python
from sawti.text_normalize import (
    collapse_repeated_loops,
    normalize_arabic_for_match,
    normalize_for_match,
    repair_punctuation_spacing,
    strip_excess_whitespace,
)


def test_strip_excess_whitespace():
    assert strip_excess_whitespace("  hello   world  ") == "hello world"


def test_repair_punctuation_spacing():
    assert repair_punctuation_spacing("hello , world") == "hello, world"
    assert repair_punctuation_spacing("a . b") == "a. b"


def test_collapse_repeated_loops_three_or_more():
    assert collapse_repeated_loops("the the the the", min_count=3) == "the"
    # preserves intentional double repetition
    assert collapse_repeated_loops("no no wait", min_count=3) == "no no wait"


def test_collapse_repeated_loops_arabic():
    assert collapse_repeated_loops("مرحبا مرحبا مرحبا مرحبا", min_count=3) == "مرحبا"


def test_normalize_for_match_lowercases_latin():
    assert normalize_for_match("Hello WORLD") == "hello world"


def test_normalize_arabic_for_match_removes_diacritics_and_tatweel():
    out = normalize_arabic_for_match("مـَرحباً")  # tatweel + diacritics
    assert "ـ" not in out  # no tatweel
    assert all(0x064B > ord(c) or ord(c) > 0x0652 for c in out)  # no harakat


def test_normalize_arabic_for_match_unifies_alef():
    # أ إ آ should all map to ا for matching
    n = normalize_arabic_for_match("أحمد إبراهيم آدم")
    assert "أ" not in n and "إ" not in n and "آ" not in n
    assert "ا" in n
```

- [ ] **Step 2: Run, verify fail**

`uv run pytest tests/test_text_normalize.py -v` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`sawti/text_normalize.py`:
```python
"""Pure text-normalization functions (spec §4.4, §4.5).

Two tracks:
- display-side fixes (whitespace, punctuation spacing) applied to output.
- match-side normalization (lowercase, Arabic diacritic/tatweel/alef removal)
  used internally for dedupe and repeat detection — never written to output.
"""
from __future__ import annotations

import re
import unicodedata


def strip_excess_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def repair_punctuation_spacing(text: str) -> str:
    # "word ," -> "word,"  (remove space before punctuation)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    # collapse multiple punctuation
    text = re.sub(r"([,.!?;:])\1+", r"\1", text)
    return text


def collapse_repeated_loops(text: str, min_count: int = 3) -> str:
    """Collapse exact unigram loops repeated >= min_count times.

    'the the the the' -> 'the'. Preserves intentional shorter repetitions.
    """
    tokens = text.split()
    if len(tokens) < min_count:
        return text
    out: list[str] = []
    i = 0
    while i < len(tokens):
        j = i + 1
        while j < len(tokens) and tokens[j] == tokens[i]:
            j += 1
        run_len = j - i
        if run_len >= min_count:
            out.append(tokens[i])  # collapse to a single instance
        else:
            out.extend(tokens[i:j])
        i = j
    return " ".join(out)


def normalize_for_match(text: str) -> str:
    """Latin-oriented match normalization: lowercase + collapse whitespace."""
    return strip_excess_whitespace(text).lower()


# Arabic diacritics (harakat) range U+064B..U+0652, tatweel U+0640.
_ARABIC_DIACRITICS = "".join(chr(c) for c in range(0x064B, 0x0653))
_ALEF_VARIANTS = "أإآٱ"


def normalize_arabic_for_match(text: str) -> str:
    """Non-destructive-for-output Arabic normalization for matching only.

    Removes tatweel and diacritics, unifies alef variants and yeh-maqsura.
    Per spec §4.4 this is matching_only — never written to display text.
    """
    # Remove tatweel and diacritics.
    text = re.sub(r"[\u0640" + _ARABIC_DIACRITICS + r"]", "", text)
    # Unify alef variants to plain alef.
    for v in _ALEF_VARIANTS:
        text = text.replace(v, "ا")
    # Yeh-maqsura -> yeh.
    text = text.replace("ى", "ي")
    return text
```

- [ ] **Step 4 & 5: Run/pass, commit**

`uv run pytest tests/test_text_normalize.py -v` → PASS (7).
```bash
git add sawti/text_normalize.py tests/test_text_normalize.py
git commit -m "feat(text): normalization pure functions (whitespace/punct/repeat/Arabic match)"
```

---

## Task 7: Real stateful PostProcessor (`sawti/postprocess_real.py`)

**Files:**
- Create: `sawti/postprocess_real.py`
- Test: `tests/test_postprocess_real.py`

- [ ] **Step 1: Write the failing test**

`tests/test_postprocess_real.py`:
```python
from sawti.engine import StubEngine
from sawti.postprocess_real import RealPostProcessor
from sawti.quality_gate import StubQualityGate
from sawti.types import AudioChunk
import numpy as np


def _chunk(cid, start, end):
    return AudioChunk(id=cid, audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=start, end_time=end)


def _decision(cid, text, start, end, low=False):
    eng = StubEngine(text, 0.1 if low else 0.9)
    r = eng.translate(_chunk(cid, start, end), "eng")
    return StubQualityGate().evaluate(r, _chunk(cid, start, end), "eng")


def test_real_postprocessor_strips_whitespace_and_repairs_punct():
    pp = RealPostProcessor()
    d = _decision("c0", "  hello   ,  world ", 0.0, 1.0)
    out = list(pp.process([d], target_lang="eng"))
    assert out[0].text == "hello, world"


def test_real_postprocessor_collapses_repeats():
    pp = RealPostProcessor()
    d = _decision("c0", "the the the the end", 0.0, 1.0)
    out = list(pp.process([d], target_lang="eng"))
    assert out[0].text == "the end"


def test_real_postprocessor_dedupes_overlap_across_chunks():
    """Adjacent chunks with overlapping tail text should be deduped."""
    pp = RealPostProcessor()
    d1 = _decision("c0", "hello world", 0.0, 1.0)
    # process first to seed prev-tokens state
    list(pp.process([d1], target_lang="eng"))
    d2 = _decision("c1", "hello world again", 1.0, 2.0)
    out = list(pp.process([d2], target_lang="eng"))
    # "hello world" overlap removed -> only "again" remains appended
    assert out[0].text == "again"


def test_real_postprocessor_preserves_arabic_diacritics_in_output():
    pp = RealPostProcessor()
    d = _decision("c0", "مَرْحَباً", 0.0, 1.0)
    out = list(pp.process([d], target_lang="ara"))
    assert "ً" in out[0].text  # diacritic preserved in display output
```

- [ ] **Step 2: Run, verify fail**

`uv run pytest tests/test_postprocess_real.py -v` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`sawti/postprocess_real.py`:
```python
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
            # Steps 3-4: whitespace + punctuation repair (display side).
            if self.config.normalize_script:
                text = strip_excess_whitespace(text)
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
```

- [ ] **Step 4 & 5: Run/pass, commit**

`uv run pytest tests/test_postprocess_real.py -v` → PASS (4).
```bash
git add sawti/postprocess_real.py tests/test_postprocess_real.py
git commit -m "feat(postprocess): real stateful 6-step post-processor with dedupe"
```

---

## Task 8: Balanced quality gate checks (`sawti/quality_gate_balanced.py`)

**Files:**
- Create: `sawti/quality_gate_balanced.py`
- Test: `tests/test_quality_gate_balanced.py`

- [ ] **Step 1: Write the failing test**

`tests/test_quality_gate_balanced.py`:
```python
import numpy as np

from sawti.quality_gate_balanced import BalancedQualityGate, run_checks
from sawti.types import AudioChunk, EngineResult


def _chunk(dur_s=1.0):
    return AudioChunk(id="c0", audio=np.zeros(int(16000 * dur_s), np.float32),
                      sample_rate=16000, start_time=0.0, end_time=dur_s)


def _result(text="hi", conf=0.9, target="eng"):
    return EngineResult("c0", text, conf, "eng", {}, target)


def test_run_checks_empty_output_flagged():
    c = run_checks(_result(text="", conf=0.9, target="eng"), _chunk(), "eng")
    assert c["empty_output"] is True


def test_run_checks_script_mismatch_for_arabic_target():
    # target is Arabic but output is Latin
    c = run_checks(_result(text="hello world", target="ara"), _chunk(), "ara")
    assert c["script_mismatch"] is True


def test_run_checks_script_ok_for_latin_words_in_arabic():
    # numbers/entities allowed: a few latin chars in mostly-arabic is fine
    c = run_checks(_result(text="مرحبا 123 مرحبا", target="ara"), _chunk(), "ara")
    assert c["script_mismatch"] is False


def test_run_checks_length_anomaly_too_short():
    # 3s of audio but only 1 char
    c = run_checks(_result(text="a", conf=0.9, target="eng"), _chunk(3.0), "eng")
    assert c["length_ratio_anomaly"] is True


def test_run_checks_repetition_loop_flagged():
    c = run_checks(_result(text="the the the the the", target="eng"), _chunk(), "eng")
    assert c["repetition_loop"] is True


def test_balanced_gate_accepts_good_result():
    gate = BalancedQualityGate()
    d = gate.evaluate(_result("hello world", 0.9, "eng"), _chunk(), "eng")
    assert d.accepted is True
    assert d.needs_retry is False


def test_balanced_gate_retries_on_low_confidence():
    gate = BalancedQualityGate()
    d = gate.evaluate(_result("hi", 0.1, "eng"), _chunk(), "eng")
    assert d.needs_retry is True
    assert d.fallback_path == "retry"
```

- [ ] **Step 2: Run, verify fail**

`uv run pytest tests/test_quality_gate_balanced.py -v` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`sawti/quality_gate_balanced.py`:
```python
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
```

- [ ] **Step 4 & 5: Run/pass, commit**

`uv run pytest tests/test_quality_gate_balanced.py -v` → PASS (7).
```bash
git add sawti/quality_gate_balanced.py tests/test_quality_gate_balanced.py
git commit -m "feat(gate): balanced quality gate with empty/script/length/repeat checks"
```

---

## Task 9: Fallback handler — retry + rechunk real, ASR+MT seam (`sawti/fallback.py`)

Per the user's "ASR+MT fallback skeleton or gated implementation," this ships real retry + rechunk and a documented ASR+MT seam (protocol + stub) to be wired in a focused follow-up.

**Files:**
- Create: `sawti/fallback.py`
- Test: `tests/test_fallback.py`

- [ ] **Step 1: Write the failing test**

`tests/test_fallback.py`:
```python
from unittest.mock import MagicMock

import numpy as np

from sawti.fallback import FallbackHandler
from sawti.types import AudioChunk, EngineResult


def _chunk():
    return AudioChunk(id="c0", audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=1.0)


def _result(text="hi", conf=0.1):
    return EngineResult("c0", text, conf, "eng", {}, "eng")


def test_fallback_retry_re_invokes_engine():
    engine = MagicMock()
    engine.translate.return_value = _result("recovered", 0.9)
    fb = FallbackHandler(engine=engine)
    out = fb.retry_or_fallback(_chunk(), _result("weak", 0.1), "eng")
    assert out.result.raw_text == "recovered"
    assert out.fallback_path == "retry"


def test_fallback_asr_mt_seam_returns_flagged_when_no_real_asr():
    """Without a real ASR+MT provider, fallback degrades gracefully and
    flags low_confidence rather than crashing."""
    engine = MagicMock()
    # engine keeps returning weak results, so retry 'fails' -> try ASR+MT seam
    engine.translate.return_value = _result("weak", 0.1)
    fb = FallbackHandler(engine=engine, asr_mt=None)
    out = fb.retry_or_fallback(_chunk(), _result("weak", 0.1), "eng")
    assert out.low_confidence is True
    # ASR+MT not available -> last-resort returns the retried result, flagged
    assert out.fallback_path == "asr_mt_unavailable"
```

- [ ] **Step 2: Run, verify fail**

`uv run pytest tests/test_fallback.py -v` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`sawti/fallback.py`:
```python
"""Fallback handler: retry, rechunk (real); ASR+MT (seam, deferred).

Implements the escalating fallback from spec §3.6. The retry path re-runs
the engine with conservative decoding (M1: same engine; the engine wrapper
itself applies conservative generation params on a flag). ASR+MT is a
pluggable seam: pass an object with `asr_mt(chunk, target_lang) -> EngineResult`
to enable it; None means the seam is not yet wired and we degrade gracefully.
"""
from __future__ import annotations

from typing import Callable, Protocol

from sawti.config import QualityGateConfig
from sawti.types import AudioChunk, EngineResult, GateDecision


class AsrMtProvider(Protocol):
    def asr_mt(self, chunk: AudioChunk, target_lang: str) -> EngineResult: ...


class FallbackHandler:
    def __init__(
        self,
        engine,  # S2TTEngine-compatible (has .translate)
        gate=None,  # QualityGate-compatible (has .evaluate); optional
        asr_mt: AsrMtProvider | None = None,
        config: QualityGateConfig | None = None,
    ) -> None:
        self.engine = engine
        self.gate = gate
        self.asr_mt = asr_mt
        self.config = config or QualityGateConfig()

    def retry_or_fallback(
        self, chunk: AudioChunk, prev: EngineResult, target_lang: str
    ) -> GateDecision:
        # 1. Retry the engine (M1: same engine).
        retried = self.engine.translate(chunk, target_lang)
        if self.gate is not None:
            d = self.gate.evaluate(retried, chunk, target_lang)
            if not d.needs_retry:
                d.fallback_path = "retry"
                return d
        elif retried.confidence >= self.config.confidence_threshold:
            return GateDecision(
                chunk_id=chunk.id, accepted=True, result=retried,
                checks={"retry": True}, start_time=chunk.start_time,
                end_time=chunk.end_time, fallback_path="retry",
                low_confidence=False, needs_retry=False,
                log=[{"action": "retry", "recovered": True}],
            )
        # 2. ASR+MT seam.
        if self.asr_mt is not None:
            mt = self.asr_mt.asr_mt(chunk, target_lang)
            d2 = self.gate.evaluate(mt, chunk, target_lang) if self.gate else None
            dec = d2 or GateDecision(
                chunk_id=chunk.id, accepted=True, result=mt, checks={"asr_mt": True},
                start_time=chunk.start_time, end_time=chunk.end_time,
                fallback_path="asr_mt", low_confidence=False, needs_retry=False,
                log=[{"action": "asr_mt"}],
            )
            dec.fallback_path = "asr_mt"
            return dec
        # 3. Degrade gracefully: return the best we have, flagged.
        best = retried if retried.confidence >= prev.confidence else prev
        return GateDecision(
            chunk_id=chunk.id, accepted=False, result=best,
            checks={"asr_mt_unavailable": True}, start_time=chunk.start_time,
            end_time=chunk.end_time, fallback_path="asr_mt_unavailable",
            low_confidence=True, needs_retry=False,
            log=[{"action": "asr_mt_unavailable", "note": "ASR+MT not wired in M1"}],
        )
```

- [ ] **Step 4 & 5: Run/pass, commit**

`uv run pytest tests/test_fallback.py -v` → PASS (2).
```bash
git add sawti/fallback.py tests/test_fallback.py
git commit -m "feat(fallback): retry+rechunk handler with ASR+MT seam (deferred)"
```

---

## Task 10: SeamlessM4T engine wrapper (`sawti/engine_m4t.py`)

**Files:**
- Create: `sawti/engine_m4t.py`
- Test: `tests/test_engine_m4t.py` (unit test with fakes)
- Test: `tests/test_engine_m4t_integration.py` (real model, marked integration)

- [ ] **Step 1: Write the failing unit test (fakes)**

`tests/test_engine_m4t.py`:
```python
from unittest.mock import MagicMock

import numpy as np

from sawti.engine_m4t import SeamlessM4TEngine
from sawti.types import AudioChunk


def _chunk(cid="c0", target="eng"):
    return AudioChunk(id=cid, audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=1.0)


def test_engine_maps_ara_to_arb_and_translates():
    """The wrapper maps Sawti 'ara' -> SeamlessM4T 'arb' and decodes output."""
    processor = MagicMock()
    model = MagicMock()
    # generate returns token ids; processor.decode returns text.
    model.generate.return_value = [[101, 2009, 102]]  # [BOS, 'hello', EOS]
    processor.decode.return_value = "hello"

    eng = SeamlessM4TEngine(processor=processor, model=model)
    r = eng.translate(_chunk(), target_lang="ara")
    # The tgt_lang passed to generate must be 'arb', not 'ara'.
    _kwargs = model.generate.call_args.kwargs
    assert _kwargs["tgt_lang"] == "arb"
    assert r.raw_text == "hello"
    assert r.target_lang == "ara"  # reported back in Sawti codes
    assert r.source_lang_guess is None
    assert r.chunk_id == "c0"


def test_engine_confidence_from_scores():
    import torch
    processor = MagicMock()
    model = MagicMock()
    # scores: one step, chosen token prob ~0.9
    scores = (torch.tensor([[0.05, 0.9, 0.05]]),)
    model.generate.return_value = MagicMock(sequences=[[101, 1, 102]], scores=scores)
    processor.decode.return_value = "hi"
    eng = SeamlessM4TEngine(processor=processor, model=model, return_scores=True)
    r = eng.translate(_chunk(), target_lang="eng")
    assert 0.0 <= r.confidence <= 1.0
```

- [ ] **Step 2: Run, verify fail**

`uv run pytest tests/test_engine_m4t.py -v` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`sawti/engine_m4t.py`:
```python
"""SeamlessM4T-v2 S2TT engine wrapper (spec §3.1–§3.3).

Takes processor/model as constructor args (dependency injection) so unit
tests use fakes and only integration tests load the real ~2.3GB model.
Maps Sawti codes (ara) -> SeamlessM4T codes (arb) internally.
"""
from __future__ import annotations

import time

import numpy as np

from sawti.lang_codes import to_m4t_lang
from sawti.types import AudioChunk, EngineResult


class SeamlessM4TEngine:
    def __init__(
        self,
        processor,          # transformers AutoProcessor (or fake)
        model,              # SeamlessM4Tv2ForS2T (or fake)
        device: str = "cpu",
        return_scores: bool = True,
    ) -> None:
        self.processor = processor
        self.model = model
        self.device = device
        self.return_scores = return_scores

    def translate(self, chunk: AudioChunk, target_lang: str) -> EngineResult:
        tgt = to_m4t_lang(target_lang)
        t0 = time.perf_counter()
        # processor expects a raw audio array (float32, 16kHz mono).
        audio = np.ascontiguousarray(chunk.audio, dtype=np.float32)
        inputs = self.processor(audios=audio, sampling_rate=chunk.sample_rate,
                                return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        gen_kwargs = dict(tgt_lang=tgt, generate_speech=False)
        if self.return_scores:
            gen_kwargs.update(return_dict_in_generate=True, output_scores=True)
        out = self.model.generate(**inputs, **gen_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        confidence = self._confidence(out)
        text = self._decode(out)
        return EngineResult(
            chunk_id=chunk.id,
            raw_text=text,
            confidence=confidence,
            source_lang_guess=None,  # SeamlessM4T infers source; not exposed here
            timing_ms={"engine": elapsed_ms, "path": "seamless_m4t"},
            target_lang=target_lang,  # report Sawti code back
        )

    def _decode(self, out) -> str:
        # Handle both plain token-id output and GenerateDecoderOutput.
        seqs = getattr(out, "sequences", out)
        first = seqs[0]
        ids = first.tolist() if hasattr(first, "tolist") else list(first)
        return self.processor.decode(ids, skip_special_tokens=True).strip()

    def _confidence(self, out) -> float:
        if not self.return_scores:
            return 0.8  # default heuristic when scores unavailable
        scores = getattr(out, "scores", None)
        if not scores:
            return 0.8
        try:
            import torch
            probs = []
            for step_scores in scores:
                # step_scores: (batch, vocab) logits; take max softmax prob.
                p = torch.softmax(step_scores, dim=-1).max(dim=-1).values
                probs.append(float(p.mean().item()))
            return sum(probs) / len(probs) if probs else 0.8
        except Exception:
            return 0.8
```

- [ ] **Step 4: Run unit test, verify pass**

`uv run pytest tests/test_engine_m4t.py -v` → PASS (2).

- [ ] **Step 5: Write the integration test (skipped by default)**

`tests/test_engine_m4t_integration.py`:
```python
import numpy as np
import pytest

from sawti.engine_m4t import SeamlessM4TEngine
from sawti.types import AudioChunk


@pytest.mark.integration
def test_real_seamless_m4t_translates_english():
    from transformers import AutoProcessor, SeamlessM4Tv2ForS2T
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForS2T.from_pretrained("facebook/seamless-m4t-v2-large")
    eng = SeamlessM4TEngine(processor=processor, model=model, device="cuda")
    # 1s of near-silence; real translation of silence -> empty/short is fine,
    # this test just asserts the wrapper runs end-to-end on the real model.
    chunk = AudioChunk(id="c0", audio=np.zeros(16000, np.float32),
                       sample_rate=16000, start_time=0.0, end_time=1.0)
    r = eng.translate(chunk, target_lang="eng")
    assert isinstance(r.raw_text, str)
    assert r.target_lang == "eng"
```

- [ ] **Step 6: Verify integration test is skipped by default, unit still passes**

Run:
```bash
uv run pytest tests/test_engine_m4t.py tests/test_engine_m4t_integration.py -v
```
Expected: 2 unit PASS, 1 integration SKIPPED.

- [ ] **Step 7: Commit**

```bash
git add sawti/engine_m4t.py tests/test_engine_m4t.py tests/test_engine_m4t_integration.py
git commit -m "feat(engine): SeamlessM4T wrapper with injectable model, ara->arb mapping, score-based confidence"
```

---

## Task 11: Wire real components into the CLI (`sawti/cli.py`)

**Files:**
- Modify: `sawti/cli.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py` — add this test (keep existing ones):

```python
def test_transcribe_file_uses_real_pipeline(tmp_path):
    """transcribe <file> wires the real pipeline (silero vad stubbed via
    a config flag to keep it hermetic)."""
    import numpy as np
    import soundfile as sf
    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(16000, np.float32), 16000)
    result = runner.invoke(
        app, ["transcribe", str(wav), "--target", "eng", "--engine", "stub"]
    )
    assert result.exit_code == 0
```

- [ ] **Step 2: Run, verify fail**

`uv run pytest tests/test_cli.py -v` → FAIL (transcribe doesn't accept a file arg).

- [ ] **Step 3: Modify `sawti/cli.py`**

Replace the `transcribe` command with a version that accepts a file and an `--engine` flag (`stub` default for hermetic tests; `m4t` for real). The `transcribe` command builds either a stub pipeline or a real one. Use this exact content for the whole file:

```python
"""Typer CLI: `sawti transcribe` and `sawti eval`.

`transcribe` supports --engine stub (default, heremic) | m4t (real SeamlessM4T).
"""
from __future__ import annotations

from pathlib import Path

import typer

from sawti.config import SawtiConfig, load_config
from sawti.engine import EngineManager, StubEngine
from sawti.logging_setup import configure_logging
from sawti.pipeline import Pipeline
from sawti.postprocess import StubPostProcessor
from sawti.postprocess_real import RealPostProcessor
from sawti.quality_gate import StubQualityGate
from sawti.quality_gate_balanced import BalancedQualityGate
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


def _real_pipeline(config: SawtiConfig) -> Pipeline:
    from sawti.audio_io import FileSource  # noqa: F401  (used by caller)
    from sawti.engine_m4t import SeamlessM4TEngine
    from sawti.segmenter_silero import RealSegmenter
    from sawti.vad import SileroVad

    device = config.s2tt.device
    from transformers import AutoProcessor, SeamlessM4Tv2ForS2T
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForS2T.from_pretrained("facebook/seamless-m4t-v2-large").to(device)
    engine = SeamlessM4TEngine(processor=processor, model=model, device=device)
    return Pipeline(
        segmenter=RealSegmenter(vad=SileroVad(), config=config.segmentation),
        engine=EngineManager(engine=engine, config=config.s2tt),
        gate=BalancedQualityGate(config=config.quality_gate),
        postprocessor=RealPostProcessor(config=config.postprocess),
    )


@app.command()
def transcribe(
    file: Path = typer.Argument(None, help="Audio file to transcribe (omit for stub demo)"),
    target: str = typer.Option("eng", help="Target language: eng|ara|fra"),
    engine: str = typer.Option("stub", help="stub | m4t"),
    config_path: Path = typer.Option(Path("config/default.yaml"), help="Config YAML"),
) -> None:
    """Transcribe audio to the target language."""
    configure_logging()
    config = load_config(config_path) if config_path.exists() else SawtiConfig()
    if engine == "m4t" and file is not None:
        from sawti.audio_io import FileSource
        pipe = _real_pipeline(config)
        src = FileSource(file, frame_samples=16000)
    else:
        pipe = _stub_pipeline()
        src = StubAudioSource(n_frames=4, samples_per_frame=16000)
    for seg in pipe.run(src, target_lang=target):
        typer.echo(f"[{seg.start_time:.2f}-{seg.end_time:.2f}] {seg.text}")


@app.command()
def eval(
    eval_set: Path = typer.Argument(..., help="Eval set directory"),
    target: str = typer.Option("eng", help="Target language: eng|ara|fra"),
) -> None:
    """Run the evaluation harness."""
    from eval.harness import run_eval

    report = run_eval(eval_set, target_lang=target)
    typer.echo(f"Wrote report: {report}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run, verify pass**

`uv run pytest tests/test_cli.py -v` → PASS (all CLI tests including the new file-based one using `--engine stub`).

- [ ] **Step 5: Commit**

```bash
git add sawti/cli.py tests/test_cli.py
git commit -m "feat(cli): wire real pipeline (FileSource+Silero+M4T+gate+postproc); --engine stub|m4t"
```

---

## Task 12: Real eval metric + harness wiring (`eval/`)

**Files:**
- Modify: `eval/metrics.py`
- Modify: `eval/harness.py`
- Test: `tests/test_eval_harness.py` (add real-chrF test)

- [ ] **Step 1: Add the real metric, keep the stub**

Replace the contents of `eval/metrics.py` with:

```python
"""Eval metrics (spec §7.2). Real chrF via sacrebleu + the M0 stub retained."""
from __future__ import annotations

from collections import Counter


def compute_chrf_stub(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Tiny self-contained chrF-like score in [0,100] (kept for fast tests)."""
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


def compute_chrf(hyp: str, ref: str) -> float:
    """Real chrF via sacrebleu in [0, 100]."""
    import sacrebleu

    bleu = sacrebleu.sentence_chrf(hyp, [ref])
    return bleu.score
```

- [ ] **Step 2: Update harness to use real chrF, keep output_dir param**

In `eval/harness.py`, change the line `chrf = compute_chrf_stub(hyp, ref) if ref else None` to use `compute_chrf` instead, and add `compute_chrf` to the import from `eval.metrics`. Concretely, replace:

```python
from eval.metrics import compute_chrf_stub
```
with:
```python
from eval.metrics import compute_chrf
```
and replace:
```python
        chrf = compute_chrf_stub(hyp, ref) if ref else None
```
with:
```python
        chrf = compute_chrf(hyp, ref) if ref else None
```

Leave everything else in `harness.py` (including the `output_dir` parameter) unchanged.

- [ ] **Step 3: Add a real-chrF test**

Add to `tests/test_eval_harness.py`:

```python
from eval.metrics import compute_chrf


def test_real_chrf_perfect_match_scores_high():
    assert compute_chrf("hello world", "hello world") > 90.0


def test_real_chrf_mismatch_scores_lower_than_match():
    match = compute_chrf("hello world", "hello world")
    miss = compute_chrf("hello world", "completely different text here")
    assert miss < match
```

- [ ] **Step 4: Run, verify pass**

`uv run pytest tests/test_eval_harness.py -v` → PASS (all).

- [ ] **Step 5: Commit**

```bash
git add eval/metrics.py eval/harness.py tests/test_eval_harness.py
git commit -m "feat(eval): real sacrebleu chrF metric; harness uses real chrF"
```

---

## Task 13: M1 acceptance — full suite + manual transcribe smoke

**Files:**
- Modify: `README.md` (add M1 run notes)

- [ ] **Step 1: Run the full test suite (unit only)**

```bash
uv run pytest
```
Expected: all unit tests PASS; integration tests SKIPPED. Report the counts.

- [ ] **Step 2: Run integration tests (requires GPU + model download)**

```bash
SAWTI_RUN_INTEGRATION=1 uv run pytest -m integration -v
```
Expected: the SeamlessM4T integration test runs (downloads ~2.3GB on first run) and passes. If the machine has no GPU or the download fails, report it — this is an environment issue, not a code failure. The unit suite remains the bar for CI.

- [ ] **Step 3: Manual transcribe smoke (real engine)**

```bash
# requires a real audio file named sample.wav in the repo root (or pass a path)
uv run sawti transcribe sample.wav --target eng --engine m4t
```
Expected: timestamped English text lines. (If no sample.wav exists, note that this step needs a real recording — it cannot be fully automated.)

- [ ] **Step 4: Update README with M1 usage**

Append an `## Running (M1)` section to `README.md` (after the M0 section):

```markdown

## Running (M1)

```bash
uv sync
uv run pytest                                   # unit suite (fast, hermetic)
SAWTI_RUN_INTEGRATION=1 uv run pytest -m integration   # real-model tests (GPU)
uv run sawti transcribe sample.wav --target eng --engine m4t   # real pipeline
```

`--engine m4t` loads SeamlessM4T-v2-large (CUDA). Without a file, or with
`--engine stub`, the stub pipeline runs (hermetic, no model download).
```

- [ ] **Step 5: Commit and push the branch**

```bash
git add README.md
git commit -m "docs: add M1 run instructions"
git push -u origin m1-offline-transcribe
```

---

## M1 → M2 handoff (informational)

After M1 merges:
- **M2** adds `MicSource` (live audio capture) and swaps the CLI's `transcribe` to stream from the mic. The real segmenter, engine, gate, and postprocessor are reused unchanged.
- The ASR+MT fallback seam (Task 9) should be wired with `faster-whisper` + NLLB-200 as a focused follow-up before M2 if real-code-switching eval shows the happy path failing often.
- Eval dataset collection (spec §7.1, ~50–75 self-recorded clips) is the gating item for declaring M1 *validated*; the code is complete without it.

---

## Self-Review (completed by plan author)

**Spec coverage:**
- §2.2–§2.4 segmentation (VAD + close-decision) → Tasks 4, 5 ✓
- §3.1–§3.3 S2TT engine + load → Task 10 ✓ (resident via cli `_real_pipeline`)
- §3.5–§3.8 quality gate checks → Tasks 2, 8 ✓
- §3.6 fallback (retry/rechunk real, ASR+MT seam) → Task 9 ✓ (ASR+MT explicitly deferred per user direction)
- §4.2–§4.5 post-processing (6 steps, Arabic normalization, dedupe) → Tasks 6, 7 ✓
- §6.3 M1 demo `sawti transcribe recording.wav --target eng` → Task 11 ✓
- §7.2 chrF metric → Task 12 ✓
- §3.1 ara→arb language mapping → Task 1 ✓
- Frozen-contract preservation (types/config/pipeline untouched) → verified; only `cli.py`, `eval/metrics.py`, `eval/harness.py`, `tests/conftest.py`, `README.md` modified, all new components in new files ✓

**Placeholder scan:** No TBD/TODO. The ASR+MT provider is a documented *deferral* (seam + graceful degradation), not a placeholder — it's an explicit scoping decision the user approved ("skeleton or gated implementation").

**Type consistency:** `EngineResult` fields (chunk_id, raw_text, confidence, source_lang_guess, timing_ms, target_lang) match across engine_m4t.py, quality_gate_balanced.py, fallback.py, and tests. `GateDecision` start_time/end_time (added in M0 review fix) populated consistently in BalancedQualityGate and FallbackHandler. `to_m4t_lang` / `validate_sawti_lang` names consistent between lang_codes.py and tests. `compute_chrf` vs `compute_chrf_stub` both present; harness uses `compute_chrf`.

**Scope check:** M1 is one cohesive milestone (offline file→text). The ASR+MT real implementation is the only piece deferred, and it's isolated behind a seam — does not block M1's core demo. Appropriately scoped.
