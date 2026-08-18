"""Commit 6: real-speech fixture exercises decode/segmentation/evaluator
in normal CI — deterministic, no models, no downloads.

The fixture (tests/fixtures/realspeech/hello.wav) is actual spoken audio
(CC BY-SA 3.0, see PROVENANCE.md). The VAD pattern was RECORDED from the
real Silero model on this exact waveform (vad_pattern.json); FakeVad
replays it so the REAL segmenter logic runs on real speech
characteristics without loading any model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sawti.config import SegmentationConfig
from sawti.segmenter_silero import RealSegmenter
from sawti.sources import AudioFrame
from sawti.vad import FakeVad

FIX = Path(__file__).parent / "fixtures" / "realspeech"


def test_fixture_is_real_speech_not_a_tone():
    import soundfile as sf

    audio, sr = sf.read(FIX / "hello.wav", dtype="float32")
    assert sr == 16000 and audio.ndim == 1
    assert 0.3 < len(audio) / sr < 1.0                 # ~0.49s per provenance
    rms = float(np.sqrt(np.mean(audio ** 2)))
    assert rms > 0.02                                  # speech energy
    # Spectral richness: a tone has ~all energy in one bin; real speech
    # spreads it. Check the ratio of peak to total spectrum energy.
    spec = np.abs(np.fft.rfft(audio))
    assert spec.max() / spec.sum() < 0.2


def test_real_speech_segmentation_hermetic():
    """Recorded-VAD FakeVad drives the REAL segmenter over the fixture
    waveform — decode + boundary logic on genuine speech, no models."""
    import soundfile as sf

    audio, sr = sf.read(FIX / "hello.wav", dtype="float32")
    pattern = json.loads((FIX / "vad_pattern.json").read_text(encoding="utf-8"))
    w = pattern["window"]
    assert w == 512 and len(pattern["probs"]) * w <= len(audio)

    frames = [
        AudioFrame(audio=audio[i * w:(i + 1) * w], sample_rate=sr,
                   timestamp_s=i * w / sr)
        for i in range(len(pattern["probs"]))
    ]
    vad = FakeVad([(p, p >= pattern["threshold"]) for p in pattern["probs"]])
    seg = RealSegmenter(vad=vad, config=SegmentationConfig(
        pause_threshold_ms=350, min_chunk_duration_ms=0, overlap_ms=0,
        min_speech_ms=100))
    chunks = list(seg.process(iter(frames)))

    assert len(chunks) == 1
    c = chunks[0]
    # Speech starts after the 2 silent lead-in windows (~0.064s).
    assert c.start_time == pytest.approx(2 * w / sr, abs=1e-6)
    # Wall-clock-true: the chunk is the literal source slice.
    src = np.concatenate([f.audio for f in frames])
    assert np.array_equal(
        c.audio, src[int(round(c.start_time * sr)):int(round(c.end_time * sr))])
    assert len(c.audio) / sr >= 13 * w / sr - 1e-6      # the speech span


def test_fixture_flows_through_evaluator():
    """End-to-end on real audio: FileSource decode -> full pipeline seam
    (stub components injected) -> evaluator report with chrF vs the
    reference 'Hello.'"""
    from eval.harness import run_eval
    from eval.transcribers import Transcription, make_pipeline_transcriber
    from sawti.engine import EngineManager, StubEngine
    from sawti.pipeline import Pipeline
    from sawti.postprocess import StubPostProcessor
    from sawti.quality_gate import StubQualityGate
    from sawti.segmenter import StubSegmenter

    tmp = FIX  # fixture dir doubles as a one-clip eval set
    def factory(on_decision=None):
        return Pipeline(
            segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
            engine=EngineManager(engine=StubEngine("Hello.", 0.9)),
            gate=StubQualityGate(),
            postprocessor=StubPostProcessor(),
            on_decision=on_decision,
        )

    transcriber = make_pipeline_transcriber(factory, "eng", frame_samples=16000)
    out = run_eval(tmp, target_lang="eng", transcriber=transcriber,
                   output_dir=tmp / "_report_test")
    report = json.loads(Path(out).read_text(encoding="utf-8"))
    clip = next(c for c in report["clips"] if c["clip"] == "hello.wav")
    assert clip["has_reference"] is True
    assert clip["hypothesis"].strip() == "Hello."
    assert clip["chrf"] is not None and clip["chrf"] > 90.0
    (tmp / "_report_test").exists() and None
    # cleanup the transient report dir
    import shutil
    shutil.rmtree(tmp / "_report_test", ignore_errors=True)
