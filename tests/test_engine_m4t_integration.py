import numpy as np
import pytest

from sawti.engine_m4t import SeamlessM4TEngine
from sawti.types import AudioChunk

# Optional: use a real recorded sample if present, so the integration test
# exercises actual speech translation rather than silence. Falls back to a
# 3s synthetic tone if no sample is found (keeps the test self-contained).
SAMPLE_PATH = "sample/test01.wav"


def _load_chunk() -> AudioChunk:
    """Return an AudioChunk with real audio if a sample exists, else a tone."""
    from pathlib import Path

    if Path(SAMPLE_PATH).exists():
        import librosa

        audio, _ = librosa.load(SAMPLE_PATH, sr=16000, mono=True)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
    else:
        # 3s low-amplitude tone as a fallback (silence would translate to "").
        t = np.linspace(0, 3.0, int(16000 * 3.0), endpoint=False)
        audio = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return AudioChunk(
        id="c0",
        audio=audio,
        sample_rate=16000,
        start_time=0.0,
        end_time=len(audio) / 16000,
    )


@pytest.mark.integration
def test_real_seamless_m4t_translates_english():
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        "facebook/seamless-m4t-v2-large"
    )
    eng = SeamlessM4TEngine(processor=processor, model=model, device="cuda")
    chunk = _load_chunk()
    r = eng.translate(chunk, target_lang="eng")
    assert isinstance(r.raw_text, str)
    assert r.target_lang == "eng"
    # Print the result so the test log shows what the model actually produced.
    print(f"\n[integration] target=eng raw_text={r.raw_text!r} confidence={r.confidence:.3f}")


@pytest.mark.integration
def test_production_builder_on_real_speech():
    """The full production graph (build.py — segmenter, M4T engine, gate,
    FallbackHandler with conservative retry + rechunker + real ASR+MT
    provider, postprocessor) runs end-to-end on the licensed real-speech
    fixture and produces meaningful, non-canned output."""
    import sawti.env

    sawti.env.load_env(override=True)  # corrected HF cache path
    from pathlib import Path as _P

    from sawti.audio_io import FileSource
    from sawti.build import build_real_pipeline
    from sawti.config import load_config

    from sawti.config import SawtiConfig, SegmentationConfig

    # The fixture is a single 0.49s word; production defaults
    # (min_chunk_duration_ms=600) would correctly drop it at EOF flush.
    # Short-clip thresholds make the fixture evaluable end-to-end.
    cfg = SawtiConfig(segmentation=SegmentationConfig(
        min_chunk_duration_ms=0, min_speech_ms=100))
    pipe = build_real_pipeline(cfg)
    assert pipe.fallback is not None
    assert pipe.fallback.asr_mt is not None          # real provider wired
    assert pipe.fallback.conservative is not None    # conservative seam bound
    assert pipe.fallback.rechunker is not None

    src = FileSource("tests/fixtures/realspeech/hello.wav", frame_samples=16000)
    out = list(pipe.run(src, target_lang="eng"))
    assert out, "no segments emitted for real speech"
    text = " ".join(s.text for s in out).strip()
    assert text                                    # non-empty transcription
    # Not a canned/stub value ("hello" = StubEngine's string; "[stub]"
    # = the old harness literal). Note "Hello." is a GENUINE M4T output
    # for this fixture — the correct transcription — so it must NOT be
    # in this blacklist.
    assert text not in ("hello", "[stub]", "hello world")
    for s in out:
        assert 0.0 <= s.start_time < s.end_time <= 1.0  # within clip duration
