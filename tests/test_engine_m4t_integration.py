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
