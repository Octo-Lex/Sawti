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
