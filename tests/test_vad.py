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
