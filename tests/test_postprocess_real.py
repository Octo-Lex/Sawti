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
