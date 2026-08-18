"""SA Task 1: shared Saudi eval utils — one authoritative metric recipe.
The loop detector is the SHARED production module (pinned here); the
aggregation exposes the full four-metric selection regime."""
from pathlib import Path

import pytest

from sawti.loop_detect import is_loop
from sawti.training.eval_utils import (aggregate, annotate_degenerate, norm,
                                       run_eval, wer_clean)


def test_norm_unifies_arabic_and_strips_punct():
    assert norm("مَرْحَباً ، World!") == "مرحبا ، World"
    assert norm("أحمد  إبراهيم") == "احمد ابراهيم"


def test_wer_clean_basic():
    assert wer_clean("احمد ذهب", "احمد ذهب") == 0.0
    # 1 insertion over 1 reference word = WER 1.0 (jiwer convention).
    assert wer_clean("احمد", "احمد ذهب") == pytest.approx(1.0)


def test_shared_detector_pinned_through_eval_utils():
    # The shared M1 detector, imported via eval_utils: phrase loops catch,
    # frequency alone never gates (the dominance fork is gone).
    assert is_loop("اشتركوا في القناه " * 3) is True
    assert is_loop("لا " * 12) is True
    assert is_loop("no no wait no no stop no no listen") is False


def test_annotate_degenerate_sets_metric_fields():
    rows = [
        {"dialect": "Najdi", "duration_s": 5.0, "cleaned_text": "مرحبا بك",
         "hyp": "مرحبا بك", "wer": 0.0},
        {"dialect": "Najdi", "duration_s": 5.0, "cleaned_text": "كلمة",
         "hyp": "اشتركوا في القناه اشتركوا في القناه اشتركوا في القناه", "wer": 9.0},
        {"dialect": "Hijazi", "duration_s": 0.5, "cleaned_text": "نعم",
         "hyp": "نعم", "wer": 0.0},
    ]
    out = annotate_degenerate(rows)
    assert out[0]["loop"] is False and out[0]["valid_ref"] is True
    assert out[0]["n_ref_words"] == 2
    assert out[1]["loop"] is True and out[1]["degenerate"] is True  # phrase loop
    assert out[2]["degenerate"] is True  # short clip


def test_aggregate_exposes_full_metric_set():
    rows = [
        {"dialect": "Najdi", "duration_s": 5.0, "cleaned_text": "مرحبا بك",
         "hyp": "مرحبا بك", "wer": 0.0, "loop": False, "degenerate": False,
         "valid_ref": True, "n_ref_words": 2},
        {"dialect": "Najdi", "duration_s": 5.0, "cleaned_text": "كلمة واحدة",
         "hyp": "كلمة", "wer": 0.5, "loop": False, "degenerate": False,
         "valid_ref": True, "n_ref_words": 2},
        {"dialect": "Hijazi", "duration_s": 5.0, "cleaned_text": "كلمة",
         "hyp": "اشتركوا في القناه " * 3, "wer": 9.0, "loop": True,
         "degenerate": True, "valid_ref": True, "n_ref_words": 1},
    ]
    out = aggregate(rows)
    assert out["clean_macro_wer"] == pytest.approx(25.0)          # (0 + 0.5) / 2
    assert out["all_valid_macro_wer"] == pytest.approx(100 * (0 + 0.5 + 9) / 3)
    assert out["all_valid_corpus_wer"] == pytest.approx(100 * (0 * 2 + 0.5 * 2 + 9 * 1) / 5)
    assert out["loop_pct"] == pytest.approx(100 / 3)
    assert out["per_dialect"]["Najdi"]["clean_macro_wer"] == pytest.approx(25.0)
    assert out["per_dialect"]["Hijazi"]["n_clean"] == 0


def test_run_eval_annotates(tmp_path: Path):
    import json as _json

    import numpy as np
    import soundfile as sf

    sf.write(tmp_path / "a.wav", np.zeros(16000, np.float32), 16000)
    (tmp_path / "manifest.jsonl").write_text(
        _json.dumps({"clip_id": "a", "dialect": "Najdi", "cleaned_text": "مرحبا",
                     "duration_s": 1.0}, ensure_ascii=False), encoding="utf-8")
    rows = run_eval(lambda w: "مرحبا", tmp_path)
    assert rows[0]["wer"] == 0.0 and rows[0]["loop"] is False
    assert aggregate(rows)["clean_macro_wer"] == pytest.approx(0.0)
