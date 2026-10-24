"""Importer contract tests (2026-10-15): pairing, long-take split,
idempotence — synthetic audio only, no mics."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.import_eval_recordings import (OUT, clip_id_from_name,
                                            recordable_clips,
                                            split_on_silence)


def _kit_ids():
    return [c["id"] for c in recordable_clips()]


def test_clip_id_from_name_matches_kit():
    assert clip_id_from_name(Path("take_en_07.m4a")) == "en_07"
    assert clip_id_from_name(Path("ar 3.wav")) == "ar_03"
    assert clip_id_from_name(Path("ENAR_12 take2.ogg")) == "enar_12"
    assert clip_id_from_name(Path("random_name.wav")) is None
    assert clip_id_from_name(Path("en_99.wav")) is None  # not in kit


def test_recordable_order_is_slices_then_id():
    ids = _kit_ids()
    assert ids[:3] == ["en_01", "en_02", "en_03"]
    assert "ar_01" in ids and "enar_15" in ids


def test_split_on_silence_separates_gaps():
    sr = TARGET = 16000
    speech = np.full(int(sr * 1.0), 0.3, np.float32)
    quiet = np.zeros(int(sr * 2.5), np.float32)
    audio = np.concatenate([speech, quiet, speech])
    parts = split_on_silence(audio, gap_s=2.0)
    assert len(parts) == 2
    assert all(np.max(np.abs(p)) > 0.2 for p in parts)


def test_split_on_silence_keeps_continuous_speech_whole():
    sr = 16000
    audio = np.full(int(sr * 5), 0.3, np.float32)
    assert len(split_on_silence(audio, gap_s=2.0)) == 1


def test_recordable_excludes_french_slices():
    slices = {c["slice"] for c in recordable_clips()}
    assert slices == {"en", "ar", "en-ar"}
    assert len(recordable_clips()) == 39
