"""SA Task 2: SADA data prep — census, Saudi filter, materialization rules."""
from sawti.training.data_prep import census_labels, keep_row, saudi_label_set


def test_census_labels_counts():
    rows = [{"speaker_dialect": "Najdi"}, {"speaker_dialect": "Najdi"},
            {"speaker_dialect": "MSA"}, {"speaker_dialect": None}]
    assert census_labels(rows) == {"Najdi": 2, "MSA": 1, "unknown": 1}


def test_saudi_label_set_from_census():
    census = {"Najdi": 10, "Hijazi": 5, "Khaliji": 5, "MSA": 99, "Yemeni": 3}
    # Saudi core confirmed by the spike; census adds any other Saudi-specific
    # labels the operator supplies via --extra-label after inspecting counts.
    assert saudi_label_set(census, extra=[]) == {"Najdi", "Hijazi", "Khaliji"}
    assert saudi_label_set(census, extra=["Southern Saudi"]) == {
        "Najdi", "Hijazi", "Khaliji", "Southern Saudi"}


def test_keep_row_filters_duration_and_text():
    base = {"duration_s": 5.0, "cleaned_text": "مرحبا", "speaker_dialect": "Najdi"}
    assert keep_row(base, {"Najdi"}) is True
    assert keep_row({**base, "duration_s": 45.0}, {"Najdi"}) is False  # >30s
    assert keep_row({**base, "duration_s": 0.3}, {"Najdi"}) is False  # <0.5s
    assert keep_row({**base, "cleaned_text": "  "}, {"Najdi"}) is False
    assert keep_row(base, {"Hijazi"}) is False  # dialect not selected


# --- corrective pass: genuine census, provenance, disjointness ---

def test_census_only_counts_all_labels_no_decode(tmp_path, monkeypatch):
    from sawti.training import data_prep as dp

    rows = [
        {"speaker_dialect": "Najdi", "cleaned_text": "أ"},
        {"speaker_dialect": "MSA", "cleaned_text": ""},
        {"speaker_dialect": None, "cleaned_text": "ب"},
    ]

    class FakeDs:
        def __iter__(self):
            return iter(rows)

    monkeypatch.setattr(dp, "_stream", lambda split: (FakeDs(), split))
    stats = dp.census_only("validation", str(tmp_path))
    assert stats["total_rows"] == 3
    assert stats["label_inventory"] == {"Najdi": 1, "MSA": 1, "unknown": 1}
    assert stats["empty_transcript_rows"] == 1
    import json as _j
    from pathlib import Path as _P
    assert _j.loads((_P(tmp_path) / "census.json").read_text(
        encoding="utf-8"))["total_rows"] == 3


def test_materialize_records_provenance_and_label_census(tmp_path, monkeypatch):
    import io as _io

    import numpy as np
    import soundfile as _sf

    from sawti.training import data_prep as dp

    def wav_bytes():
        buf = _io.BytesIO()
        _sf.write(buf, np.zeros(8000, np.float32), 16000, format="WAV")  # 0.5s >= MIN_S
        return buf.getvalue()

    b1, b2, b3 = wav_bytes(), wav_bytes(), wav_bytes()
    rows = [
        {"speaker_dialect": "Najdi", "cleaned_text": "أ", "audio": {"bytes": b1}},
        {"speaker_dialect": "MSA", "cleaned_text": "ب", "audio": {"bytes": b2}},
        {"speaker_dialect": "Najdi", "cleaned_text": "ج", "audio": {"bytes": b3}},
    ]

    class FakeDs:
        def __iter__(self):
            return iter(rows)

    monkeypatch.setattr(dp, "_stream", lambda split: (FakeDs(), split))
    stats = dp.materialize("validation", str(tmp_path), [], None)
    import hashlib, json as _j
    from pathlib import Path as _P

    manifest = [_j.loads(l) for l in
                (_P(tmp_path) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(manifest) == 2                      # MSA dropped
    assert stats["label_inventory"] == {"Najdi": 2, "MSA": 1}   # genuine census
    assert stats["dropped_by_label"] == {"MSA": 1}               # by NAME
    for m, ordinal, blob in zip(manifest, [0, 2], [b1, b3]):
        assert m["source_split"] == "validation"
        assert m["source_ordinal"] == ordinal
        assert m["audio_sha256"] == hashlib.sha256(blob).hexdigest()
    assert stats["scanned"] == 3


def test_assert_no_overlap_detects_leakage(tmp_path):
    import json as _j
    from pathlib import Path as _P

    from sawti.training.data_prep import assert_no_overlap, manifest_audio_hashes

    def mk(d, hashes):
        d = _P(d); d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.jsonl").write_text(
            "\n".join(_j.dumps({"audio_sha256": h}) for h in hashes),
            encoding="utf-8")

    a, b = tmp_path / "a", tmp_path / "b"
    mk(a, ["h1", "h2"])
    mk(b, ["h2", "h3"])                            # h2 leaked
    import pytest as _pt
    with _pt.raises(AssertionError, match="split leakage"):
        assert_no_overlap(a, b)
    mk(b, ["h3", "h4"])                            # disjoint now
    assert_no_overlap(a, b)
    assert manifest_audio_hashes(a) == {"h1", "h2"}
