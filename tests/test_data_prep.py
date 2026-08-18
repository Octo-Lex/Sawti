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
