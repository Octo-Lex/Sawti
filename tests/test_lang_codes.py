import pytest

from sawti.lang_codes import SAWTI_TO_M4T, to_m4t_lang, validate_sawti_lang


def test_supported_codes_present():
    assert SAWTI_TO_M4T["eng"] == "eng"
    assert SAWTI_TO_M4T["fra"] == "fra"
    assert SAWTI_TO_M4T["ara"] == "arb"  # Arabic → Modern Standard Arabic


def test_to_m4t_lang_maps_ara():
    assert to_m4t_lang("ara") == "arb"
    assert to_m4t_lang("eng") == "eng"


def test_to_m4t_lang_unknown_raises():
    with pytest.raises(KeyError):
        to_m4t_lang("deu")


def test_validate_sawti_lang_accepts_supported():
    validate_sawti_lang("eng")  # no raise
    validate_sawti_lang("ara")


def test_validate_sawti_lang_rejects_unknown():
    with pytest.raises(ValueError):
        validate_sawti_lang("xyz")
