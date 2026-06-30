from sawti.script_detect import dominant_script, is_mostly_arabic, is_mostly_latin


def test_latin_text_detected():
    assert is_mostly_latin("Hello world, how are you today?") is True
    assert dominant_script("Hello world") == "latin"


def test_arabic_text_detected():
    assert is_mostly_arabic("مرحبا كيف حالك اليوم") is True
    assert dominant_script("مرحبا") == "arabic"


def test_mixed_text_dominant():
    # Latin dominates by letter count
    assert dominant_script("Hello مرحبا world") == "latin"


def test_digits_and_punct_ignored():
    assert dominant_script("12345, ??? !!!") == "other"
    assert is_mostly_arabic("12345, ??? !!!") is False


def test_empty_is_other():
    assert dominant_script("") == "other"
