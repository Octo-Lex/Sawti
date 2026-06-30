from sawti.text_normalize import (
    collapse_repeated_loops,
    normalize_arabic_for_match,
    normalize_for_match,
    repair_punctuation_spacing,
    strip_excess_whitespace,
)


def test_strip_excess_whitespace():
    assert strip_excess_whitespace("  hello   world  ") == "hello world"


def test_repair_punctuation_spacing():
    assert repair_punctuation_spacing("hello , world") == "hello, world"
    assert repair_punctuation_spacing("a . b") == "a. b"


def test_collapse_repeated_loops_three_or_more():
    assert collapse_repeated_loops("the the the the", min_count=3) == "the"
    # preserves intentional double repetition
    assert collapse_repeated_loops("no no wait", min_count=3) == "no no wait"


def test_collapse_repeated_loops_arabic():
    assert collapse_repeated_loops("مرحبا مرحبا مرحبا مرحبا", min_count=3) == "مرحبا"


def test_normalize_for_match_lowercases_latin():
    assert normalize_for_match("Hello WORLD") == "hello world"


def test_normalize_arabic_for_match_removes_diacritics_and_tatweel():
    out = normalize_arabic_for_match("مـَرحباً")  # tatweel + diacritics
    assert "ـ" not in out  # no tatweel
    assert all(0x064B > ord(c) or ord(c) > 0x0652 for c in out)  # no harakat


def test_normalize_arabic_for_match_unifies_alef():
    # أ إ آ should all map to ا for matching
    n = normalize_arabic_for_match("أحمد إبراهيم آدم")
    assert "أ" not in n and "إ" not in n and "آ" not in n
    assert "ا" in n
