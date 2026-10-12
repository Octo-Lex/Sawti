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


def test_collapse_repeated_loops_ngram():
    # spec §4.5: n-gram loops collapse to ONE instance of the phrase
    assert collapse_repeated_loops("a b a b a b", min_count=3) == "a b"
    assert collapse_repeated_loops("x y x y x y x y", min_count=3) == "x y"
    # two repeats stay (below min_count) even for phrases
    assert collapse_repeated_loops("a b a b", min_count=3) == "a b a b"


def test_collapse_repeated_loops_session_arabic_phrase_loop():
    # The exact failure shape from the 2026-10-09 Task 7 smoke: the
    # exhausted chunk's Whisper output looped a 5-token Arabic phrase.
    phrase = "لكي نتحدث عن هذه المشكلة"
    text = " ".join([phrase] * 40)
    assert collapse_repeated_loops(text, min_count=3) == phrase


def test_collapse_repeated_loops_mixed_runs_and_normal_text():
    assert (collapse_repeated_loops(
        "the the the the cat cat cat and then we left",
        min_count=3) == "the cat and then we left")
    # a unigram loop followed by another unigram loop in one utterance
    assert (collapse_repeated_loops(
        "يلا يلا يلا go go go go go", min_count=3) == "يلا go")


def test_collapse_repeated_loops_preserves_intentional():
    # spec §4.5 preserve-list
    assert collapse_repeated_loops("no no, wait", min_count=3) == "no no, wait"
    assert collapse_repeated_loops("very very important", min_count=3) == \
        "very very important"
    assert collapse_repeated_loops("لا لا", min_count=3) == "لا لا"


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
