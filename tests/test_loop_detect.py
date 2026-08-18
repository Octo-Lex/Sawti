"""Loop-detector contract: consecutive n-gram blocks, no frequency false-positives."""
from sawti.loop_detect import is_loop


def test_unigram_loop():
    assert is_loop("نحن " * 35) is True
    assert is_loop("the the the the the") is True


def test_phrase_loop_x3():
    # The Saudi-spike failure mode invisible to unigram-only rules
    # (uniq ratio 0.33, top-token share 0.33).
    assert is_loop("اشتركوا في القناه " * 3) is True
    assert is_loop("و اشتركوا في القناه " * 8) is True


def test_longest_span_within_8_tokens():
    # 8-token span repeated 3x -> caught at n=8.
    span = "a b c d e f g h "
    assert is_loop(span * 3) is True


def test_no_false_positive_legitimate_repetition():
    assert is_loop("لا لا انتظر") is False            # x2 unigram
    assert is_loop("very very important") is False    # x2 unigram
    assert is_loop("شكرا شكرا") is False              # x2, short
    assert is_loop("مرحبا كيف حالك اليوم أتمنى أن تكون بخير") is False
    assert is_loop("short") is False                  # below min length
    assert is_loop("") is False


def test_no_false_positive_nonconsecutive_recurrence():
    # Frequency without consecutive blocks must NOT trigger.
    assert is_loop("الله أكبر قال محمد الله أكبر في كل وقت الله أكبر") is False
    assert is_loop("the cat sat on the mat the dog sat on the rug") is False


def test_min_repeats_parameter():
    text = "لا لا لا"                                   # x3 unigram
    assert is_loop(text, min_repeats=3) is True
    assert is_loop(text, min_repeats=4) is False
