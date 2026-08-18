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


def test_dominance_alone_never_triggers():
    # Reviewer's exact regression: "no" is 6/9 tokens (66.7% > the old 0.6
    # dominance threshold), but there is no run of three consecutive
    # repeats of any block. Frequency must NOT gate.
    assert is_loop("no no wait no no stop no no listen") is False


def test_dominance_uniq_side_never_triggers():
    # High recurrence density (2 unique tokens in 9) but the only 3x block
    # repeats are the genuine-loop pattern this text deliberately lacks
    # beyond two: "لا" recurs 6/9 without three consecutive.
    assert is_loop("لا انتظر لا انتظر لا انتظر") is True  # genuine 2-gram x3
    assert is_loop("لا لا انتظر لا لا انتظر") is False     # x2 only
