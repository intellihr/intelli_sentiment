from intelli_sentiment import sentence_sentiment


def test_basic_positives():
    sentences = [
        'Eric is smart, handsome, and funny.', 'The book was good',
        "At least it isn't a horrible book."
        'Make sure you :) or :D today!'
    ]

    for sentence in sentences:
        assert_pos(sentence)


def test_basic_negatives():
    sentences = [
        'Bobo is not smart, handsome, nor funny.',
        'The plot was good, but the characters are uncompelling and the dialog is not great.',
        'Today sucks!', "Today only kinda sux! But I'll get by, lol"
    ]

    for sentence in sentences:
        assert_neg(sentence)


def test_punctuation_emphasis():
    s1 = sentence_sentiment('Eric is smart, handsome, and funny.').compound
    s2 = sentence_sentiment('Eric is smart, handsome, and funny!').compound
    assert s1 < s2


def test_allcaps_emphasis():
    s1 = sentence_sentiment('Eric is smart and handsome.').compound
    s2 = sentence_sentiment('Eric is SMART and HANDSOME.').compound
    assert s1 < s2


def test_kind_of():
    s1 = sentence_sentiment('The book was good.').compound
    s2 = sentence_sentiment('The book was kind of good.').compound
    assert s1 > s2


def test_spell_correct():
    s1 = sentence_sentiment('The book was aweosme.').compound
    s2 = sentence_sentiment('The book was kind of awesoem.').compound

    assert s1 > 0
    assert s2 > 0
    assert s1 > s2


def test_phrases():
    assert_neg('it is kiss of death!')
    assert_neg('it is kiss of deaht!')
    assert_pos('he is the shit!')


def test_least():
    assert_neg('he is the least cool man!')
    assert_neu('feed him at least!')
    assert_neu('feed him very least!')
    assert_neu('feed him at very least!')


def assert_pos(text):
    scores = sentence_sentiment(text)
    assert scores.compound > 0


def assert_neg(text):
    scores = sentence_sentiment(text)
    assert scores.compound < 0


def assert_neu(text):
    scores = sentence_sentiment(text)
    assert scores.compound == 0
