import json

from intelli_sentiment import paragraph_sentiment
from intelli_sentiment.vader_lexicon import build_lexicon
from intelli_sentiment.analyzer import nlp
from intelli_sentiment.nlp_matcher import build_matcher
from tests.util import (assert_neg, assert_neu, assert_pos, vader_sentiment,
                        run_evaluation, read_dataset, file_path)



_test_lexicon = \
    build_lexicon(additions=json.load(open(file_path('lexicon_addition.json'))))
_test_matcher = build_matcher(nlp, _test_lexicon)


def test_basic_positives():
    text = """
    I feel that my role is impacted not only by myself, but from all of our team, both locally in NSW and across the country.
    I feel like my direct supervisor is responsible for too much within our office and should have some managerial support.
    In essence I feel like he is a talented person in our field, however is wasted by being responsible for low level responsibilities that take up too much time.
    I also am not a fan of some of the new training material we have been provided.
    I feel that it shows that there is a disconnect with what we are as a company and what our clients needs are.
    """.strip()

    assert vader_sentiment(text) < 0
    assert _paragraph_sentiment(text) < 0


def test_with_f5mgroup_dataset():
    evaluation = run_evaluation('sentiment_dataset_large.tsv',
                                vader_sentiment)

    print(evaluation.results)

    evaluation = run_evaluation('sentiment_dataset_large.tsv',
                                paragraph_sentiment)

    print(evaluation.results)
    # dataset = read_dataset('sentiment_dataset_f5mgroup.tsv')
    # for score, text in dataset:
    #     s1 = vader_sentiment(text)
    #     s2 = _paragraph_sentiment(text)
    #     if (score < 0 and s2 >= 0) or (score >= 0 and s2 < 0):
    #         print(f'{score} {s1} {s2} : {text}')


def _paragraph_sentiment(text):
    return paragraph_sentiment(
        text, lexicon=_test_lexicon, matcher=_test_matcher)
