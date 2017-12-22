import json
import logging

from intelli_sentiment import paragraph_sentiment
from intelli_sentiment.vader_lexicon import build_lexicon
from intelli_sentiment.analyzer import nlp
from intelli_sentiment.nlp_matcher import build_matcher
from tests.util import (vader_sentiment, run_evaluation, file_path)

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)

_test_lexicon = build_lexicon(
    additions=json.load(open(file_path('lexicon_addition.json'))))
_test_matcher = build_matcher(nlp, _test_lexicon)


def test_basic_positives():
    text = """
    I feel that my role is impacted not only by myself, but from all of
    our team, both locally in NSW and across the country.
    I feel like my direct supervisor is responsible for too much within our
    office and should have some managerial support.
    In essence I feel like he is a talented person in our field,
    however is wasted by being responsible for low level responsibilities
    that take up too much time.
    I also am not a fan of some of the new training material
    we have been provided.
    I feel that it shows that there is a disconnect with what
    we are as a company and what our clients needs are.
    """.strip()

    assert vader_sentiment(text) < 0
    assert _paragraph_sentiment(text) < 0


def test_with_f5mgroup_dataset():
    evaluation = run_evaluation('sentiment_dataset_f5mgroup.tsv',
                                vader_sentiment)
    vader_score = evaluation.results['fscore']
    logging.info(f'vader: {evaluation.results}')

    evaluation = run_evaluation('sentiment_dataset_f5mgroup.tsv',
                                paragraph_sentiment)
    score = evaluation.results['fscore']
    logging.info(f'intelli: {evaluation.results}')

    assert score > vader_score


def _paragraph_sentiment(text):
    return paragraph_sentiment(
        text, lexicon=_test_lexicon, matcher=_test_matcher)
