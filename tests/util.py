import re

import numpy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from intelli_sentiment import sentence_sentiment
from intelli_sentiment.analyzer import nlp


def assert_pos(text):
    scores = sentence_sentiment(text)
    assert scores.compound > 0


def assert_neg(text):
    scores = sentence_sentiment(text)
    assert scores.compound < 0


def assert_neu(text):
    scores = sentence_sentiment(text)
    assert scores.compound == 0


def vader_sentiment(text, alpha=0.87):
    scores = [_compute_sentence_sentiment(tok) for tok in _tokenizer(text)]
    negatives = [s for s in scores if s < 0]
    positives = [s for s in scores if s > 0]

    if len(negatives) == 0 and len(positives) == 0:
        return 0
    elif len(negatives) == 0 and len(positives) > 0:
        return numpy.mean(positives)
    elif len(positives) == 0 and len(negatives) > 0:
        return numpy.mean(negatives)

    neg = numpy.mean(negatives) * alpha
    pos = numpy.mean(positives) * (1 - alpha)

    return max(-1, min(1, neg + pos))


def _tokenizer(text):
    """
    break up text by 'however'
    """
    for sent in nlp(text).sents:
        sent_text = sent.text.strip()
        matches = re.match(r'(.+)([^a-zA-Z\d]\s*however\s*)(.+)', sent_text)

        if matches:
            yield matches[1]
            yield matches[3]
            continue

        matches = re.match(r'(.+)(\s*but\s*)(.+)(\s*so that\s*)(.+)',
                           sent_text)
        if matches:
            yield matches[1]
            yield matches[3]
            yield matches[5]
            continue

        yield _cleanse(sent_text)


def _cleanse(text):
    return re.sub(r'([^\s]+)(/)([^\s]+)', r'\1 / \3', text)


_vader_sentiment = SentimentIntensityAnalyzer()


def _compute_sentence_sentiment(sentence):
    score = _vader_sentiment.polarity_scores(sentence)['compound']

    return score
