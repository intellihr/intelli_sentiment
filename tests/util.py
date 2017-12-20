import os
import re
import csv

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


def run_evaluation(dataset_file, predictor):
    evaluation = Evaluation()
    dataset = read_dataset(dataset_file)
    for record in dataset:
        evaluation.add(record[0], predictor(record[1]))

    return evaluation


def read_dataset(dataset_file):
    _file_path = file_path(dataset_file)

    records = []
    with open(_file_path) as dataset_file:
        reader = csv.reader(dataset_file, delimiter='\t')
        for row in reader:
            records.append((float(row[0]), row[1]))

    return records


def file_path(file):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), file)


class Evaluation:
    def __init__(self):
        self._records = []

    def add(self, actual, predict):
        self._records.append((actual, predict))

    @property
    def results(self):
        pos = self.pos
        neg = self.neg
        fscore = (pos['fscore'] + neg['fscore']) / 2

        return dict(mse=self.mse, pos=pos, neg=neg, fscore=fscore)

    @property
    def pos(self):
        if len(self._records) <= 0:
            return 0

        pos_actual = len([r for r in self._records if r[0] >= 0])
        pos_predicted = len([r for r in self._records if r[1] >= 0])
        pos_correct = len(
            [r for r in self._records if r[0] >= 0 and r[1] >= 0])

        precision = pos_correct / pos_predicted
        recall = pos_correct / pos_actual
        fscore = 2 * ((precision * recall) / max(1, precision + recall))

        return dict(precision=precision, recall=recall, fscore=fscore)

    @property
    def neg(self):
        if len(self._records) <= 0:
            return 0

        neg_actual = len([r for r in self._records if r[0] < 0])
        neg_predicted = len([r for r in self._records if r[1] < 0])
        neg_correct = len([r for r in self._records if r[0] < 0 and r[1] < 0])

        precision = neg_correct / neg_predicted
        recall = neg_correct / neg_actual
        fscore = 2 * ((precision * recall) / max(1, precision + recall))

        return dict(precision=precision, recall=recall, fscore=fscore)

    @property
    def mse(self):
        if len(self._records) <= 0:
            return 0

        square_errors = [(r[0] - r[1])**2 for r in self._records]
        return numpy.mean(square_errors)


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
    for sent in nlp(text).sents:
        # split text by list style
        for sent_text in _break_lines_by_structure(sent.text):
            if len(sent_text) <= 0:
                continue

            matches = re.match(r'(.+)([^a-zA-Z\d]\s*however\s*)(.+)',
                               sent_text)

            if matches:
                yield _cleanse(matches[1])
                yield _cleanse(matches[3])
                continue

            matches = re.match(r'(.+)(\s*but\s*)(.+)(\s*so that\s*)(.+)',
                               sent_text)
            if matches:
                yield _cleanse(matches[1])
                yield _cleanse(matches[3])
                yield _cleanse(matches[5])
                continue

            yield _cleanse(sent_text)


def _break_lines_by_structure(text):
    return _split(text, '^\s*[0-9-*•]+[.,]?\s*', '\n\n+', '^\s+')


def _split(text, *regexes):
    toks = [text]
    for regex in regexes:
        _toks = []
        for tok in toks:
            for t in re.split(regex, tok, flags=re.MULTILINE):
                t = t.strip()
                if len(t) > 0:
                    _toks.append(t)
        toks = _toks

    return toks


def _cleanse(text):
    return re.sub(r'([^\s]+)(/)([^\s]+)', r'\1 / \3', text)


_vader_sentiment = SentimentIntensityAnalyzer()


def _compute_sentence_sentiment(sentence):
    score = _vader_sentiment.polarity_scores(sentence)['compound']

    return score
