import os
import csv
import logging

from itertools import chain

from intelli_sentiment.symspell_python import (create_dictionary_entry,
                                               best_word)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

NEGATES = [
    "neither", "never", "none", "nope", "nor", "nothing", "nowhere", "uhuh",
    "without", "rarely", "seldom", "despite", "least", "not", "no"
]

NEGATE_VERBS = [
    'be', 'can', 'could', 'do', 'have', 'may', 'must', 'ne', 'ought', 'shall',
    'should', 'will', 'would', 'dare'
]

CONTRACTIVES = [
    'although', 'as opposed', 'but', 'by contrast', 'contrastively', 'despite',
    'either', 'even though', 'however', 'in contrast', 'in spite', 'instead',
    'nevertheless', 'notwithstanding', 'on the one hand', 'on the other hand',
    'though', 'whereas', 'while'
]


class Lexicon():
    def __init__(self):
        self._lexicon = None
        self.boosters = None
        self.phrases = None
        self.negates = set(NEGATES)
        self.negate_verbs = set(NEGATE_VERBS)
        self.contractives = set(CONTRACTIVES)
        self.words = set()

    def load(self,
             lexicon_file='vader_lexicon.txt',
             boosters_file='boosters.txt',
             phrases_file='phrases.txt',
             additions=None):

        logger.debug('loading lexicon from file')
        self._lexicon = self._load_csv(lexicon_file)
        self.boosters = self._load_csv(boosters_file)
        self.phrases = self._load_csv(phrases_file)

        logger.debug('apply lexicon additions')
        self._apply_additions(additions)

        logger.debug('build words collection')
        for phrase in chain(self.boosters.keys(),
                            self.phrases.keys(), self.contractives,
                            self.negates, self.negate_verbs,
                            self._lexicon.keys()):
            for word in phrase.split():
                self.words.add(word.lower())

        logger.debug('build spelling dictionary')
        for word in self.words:
            if word.isalpha():
                create_dictionary_entry(word)
        logger.debug('finish build spelling dictionary')

    def lookup(self, word):
        return self._lexicon.get(word)

    def contains(self, word):
        return word in self._lexicon

    def spell(self, word):
        if word in self.words:
            return []

        correction = best_word(word, True)
        if not correction:
            logging.debug(f'unable to re-spell: {word}')
            return []

        return [correction[0]]

    def _load_csv(self, csv_file):
        file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), csv_file)

        result = {}
        with open(file_path) as file:
            for row in csv.reader(file, delimiter='\t'):
                word, score = row[0:2]
                result[word] = float(score)

        return result

    def _apply_additions(self, additions):
        if not additions:
            return

        self._lexicon.update(additions.get('lexicon', {}))
        self.boosters.update(additions.get('boosters', {}))
        self.phrases.update(additions.get('phrases', {}))
        self.negates.update(additions.get('negates', []))
        self.negate_verbs.update(additions.get('negate_verbs', []))
        self.contractives.update(additions.get('contractives', []))


def build_lexicon(**kwargs):
    lexicon = Lexicon()
    lexicon.load(**kwargs)

    return lexicon


_english_words = set()
with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'english_words.txt')) as file:
    for line in file:
        word = line.strip().lower()
        _english_words.add(word)


# there is a issue in spacy's is_oov implementation, need to revisit later
def is_oov(word):
    return not word.lower() in _english_words
