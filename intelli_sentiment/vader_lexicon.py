import os
import csv
from collections import defaultdict


NEGATES = [
    "neither", "hasnt", "havent", "hasn't", "haven't", "never", "none", "nope",
    "nor", "nothing", "nowhere", "uhuh", "without", "rarely", "seldom",
    "despite", "darent", "daren't", "least"
]

NEGATE_VERBS = [
    'be', 'can', 'could', 'do', 'have', 'may', 'must', 'ne', 'ought', 'shall',
    'should', 'will', 'would'
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
        self._words = defaultdict(set)

    def load(self,
             lexicon_file='vader_lexicon.txt',
             boosters_file='boosters.txt',
             phrases_file='phrases.txt'):
        self._lexicon = self._load_csv(lexicon_file)
        self.boosters = self._load_csv(boosters_file)
        self.phrases = self._load_csv(phrases_file)

    def lookup(self, word):
        return self._lexicon.get(word)

    def contains(self, word):
        return word in self._lexicon

    def lookup_booster(self, word):
        return self.boosters.get(word)

    def _load_csv(self, csv_file):
        file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), csv_file)

        result = {}
        with open(file_path) as file:
            for row in csv.reader(file, delimiter='\t'):
                word, score = row[0:2]
                result[word] = float(score)

        return result

lexicon = Lexicon()
lexicon.load()