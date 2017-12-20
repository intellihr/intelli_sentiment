import re
import math
import logging
from enum import Enum
from collections import namedtuple

import numpy
import spacy
from spacy.tokens import Token

from intelli_sentiment.vader_lexicon import build_lexicon, is_oov
from intelli_sentiment.nlp_matcher import build_matcher

# (empirically derived mean sentiment intensity rating increase for using
# ALLCAPs to emphasize a word)
C_INCR = 0.733

N_SCALAR = -0.74

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

nlp = spacy.load('en')
Token.set_extension('_custom_pos', default=None)
Token.set_extension(
    'pos', getter=lambda token: token._._custom_pos or token.pos_)


class TokenType(Enum):
    UNMATCH = 0
    MATCH = 1
    BOOSTER = 2
    NEGATE = 3
    CONTRACTIVE = 4


class SentenceAnalyzer:
    __default_lexicon = None
    __default_matcher = None

    def __init__(self,
                 raw_text,
                 spell_correct=True,
                 lexicon=None,
                 matcher=None):
        self._lexicon = lexicon
        self._matcher = matcher
        self.text = self._tokenize(raw_text)
        self.is_cap_diff = self._check_is_cap_diff()
        self._sentiments = []
        self._corrections = {}
        self.spell_correct = spell_correct

    def analyze(self):
        i = 0
        while i < len(self.text):
            # 1) handle custom exceptions
            is_exception, length = self._check_exception(i)
            if is_exception:
                i += length
                continue

            # 2) resolve long phrases first
            phrase_score, length, corrections = self._add_phrase_sentiment(i)
            if phrase_score is not False:
                self._sentiments.append(
                    dict(
                        score=phrase_score,
                        pos=i,
                        length=length,
                        corrections=corrections,
                        type=TokenType.MATCH))
                i += length
                continue

            # 3) check if contractive phrases
            is_contractive, length, corrections = self._check_contractive(i)
            if is_contractive:
                self._sentiments.append(
                    dict(
                        score=0,
                        pos=i,
                        corrections=corrections,
                        type=TokenType.CONTRACTIVE))
                i += length
                continue

            # 4) check if it is negate phrase
            is_negate, length, corrections = self._check_negate(i)
            if is_negate:
                self._sentiments.append(
                    dict(
                        score=0,
                        pos=i,
                        corrections=corrections,
                        type=TokenType.NEGATE))
                i += length
                continue

            # 5) check if it contains booster phrase
            booster_score, length, corrections = self._check_booster(i)
            if booster_score is not False:
                self._sentiments.append(
                    dict(
                        score=0,
                        pos=i,
                        corrections=corrections,
                        booster=booster_score,
                        type=TokenType.BOOSTER))
                i += length
                continue

            self._add_word_sentiment(i)
            i += 1

        self._apply_contractive()

        return SentenceResult(self.text, self._sentiments)

    @property
    def lexicon(self):
        if self._lexicon:
            return self._lexicon

        return self._default_lexicon()

    @classmethod
    def _default_lexicon(cls):
        if not cls.__default_lexicon:
            cls.__default_lexicon = build_lexicon()

        return cls.__default_lexicon

    @property
    def matcher(self):
        if self._matcher:
            return self._matcher

        return self._default_matcher(self.lexicon)

    @classmethod
    def _default_matcher(cls, lexicon):
        if not cls.__default_matcher:
            cls.__default_matcher = build_matcher(nlp, lexicon)

        return cls.__default_matcher

    def _apply_contractive(self):
        contractives = (sentiment for sentiment in self._sentiments
                        if sentiment['type'] == TokenType.CONTRACTIVE)
        contractive = next(contractives, None)
        if not contractive:
            return

        sentiments = (s for s in self._sentiments
                      if s['type'] == TokenType.MATCH)
        for sentiment in sentiments:
            if sentiment['pos'] < contractive['pos']:
                sentiment['score'] *= 0.5
            elif sentiment['pos'] > contractive['pos']:
                sentiment['score'] *= 1.5

    def _check_exception(self, current):
        term_1 = self.text[current]
        term_2 = self.text[current
                           + 1] if current + 1 < len(self.text) else None
        if term_1.lemma_ == 'please' and term_1._.pos == 'INTJ':
            return (True, 1)
        elif term_1.lemma_ in ['no', 'none'] and term_2 and term_2.is_punct:
            return (True, 2)
        elif term_1.lemma_ == 'certain' and term_1._.pos == 'ADJ':
            return (True, 1)

        return (False, 0)

    def _check_booster(self, current):
        for phrase, score in self.lexicon.boosters.items():
            result = self._match_next(current, phrase)
            if result[0]:
                return (score, result[1], result[2])

        return (False, 0, None)

    def _check_negate(self, current):
        for phrase in self.lexicon.negates:
            result = self._match_next(current, phrase)
            if result[0]:
                return (True, result[1], result[2])

        if current + 1 < len(self.text):
            term_1 = self.text[current]
            term_2 = self.text[current + 1]
            if (term_2.lemma_ == 'not' and term_2._.pos == 'ADV'
                    and term_1._.pos == 'VERB'
                    and term_1.lemma_ in self.lexicon.negate_verbs):
                return (True, 2, None)

        return (False, 0, None)

    def _check_contractive(self, current):
        for phrase in self.lexicon.contractives:
            result = self._match_next(current, phrase)
            if result[0]:
                return (True, result[1], result[2])

        return (False, 0, None)

    def _add_word_sentiment(self, current):
        word = self.text[current]

        if word._.pos in [
                'SPACE', 'ADP', 'CONJ', 'CCONJ', 'NUM', 'PART', 'SCONJ', 'PRON'
        ]:
            return

        # handle "is COOL or Nice", might deal using training based method
        if word.ent_type > 0:
            last_texts = [
                self.text[current - i] for i in range(1, 5) if current - i > 0
            ]
            if all(t.lemma_ != 'be' for t in last_texts):
                return

        # spacy sometimes detect no as det
        if word._.pos == 'DET' and word.lemma_ != 'no':
            return

        word, corrections = self._correct(self.text[current])

        score = None
        if corrections:
            score = self.lexicon.lookup(corrections[0])
        else:
            score = self.lexicon.lookup(word.lower_)

        if score is None:
            if not word.is_punct:
                self._sentiments.append(
                    dict(score=0, pos=current, type=TokenType.UNMATCH))
        else:
            # boost if word is upper case
            if self.is_cap_diff and word.is_upper:
                if score >= 0:
                    score += C_INCR
                else:
                    score -= C_INCR

            # apply boosters
            score = self._apply_previous_boosters(current, score)

            # apply negate
            score = self._apply_previous_negate(current, score)

            self._sentiments.append(
                dict(
                    score=score,
                    pos=current,
                    corrections=corrections,
                    type=TokenType.MATCH))

    def _add_phrase_sentiment(self, current):
        found = False
        length = 0
        corrections = None
        score = None

        for phrase, _score in self.lexicon.phrases.items():
            found, length, corrections = self._match_next(current, phrase)
            if found:
                score = _score
                break

        if not found:
            return (False, 0, corrections)

        # boost if word is upper case
        has_upper = any(w.is_upper
                        for w in self.text[current:current + length])
        if self.is_cap_diff and has_upper:
            if score >= 0:
                score += C_INCR
            else:
                score -= C_INCR

        # apply boosters
        score = self._apply_previous_boosters(current, score)

        # apply negate
        score = self._apply_previous_negate(current, score)

        return (score, length, corrections)

    def _apply_previous_boosters(self, current, score):
        lookup_limit = 3
        _score = score
        booster_tokens = self._lookup_previous_tokens(TokenType.BOOSTER,
                                                      lookup_limit)
        for token in booster_tokens:
            distance = current - token['pos']

            if distance > lookup_limit * 1.5:
                continue

            booster_score = token['booster']
            if score < 0:
                booster_score = booster_score * -1

            if self.is_cap_diff and self.text[token['pos']].is_upper:
                if score < 0:
                    booster_score -= C_INCR
                else:
                    booster_score += C_INCR

            decay = 0.05 * max(distance - 1, 0)
            _score += booster_score * (1 - decay)

        return _score

    def _apply_previous_negate(self, current, score):
        lookup_limit = 3
        _score = score
        negate_tokens = self._lookup_previous_tokens(TokenType.NEGATE,
                                                     lookup_limit)
        for token in negate_tokens:
            distance = current - token['pos']

            if distance > lookup_limit * 1.5:
                continue

            _score *= N_SCALAR

        return _score

    def _lookup_previous_tokens(self, token_type, limit):
        tokens = []
        length = len(self._sentiments)
        for i in range(1, limit + 1):
            if length - i >= 0:
                token = self._sentiments[length - i]
                if token['type'] == token_type:
                    tokens.append(token)

        return tokens

    def _check_is_cap_diff(self):
        upper = 0
        count = 0
        toks = (tok for tok in self.text if not tok.is_punct)
        for tok in toks:
            count += 1
            if tok.is_upper:
                upper += 1

        return upper > 1 and upper < count

    def _match_next(self, current, words):
        _words = words.split()
        _corrections = []
        for i, word in enumerate(_words):
            if current + i >= len(self.text):
                return (False, len(_words), None)

            target, corrections = self._correct(self.text[current + i])
            if target.lemma_ == word or target.lower_ == word:
                continue

            if corrections and \
               any(correction == word for correction in corrections):
                logger.debug(f'correct {target.text} to {corrections}')
                _corrections += corrections
                continue

            return (False, len(_words), None)

        return (True, len(_words), _corrections)

    def _correct(self, word):
        if not self.spell_correct:
            return (word, None)

        if word.lower_ in self._corrections:
            return (word, self._corrections[word.lower_])

        # later we should use word.is_oov
        if not is_oov(word.text) or not word.is_alpha or word.is_upper:
            self._corrections[word.lower_] = None
            return (word, None)

        word_corrections = self.lexicon.spell(word.lower_)
        if not word_corrections:  # already correct
            self._corrections[word.lower_] = None
            return (word, None)

        self._corrections[word.lower_] = word_corrections
        return (word, word_corrections)

    def _tokenize(self, raw_text):
        toks = nlp(raw_text)
        matcher = self.matcher
        matcher(toks)
        return toks


Score = namedtuple('Score', 'pos neg neu compound')


class SentenceResult:
    def __init__(self, text, sentiments):
        self.text = text
        self.sentiments = sentiments
        self._scores = None

    @property
    def scores(self):
        if not self._scores:
            if self.sentiments:
                sum_s = float(
                    sum(s['score'] for s in self.sentiments
                        if s['type'] == TokenType.MATCH))

                # compute and add emphasis from punctuation in text
                punct_emph_amplifier = self._punctuation_emphasis()
                if sum_s > 0:
                    sum_s += punct_emph_amplifier
                elif sum_s < 0:
                    sum_s -= punct_emph_amplifier

                # discriminate between positive, negative and
                # neutral sentiment scores
                pos_sum, neg_sum, neu_count = self._sift_sentiment_scores()
                if pos_sum > math.fabs(neg_sum):
                    pos_sum += punct_emph_amplifier
                elif pos_sum < math.fabs(neg_sum):
                    neg_sum -= punct_emph_amplifier

                total = pos_sum + math.fabs(neg_sum) + neu_count

                self._scores = Score(
                    round(fabs_div(pos_sum, total), 3),
                    round(fabs_div(neg_sum, total), 3),
                    round(fabs_div(neu_count, total), 3),
                    round(_normalize(sum_s), 3))
            else:
                self._scores = Score(0.0, 0.0, 0.0, 0.0)

        return self._scores

    def _punctuation_emphasis(self):
        amplifier = 0

        # 1) handle exclamation points
        count = len([tok for tok in self.text if tok.lemma_ == '!'])
        amplifier += min(count, 4) * 0.292

        # 2) handle question mark
        count = len([tok for tok in self.text if tok.lemma_ == '?'])
        amplifier += min(count * 0.18, 0.96)

        return amplifier

    def _sift_sentiment_scores(self):
        # want separate positive versus negative sentiment scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        scores = (s['score'] for s in self.sentiments
                  if s['type'] == TokenType.MATCH)
        for score in scores:
            if score > 0:
                pos_sum += (
                    float(score) + 1
                )  # compensates for neutral words that are counted as 1
            elif score < 0:
                neg_sum += (
                    float(score) - 1
                )  # when used with math.fabs(), compensates for neutrals
            elif score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count

    def debug(self):
        debugs = []
        for sent in self.sentiments:
            word = self.text[sent['pos']]
            debugs.append(f"{sent['score']} {word.text} {sent['type']}")

        return '\n'.join(debugs)


def sentence_sentiment(text, **kwargs):
    result = SentenceAnalyzer(text, **kwargs).analyze()

    return result.scores


def paragraph_sentiment(text, alpha=0.87, **kwargs):
    scores = [
        sentence_sentiment(tok, **kwargs).compound
        for tok in _paragraph_tokenizer(text)
    ]
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


def _normalize(score, alpha=15):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score / math.sqrt((score * score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score


def fabs_div(sum, total):
    if total == 0.0:
        return 0.0

    return math.fabs(sum / total)


def _paragraph_tokenizer(text):
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
