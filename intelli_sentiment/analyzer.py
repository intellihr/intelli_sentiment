import math
import sys
import logging
from enum import Enum
from collections import namedtuple

import spacy
from autocorrect import spell

from intelli_sentiment.vader_lexicon import lexicon

# (empirically derived mean sentiment intensity rating increase for using
# ALLCAPs to emphasize a word)
C_INCR = 0.733

N_SCALAR = -0.74

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger()

nlp = spacy.load('en')


class TokenType(Enum):
    UNMATCH = 0
    MATCH = 1
    BOOSTER = 2
    NEGATE = 3
    CONTRACTIVE = 4


class SentenceAnalyzer:
    def __init__(self, raw_text):
        self.text = nlp(raw_text)
        self.is_cap_diff = self._check_is_cap_diff()
        self._sentiments = []

    def analyze(self):
        i = 0
        while i < len(self.text):
            # 1) resolve long phrases first
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

            # 2) check if contractive phrases
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

            # 3) check if it is negate phrase
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

            # 4) check if it contains booster phrase
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

    def _check_booster(self, current):
        for phrase, score in lexicon.boosters.items():
            result = self._match_next(current, phrase)
            if result[0]:
                return (score, result[1], result[2])

        return (False, 0, None)

    def _check_negate(self, current):
        for phrase in lexicon.negates:
            result = self._match_next(current, phrase)
            if result[0]:
                return (True, result[1], result[2])

        if current + 1 < len(self.text):
            term_1 = self.text[current]
            term_2 = self.text[current + 1]
            if (term_2.lemma_ == 'not' and term_2.pos_ == 'ADV'
                    and term_1.pos_ == 'VERB'
                    and term_1.lemma_ in lexicon.negate_verbs):
                return (True, 2, None)

        return (False, 0, None)

    def _check_contractive(self, current):
        for phrase in lexicon.contractives:
            result = self._match_next(current, phrase)
            if result[0]:
                return (True, result[1], result[2])

        return (False, 0, None)

    def _add_word_sentiment(self, current):
        word = self.text[current]

        if word.pos_ in [
                'SPACE', 'ADP', 'CONJ', 'CCONJ', 'DET', 'NUM', 'PART', 'SCONJ',
                'X', 'PRON'
        ]:
            return

        word, corrections = self._correct(self.text[current])

        score = None
        if corrections:
            score = lexicon.lookup(corrections[0])
        else:
            score = lexicon.lookup(word.lower_)

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

        for phrase, _score in lexicon.phrases.items():
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

            target, corrections = self._correct(self.text[current + i], [word])
            if target.lemma_ == word or target.lower_ == word:
                continue

            if corrections and \
                any(correction == word for correction in corrections):
                logger.debug(f'correct {target.text} to {corrections}')
                _corrections += corrections
                continue

            return (False, len(_words), None)

        return (True, len(_words), _corrections)

    def _correct(self, word, hints=[]):
        if not word.is_oov or word.is_punct:
            return (word, None)

        if hints and (word.lower_ in hints or word.lemma_ in hints):
            return (word, None)
        elif not hints and (lexicon.contains(word.lower_)
                            or lexicon.contains(word.lemma_)):
            return (word, None)

        # attempt word correction
        corrected_words = (spell(word.lower_), spell(word.lemma_))
        if word.lower_ != corrected_words[0]:
            if hints and (corrected_words[0] in hints
                          or corrected_words[1] in hints):
                return (word, corrected_words)
            elif not hints and (lexicon.contains(corrected_words[0])
                                or lexicon.contains(corrected_words[1])):
                return (word, corrected_words)

        return (word, None)


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

                # discriminate between positive, negative and neutral sentiment scores
                pos_sum, neg_sum, neu_count = self._sift_sentiment_scores()
                if pos_sum > math.fabs(neg_sum):
                    pos_sum += punct_emph_amplifier
                elif pos_sum < math.fabs(neg_sum):
                    neg_sum -= punct_emph_amplifier

                total = pos_sum + math.fabs(neg_sum) + neu_count

                self._scores = Score(
                    round(math.fabs(pos_sum / total), 3),
                    round(math.fabs(neg_sum / total), 3),
                    round(math.fabs(neu_count / total), 3),
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


def sentence_sentiment(text):
    result = SentenceAnalyzer(text).analyze()

    return result.scores


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
