from spacy.matcher import Matcher


def build_matcher(nlp, lexicon):
    matcher = Matcher(nlp.vocab)

    # handle re- rule
    matcher.add('PLEASE_RE', _mark_please_interjection, [{
        'LOWER': 'please'
    }, {
        'LOWER': 're'
    }, {
        'ORTH': '-'
    }, {
        'IS_ALPHA': True
    }])

    matcher.add('RE', _merge_words, [{
        'LOWER': 're'
    }, {
        'ORTH': '-'
    }, {
        'IS_ALPHA': True
    }])

    return matcher


def _merge_words(matcher, doc, i, matches):
    _, start, end = matches[i]
    span = doc[start:end]
    span.merge()


def _mark_please_interjection(matcher, doc, i, matches):
    _, start, end = matches[i]
    doc[start]._._custom_pos = 'INTJ'
