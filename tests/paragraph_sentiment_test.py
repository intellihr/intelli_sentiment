from intelli_sentiment import paragraph_sentiment
from tests.util import assert_neg, assert_neu, assert_pos, vader_sentiment


def test_basic_positives():
    text = """
    I feel that my role is impacted not only by myself, but from all of our team, both locally in NSW and across the country.
    I feel like my direct supervisor is responsible for too much within our office and should have some managerial support.
    In essence I feel like he is a talented person in our field, however is wasted by being responsible for low level responsibilities that take up too much time.
    I also am not a fan of some of the new training material we have been provided.
    I feel that it shows that there is a disconnect with what we are as a company and what our clients needs are.
    """.strip()

    assert vader_sentiment(text) < 0
    assert paragraph_sentiment(text) < 0
