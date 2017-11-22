from intelli_sentiment import paragraph_sentiment
from tests.util import (assert_neg, assert_neu, assert_pos, vader_sentiment,
                        run_evaluation, read_dataset)


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


def test_with_f5mgroup_dataset():
    # evaluation = run_evaluation('sentiment_dataset_f5mgroup.tsv',
    #                             vader_sentiment)
    #
    # print(evaluation.results)
    #
    # evaluation = run_evaluation('sentiment_dataset_f5mgroup.tsv',
    #                             paragraph_sentiment)
    #
    # print(evaluation.results)
    dataset = read_dataset('sentiment_dataset_f5mgroup.tsv')
    for score, text in dataset:
        s1 = vader_sentiment(text)
        s2 = paragraph_sentiment(text)
        if (s1 < 0 and s2 >=0) or (s1 >= 0 and s2 < 0):
            print(f'{score} {s1} {s2} : {text}')
