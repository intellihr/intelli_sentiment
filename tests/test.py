import sys

from intelli_sentiment import SentenceAnalyzer

def test():
    text = """
    it is kind of good!
    """.strip()
    print(SentenceAnalyzer(text).analyze())

if __name__ == '__main__':
    test()
