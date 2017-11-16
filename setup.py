import setuptools

try:
   import pypandoc
   readme = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError, OSError, RuntimeError):
   readme = ''

setuptools.setup(
    name="intelli_sentiment",
    version="0.0.1",
    url="https://github.com/intellihr/intelli_sentiment",

    author="Soloman Weng",
    author_email="soloman1124@gmail.com",

    description="Sentiment Predictor Based on Vader",
    long_description=readme,

    packages=setuptools.find_packages(),

    install_requires=['spacy', 'autocorrect'],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
