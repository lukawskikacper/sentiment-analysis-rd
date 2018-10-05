import os

from setuptools import setup


with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    required = f.read().splitlines()

setup(
    name='sentiment-analysis-rd',
    version='1.2.0',
    package_dir={'sentiment': './sentiment',
                 'sentiment.data': './sentiment/data',
                 'sentiment.loader': './sentiment/loader',
                 'sentiment.message': './sentiment/message',
                 'sentiment.preprocessing': './sentiment/preprocessing',
                 'sentiment.tokenizing': './sentiment/tokenizing',
                 'sentiment.vectorizer': './sentiment/vectorizer'},
    packages=['sentiment', 'sentiment.data', 'sentiment.loader', 'sentiment.message', 'sentiment.tokenizing',
              'sentiment.preprocessing', 'sentiment.vectorizer'],
    package_data={"sentiment.data": ["emoji_mapping.properties", "twitter-airlines-sentiment.csv",
                                     "twitter-thinknook-sentiment.zip"]},
    install_requires=required,
    extras_require={},
    license='MIT License',
    description='A sentiment analysis library',
    long_description='A sentiment analysis library',
    url='https://github.com/lukawskikacper/sentiment-analysis-rd',
    author='Kacper Łukawski',
    author_email='kacper.lukawski@codete.com',
    entry_points={},
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
