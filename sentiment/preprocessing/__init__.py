import re
import html
import logging

from nltk import PorterStemmer
from nltk.corpus import stopwords
from sentiment.loader import load_emoji_mapping

# Get logger for current module
logger = logging.getLogger(__name__)


class TextPreprocessor(object):
    """
    An abstract text preprocessor.
    """

    def preprocess(self, message):
        raise NotImplementedError("Method preprocess is not implemented")

    def __call__(self, message):
        """Calls the preprocess method on given message. It is done to fulfill the scikit design."""
        return self.preprocess(message)

    def __str__(self):
        return "{}()".format(self.__class__.__name__)


class TwitterTextPreprocessor(TextPreprocessor):
    """
    A preprocessor which performs the following operations:
     - converts texts to lowercase
     - unescape HTML characters
     - replaces emojis with their textual representation
     - removes urls
     - changes hashtags to extract the words
     - removes special characters
     - removes duplicated letters
    """

    EMOJI_MAPPING = load_emoji_mapping()
    ENGLISH_STOPWORDS = stopwords.words("english")
    URL_REGEX = re.compile(r"(?:(http://)|(www\.))(\S+\b/?)"
                           r"([!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]*)(\s|$)",
                           re.UNICODE | re.I)
    HASHTAG_REGEX = re.compile(r"#([^ ]+)", re.UNICODE | re.I)
    MENTION_REGEX = re.compile(r"@([a-z0-9_]+)", re.UNICODE | re.I)
    SPECIAL_CHARACTERS_REGEX = re.compile(r"\.|…|!|,|\?|:|\-|\(|\)|/|\\|\+|\*|~|`|\"|'|\[|\]|=|;", re.UNICODE | re.I)
    DUPLICATED_LETTER_REGEX = re.compile(r"([a-zA-Z])\1+", re.UNICODE | re.I)

    def preprocess(self, raw_message):
        message = raw_message.lower()
        # encoding HTML entitites
        message = html.unescape(message.strip())
        # replace emojis with their textual representation
        for emoji, text_description in self.EMOJI_MAPPING.items():
            message = message.replace(emoji, text_description)
        # removing non-informative parts
        message = self.URL_REGEX.sub("", message)
        message = self.HASHTAG_REGEX.sub("\\1", message)
        message = self.MENTION_REGEX.sub("", message)
        message = self.SPECIAL_CHARACTERS_REGEX.sub(" ", message)
        # handling recognized common behaviour
        message = self.DUPLICATED_LETTER_REGEX.sub("\\1", message)
        return message.strip()


class StemmingTextPreprocessor(TwitterTextPreprocessor):
    """
    An extension of TwitterTextProcessor which additionally performs stemming.
    """

    def __init__(self):
        super().__init__()
        self._stemmer = PorterStemmer()

    def preprocess(self, raw_message):
        preprocessed_message = super().preprocess(raw_message)
        words = preprocessed_message.split()
        message = " ".join(map(self._stemmer.stem, words))
        return message
