import re
import logging

from nltk.corpus import stopwords
from sentiment.loader import load_emoji_mapping

# Get logger for current module
logger = logging.getLogger(__name__)


class TextPreprocessor(object):
    def preprocess(self, message):
        raise NotImplementedError("Method .preprocess is not implemented")


class TwitterTextPreprocessor(TextPreprocessor):

    ENGLISH_STOPWORDS = stopwords.words("english")
    HASHTAG_REGEX = re.compile("#([a-z0-9_]+)", re.UNICODE)
    MENTION_REGEX = re.compile("@([a-z0-9_]+)", re.UNICODE)
    URL_REGEX = re.compile(r"(?:(http://)|(www\.))(\S+\b/?)"
                           r"([!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]*)(\s|$)",
                           re.UNICODE | re.I)
    EMOJI_MAPPING = load_emoji_mapping()

    def preprocess(self, raw_message):
        message = raw_message.lower()
        for emoji, text_description in self.EMOJI_MAPPING.items():
            message = message.replace(emoji, text_description)
        message = self.HASHTAG_REGEX.sub("\\1", message)
        message = self.MENTION_REGEX.sub("twitter_account", message)
        message = self.URL_REGEX.sub("external_link ", message)
        logger.debug("Message '%s' preprocessed to '%s'", raw_message, message)
        return message
