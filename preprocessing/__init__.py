import re

from nltk.corpus import stopwords


class TextPreprocessor(object):
    def preprocess(self, message):
        raise NotImplementedError("Method .preprocess is not implemented")


class TwitterTextPreprocessor(TextPreprocessor):

    ENGLISH_STOPWORDS = stopwords.words('english')
    HASHTAG_REGEX = re.compile("#([a-z0-9_]+)", re.UNICODE)
    MENTION_REGEX = re.compile("@([a-z0-9_]+)", re.UNICODE)
    URL_REGEX = re.compile(r"(?:(http://)|(www\.))(\S+\b/?)"
                           r"([!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]*)(\s|$)",
                           re.UNICODE | re.I)

    def preprocess(self, message):
        message = message.lower()
        message = self.HASHTAG_REGEX.sub("\\1", message)
        message = self.MENTION_REGEX.sub("twitter_account", message)
        message = self.URL_REGEX.sub("external_link ", message)
        message = " ".join(word for word in message.split()
                           if word not in self.ENGLISH_STOPWORDS)
        return message