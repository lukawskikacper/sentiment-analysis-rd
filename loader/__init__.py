# -*- coding: UTF-8 -*-
import csv
import logging
import ftfy

from configparser import ConfigParser
from itertools import chain

from message import Sentiment

# Get logger for current module
logger = logging.getLogger(__name__)


def airline_sentiment_to_sentiment(airline_sentiment):
    if "positive" == airline_sentiment:
        return Sentiment.POSITIVE
    if "negative" == airline_sentiment:
        return Sentiment.NEGATIVE
    return Sentiment.NEUTRAL


def thinknook_sentiment_to_sentiment(thinknook_sentiment):
    thinknook_sentiment_int = int(thinknook_sentiment)
    if 1 == thinknook_sentiment_int:
        return Sentiment.POSITIVE
    if 0 == thinknook_sentiment_int:
        return Sentiment.NEGATIVE
    return Sentiment.NEUTRAL


def load_airlines_data():
    """
    Loads the data from twitter-airlines-sentiment.csv file and returns it as
    two iterables - one of raw tweets and another one as a list of targets
    (negative, neutral, positive). The order of both lists is kept.
    :return: messages, targets
    """
    messages, targets = [], []
    with open("data/twitter-airlines-sentiment.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
        headers = None
        for row in reader:
            if 1 == reader.line_num:
                headers = row
                continue
            zipped = dict(zip(headers, row))
            messages.append(zipped["text"].strip())
            targets.append(
                airline_sentiment_to_sentiment(zipped["airline_sentiment"]))
    logger.info("Loaded %d messages from airlines dataset", len(messages))
    return messages, targets


def load_thinknook_data():
    """
    Loads the data from twitter-thinknook-sentiment.csv file and returns it as
    two iterables - one of raw tweets and another one as a list of targets
    (negative, neutral, positive). The order of both lists is kept.
    :return: messages, targets
    """
    messages, targets = [], []
    with open("data/twitter-thinknook-sentiment.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
        headers = None
        num_messages = 0
        for row in reader:
            if 1 == reader.line_num:
                headers = row
                continue
            zipped = dict(zip(headers, row))
            messages.append(zipped["SentimentText"].strip())
            targets.append(
                thinknook_sentiment_to_sentiment(zipped["Sentiment"]))
            num_messages += 1
            if num_messages == 100000:
                break
    logger.info("Loaded %d messages from thinknook dataset", len(messages))
    return messages, targets


def load_emoji_mapping():
    """
    Loads the data from emoji_mapping.properties file and returns it as
    a dict-like structure, where key is a unicode representation of the emoji
    and value is its description.
    :return: emojis mapping
    """
    with open("data/emoji_mapping.properties", "r",
              encoding="unicode_escape", errors="replace") as file:
        return dict(ftfy.fix_text(line).split("=", 1) for line in file)
