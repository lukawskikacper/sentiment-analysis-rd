# -*- coding: UTF-8 -*-
import io
import os
import csv
import logging
import ftfy

from pkg_resources import resource_filename
from zipfile import ZipFile
from sentiment.message import Sentiment

# Get logger for current module
logger = logging.getLogger(__name__)


def airline_sentiment_to_sentiment(airline_sentiment):
    airline_sentiment_lower = airline_sentiment.lower()
    if "positive" == airline_sentiment_lower:
        return Sentiment.POSITIVE
    if "negative" == airline_sentiment_lower:
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
    input_file_path = resource_filename(__name__, "../data/twitter-airlines-sentiment.csv")
    with open(input_file_path, "r") as csvfile:
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
    Loads the data from twitter-thinknook-sentiment.zip file and returns it as
    two iterables - one of raw tweets and another one as a list of targets
    (negative, neutral, positive). The order of both lists is kept.
    :return: messages, targets
    """
    num_messages = 0
    messages, targets = [], []
    input_file_path = resource_filename(__name__, "../data/twitter-thinknook-sentiment.zip")
    with ZipFile(input_file_path, "r") as zipfile:
        with zipfile.open("twitter-thinknook-sentiment.csv", "r") as csvfile:
            reader = csv.reader(io.TextIOWrapper(csvfile), delimiter=",", quotechar="\"")
            headers = None
            for row in reader:
                if 1 == reader.line_num:
                    headers = row
                    continue
                zipped = dict(zip(headers, row))
                messages.append(zipped["SentimentText"].strip())
                targets.append(
                    thinknook_sentiment_to_sentiment(zipped["Sentiment"]))
                num_messages += 1
                if num_messages % 100000 == 0:
                    logger.info("Loaded %d messages from thinknook dataset", num_messages)
    logger.info("Finally: %d messages from thinknook dataset have been loaded", num_messages)
    return messages, targets


def load_emoji_mapping():
    """
    Loads the data from emoji_mapping.properties file and returns it as
    a dict-like structure, where key is a unicode representation of the emoji
    and value is its description.
    :return: emojis mapping
    """
    input_file_path = resource_filename(__name__, "../data/emoji_mapping.properties")
    with open(input_file_path, "r", encoding="unicode_escape", errors="replace") as file:
        return dict(ftfy.fix_text(line).split("=", 1) for line in file)


def load_codete_sentiment_data():
    """
    Loads the data from codete-sentiment-dataset.csv file and returns it as
    two iterables - one of raw tweets and another one as a list of targets
    (negative, neutral, positive). The order of both lists is kept.
    :return: messages, targets
    """
    messages, targets = [], []
    input_file_path = resource_filename(__name__, "../data/codete-sentiment-dataset.csv")
    with open(input_file_path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
        headers = None
        for row in reader:
            if 1 == reader.line_num:
                headers = row
                continue
            zipped = dict(zip(headers, row))
            messages.append(zipped["content"].strip())
            targets.append(
                airline_sentiment_to_sentiment(zipped["most_probable_sentiment"]))
    logger.info("Loaded %d messages from codete dataset", len(messages))
    return messages, targets
