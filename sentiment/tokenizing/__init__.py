import logging
import collections

from typing import Sequence

# Creates the package-level logger
from sentiment.tokenizing.exception import WrongArgumentException, EmptyTextValueException, RedundantValueException

logger = logging.getLogger(__name__)


class Token(object):
    """
    A basic unit of the text. Can be thought to be a single word, but can be anything that is semantically atomic.
    """

    TOKEN_SEPARATOR = " "

    def __init__(self, text=None, subtokens=None):
        if (text is None or 0 == len(text)) and \
                (subtokens is None or 0 == len(subtokens)):
            raise EmptyTextValueException(
                "Cannot instantiate a Token instance with text: '{}' and subtokens: '{}'".format(text, subtokens))
        if text is not None and len(text) > 0 and \
                subtokens is not None and len(subtokens) > 0:
            raise RedundantValueException("Either text or subtokens has to be provided. Both given.")
        text_value = text if text is not None and len(text) > 0 else self.TOKEN_SEPARATOR.join(map(str, subtokens))
        self._text_value = text_value
        self._subtokens = subtokens

    @classmethod
    def from_string(cls, string):
        """Creates an instance of Token for given string."""
        return Token(text=string)

    @classmethod
    def from_subtokens(cls, *args):
        """Creates an instance of Token from given subtokens."""
        return Token(subtokens=args)

    def __eq__(self, other):
        if not isinstance(other, Token) or other is None:
            return False
        return self._text_value == self._text_value

    def __hash__(self):
        return (hash(self._text_value) + 7) >> 5

    def __str__(self):
        return self._text_value

    def __repr__(self):
        return "Token(text={})".format(self._text_value)

    def __add__(self, other):
        if not isinstance(other, Token) or other is None:
            raise WrongArgumentException("Given object {} is not valid Token instance".format(other))
        if self.length() > 1:
            return Token.from_subtokens(*(self._subtokens + (other, )))
        return Token.from_subtokens(self, other)

    def __len__(self):
        return self.length()

    def length(self):
        return 1 if self._subtokens is None or len(self._subtokens) == 0 else len(self._subtokens)


class BaseTokenizer(object):
    """
    A base class for all the tokenizers. A tokenizer is a class which divides given string into basic unit. For
    the simplest case, tokens may be thought to be the words of given sentences.
    """

    def fit(self, documents):
        """
        Fits the tokenizer on given documents. Applicable not for all the tokenizers, as some simplest ones may just use
        hard-coded or heuristic approach, but still has to be implemented in all the inherited classes.
        :param documents: an iterable of strings with the documents to be tokenized
        """
        raise NotImplementedError("Method fit of the {} class is not implemented".format(self.__class__))

    def tokenize(self, document: str) -> Sequence[Token]:
        """
        Tokenizes given document and returns an iterable of all the found tokens.
        :param document: string to be tokenized
        :return: an Sequence of Token instances
        """
        raise NotImplementedError("Method tokenize of the {} class is not implemented".format(self.__class__))

    def __call__(self, document):
        """Calls the tokenize method on given document. Made to fulfill the scikit internal convention."""
        return self.tokenize(document)

    def __str__(self):
        return "{}()".format(self.__class__.__name__)


class SplitTokenizer(BaseTokenizer):
    """
    A simple tokenizer using built-in Python split method of string.
    """

    def fit(self, documents):
        """This kind of tokenizer does not require any training"""
        return

    def tokenize(self, document: str) -> Sequence[Token]:
        """Splits the document by white-characters"""
        for substring in document.split():
            yield Token.from_string(substring)


class PhraseTokenizer(BaseTokenizer):
    """
    A tokenizer which treats words which are commonly collocated, as if they were a single word. It is done in order to
    recognize words which do not make too much sense without the context.
    """

    def __init__(self, max_length=2, dicounting_coeff=2, phrase_score_threshold=0.1):
        self._max_length = max_length
        self._vocabulary = set()
        self._dicounting_coeff = dicounting_coeff
        self._phrase_score_threshold = phrase_score_threshold

    def fit(self, documents):
        # Count the occurences of subtokens
        temp_vocabulary = collections.defaultdict(int)
        for document in (self._split_words(document) for document in documents):
            for subtokens in self._iterate_tokens(document):
                temp_vocabulary[subtokens] += 1
        # Print temporary vocabulary
        logger.debug("Temporary vocabulary: {}".format(temp_vocabulary))
        # Remove the collocations which are not frequent enough
        self._vocabulary = set()
        for token_tuple, occurences in temp_vocabulary.items():
            if len(token_tuple) == 1:
                # Base tokens, like words, should be always present
                token = token_tuple[0]
                logger.debug("Adding single token to the vocabulary: {}".format(token))
                self._vocabulary.add(token)
                continue
            # Handling tokens combined from many subtokens
            prefix_subtoken_tuple, last_token = token_tuple[:-1], token_tuple[-1:]
            denominator = temp_vocabulary[prefix_subtoken_tuple] * temp_vocabulary[token_tuple]
            score = (occurences - self._dicounting_coeff) / denominator
            logger.debug("Score {} for token: {}".format(score, token_tuple))
            # Include only tokens with the score higher than given threshold
            if score > self._phrase_score_threshold:
                token = self._add_subtokens(token_tuple)
                logger.debug("Adding collocation to the vocabulary: {}".format(token))
                self._vocabulary.add(token)
        # Display the statistics of collected vocabulary
        self.__display_vocabulary_stats()

    def tokenize(self, document: str) -> Sequence[Token]:
        document_tokens = []
        for subtokens in self._iterate_tokens(self._split_words(document)):
            token = self._add_subtokens(subtokens)
            if token not in self._vocabulary:
                continue
            document_tokens.append(token)
        return document_tokens

    def __str__(self):
        return "{}(max_length={}, dicounting_coeff={}, phrase_score_threshold={})".format(self.__class__.__name__,
                                                                                          self._max_length,
                                                                                          self._dicounting_coeff,
                                                                                          self._phrase_score_threshold)

    def _split_words(self, document: str):
        """Perform basic split of given document to basic, word-like, units."""
        return tuple(Token.from_string(word) for word in document.split())

    def _iterate_tokens(self, words: Sequence[Token]):
        """Iterates through all the tokens in given document and yields them."""
        for entry_length in range(self._max_length):
            document_list = list(words)
            document_length = len(document_list)
            # Count the occurences of the subtoken sequences, in order to find collocations
            for start_index in range(document_length - entry_length):
                subtokens = tuple(document_list[start_index:start_index + entry_length + 1])
                logger.debug("start = {}, end = {}, length = {}, "
                             "subtokens = {}".format(start_index, start_index + entry_length + 1,
                                                     document_length, subtokens))
                yield subtokens

    def _add_subtokens(self, token_tuple):
        """Adds all the tokens and creates a single token representing a collocation."""
        if len(token_tuple) == 0:
            return None
        token = token_tuple[0]
        for subtoken in token_tuple[1:]:
            token = token + subtoken
        return token

    def __display_vocabulary_stats(self):
        """Displays a summary of collected collocations as a dict, where keys are the token lengths."""
        counter = collections.defaultdict(int)
        for token in self._vocabulary:
            token_length = token.length()
            counter[token_length] += 1
        logger.info("Collected vocabulary of the following token lengths: {}".format(counter))
