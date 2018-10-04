import unittest
import logging

from sentiment.tokenizing import PhraseTokenizer, Token

# Get the root logger instance and configure it
root_logger = logging.getLogger()
root_logger.addHandler(logging.StreamHandler())
root_logger.setLevel(logging.DEBUG)


class TestPhraseTokenizer(unittest.TestCase):

    DOCUMENTS = (
        "it is my last call",
        "is it last call of mine",
        "this is mine",
        "this this this"
    )

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.tokenizer = None

    def setUp(self):
        super(TestPhraseTokenizer, self).setUp()
        self.tokenizer = PhraseTokenizer(max_length=2, dicounting_coeff=1, phrase_score_threshold=0.05)

    def test_fit_reads_proper_sequences(self):
        """
        Checks if model recognizes collocations which occur much often than their single components.
        """
        # Prepare
        self.tokenizer.fit(self.DOCUMENTS)

        # Check
        self.assertTrue(Token("last") in self.tokenizer._vocabulary)
        self.assertTrue(Token("call") in self.tokenizer._vocabulary)
        self.assertTrue(Token("last") + Token("call") in self.tokenizer._vocabulary)

    def test_tokenize_return_all_recognized_tokens(self):
        """
        Checks if tokenize method, after being fitted, recognizes the collocations properly.
        """
        # Prepare
        self.tokenizer.fit(self.DOCUMENTS)

        # Run
        tokens = self.tokenizer.tokenize("The last call of mine")

        # Check
        self.assertTrue(Token("The") in tokens)
        self.assertTrue(Token("last") in tokens)
        self.assertTrue(Token("call") in tokens)
        self.assertTrue(Token("last") + Token("call") in tokens)
        self.assertTrue(Token("of") in tokens)
        self.assertTrue(Token("mine") in tokens)
