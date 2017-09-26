import numpy as np

from scipy.sparse import hstack

from sklearn.feature_extraction.text import CountVectorizer

from loader import load_emoji_mapping


class FeatureAndCountVectorizer(CountVectorizer):
    """
    An extended version of CountVectorizer which additionally puts general
    features of given texts.
    """

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, stop_words=None,
                 token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1),
                 analyzer='word', max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):
        super().__init__(input, encoding, decode_error, strip_accents,
                         lowercase, preprocessor, tokenizer, stop_words,
                         token_pattern, ngram_range, analyzer, max_df, min_df,
                         max_features, vocabulary, binary, dtype)
        self._features = [
            (self._length_feature, None),
            (self._character_count_feature, ("!",)),
            (self._character_count_feature, ("...",)),
            (self._character_count_feature, ("?",))
        ]
        for emoji, description in load_emoji_mapping().items():
            self._features.append((self._character_count_feature, (emoji, )))

    def fit_transform(self, raw_documents, y=None):
        count_vectors = super().fit_transform(raw_documents, y)
        extended_vectors = self._append_text_features(count_vectors,
                                                      raw_documents)
        return extended_vectors

    def transform(self, raw_documents):
        count_vectors = super().transform(raw_documents)
        extended_vectors = self._append_text_features(count_vectors,
                                                      raw_documents)
        return extended_vectors

    def _append_text_features(self, count_vectors, raw_documents):
        extended_vectors = count_vectors
        for feature_cb, args in self._features:
            args = () if args is None else args
            vectors_feature = np.matrix(feature_cb(raw_documents, *args)) \
                .reshape((len(raw_documents), 1))
            extended_vectors = hstack([count_vectors, vectors_feature])
        return extended_vectors

    def _length_feature(self, raw_documents):
        return list(map(len, raw_documents))

    def _character_count_feature(self, raw_documents, character):
        return list(map(lambda x: x.count(character), raw_documents))
