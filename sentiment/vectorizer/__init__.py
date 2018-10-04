import six
import numpy as np
import logging

from scipy.sparse import hstack
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, VectorizerMixin
from sentiment.loader import load_emoji_mapping

# Get logger for current module
logger = logging.getLogger(__name__)


class CountingVectorizer(CountVectorizer):
    """
    Extended CountVectorizer that additionally fits the preprocessor while fitting itself.
    """

    def fit_transform(self, raw_documents, y=None):
        if self.tokenizer is not None:
            preprocess = self.build_preprocessor()
            self.tokenizer.fit(map(preprocess, raw_documents))
        return super().fit_transform(raw_documents, y)


class TFIDFVectorizer(TfidfVectorizer):
    """
    Extended TfidfVectorizer that fits the preprocessor while fitting itself.
    """

    def fit_transform(self, raw_documents, y=None):
        if self.tokenizer is not None:
            preprocess = self.build_preprocessor()
            self.tokenizer.fit(map(preprocess, raw_documents))
        return super().fit_transform(raw_documents, y)


class PCATfidfVectorizer(TFIDFVectorizer):
    """
    An extension of TFIDFVectorizer with additional PCA, which limits the dimensionality.
    """

    def __init__(self, pca_n_component=500, pca_n_iter=2, pca_random_state=42, **kwargs):
        super().__init__(input, **kwargs)
        self._pca = TruncatedSVD(n_components=pca_n_component, n_iter=pca_n_iter, random_state=pca_random_state)

    def fit(self, raw_documents, y=None):
        vectorized_documents = self.fit_transform(raw_documents, y)
        return self._pca.fit(vectorized_documents, y)

    def transform(self, raw_documents, copy=True):
        vectorized_documents = super().transform(raw_documents)
        return self._pca.transform(vectorized_documents)

    def fit_transform(self, raw_documents, y=None):
        vectorized_document = super().fit_transform(raw_documents, y)
        return self._pca.fit_transform(vectorized_document, y)


class FeatureAndCountVectorizer(CountVectorizer):
    """
    An extended version of CountVectorizer which additionally puts general
    features of given texts.
    """

    def __init__(self, **kwargs):
        super().__init__(input, **kwargs)
        self._features = [
            (self._length_feature, None),
            (self._character_count_feature, ("!",)),
            (self._character_count_feature, ("...",)),
            (self._character_count_feature, ("?",))
        ]
        for emoji, description in load_emoji_mapping().items():
            self._features.append((self._character_count_feature, (emoji, )))

    def fit_transform(self, raw_documents, y=None):
        if self.tokenizer is not None:
            preprocess = self.build_preprocessor()
            self.tokenizer.fit(map(preprocess, raw_documents))
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


class Doc2VecVectorizer(BaseEstimator, VectorizerMixin):
    """
    A vectorizer performing a document embedding, so called doc2vec.
    """

    def fit(self, raw_documents, y=None):
        self.fit(raw_documents, y)
        return self

    def fit_transform(self, raw_documents, y=None):
        if self.tokenizer is not None:
            self.tokenizer.fit(raw_documents)
        X = np.matrix
        return X

    def transform(self, raw_documents):
        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        X = np.matrix
        return X
