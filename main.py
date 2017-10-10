# -*- coding: UTF-8 -*-
import copy
import random
import dill
import itertools
import logging
import datetime

from multiprocessing import Pool
from loader import load_airlines_data, load_thinknook_data, load_emoji_mapping
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD

from preprocessing import TwitterTextPreprocessor
from vectorizer import FeatureAndCountVectorizer, PCATfidfVectorizer

# Get logger for current module
logger = logging.getLogger(__name__)

# Load all the messages along with their targets
messages, targets = [], []
loaders = (load_airlines_data, load_thinknook_data, )
for loader in loaders:
    loader_messages, loader_targets = loader()
    messages.extend(loader_messages)
    targets.extend(loader_targets)

# Preprocess the messages
preprocessor = TwitterTextPreprocessor()
messages = map(preprocessor.preprocess, messages)

# Merge messages with targets and shuffle
messages_with_targets = list(zip(messages, targets))
random.shuffle(messages_with_targets)
messages, targets = zip(*messages_with_targets)

# Choose random test dataset
test_fraction = 0.2
test_fraction_index = int(len(messages) * test_fraction)
# train_messages, test_messages = messages[:-test_fraction_index], \
#                                 messages[-test_fraction_index:]
# train_targets, test_targets = list(map(int, targets[:-test_fraction_index])), \
#                               list(map(int, targets[-test_fraction_index:]))
train_messages = test_messages = messages
train_targets = test_targets = list(map(int, targets))
del messages_with_targets, messages, targets

# Display dataset summary
print("Got %d train and %d test messages" % (len(train_messages),
                                             len(test_messages)))

# Define vectorizers to be tested
vectorizers = [
    # CountVectorizer(analyzer="word"),
    # TfidfVectorizer(),
    PCATfidfVectorizer(pca_n_component=100, pca_n_iter=2, pca_random_state=42),
    # FeatureAndCountVectorizer(analyzer="word")
]

# Create all the classifiers for the comparison
classifiers = [
    # LogisticRegression(C=0.000000001, solver='liblinear', max_iter=10000),
    # KNeighborsClassifier(3),
    # SVC(kernel="rbf", C=0.025, probability=True),
    # LinearSVC(),
    # DecisionTreeClassifier(),
    # GradientBoostingClassifier(),
    RandomForestClassifier(n_estimators=35, verbose=3, n_jobs=5,
                           bootstrap=False, random_state=1,
                           min_samples_leaf=2),
    # ExtraTreesClassifier(n_estimators=200),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # BernoulliNB()
]


def process_vectorizer_with_classifier(vc):
    """
    Perform the process of creating and validating given classifier with
    provided vectorizer.
    :param vc: a tuple with vectorizer and classifier
    :return: vectorizer, classifier and accuracy
    """
    try:
        vectorizer, classifier = copy.deepcopy(vc)
        train_features = vectorizer.fit_transform(train_messages)
        test_features = vectorizer.transform(test_messages)
        try:
            fit = classifier.fit(train_features, train_targets)
            train_pred = fit.predict(train_features)
            test_pred = fit.predict(test_features)
        except ValueError as e:
            print("Could not fit given model " +
                  classifier.__class__.__name__ + " due to " + str(e))
            return vectorizer.__class__, classifier.__class__, None
        except Exception:
            dense_features = train_features.toarray()
            dense_test = test_features.toarray()
            fit = classifier.fit(dense_features, train_targets)
            train_pred = fit.predict(dense_features)
            test_pred = fit.predict(dense_test)
            del dense_features, dense_test
        accuracy_train = accuracy_score(train_pred, train_targets)
        print(
            "Train accuracy of " + classifier.__class__.__name__ + " is " +
            str(accuracy_train) + " for vectorizer " +
            vectorizer.__class__.__name__)
        accuracy_test = accuracy_score(test_pred, test_targets)
        print(
            "Test accuracy of " + classifier.__class__.__name__ + " is " +
            str(accuracy_test) + " for vectorizer " +
            vectorizer.__class__.__name__)
        result = vectorizer.__class__, classifier.__class__, accuracy_test
    except Exception as e:
        print("Interrupted " + classifier.__class__.__name__ + " due to " + str(e))
        result = vectorizer.__class__, classifier.__class__, None
    pickle_filename = "model_%s_%s_%s.pkl" % (str(classifier.__class__.__name__),
                                              str(vectorizer.__class__.__name__),
                                              str(datetime.datetime.utcnow()))
    with open(pickle_filename, 'wb') as fp:
        dill.dump({
            "preprocessor": preprocessor,
            "classifier": classifier,
            "vectorizer": vectorizer
        }, fp)
    return result


with Pool(2) as pool:
    results = pool.map(process_vectorizer_with_classifier,
                       itertools.product(vectorizers, classifiers))
