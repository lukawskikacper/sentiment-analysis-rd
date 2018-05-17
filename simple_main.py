# -*- coding: UTF-8 -*-
import copy
import itertools
import logging
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score

from sentiment.loader import load_airlines_data, load_thinknook_data
from sentiment.preprocessing import TwitterTextPreprocessor, StemmingTextPreprocessor
from sentiment.vectorizer import FeatureAndCountVectorizer, PCATfidfVectorizer

# Get logger for current module
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# Load all the messages along with their targets
raw_messages, targets = [], []
loaders = (load_airlines_data, load_thinknook_data, )
for loader in loaders:
    loader_messages, loader_targets = loader()
    raw_messages.extend(loader_messages)
    targets.extend(loader_targets)

# Preprocess the messages
preprocessor = StemmingTextPreprocessor() # TwitterTextPreprocessor()
messages = map(preprocessor.preprocess, raw_messages)
logger.info("Preprocessed all the input messages")

# Merge messages with targets and shuffle
messages_with_targets = list(zip(messages, targets))
random.shuffle(messages_with_targets)
messages, targets = zip(*messages_with_targets)
logger.info("Zipped messages with targets")

# Choose random test dataset
test_fraction = 0.2
test_fraction_index = int(len(messages) * test_fraction)
train_messages, test_messages = messages[:-test_fraction_index], \
                                messages[-test_fraction_index:]
train_targets, test_targets = list(map(int, targets[:-test_fraction_index])), \
                              list(map(int, targets[-test_fraction_index:]))

# Display dataset summary
print("Got %d train and %d test messages" % (len(train_messages),
                                             len(test_messages)))

# Define vectorizers to be tested
vectorizers = [
    CountVectorizer(analyzer="word"),
    TfidfVectorizer(),
    PCATfidfVectorizer(pca_n_component=100, pca_n_iter=2, pca_random_state=42),
    FeatureAndCountVectorizer(analyzer="word")
]

# Create all the classifiers for the comparison
classifiers = [
    RandomForestClassifier(n_estimators=35, verbose=2, n_jobs=6,
                           bootstrap=False, random_state=1,
                           min_samples_leaf=4)
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
        # Convert the messages to features
        train_features = vectorizer.fit_transform(train_messages)
        test_features = vectorizer.transform(test_messages)
        # Train the classifier
        fit = classifier.fit(train_features, train_targets)
        # Perform a prediction on training and test messages
        train_pred = fit.predict(train_features)
        test_pred = fit.predict(test_features)
        # Calculate the accuracy of predictions on training dataset
        accuracy_train = accuracy_score(train_pred, train_targets)
        print(
            "Train accuracy of " + classifier.__class__.__name__ + " is " +
            str(accuracy_train) + " for vectorizer " +
            vectorizer.__class__.__name__)
        # Calculate the accuracy of predictions on test dataset
        accuracy_test = accuracy_score(test_pred, test_targets)
        print(
            "Test accuracy of " + classifier.__class__.__name__ + " is " +
            str(accuracy_test) + " for vectorizer " +
            vectorizer.__class__.__name__)
        result = vectorizer.__class__, classifier.__class__, accuracy_test
    except Exception as e:
        print("Interrupted " + classifier.__class__.__name__ + " due to " + str(e))
        result = vectorizer.__class__, classifier.__class__, None
    return result


# Run the processing for all the possible pairs of vectorizers
# and classifiers, finally collect and display results
results = list(map(process_vectorizer_with_classifier,
                   itertools.product(vectorizers, classifiers)))
for vectorizer, classifier, accuracy_test in results:
    print(
        "Test accuracy of " + classifier.__name__ + " is " +
        str(accuracy_test) + " for vectorizer " +
        vectorizer.__name__)
