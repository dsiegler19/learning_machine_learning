# A trainable text classifier to determine whether a piece of writing is positive or negative

import nltk
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI
import pickle
from statistics import mode
import os

# TODO:
# Add more classifier functions
# Retrain to include verbs (not just adjectives)
# Make a sentiment analysis for finance articles

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

folder_name = "-"

def set_pickle_location(name):
    global folder_name
    folder_name = name

documents = None
word_features = None

MultinomialNB_classifier = None
BernoulliNB_classifier = None
LogisticRegression_classifier = None
LinearSVC_classifier = None
NuSVC_classifier = None
voted_classifier = None

def load_pickles():

    global documents
    global word_features

    global MultinomialNB_classifier
    global BernoulliNB_classifier
    global LogisticRegression_classifier
    global LinearSVC_classifier
    global NuSVC_classifier
    global voted_classifier

    documents_f = open(os.getcwd() + "/pickles/" + folder_name + "/documents.pickle", "rb")
    documents = pickle.load(documents_f)
    documents_f.close()

    word_features_f = open(os.getcwd() + "/pickles/" + folder_name + "/word_features.pickle", "rb")
    word_features = pickle.load(word_features_f)
    word_features_f.close()

    MultinomialNB_classifier_f = open(os.getcwd() + "/pickles/" + folder_name + "/MultinomialNB_classifier.pickle", "rb")
    MultinomialNB_classifier = pickle.load(MultinomialNB_classifier_f)
    MultinomialNB_classifier_f.close()

    BernoulliNB_classifier_f = open(os.getcwd() + "/pickles/" + folder_name + "/BernoulliNB_classifier.pickle", "rb")
    BernoulliNB_classifier = pickle.load(BernoulliNB_classifier_f)
    BernoulliNB_classifier_f.close()

    LogisticRegression_classifier_f = open(os.getcwd() + "/pickles/" + folder_name + "/LogisticRegression_classifier.pickle", "rb")
    LogisticRegression_classifier = pickle.load(LogisticRegression_classifier_f)
    LogisticRegression_classifier_f.close()

    LinearSVC_classifier_f = open(os.getcwd() + "/pickles/" + folder_name + "/LinearSVC_classifier.pickle", "rb")
    LinearSVC_classifier = pickle.load(LinearSVC_classifier_f)
    LinearSVC_classifier_f.close()

    NuSVC_classifier_f = open(os.getcwd() + "/pickles/" + folder_name + "/NuSVC_classifier.pickle", "rb")
    NuSVC_classifier = pickle.load(NuSVC_classifier_f)
    NuSVC_classifier_f.close()

    voted_classifier = VoteClassifier(MultinomialNB_classifier, BernoulliNB_classifier,
                    LogisticRegression_classifier, LinearSVC_classifier, NuSVC_classifier)

all_words = []

all_words = nltk.FreqDist(all_words)

# This method returns a set of tuples which each contain 1 word in the review and whether that word is in the top
# 5000 words (by frequency) or not
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for wrd in word_features:
        features[wrd] = (wrd in words)

    return features

def sentiment(text):

    feats = find_features(text)
    try:
        return voted_classifier.classify(feats), voted_classifier.confidence(feats)
    except:
        return (0, 0)
