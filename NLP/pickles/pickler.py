# Repickles all of the values of sentiment.py

import random
import os
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from statistics import mode

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

folder_name = "short_reviews"
pos_path = "/Users/dsiegler19/PycharmProjects/Language Processing and Machine Learning/NLP/short_reviews/positive.txt"
neg_path = "/Users/dsiegler19/PycharmProjects/Language Processing and Machine Learning/NLP/short_reviews/negative.txt"

pos = open(pos_path, encoding="utf-8", errors="replace").read()
neg = open(neg_path, encoding="utf-8", errors="replace").read()

if pos_path.__eq__("-") | neg_path.__eq__("-"):
    pos = open(os.getcwd() + "/short_reviews/positive.txt", "r", encoding="utf-8", errors="replace").read()
    neg = open(os.getcwd() + "/short_reviews/negative.txt", "r", encoding="utf-8", errors="replace").read()

all_words = []
documents = []

#  J is adjective, R is adverb, and V is verb
allowed_word_types = ["J", "V"]

for p in pos.split('\n'):
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in neg.split('\n'):
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

documents_f = open(os.getcwd() + "/" + folder_name + "/documents.pickle", "wb")
pickle.dump(documents, documents_f)
documents_f.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

word_features_f = open(os.getcwd() + "/" + folder_name + "/word_features.pickle", "wb")
pickle.dump(word_features, word_features_f)
word_features_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

testing_set = featuresets[10000:]
training_set = featuresets[:10000]

MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_classifier.train(training_set)
print("MultinomialNB_classifier accuracy percent:", nltk.classify.accuracy(MultinomialNB_classifier, testing_set) * 100)

save_classifier = open(os.getcwd() + "/" + folder_name + "/MultinomialNB_classifier.pickle", "wb")
pickle.dump(MultinomialNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", nltk.classify.accuracy(BernoulliNB_classifier, testing_set) * 100)

save_classifier = open(os.getcwd() + "/" + folder_name + "/BernoulliNB_classifier.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", nltk.classify.accuracy(LogisticRegression_classifier, testing_set) * 100)

save_classifier = open(os.getcwd() + "/" + folder_name + "/LogisticRegression_classifier.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", nltk.classify.accuracy(LinearSVC_classifier, testing_set) * 100)

save_classifier = open(os.getcwd() + "/" + folder_name + "/LinearSVC_classifier.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", nltk.classify.accuracy(NuSVC_classifier, testing_set) * 100)

save_classifier = open(os.getcwd() + "/" + folder_name + "/NuSVC_classifier.pickle", "wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()
