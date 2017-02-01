# For tuning parameters

import nltk
import random
import os
from statistics import mode
from NLP.opinion_lexicon_classifier import OpinionLexiconClassifier
from nltk.classify import DecisionTreeClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
from sklearn.svm import NuSVC
from sklearn import tree

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

def bigrams(text):
    bigrams = []
    for i in range(len(text) - 1):
        bigrams.append([text[i] + " " + text[i + 1]])

    return bigrams

folder_name = "short_reviews"
pos_path = "/Users/dsiegler19/PycharmProjects/Language Processing and Machine Learning/NLP/short_reviews/positive.txt"
neg_path = "/Users/dsiegler19/PycharmProjects/Language Processing and Machine Learning/NLP/short_reviews/negative.txt"

pos = open(pos_path, encoding="utf-8", errors="replace").read()
neg = open(neg_path, encoding="utf-8", errors="replace").read()

if pos_path.__eq__("-") | neg_path.__eq__("-"):
    pos = open(os.getcwd() + "/short_reviews/positive.txt", "r", encoding="utf-8", errors="replace").read()
    neg = open(os.getcwd() + "/short_reviews/negative.txt", "r", encoding="utf-8", errors="replace").read()

pos_fileids = movie_reviews.fileids("pos")
neg_fileids = movie_reviews.fileids("neg")

all_words = []
documents = []

stop_words = stopwords.words("english")

print("here")

#  J is adjective, R is adverb, and V is verb
allowed_word_types = ["J"]

for f in pos_fileids:
    p = movie_reviews.raw(f)
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w[0] not in stop_words:
            all_words.append(w[0].lower())

for f in neg_fileids:
    p = movie_reviews.raw(f)
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types and w[0] not in stop_words:
            all_words.append(w[0].lower())

print("here")
all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:9000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

print("here")

featuresets = [(find_features(rev), category) for (rev, category) in documents]

'''DecisionTreeClassifier_classifier = SklearnClassifier(tree.DecisionTreeClassifier())
DecisionTreeClassifier_classifier.train(training_set)
print(nltk.classify.accuracy(DecisionTreeClassifier_classifier, testing_set))'''

print("here")

accuracy_sum = 0

for j in range(0, 10):

    random.shuffle(featuresets)

    testing_set = featuresets[1900:]
    training_set = featuresets[:1900]

    # classifier = OpinionLexiconClassifier()
    # accuracy = nltk.classify.accuracy(classifier)
    NuSVC_classifier = SklearnClassifier(NuSVC(nu=0.8))
    NuSVC_classifier.train(training_set)
    accuracy = nltk.classify.accuracy(NuSVC_classifier, testing_set)

    accuracy_sum += accuracy
    print("NuSVC_classifier accuracy percent:", str(accuracy * 100))

print("Average of the ten accuracies with top 4000 features:", str(accuracy_sum / 10))
