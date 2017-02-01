# A simple classifier based on the opinion lexicon
from nltk.classify import ClassifierI
from nltk.corpus import opinion_lexicon

pos_words = opinion_lexicon.words("positive-words.txt")
neg_words = opinion_lexicon.words("negative-words.txt")


class OpinionLexiconClassifier (ClassifierI):
    def __init__(self):
        pass

    def classify(self, featureset):
        score = 0
        for w in featureset.keys():
            if featureset[w] and w in pos_words:
                score += 1
            if featureset[w] and w in neg_words:
                score -= 1

        if score >= 0:
            return "pos"
        else:
            return "neg"
