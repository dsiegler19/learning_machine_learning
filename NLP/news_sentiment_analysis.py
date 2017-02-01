# Possible modules for training:
# movie_reviews √
# sentence_polarity NO
# pros_cons NO
# opinion_lexicon NO
# short_reviews NO
# https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences (sentiment labeled sentences)
# SAR14 (in downloads and found at: https://sites.google.com/site/nquocdai/resources, sentiment labeled movie reviews) √
# aclImdb (in downloads and found at: http://ai.stanford.edu/~amaas/data/sentiment/, sentiment labeled movie reviews)
# MAYBE

# Possible modules for testing:
# Reuters
# MPQA (http://mpqa.cs.pitt.edu/)

# TODO
# Categorize the positive and negative words by POS
# Come up with better featuresets using POS tagging and context
# Organize pickling

import os

SAR14_f = open(os.getcwd() + "/SAR14.txt")
SAR14_raw = SAR14_f.read()
raw_reviews = SAR14_raw.split("\n")
print(len(raw_reviews))
reviews_scores = []

for r in raw_reviews:
    if r[-2:-1] == "1":
        score = 10
        review = r[2:-6]

    elif r[-2:-1] == ",":
        score = int(r[-1:])
        review = r[2:-5]

    reviews_scores.append((review, score))

print(len(reviews_scores))
