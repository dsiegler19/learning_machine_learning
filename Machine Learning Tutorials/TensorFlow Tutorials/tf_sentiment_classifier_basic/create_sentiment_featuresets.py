# Turns sentences into standard length vectors using the following process:
# Collect an array of all of the unique words in the training sets. This array may look something like this:
# [the, product, was, complete, garbage, I, loved, this, thing, it, fantastic...].
# Then when a sentence needs to be converted into an array the following process is used:
# First, create an array of all 0s. Next, go through each word in the sentence and see if it is in the lexicon of words
# collected earlier. If it isn't, do nothing. If the word is in the lexicon, add 1 to ith element of the vector (the one
# that started as all 0s) where i is the index of the word that is found in the lexicon.
# For example, if the lexicon is:
# [the, product, was, complete, garbage, I, loved, this, thing, it, fantastic]
# And the sentence is:
# this product is cool but the service is garbage.
# Then the array for this sentence would be:
#                                                  [1,    1,     0, 0, 1, 0, 0, 1, 0, 0, 0]
# This element is 1 because the sentence contains: the  product      garbage   this

from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
hm_lines = 100000


def create_lexicon(pos, neg, max_occurrence, min_occurrence):

    """
    Creates a lexicon of all of the words in pos and and neg which have less than max_occurrence occurrence and less
    than min_occurrence occurrence.
    :param pos: The first file.
    :param neg: The second file.
    :param max_occurrence: The maximum number of occurrence allowed to be included in the lexicon.
    :param min_occurrence: The minimum number of occurrence allowed to be included in the lexicon.
    :return: A list of all of the words within the tolerated amount of occurrences.
    """

    lexicon = []

    with open(pos, "r") as f:

        contents = f.readlines()

        for l in contents[:hm_lines]:

            all_words = word_tokenize(l)
            lexicon += list(all_words)

    with open(neg, "r") as f:

        contents = f.readlines()

        for l in contents[:hm_lines]:

            all_words = word_tokenize(l)
            lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []

    for w in w_counts:

        if max_occurrence > w_counts[w] > min_occurrence:

            l2.append(w)

    return l2


def sample_handling(sample, lexicon, classification):
    """
    Creates a vector representation of the sample given the lexicon and classification (see above).
    :param sample: The sample to vectorize.
    :param lexicon: The lexicon to use.
    :param classification: The classification of the sample.
    :return: A list of the the features in vector the classification.
    """

    featureset = []

    with open(sample, "r") as f:

        contents = f.readlines()

        for l in contents[:hm_lines]:

            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))

            for word in current_words:

                if word.lower() in lexicon:

                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            featureset.append([features, classification])

    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    """
    Creates a training and testing set from pos and neg files.
    :param pos: The first document.
    :param neg: The second document.
    :param test_size: The size of the testing set (as a percentage of all the data).
    :return: The training and testing sets.
    """

    lexicon = create_lexicon(pos, neg, 1000, 50)
    features = []
    features += sample_handling("pos.txt", lexicon,[1,0])
    features += sample_handling("neg.txt", lexicon,[0,1])
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':

    train_x, train_y, test_x, test_y = create_feature_sets_and_labels("pos.txt", "neg.txt")

    with open("pickles/sentiment_set.pickle", "wb") as f:

        pickle.dump([train_x,train_y,test_x,test_y],f)
