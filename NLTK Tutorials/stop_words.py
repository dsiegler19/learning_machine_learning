from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = 'This is an example showing off stop word filtration.'
# A lexicon of all of the stop words in nltk (one could make a different one though)
stop_words = set(stopwords.words('english'))

words = word_tokenize(example_sentence)

filtered_sentence = []

for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)

print(filtered_sentence)

# Or to replace the for loop with one line:
# [w for w in words if not w in stop_words]
