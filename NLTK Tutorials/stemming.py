from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

text = "It is very important to be pythonly while you are pythoning with python. " \
       "All pythoners are pythoned poorely at least once"

words = word_tokenize(text)

for w in words:
    # The method to actually stem
    print(ps.stem(w))
