from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import state_union
from nltk import PunktSentenceTokenizer

example_text = "This is some example text. This is to show off the ability of tokenizing."

# Using nltk's default word and sentence tokenizer
words = word_tokenize(example_text)
sentences = sent_tokenize(example_text)

for w in words:
    print(w)

print()

for s in sentences:
    print(s)

print()

# Using PunktSentenceTokenizer and training it
train_text = state_union.raw("2005-GWBush.txt")

custom_sentence_tokenizer_trained = PunktSentenceTokenizer(train_text)

sentences = custom_sentence_tokenizer_trained.tokenize(example_text)

for s in sentences:
    print(s)

print()

# Using PunktSentenceTokenizer with no training (it comes pretrained)
custom_sentence_tokenizer_untrained = PunktSentenceTokenizer()

sentences = custom_sentence_tokenizer_untrained.tokenize(example_text)

for s in sentences:
    print(s)