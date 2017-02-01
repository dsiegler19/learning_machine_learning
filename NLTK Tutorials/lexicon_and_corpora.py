from nltk.corpus import gutenberg
from nltk.corpus import state_union
from nltk.corpus import words
from nltk.corpus import cmudict

# Examples of corpora from gutenberg and state_union

emma = gutenberg.words('austen-emma.txt')
state_union_2006 = state_union.raw("2006-GWBush.txt")
bible = gutenberg.words("bible-kjv.txt")

for s in bible[:100]:
    print(s)

for w in emma[:100]:
    print(w)

print()

for w in state_union_2006[:100]:
    print(w)

print()

# Examples of a lexicon of all the words in the english language and a pronunciation dictionary

english_language = words.words()
pronunciation_dictionary = cmudict.entries()

for w in english_language[:100]:
    print(w)

print()

for w in pronunciation_dictionary[:100]:
    print(w)
