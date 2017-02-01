from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))

print()

# The default parameter for pos (part of speech) is noun, so one must specify the pos if it is not a noun. This pos
# argument uses codes then the POS tagging codes.

# This produces an incorrect result because it assumes better is a noun.
print(lemmatizer.lemmatize("better"))
# This produces the correct results
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))

# Run can be both a verb and a noun, so one must specify which one it is for the lemmatizer
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run", pos="v"))


