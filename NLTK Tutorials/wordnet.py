from nltk.corpus import wordnet

# The synset for the word program
synonym_set = wordnet.synsets("program")

# The whole synset
print(synonym_set)

# Just the first word
print(synonym_set[0].lemmas()[0].name())

# The definition of the first word in the synset
print(synonym_set[0].definition())

# Some examples for the first word in the synset
print(synonym_set[0].examples())

print()

# Finding all of the synonyms and antonyms of the word good

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

print()

# Finding if word1 and word2 are semantically similar

word1 = wordnet.synset("ship.n.01")
word2 = wordnet.synset("boat.n.01")
# wup stands for Wu and Palmer, who wrote a paper that outlines the method used for semantic similarity
print(word1.wup_similarity(word2))


word1 = wordnet.synset("ship.n.01")
word2 = wordnet.synset("car.n.01")
print(word1.wup_similarity(word2))


word1 = wordnet.synset("ship.n.01")
word2 = wordnet.synset("cat.n.01")
print(word1.wup_similarity(word2))
