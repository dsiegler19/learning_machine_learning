import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

try:
    for i in tokenized:
        words = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(words)

        # Regex for 0 or more adverb (because all adverbs have code RB + 0 or 1 other letters) then 0 or more verbs
        # (because all verbs have the code VB + 0 or 1 other letters) then one or more proper noun and finally
        # 0 or 1 singular nouns
        chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
        chunkParser = nltk.RegexpParser(chunkGram)
        chunked = chunkParser.parse(tagged)

        chunked.draw()

except Exception as e:
    print(str(e))