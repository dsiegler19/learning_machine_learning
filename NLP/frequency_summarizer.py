from collections import defaultdict
from heapq import nlargest
from string import punctuation
from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


class FrequencySummarizer:
    def __init__(self, min_cut=0.1, max_cut=0.9):
        self.min_cut = min_cut
        self.max_cut = max_cut
        self.stopwords = list(stopwords.words("english")) + list(punctuation)

    # Return a dictionary of the frequencies of every word given that word_sent is already word and sentence tokenized.
    def compute_frequencies(self, word_sent):
        freq = defaultdict(int)
        for s in word_sent:
            for word in s:
                if word not in self.stopwords:
                    freq[word] += 1
        # Normalize and filter all of the frequencies
        m = float(max(freq.values()))
        to_delete = []
        for w in freq.keys():
            freq[w] /= m
            if freq[w] >= self.max_cut or freq[w] <= self.min_cut:
                to_delete.append(w)
        for td in to_delete:
            del freq[td]
        return freq

    # Returns the top n sentences that contain the most of the most frequent words.
    def summarize(self, text, n):
        sents = sent_tokenize(text)
        word_sent = [word_tokenize(s.lower()) for s in sents]
        self.freq = self.compute_frequencies(word_sent)
        ranking = defaultdict(int)
        for i, sent in enumerate(word_sent):
            for w in sent:
                if w in self.freq:
                    ranking[i] += self.freq[w]
        sents_idx = self.rank(ranking, n)
        return [sents[j] for j in sents_idx]

    # Returns the first n sentences with the highest ranking.
    @staticmethod
    def rank(ranking, n):
        return nlargest(n, ranking, key=ranking.get)


def get_only_text(url):
    page = urlopen(url).read().decode("utf8")
    soup = BeautifulSoup(page, "html.parser")
    text = " ".join(map(lambda p: p.text, soup.find_all("p")))
    return soup.title.text, text

feed_xml = urlopen("http://feeds.bbci.co.uk/news/rss.xml").read()
feed = BeautifulSoup(feed_xml.decode('utf8'), "html.parser")
to_summarize = list(map(lambda p: p.text, feed.find_all('guid')))

fs = FrequencySummarizer()
for article_url in to_summarize[:5]:
    title, text = get_only_text(article_url)
    print("-----------------------------------------------------------------------------------------------------------")
    print(title)
    num_sents = int(len(sent_tokenize(text)) / 12)
    for s in fs.summarize(text, 4):
        print("*", s)
