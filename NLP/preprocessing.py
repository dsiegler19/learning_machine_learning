import os
import random
import re
import urllib
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk import pos_tag
from nltk.chunk import conlltags2tree
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from urllib.parse import quote_plus
from wikipedia import wikipedia
from yahoo_finance import Share
from nltk.corpus import reuters


def yahoo_request(symbol, tag):
    url = "http://finance.yahoo.com/d/quotes.csv?s=%s&f=%s" % (symbol, tag)
    request = Request(url)
    response = urlopen(request)
    content = response.read().decode().strip().strip('"')
    return content


def get_name(symbol):
    return yahoo_request(symbol, "n")


def is_valid_ticker(ticker):
    if ticker is None:
        return False
    ticker = ticker.upper()
    return True if get_name(ticker) != 'N/A' else False


def find_parent(company_name):
    url = "https://en.wikipedia.org/wiki/" + company_name
    req = urllib.request.Request(url)
    try:
        page = urllib.request.urlopen(req)
        # Get info box > Get parent
        soup = BeautifulSoup(page, "html.parser")
        row_re = re.compile(r"(Parent|Owner)")
        row = soup.find(text=row_re)
        if row is None:
            return None
        row = str(row.parent.parent.parent)
        soup = BeautifulSoup(row, "html.parser")
        markup = soup.find_all("a")
        for m in markup:
            if "Parent" not in m.get_text() and "Owner" not in m.get_text():
                return m.get_text()
        return None
    except:
        return None


def get_ticker(company_name, search_for_parent=True, sources=None):
    if not sources:
        sources = ["bing", "wikipedia"]

    ticker = None

    try:

        if "bing" in sources:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
            url = "https://www.bing.com/search?q=" + quote_plus(company_name + " stock price")
            req = urllib.request.Request(url, headers=headers)
            page = urllib.request.urlopen(req)
            soup = BeautifulSoup(page, "html.parser")
            markup = str(soup.find_all("div", {"class": "fin_metadata"}))[1:-1]
            text = BeautifulSoup(markup, "html.parser").get_text()
            ticker_start = 0
            ticker_end = 0
            for i in range(len(text)):
                if text[i] == ":":
                    ticker_start = i + 2
                    break

            for i in range(len(text)):
                if text[i] == "·":
                    ticker_end = i - 1
                    break

            if not (ticker_start == 0 and ticker_end == 0):
                ticker = text[ticker_start:ticker_end]
        if not is_valid_ticker(ticker):
            ticker = None
    except:
        pass

    try:
        if "bing" in sources and ticker is None:
            headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
            url = "https://www.bing.com/search?q=" + quote_plus(company_name)
            req = urllib.request.Request(url, headers=headers)
            page = urllib.request.urlopen(req)
            soup = BeautifulSoup(page, "html.parser")
            markup = str(soup.find_all("div", {"class": "fin_metadata"}))[1:-1]
            text = BeautifulSoup(markup, "html.parser").get_text()
            ticker_start = 0
            ticker_end = 0
            for i in range(len(text)):
                if text[i] == ":":
                    ticker_start = i + 2
                    break

            for i in range(len(text)):
                if text[i] == "·":
                    ticker_end = i - 1
                    break

            if not (ticker_start == 0 and ticker_end == 0):
                ticker = text[ticker_start:ticker_end]

        if not is_valid_ticker(ticker):
            ticker = None
    except:
        pass

    try:
        if "wikipedia" in sources and ticker is None:
            url = "https://en.wikipedia.org/wiki/" + "_".join(company_name.split(" "))
            req = urllib.request.Request(url)
            try:
                page = urllib.request.urlopen(req)
                # Get info box > Get traded as > Get the ticker
                soup = BeautifulSoup(page, "html.parser")
                infobox = BeautifulSoup(str(soup.find_all("table", {"class": "infobox vcard"}))[1:-1], "html.parser")
                row_re = re.compile(r"Traded")
                row = infobox.find(text=row_re)
                if row is not None:
                    row = str(row.parent.parent.parent)
                    soup = BeautifulSoup(row, "html.parser")
                    markup = soup.find_all("a", {"rel": "nofollow"})
                    for m in markup:
                        if m.get_text().isupper():
                            ticker = m.get_text()
            except:
                company_name = wikipedia.search(company_name)[0]
                url = "https://en.wikipedia.org/wiki/" + "_".join(company_name.split(" "))
                req = urllib.request.Request(url)
                try:
                    page = urllib.request.urlopen(req)
                    # Get info box > Get traded as > Get the ticker
                    soup = BeautifulSoup(page, "html.parser")
                    infobox = BeautifulSoup(str(soup.find_all("table", {"class": "infobox vcard"}))[1:-1], "html.parser")
                    row_re = re.compile(r"Traded")
                    row = infobox.find(text=row_re)
                    if row is not None:
                        row = str(row.parent.parent.parent)
                        soup = BeautifulSoup(row, "html.parser")
                        markup = soup.find_all("a", {"rel": "nofollow"})
                        for m in markup:
                            if m.get_text().isupper():
                                ticker = m.get_text()
                except:
                    pass
    except:
        pass

    if not is_valid_ticker(ticker):
        ticker = None

    if search_for_parent and ticker is None:
        parent = find_parent(company_name)
        if parent:
            ticker = get_ticker(parent)

    return ticker


# Process text
def process_text(raw_text):
    token_text = word_tokenize(raw_text)
    return token_text


# Stanford NER tagger
def stanford_tagger(token_text):
    st = StanfordNERTagger(os.getcwd() + "/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz",
                           os.getcwd() + "/stanford-ner/stanford-ner.jar", encoding='utf-8')
    ne_tagged = st.tag(token_text)
    return ne_tagged


# Tag tokens with standard NLP BIO tags
def bio_tagger(ne_tagged):
    bio_tagged = []
    prev_tag = "O"
    for token, tag in ne_tagged:
        if tag == "O":  # O
            bio_tagged.append((token, tag))
            prev_tag = tag
            continue
        if tag != "O" and prev_tag == "O":  # Begin NE
            bio_tagged.append((token, "B-" + tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag == tag:  # Inside NE
            bio_tagged.append((token, "I-" + tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag != tag:  # Adjacent NE
            bio_tagged.append((token, "B-" + tag))
            prev_tag = tag
    return bio_tagged


# Create tree
def stanford_tree(bio_tagged):
    tokens, ne_tags = zip(*bio_tagged)
    pos_tags = [pos for token, pos in pos_tag(tokens)]

    conlltags = [(token, pos, ne) for token, pos, ne in zip(tokens, pos_tags, ne_tags)]
    ne_tree = conlltags2tree(conlltags)
    return ne_tree


# Parse named entities from tree
def structure_ne(ne_tree):
    ne = []
    for subtree in ne_tree:
        if type(subtree) == Tree:  # If subtree is a noun chunk, i.e. NE != "O"
            ne_label = subtree.label()
            ne_string = " ".join([token for token, pos in subtree.leaves()])
            ne.append((ne_string, ne_label))
    return ne


def stanford_tag(text):
    return structure_ne(stanford_tree(bio_tagger(stanford_tagger(process_text(text)))))


def find_all_stocks_mentioned(text):
    NEs = stanford_tag(text)
    tickers = []
    for NE in NEs:
        if NE[1] == "ORGANIZATION":
            ticker = get_ticker(NE[0])
            if ticker is not None:
                tickers.append(ticker)
    return tickers


def get_only_text(url):
    page = urlopen(url).read().decode('utf8')
    soup = BeautifulSoup(page, "html.parser")
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return soup.title.text, text

found_stocks = {}  # {ticker: num_of_occurrences}
article_fileids = [f for f in random.sample(reuters.fileids(), len(reuters.fileids()))[:1500]]
i = 0
for f in article_fileids[:50]:
    i += 1
    if i % 5 == 0:
        print("Currently on:", str(i))
        print("found_stocks is")
        print(found_stocks)
        print()
    text = reuters.raw(f)
    tickers = find_all_stocks_mentioned(text)
    for ticker in tickers:
        if ticker is not None and ticker is not "":
            try:
                found_stocks[ticker] += 1
            except KeyError:
                found_stocks[ticker] = 1

print(found_stocks)
print("Remember TRI is the ticker for Reuters")
