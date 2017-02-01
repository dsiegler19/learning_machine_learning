# For tuning parameters

import nltk
import random
import os
from statistics import mode, StatisticsError
import time as gettime
from sklearn.linear_model import LogisticRegression, SGDClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
from sklearn.svm import NuSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers, default=0):
        self._classifiers = classifiers
        self._default = default

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        try:
            return mode(votes)
        except StatisticsError:
            return votes[self._default]

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

# Input is a list of POS categorized words, output is a list of tuples containing the bigrams (excluding all punctuation
# except n't) and a list of the POSs
# text = [('Hello', 'NN'), ('my', 'PRP$'), ('name', 'NN'), ('is', 'VBZ'), ('Dylan', 'NNP')]
# bigrams = [('Hello my', ['NN', 'PRP$']), ('my name', ['PRP$', 'NN']), ('name is', ['NN', 'VBZ'])...]
"""def pos_tagged_bigrams(text):
    bigrams = []

    for i in range(len(text) - 1):
        if text[i][0][0].isalnum() and text[i + 1][0][0].isalnum():
            word_pair = " ".join([text[i][0], text[i + 1][0]])
            pos_pair = [text[i][1], text[i + 1][1]]
            bigrams.append((word_pair.lower(), pos_pair))

    return bigrams

# Simply returns a list of all of the bigrams in text (excluding the punctuation)
def bigrams(text):
    bigrams = []
    words = word_tokenize(text)
    for i in range(len(words) - 1):
        if words[i][0].isalnum() and words[i + 1][0].isalnum():
            bigrams.append(words[i] + " " + words[i + 1])

    return bigrams"""

folder_name = "short_reviews"
pos_path = "/Users/dsiegler19/PycharmProjects/Language Processing and Machine Learning/NLP/short_reviews/positive.txt"
neg_path = "/Users/dsiegler19/PycharmProjects/Language Processing and Machine Learning/NLP/short_reviews/negative.txt"

pos_tagged = open(pos_path, encoding="utf-8", errors="replace").read()
neg = open(neg_path, encoding="utf-8", errors="replace").read()

if pos_path.__eq__("-") | neg_path.__eq__("-"):
    pos_tagged = open(os.getcwd() + "/short_reviews/positive.txt", "r", encoding="utf-8", errors="replace").read()
    neg = open(os.getcwd() + "/short_reviews/negative.txt", "r", encoding="utf-8", errors="replace").read()

pos_fileids = movie_reviews.fileids("pos")
neg_fileids = movie_reviews.fileids("neg")

SAR14_f = open(os.getcwd() + "/SAR14.txt")
SAR14_raw = SAR14_f.read()
raw_reviews = SAR14_raw.split("\n")
reviews_scores = []

for r in raw_reviews[:10]:
    if r[-2:-1] == "1":
        score = 10
        review = r[2:-6]

    elif r[-2:-1] == ",":
        score = int(r[-1:])
        review = r[2:-5]

    reviews_scores.append((review, score))

all_words = []
documents = []

# all_bigrams = []

stop_words = stopwords.words("english")
# stop_words_without_not = [w for w in stop_words if w != "not"]

#  J is adjective, R is adverb, V is verb, and N is noun
allowed_word_types = ["J"]

# bigrams_must_consist_of = ["V", "J", "N", "R"]

for rs in reviews_scores:
    p = rs[0]
    if rs[1] > 5:
        sentiment = "pos"
    else:
        sentiment = "neg"
    documents.append((p, sentiment))
    words = word_tokenize(p)
    pos_tagged = nltk.pos_tag(words)
    for w in pos_tagged:
        if w[1][0] in allowed_word_types and w[0] not in stop_words:
            all_words.append(w[0].lower())

for f in pos_fileids:
    p = movie_reviews.raw(f)
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos_tagged = nltk.pos_tag(words)
    # positive_bigrams = pos_tagged_bigrams(pos_tagged)
    for w in pos_tagged:
        if w[1][0] in allowed_word_types and w[0] not in stop_words:
            all_words.append(w[0].lower())

    # for b in positive_bigrams:
    #     if b[1][0][0] in bigrams_must_consist_of and b[1][0][0] in bigrams_must_consist_of and \
    #             b[0].split(" ")[0] not in stop_words_without_not and b[0].split(" ")[1] not in stop_words_without_not:
    #         all_bigrams.append(b[0])

for f in neg_fileids:
    p = movie_reviews.raw(f)
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos_tagged = nltk.pos_tag(words)
    # negative_bigrams = pos_tagged_bigrams(pos_tagged)
    for w in pos_tagged:
        if w[1][0] in allowed_word_types and w[0] not in stop_words:
            all_words.append(w[0].lower())

    # for b in negative_bigrams:
    #     if b[1][0][0] in bigrams_must_consist_of and b[1][0][0] in bigrams_must_consist_of and \
    #             b[0].split(" ")[0] not in stop_words_without_not and b[0].split(" ")[1] not in stop_words_without_not:
    #         all_bigrams.append(b[0])

all_words = nltk.FreqDist(all_words)
# all_bigrams = nltk.FreqDist(all_bigrams)

word_features = list(all_words.keys())[:20000]  # 9000, 20000


def find_features(document):
    # print("find_features called. document =", document)
    words = word_tokenize(document)
    # grams = bigrams(document)
    features = {}
    for wrd in word_features:
        features[wrd] = (wrd in words)

    return features

print("documents is this long:")
print(len(documents))
print("and is of type:")
print(type(documents))

featureset = [(find_features(rev), category) for (rev, category) in documents]

accuracy_sum = 0
time_sum = 0

NuSVClassifier_accuracy_sum = 0
NuSVClassifier_time_sum = 0

RFC_accuracy_sum = 0
RFC_time_sum = 0

OLC_accuracy_sum = 0
OLC_time_sum = 0

LinearSVClassifier_accuracy_sum = 0
LinearSVClassifier_time_sum = 0

SVClassifier_accuracy_sum = 0
SVClassifier_time_sum = 0

SGDC_accuracy_sum = 0
SGDC_time_sum = 0

LogisticRegressionClassifier_accuracy_sum = 0
LogisticRegressionClassifier_time_sum = 0

BNB_accuracy_sum = 0
BNB_time_sum = 0

MNB_accuracy_sum = 0
MNB_time_sum = 0

ABC_accuracy_sum = 0
ABC_time_sum = 0

news1 = """Nintendo shares had another banner day on Tuesday, surging more than 14% in Tokyo amid widespread mania over the company's sensational Pokemon Go game.

The stock's performance following the release of the augmented reality game is staggering: Shares have risen by more than 120% since July 6, adding $23 billion to Nintendo's market value.

The company is now worth $42.5 billion (4.5 trillion yen), more than Sony (4.1 trillion yen), Canon (4 trillion yen), Panasonic (2.4 trillion yen) or Toshiba (1.3 trillion yen).
Pokemon Go is a legitimate sensation -- ranking as the top free downloaded app on both Apple's App Store as well as Google's  Play store for Android devices.

Investors are hoping that Nintendo will be able to cash in on the purchase of add-on features for the game.
It's a model that many mobile game developers use. Make the game free for download, get them hooked and then sell extra items to enhance the game to generate big revenue.
The game's immense popularity could also lead to the proverbial halo effect for Nintendo, as people who may not have been familiar with the company's many offerings decide to buy more Nintendo products."""
news2 = """To quote 1980s standup comedian Judy Tenuta, "It could happen!" And Dow 20,000 isn't as crazy a prediction as the famous Dow 36,000 one from the book by James Glassman -- way back in 1999. But who wasn't drinking the market Kool-Aid then?

Here's why Dow 20K isn't that farfetched of an expectation.
U.S. stocks have been riding a bullish wave of momentum for the past few weeks.
The Dow and S&P 500 are back at all-time highs. The Nasdaq still has a ways to go before it hits a new record, but the tech-heavy index is once again above 5,000 -- a key psychological level.
Simply put, investors are realizing that they have to put their money somewhere.
European stocks are perceived as being a huge risk following Brexit.
There are also still big concerns about the health of China and other emerging markets.
Oil? Commodities have started to retreat again after a temporary supply shock pushed it back above $50.
Bonds? Have fun doing a word that rhymes with "kissing" your money away in low-yielding Treasuries or European and Japanese bonds that actually have yields below zero.
So American stocks, for better or for worse, still seem to be the cleanest dirty shirt -- a reference to a Johnny Cash lyric that bond king Bill Gross has often used to describe the U.S. markets.
Investors seem to appreciate the fact that the U.S. -- despite concerns about sluggish growth and the upcoming presidential election -- is still much more stable than most other major developed and developing countries.
"The U.S. is our favorite place to be. Slow growth is better than no growth," said Vince Rivers, manager of the JOHCM US Small Mid Cap Equity Fund.
He added that he sat tight when the global financial markets were briefly worried about Brexit last month. He does not think that Brexit will wind up being the 2016 equivalent of Lehman Brothers.
"We traded not one share for two weeks after Brexit. It was pure panic," he said.
John Augustine, chief investment officer with Huntington Bank, agreed that the U.S. is the best place to invest in right now. He's a little concerned about how long that can last.
But he said that as long as companies report decent, if not spectacular, earnings for the second quarter and also give solid outlooks for the rest of the year then stocks should keep climbing.
That seems to be the case so far. Only a handful of major companies have reported their latest results. But Walgreens and Pepsi -- which both have significant exposure to the U.K. -- did well.
So did big banks JPMorgan Chase and Citigroup and global fast food restaurant giantYum Brands.
Augustine said that he thinks consumers will continue to spend and travel and that the housing market will remain strong.
His firm's core stock portfolio owns wine and beer company Constellation Brands, household products (and Trojan condom maker) Church & Dwight, Marriott, Disney, Lowe's and home improvement product maker Masco.
Matthew Pistorio, client portfolio manager with Henderson Geneva Capital Management, also thinks housing-related stocks are a good bet. He likes Lowe's too, and holds building supply companies Beacon Roofing, Fortune Brands and Acuity.

Pistorio said he's a little surprised by how fast the overall market has rebounded from Brexit-related worries. But he said he wouldn't bet against the market right now.
The Federal Reserve seems to be suggesting that it's in no hurry to raise interest rates again. And the latest retail sales figures for June showed that consumers are still spending -- especially on their homes.
"The rally has happened a lot more quickly than we and others had thought it would. But it's a Goldilocks environment. The Fed is unlikely do anything soon and consumer spending is healthy," Pistorio said.
As for the other big wild card out there, Augustine said he's not too worried about the presidential election. President Trump? President Clinton? It probably won't matter much since there could be more gridlock in Congress if either of them win.
Augustine thinks there is one good election bet though -- the U.S. military. Both Trump and Clinton are more hawkish than dovish when it comes to national security. So defense contractors could be winners regardless of what happens in November.
"Defense is the Energizer bunny," he said."""
news3 = """ARM Holdings, one of Britain's most successful tech companies, is being snapped up by Softbank for £24.3 billion ($32 billion) in the biggest foreign takeover by a Japanese company.
The cash purchase, which has been agreed to by the boards of both companies, represents a major strategic bet by Softbank CEO Masayoshi Son on mobile communications and the "Internet of Things."

ARM shares jumped 43% in early trade on Monday to match the £17 pounds ($22.50) per share offered by Softbank. Markets in Japan were closed for a holiday.
Cambridge-based ARM is a leader in mobile computing, designing technology that can be found in popular smartphones including Apple's iPhone, making it a juicy takeover target for mobile-focused Softbank.
Son told reporters that the acquisition was his "big bet for the future," explaining that ARM was well positioned to capitalize as more and more devices -- household appliances, cars and censors -- are connected to the Internet.
"This is the company I've wanted to [build] for 10 years," he said. "The 'Internet of Things' is growing big time."
The deal is the largest investment ever from Asia into the U.K. It's also the biggest Japanese purchase of a foreign company based on the value of the equity, according to Dealogic.
The ARM takeover comes less than one month after the U.K. voted to leave the European Union, a shock event that rattled investors and caused the pound to plummet.
Son said he had discussed the deal with U.K. officials, and found them to be receptive. Philip Hammond, the U.K.'s new Treasury chief, said the takeover showed that "Britain has lost none of its allure to international investors."

But the deal was also a chance for Softbank to buy a prized asset on the cheap. The pound has fallen by more than 27% against the yen since this time last year, with more than half of those losses coming before the British vote to exit the EU.
ARM does very little business in Europe, which should help shield it from Brexit fallout. The company generated only 1% of its revenue in the U.K. in 2015, 9% in the rest of Europe and 52% in Asia.
Softbank, which also owns a controlling stake in U.S. telecom Sprint, has been raising cash in recent months. In June, the company announced a plan to unload roughly $8 billion in Alibaba shares, the fruits of an early investment in the Chinese e-commerce giant.
Son said that while he only approached ARM about a potential deal in early July, he wasn't bargain hunting.
"Brexit did not effect my decision," he told reporters. "I was waiting to have the cash on hand."
As part of the deal, Softbank committed to doubling the number of ARM employees in the U.K. over the next five years. Roughly 40% of ARM's 3,975 workers are currently based in Britain.
There have been major changes to Softbank's leadership recently. Superstar executive Nikesh Arora, who was being groomed as Son's successor, resigned as company president in June."""

print("Starting the first round of training")
print("Length of featureset is:", len(featureset))

for i in range(0, 30):

    random.shuffle(featureset)

    testing_set = featureset[(int(len(featureset) * 0.9)):]
    training_set = featureset[:(int(len(featureset) * 0.9))]

    start_time = gettime.time()

    NuSVClassifier = SklearnClassifier(NuSVC(nu=0.8, decision_function_shape="ovr"))
    NuSVClassifier.train(training_set)
    NuSVClassifier_accuracy = nltk.classify.accuracy(NuSVClassifier, testing_set)

    print("NuSVC done.")

    RFC = SklearnClassifier(RandomForestClassifier(n_estimators=25, min_samples_leaf=6))
    RFC.train(training_set)
    RFC_accuracy = nltk.classify.accuracy(RFC, testing_set)

    print("RFC done.")

    # OLC = OpinionLexiconClassifier()
    # OLC_accuracy = nltk.classify.accuracy(OLC, testing_set)

    # print("OLC done.")

    LinearSVClassifier = SklearnClassifier(LinearSVC())
    LinearSVClassifier.train(training_set)
    LinearSVClassifier_accuracy = nltk.classify.accuracy(LinearSVClassifier, testing_set)

    print("LinearSVC done.")

    SVClassifier = SklearnClassifier(SVC(C=15.0, decision_function_shape="ovr"))
    SVClassifier.train(training_set)
    SVClassifier_accuracy = nltk.classify.accuracy(SVClassifier, testing_set)

    print("SVC done.")

    SGDC = SklearnClassifier(SGDClassifier())
    SGDC.train(training_set)
    SGDC_accuracy = nltk.classify.accuracy(SGDC, testing_set)

    print("SGDC done.")

    LogisticRegressionClassifier = SklearnClassifier(LogisticRegression(C=2.0))
    LogisticRegressionClassifier.train(training_set)
    LogisticRegressionClassifier_accuracy = nltk.classify.accuracy(LogisticRegressionClassifier, testing_set)

    print("LogisticRegressionClassifier done.")

    BNB = SklearnClassifier(BernoulliNB())
    BNB.train(training_set)
    BNB_accuracy = nltk.classify.accuracy(BNB, testing_set)

    print("BNB done.")

    MNB = SklearnClassifier(MultinomialNB())
    MNB.train(training_set)
    MNB_accuracy = nltk.classify.accuracy(MNB, testing_set)

    print("MNB done.")

    ABC = SklearnClassifier(AdaBoostClassifier())
    ABC.train(training_set)
    ABC_accuracy = nltk.classify.accuracy(ABC, testing_set)

    print("ABC done.")

    # NuSVC, RandomForestClassifier, OpinionLexiconClassifier, LinearSVC, SVC, SGDC, LogisticRegression, BernoulliNB,
    # MultinomialNB, AdaBoostClassifier

    VC = VoteClassifier(NuSVClassifier, RFC, LinearSVClassifier, SVClassifier, SGDC,
                        LogisticRegressionClassifier, BNB, MNB, ABC)

    try:
        VC_accuracy = nltk.classify.accuracy(VC, testing_set)
    except Exception as e:
        print("Exception in accuracy")
        print(e)
        VC_accuracy = 0

    end_time = gettime.time()
    time = end_time - start_time

    time_sum += time
    accuracy_sum += VC_accuracy

    NuSVClassifier_accuracy_sum += NuSVClassifier_accuracy
    RFC_accuracy_sum += RFC_accuracy
    # OLC_accuracy_sum += OLC_accuracy
    LinearSVClassifier_accuracy_sum += LinearSVClassifier_accuracy
    SVClassifier_accuracy_sum += SVClassifier_accuracy
    SGDC_accuracy_sum += SGDC_accuracy
    LogisticRegressionClassifier_accuracy_sum += LogisticRegressionClassifier_accuracy
    BNB_accuracy_sum += BNB_accuracy
    MNB_accuracy_sum += MNB_accuracy
    ABC_accuracy_sum += ABC_accuracy

    print("NuSVClassifier:", str(NuSVClassifier_accuracy))
    print("RFC:", str(RFC_accuracy))
    # print("OLC:", str(OLC_accuracy))
    print("LinearSVClassifier:", str(LinearSVClassifier_accuracy))
    print("SVClassifier:", str(SVClassifier_accuracy))
    print("SGDClassifier:", str(SGDC_accuracy))
    print("LogisticRegressionClassifier:", str(LogisticRegressionClassifier_accuracy))
    print("BNB:", str(BNB_accuracy))
    print("MNB:", str(MNB_accuracy))
    print("ABC:", str(ABC_accuracy))

    print()

    print("Time:", str(time))
    print("Accuracy:", str(VC_accuracy))
    print("The first article is categorized as:")
    print(VC.classify(find_features(news1)))
    print("The second article is categorized as:")
    print(VC.classify(find_features(news2)))
    print("The third article is categorized as:")
    print(VC.classify(find_features(news3)))
    print()
    print()
    print()

print()
print("===========================================FINAL RESULTS=======================================================")
print()

print("NuSVClassifier:", str(NuSVClassifier_accuracy_sum / 30))
print("RFC:", str(RFC_accuracy_sum / 30))
# print("OLC:", str(OLC_accuracy_sum / 30))
print("LinearSVClassifier:", str(LinearSVClassifier_accuracy_sum / 30))
print("SVClassifier:", str(SVClassifier_accuracy_sum / 30))
print("SGDClassifier:", str(SGDC_accuracy_sum / 30))
print("LogisticRegressionClassifier:", str(LogisticRegressionClassifier_accuracy_sum / 30))
print("BNB:", str(BNB_accuracy_sum / 30))
print("MNB:", str(MNB_accuracy_sum / 30))
print("ABC:", str(ABC_accuracy_sum / 30))

print()

print("Time:", str(time_sum / 30))
print("Accuracy:", str(accuracy_sum / 30))
