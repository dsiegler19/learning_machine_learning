from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from tweepy.streaming import json
from NLP import sentiment
import os

# Consumer key, consumer secret, access token, access secret.
ckey="yc9hmrcLMy0NXJCdSHzSS7D2z"
csecret="VeYgbUnpXnFjHyPGXdqQTngkn7TSYc9OBgeQzwnBMizjdwjHjj"
atoken="4089934096-G4M7yiBG9tcoYO1t4n434n28DBM02NAdUJDSowL"
asecret="xYK6gNgBhWOJFB7K5bLGQycsoG2MIvNwqkHRSTdnlIq1I"

min_confidence = 80


class Listener(StreamListener):

    def on_data(self, data):

        all_data = json.loads(data)

        tweet = all_data["text"]
        sentiment_value, confidence = sentiment.sentiment(tweet)

        print(tweet, "\n", sentiment_value, confidence)

        if confidence * 100 > min_confidence:

            output = open(os.getcwd() + "/twitter-output.txt", "a")
            output.write(sentiment_value)
            output.write("\n")
            output.close()

        return True

    def on_error(self, status):
        print(status)

sentiment.set_pickle_location("short_reviews")
sentiment.load_pickles()

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, Listener())
twitterStream.filter(track=["trump"], follow="@barackobama")