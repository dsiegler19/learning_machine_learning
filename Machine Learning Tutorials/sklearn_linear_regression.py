import os
import math
import quandl
import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
from sklearn import cross_validation

style.use("ggplot")

df = quandl.get("WIKI/GOOGL")
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100
df["PCT_Change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100

# The features to train against
df = df[["Adj. Close", "HL_PCT", "PCT_Change", "Adj. Volume"]]

forecast_col = "Adj. Close"
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1 * len(df)))
print(forecast_out)

df["label"] = df[forecast_col].shift(-forecast_out)

# X is features, y is labels
X = np.array(df.drop(["label"], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df["label"])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

classifier_f = open(os.getcwd() + "/pickles/linear_regression_pickles/linear_regression.pickle", "rb")
classifier = pickle.load(classifier_f)

accuracy = classifier.score(X_test, y_test)

forecast_set = classifier.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df["Forecast"] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
