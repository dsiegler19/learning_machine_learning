import os
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import neighbors

df = pd.read_csv(os.getcwd() + "/breast-cancer-wisconsin.data.txt")
df.replace("?", -99999, inplace=True)
df.drop(["id"], 1, inplace=True)

X = np.array(df.drop(["class"], 1))
y = np.array(df["class"])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

classifier = neighbors.KNeighborsClassifier()
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = classifier.predict(example_measures)
print(prediction)
