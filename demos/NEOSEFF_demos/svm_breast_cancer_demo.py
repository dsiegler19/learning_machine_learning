# Support Vector Machines are a very common machine learning algorithm. An SVM attempts to find the best separating
# hyperplane (since an SVM works in multidimensional, it is a hyperplane, in 2d space this would be just a line).
# An SVM is binary, meaning that it can only classify data into 2 sets at a time. This doesn't mean that an SVM can't
# classify data into more than 2 categories, it just means it only categorizes into 2 categories at a time.
# ^
# |
# |                +
# |              +
# |                +
# |
# |
# |
# |    -
# | -
# |   -
# |------------------>
# An SVM would look to find the separating hyperplane that maximizes the distance from that hyperplane to each point
# ^
# |    \
# |     \          +
# |      \       +/
# |      /\     //  +
# |     / /\   //  /
# |    / // \ //  /
# |   / //   \/  /
# |  / -/     \ /
# | -  /       \
# |   -         \
# |------------------>
# Imagine each of the lines were perpendicular to the separating hyperplane. Although there are other separating
# hyperplanes, this one maximizes the distance to each of the points. This concept can be applied in higher dimensions.
# ^
# |    \
# |     \   p1       +
# |      \       +
# |  p2   \           +
# |        \
# |         \
# |          \
# |    -      \
# | -          \
# |   -         \
# |------------------>
# Since p1 is to the right of the separating hyperplane, p1 is +. Since p2 is to the left of the separating hyperplane,
# it is -.
# Vectors have both magnitude and direction. Vectors are represented like this:
# ->
# A  = [3, 4] represents a vector going from [0, 0] to [3, 4]. Its magnitude is calculated like its length (or norm as
# it is called in linear algebra):
# ||A|| = sqrt((3 ^ 2) + (4 ^ 2))
# Multiplying vectors is done with the dot (·) operator and it results in a scalar (regular number), or a vector with
# just magnitude
# ->           ->
# B = [1, 3]   C = [4, 2]
# ->  ->
# A · B = (1 * 4) + (3 * 2) = 10
# Once the SVM has been trained, how does it classify something? This is done through support vector assertion (see
# Support Vector Assertion.png). A vector W is a vector perpendicular to the separating hyperplane. When the SVM goes to
# classify a point, it makes a vector U to that point and then projects U onto W and calculates, accounting for the
# bias, whether U "goes past" the separating hyperplane (+) or "stays behind" it (-). This is calculated by this
# formula:
# U · W + b
# If this is > 0 then it is positive, if it = 0 then it on the decision boundary, and if it is < 0 then it is negative.
# However, the SVM still has to calculate W and b. Math tells us that if point X is in the positive SV domain then
# the equation = 1 and if X is in the negative SV domain then the equation = -1. This means:
# X(+) · W + b = 1
# X(-) · W + b = -1
# The SVM can introduce variable y sub i (denoted yi), which is dependant on the class of X. This means when X is in
# the positive SV domain yi = 1 and when X is in the negative SV domain, yi = -1
# Going back to the previous equations, the SVM can now use yi to manipulate the equations:
# X(+) · W + b = 1      yi = 1
# X(-) · W + b = -1     yi = -1
# Multiplying both sides by yi:
# yi(X(+) · W + b) = 1 * 1 = 1
# yi(X(-) · W + b) = -1 * -1 = 1
# Subtracting 1 from both sides to make the right hand = 0:
# yi(X · W + b) - 1 = 0
# And this is the equation that the SVM uses to classify Xs. However, this equation doesn't define a support vector,
# but for a SVM to be good its support vectors must satisfy this equation. A support vector is simply a vector that,
# if moved, would change the best separating hyperplane.
# ^
# |               \
# |                +
# |                 \
# |\                 +
# | \                 \
# |  \                 \
# |   \                 \
# |    -
# |     \
# |      \
# |---------------------->
# These are 2 support vectors.
# To get the best separating hyperplane the SVM simply takes the width between the support vectors, divides it by 2,
# and adds that scalar to the smaller of the to support vectors.
# ^
# |              /\
# |             /  +
# |            /    \
# |\          /      +
# | \        /        \
# |  \      /width     \      The width line should be perpendicular to the 2 support vectors
# |   \    /            \
# |    -  /
# |     \/
# |      \
# |---------------------->
# However, to make the best separating hyperplane the SVM must maximize the width.

import os
import random
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import svm
from sklearn.externals import joblib

df = pd.read_csv(os.getcwd() + "/cancer_dataset/breast-cancer-wisconsin.data.txt")
df.replace("?", -99999, inplace=True)
df.drop(["id"], 1, inplace=True)

X = np.array(df.drop(["class"], 1))
y = np.array(df["class"])

classifier = joblib.load("models/svm_breast_cancer_demo.pickle")

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.99)

accuracy = classifier.score(X_test, y_test)
print("The support vector machine is finished training.")
print("The accuracy on the testing data is: ", (accuracy * 100), "%\n")

print("Now to try your own data.")

while True:

    print("Please enter the following parameters or ? for unknown (can produce some strange results):")

    print("Clump thickness on a scale of 1 - 10")
    clump_thickness = input()

    if clump_thickness == "exit":
        break

    if clump_thickness == "retrain":

        print("\n" * 50)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

        classifier = svm.SVC()
        classifier.fit(X_train, y_train)

        accuracy = classifier.score(X_test, y_test)

        joblib.dump(classifier, "models/svm_breast_cancer_demo.pickle")
        print("\n" * 50)

        print("The support vector machine is finished training.")
        print("The accuracy on the testing data is: ", (accuracy * 100), "%\n")

        print("Now to try your own data.")

        continue

    if clump_thickness == "example":
        idx = random.randrange(0, len(X_test))
        example = X_test[idx]

        print("Here is an example data point from the testing set:\n")

        print("Uniformity of cell size on a scale of 1 - 10 (10 is very uniform):                 ", example[0])
        print("Uniformity of cell shape on a scale of 1 - 10 (10 is very uniform):                ", example[1])
        print("Marginal adhesion (stickyness of cells) on a scale of 1 - 10 (10 is very stick):   ", example[2])
        print("Single epithelial cell size on a scale of 1 - 10 (10 is very large):               ", example[3])
        print("Frequency of bare nuclei on a scale of 1 - 10 (10 is very frequent):               ", example[4])
        print("Frequency of bland chromatin on a scale of 1 - 10 (10 is very frequent):           ", example[5])
        print("Frequency of normal nucleoli on a scale of 1 - 10 (10 is very frequent):           ", example[6])
        print("Frequency of normal mitoses on a scale of 1 - 10 (10 is very frequent):            ", example[7], "\n")

        example = np.array(example)
        example = example.reshape(-1, len(example))

        example_prediction = classifier.predict(example)

        if example_prediction == 2:
            print("The support vector machine predicts that the inputted cancer is BENIGN")
        else:
            print("The support vector machine predicts that the inputted cancer is MALIGNANT")

        actual = y_test[idx]

        if actual == 2:
            print("The cancer was actually BENIGN\n")
        else:
            print("The cancer was actually MALIGNANT\n")

        if example_prediction == actual:
            print("The support vector machine is CORRECT")
        else:
            print("The support vector machine is INCORRECT")

        continue

    print("Uniformity of cell size on a scale of 1 - 10 (10 is very uniform)")
    uniformity_size = input()

    print("Uniformity of cell shape on a scale of 1 - 10 (10 is very uniform)")
    uniformity_shape = input()

    print("Marginal adhesion (stickyness of cells) on a scale of 1 - 10 (10 is very stick)")
    adhesion = input()

    print("Single epithelial cell size on a scale of 1 - 10 (10 is very large)")
    epithelial_size = input()

    print("Frequency of bare nuclei on a scale of 1 - 10 (10 is very frequent)")
    bare_nuclei = input()

    print("Frequency of bland chromatin on a scale of 1 - 10 (10 is very frequent)")
    bland_chromatin = input()

    print("Frequency of normal nucleoli on a scale of 1 - 10 (10 is very frequent)")
    normal_nucleoli = input()

    print("Frequency of normal mitoses on a scale of 1 - 10 (10 is very frequent)")
    normal_mitoses = input()

    try:
        clump_thickness = int(clump_thickness)
        uniformity_size = int(uniformity_size)
        uniformity_shape = int(uniformity_shape)
        adhesion = int(adhesion)
        epithelial_size = int(epithelial_size)
        bare_nuclei = int(bare_nuclei)
        bland_chromatin = int(bland_chromatin)
        normal_nucleoli = int(normal_nucleoli)
        normal_mitoses = int(normal_mitoses)
    except ValueError:
        print("You gave an invalid input!")
        continue

    inputted = np.array([clump_thickness, uniformity_size, uniformity_shape, adhesion, epithelial_size, bare_nuclei,
                         bland_chromatin, normal_nucleoli, normal_mitoses])
    inputted = inputted.reshape(-1, len(inputted))

    prediction = classifier.predict(inputted)

    if prediction == 2:
        print("The support vector machine predicts that the inputted cancer is BENIGN\n")
    else:
        print("The support vector machine predicts that the inputted cancer is MALIGNANT\n")
