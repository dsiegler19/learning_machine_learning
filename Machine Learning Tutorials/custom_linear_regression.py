import random
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use("fivethirtyeight")

def create_dataset(size, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(size):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == "pos":
            val += step
        elif correlation and correlation == "neg":
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
        ((mean(xs) * mean(xs)) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b


def squared_error(ys_original, ys_line):
    return sum((ys_line - ys_original) ** 2)

def coefficient_of_determination(ys_original, ys_line):
    y_mean_line = [mean(ys_original) for y in ys_original]
    squared_error_regression = squared_error(ys_original, ys_line)
    squared_error_y_mean = squared_error(ys_original, y_mean_line)
    return 1 - (squared_error_regression / squared_error_y_mean)

xs, ys = create_dataset(40, 40, step=2, correlation="pos")

m, b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m * x + b) for x in xs]

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()
