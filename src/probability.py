import numpy as np
import math


def LCG(seed, n, a=1664525, c=1013904223, m=2 ** 32):
    """
    A linear congruential generator (LCG)
    Yields a sequence of pseudo-randomized numbers calculated with a discontinuous piecewise linear equation.

    1) c and m are relatively prime
    2) a - 1 is divisible by all prime factors of m
    3) a - 1 is a multiple of 4 if m is a multiple of 4.
    """
    X = []
    for i in range(n):
        seed = (a * seed + c) % m
        X.append(seed)
    return X


def get_random_numbers_vector(m):
    return np.random.random(int((np.random.random() * m)))


def prior_probability(x_i, X):
    return X.count(x_i) / len(X)


def joint_probability(x_i, y_i, X, Y):
    return [x_i == X[i] and y_i == Y[i] for i in range(len(X))].count(True) / len(X)


def conditional_probability(x_i, y_i, X, Y):
    Z = [z for z in list(zip(X, Y)) if z[1] == y_i]
    return [x_i == z[0] for z in Z].count(True) / len(Z)


def marginal_probability(x_i, X, Y):
    return sum([conditional_probability(x_i, y_i, X, Y)
                * prior_probability(y_i, Y) for y_i in Y])


def mutual_information(X, Y):
    # buggy
    result_sum = 0
    for i in range(len(X)):
        x_i = X[i]
        y_i = Y[i]
        m = (marginal_probability(x_i, X, Y) * marginal_probability(y_i, Y, X))
        j = joint_probability(x_i, y_i, X, Y)
        result_sum += j * math.log1p(j / m)
    return result_sum
