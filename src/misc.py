from math import sqrt

from statistic import expected_value, standard_deviation
from algebra import dot_product, l2_norm


def mean_square_error(Y, Y_prediction):
    return (1 / len(Y)) * sum([(Y[i] - Y_prediction[i])**2 for i in range(len(Y))])


def euclidean_distance(v, w):
    return sqrt(sum([(v[i] - w[i])**2 for i in range(len(v))]))


def euclidean_norm(v):
    return sqrt(sum([x**2 for x in v]))


def scale(X):
    return [(X[i] - expected_value(X)) / standard_deviation(X) for i in range(len(X))]


def cosine_similarity(v, w):
    return dot_product(v, w) / (l2_norm(v) * l2_norm(w))