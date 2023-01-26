from algebra import matrix_transpose

import math


def expected_value(X):
    return sum(X) / len(X)


def sum_of_square_deviations(X):
    return sum((x_i - expected_value(X)) ** 2 for x_i in X)


def variance(X):
    return sum_of_square_deviations(X) / (len(X) - 1)


def standard_deviation(X):
    return variance(X) ** 0.5


def gaussian_probability_density(x, u, std):
    return (1 / (std * math.sqrt(2 * math.pi))) * math.exp(-((x - u) ** 2 / (2 * std ** 2)))


def prior_probability(x_i, X):
    return X.count(x_i) / len(X)


def joint_probability(x_i, y_i, X, Y):
    return [x_i == X[i] and y_i == Y[i]
            for i in range(len(X))].count(True) / len(X)


def conditional_probability(x_i, y_i, X, Y):
    Z = [z for z in list(zip(X, Y)) if z[1] == y_i]
    return [x_i == z[0] for z in Z].count(True) / len(Z)


def marginal_probability(x_i, X, Y):
    return sum([conditional_probability(x_i, y_i, X, Y)
                * prior_probability(y_i, Y) for y_i in Y])


def covariance(X, Y):
    return sum([(X[i] - expected_value(X)) * (Y[i] - expected_value(Y))
                for i in range(len(X))]) * (1 / (len(X) - 1))


def covariance_matrix(A):
    return [[covariance(i, j) for j in matrix_transpose(A)] for i in matrix_transpose(A)]
