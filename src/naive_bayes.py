import src.statistic as stc

p = stc.conditional_probability
m = stc.marginal_probability
g = stc.gaussian_probability_density


def gaussian_naive_bayes_classifier(v, X, Y):
    results = []
    for class_i in set(Y):
        probability = stc.prior_probability(class_i, Y)
        X_class_i = [X[j] for j in range(len(Y)) if Y[j] == class_i]
        for i in range(len(v)):
            w = [w[i] for w in X_class_i]
            probability *= g(v[i], stc.expected_value(w),
                             stc.standard_deviation(w))
        results += [(class_i, probability)]
    return [x[1] for x in results].index(max([x[1] for x in results]))


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    X, Y = make_blobs(n_samples=100, random_state=173)
    y_p = [gaussian_naive_bayes_classifier(v, list(X), list(Y)) for v in X]

    plt.figure(figsize=(12, 4), dpi=80)
    plt.scatter(X[:, 0], X[:, 1], c=y_p)
    plt.show()
