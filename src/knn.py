from src.misc import euclidean_distance, euclidean_norm


def k_nearest_neighbors(X, y, v, k, regression=False, d=euclidean_distance):
    neighbors = [(d(X[i], v), y[i]) for i in range(len(X))]
    neighbors.sort(key=lambda w: w[1])  # call by reference
    knn = neighbors[:k]
    labels = [x[1] for x in knn]
    if regression:
        return sum(labels) / len(labels)
    else:  # classification
        return max(labels, key=labels.count)


def knn(v, X, k):
    neighbors = [(X[i], euclidean_distance(X[i], v)) for i in range(len(X))]
    neighbors.sort(key=lambda w: w[1])  # call by reference
    knn = neighbors[1:k + 1]
    return [x[0] for x in knn]


def outlier_detection_knn(X, k=10):
    knn_table = [knn(X[j], X, k) for j in range(len(X))]
    knn_distance = [[X[i] - x for x in knn_table[i]][k - 1]
                    for i in range(len(X))]
    return list(map(euclidean_norm, knn_distance))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    import random
    import numpy as np
    import pandas as pd

    # Sample Data
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=170)
    noise = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(10)]
    X = np.append(X, noise, axis=0)
    data = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1]))

    knn_distance = outlier_detection_knn(X)
    plt.figure(figsize=(12, 4), dpi=80)
    plt.scatter(data.x, data.y, c=knn_distance, s=[x * 30 for x in knn_distance])
    plt.show()
