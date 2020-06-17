import random


def k_means_Lloyd(X, k):
    # Initialisierung: Wähle k zufällige Mittelwerte
    centroids = [list(X[random.randint(0, len(X) - 1)]) for i in range(k)]
    last_centroids, assignments = None, None

    while last_centroids != centroids:

        last_centroids = centroids.copy()

        # Datenobjekt zugeordnen, wobei Cluster-Varianz minimiert wird
        assignments = [min([(sum((X[x_i] - centroids[k_i]) ** 2), k_i)
                            for k_i in range(k)])[1]
                       for x_i in range(len(X))]

        # Mittelpunkte der Cluster neu berechnen
        for k_i in range(k):
            k_i_subset = [X[i] for i in range(len(X))
                          if assignments[i] == k_i]
            centroids[k_i] = list(1 / len(k_i_subset) * sum(k_i_subset))
    return assignments


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    X, Y = make_blobs(n_samples=100, random_state=173)

    plt.figure(figsize=(12, 4), dpi=80)
    plt.scatter(X[:, 0], X[:, 1], c=k_means_Lloyd(X, k=3))
    plt.show()
