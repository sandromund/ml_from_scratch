from math import e
import numpy as np

from src.algebra import matrix_multiplication


def sigmoid(x):
    divider = (1 + e ** (-x))
    return 1 / divider if divider else 0


def sigmoid_derivative(x):
    return x * (1 - x)


def random_matrix(i, j):
    return np.random.rand(i, j)


def matrix_transpose(A):
    return np.array(A).T


T = lambda M: matrix_transpose(M)
o = lambda A, B: matrix_multiplication(A, B)
f = lambda M: [list(map(sigmoid_derivative, x)) for x in M]


class NeuralNetwork:

    def __init__(self, x, y):
        self.input = x
        self.weights1 = random_matrix(self.input.shape[1], 4)
        self.weights2 = random_matrix(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = [list(map(sigmoid, x)) for x in
                       (o(self.input, self.weights1))]
        self.layer2 = [list(map(sigmoid, x)) for x in
                       (o(self.layer1, self.weights2))]
        return self.layer2

    def backprop(self):
        w = [2 * (self.y[i] - self.output[i]) *
             sigmoid_derivative(self.output[i][0])
             for i in range(len(self.y))]
        self.weights2 += o(T(self.layer1), w)
        self.weights1 += o(T(self.input), o(w, f(T(self.weights2))))

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    import pandas as pd

    data_classification = make_classification(n_samples=300, n_features=4, n_classes=2)
    data = pd.DataFrame.from_records(data_classification[0], columns=['x1', 'x2', 'x3', 'x4'])
    data['y'] = data_classification[1]
    plt.figure(figsize=(14, 8))
    plt.scatter(data.x3, data.x4, c=data.y)
    plt.show()

    X = data[['x1', 'x2', 'x3', 'x4']].values
    y = np.array([[i] for i in data['y']], dtype=float)

    NN = NeuralNetwork(X, y)
    for i in range(10):
        print("iteration " + str(i) + " - Loss: " + str(np.mean(np.square(y - NN.feedforward()))))
        NN.train(X, y)

