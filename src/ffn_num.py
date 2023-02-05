import numpy as np


class NeuralNetwork:

    def __init__(self):
        self.alpha = 0.2

        self.activation_function = lambda x: (x > 0) * x  # ReLu
        self.activation_derivative = lambda x: x > 0  # ReLu derivative

        self.weights_0_1 = self.__init_layer(3, 4)
        self.weights_1_2 = self.__init_layer(4, 1)

    @staticmethod
    def __init_layer(n_input_units, n_output_units):
        return 2 * np.random.random((n_input_units, n_output_units)) - 1

    def predict(self, X):
        layer_1 = self.activation_function(np.dot(X, self.weights_0_1))
        layer_2 = np.dot(layer_1, self.weights_1_2)
        return layer_2

    def learn(self, n_iterations, X, Y):
        for iteration in range(1, n_iterations + 1):
            error = 0
            for i in range(len(X)):
                layer_0 = X[i:i + 1]
                layer_1 = self.activation_function(np.dot(layer_0, self.weights_0_1))
                layer_2 = np.dot(layer_1, self.weights_1_2)

                error += np.sum((layer_2 - Y[i]) ** 2)  # MSE

                layer_2_delta = layer_2 - Y[i:i + 1]
                layer_1_delta = layer_2_delta.dot(self.weights_1_2.T) * self.activation_derivative(layer_1)

                self.weights_1_2 -= self.alpha * layer_1.T.dot(layer_2_delta)
                self.weights_0_1 -= self.alpha * layer_0.T.dot(layer_1_delta)

            if iteration % 10 == 0:
                print("Iteration:", iteration, " MSE Error:", error)


if __name__ == '__main__':
    np.random.seed(1)

    X = np.array([[1, 0, 1],
                  [0, 1, 1],
                  [0, 0, 1],
                  [1, 1, 1]])

    Y = np.array([[1, 1, 0, 0]]).T

    NN = NeuralNetwork()

    NN.learn(50, X, Y)

    print(NN.predict(X[0]))
