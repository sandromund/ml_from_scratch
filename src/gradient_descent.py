

def gradient_descent(X, Y, learning_rate=0.3, m=0, b=0, iterations=5):
    for i in range(iterations):
        b_gradient = - (2 / len(X)) * sum([ Y[i] - ( m * X[i] + b)
                                           for i in range(len(X))])
        m_gradient = - (2 / len(X)) * sum([ X[i] * (Y[i] - ( m * X[i] + b))
                                           for i in range(len(X))])
        b = b - (learning_rate * b_gradient)
        m = m - (learning_rate * m_gradient)
    return b, m


if __name__ == '__main__':

    from sklearn.datasets import make_regression
    from src.misc import mean_square_error
    import matplotlib.pyplot as plt
    import pandas as pd

    # Sample Data
    data_regression = make_regression(n_samples=300, n_features=1, n_targets=1, noise=30, bias=10)
    data = pd.DataFrame.from_records(data_regression[0], columns=['x'])
    data['y'] = data_regression[1]
    X = data.x.values
    Y = data.y.values
    b, m = gradient_descent(X, Y)
    Y_prediction = [m * X[i] + b for i in range(len(X))]
    print("mean_square_error: ", mean_square_error(Y, Y_prediction))

    f = plt.figure(figsize=(12, 4), dpi=80)
    plt.scatter(X, Y, figure=f)
    plt.plot(X, Y_prediction, "--", figure=f)
    plt.show()
