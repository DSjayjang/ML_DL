# linear regression

import numpy as np

class LinearRegression:
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # Training
    def fit(self, X, y):
        # initialize parameters
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        """
        모든 가중치 0으로 초기화
        e.g. [0.]
        """

        self.bias = 0
        """
        편향 0으로 초기화 / scalar
        """        

        # Gradient Descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            """
            y = Xw + b
            y : n x 1
            X : n x p
            w : p x 1
            b : scalar
            """

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            """
            dw = (1/n) * X^T * (y_pred - y_true)
            db = (1/n) * sum(y_pred - y_true)
            """

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias

        return y_predicted