# linear regression

import numpy as np

class LinearRegressionGD:
    def __init__(self, lr = 0.01, n_iter = 50, random_state = 1):
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        # random seed 생성
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.b_ = np.array([0.])
        self.losses_ = []
        
        # Gradient Descent
        for i in range(self.n_iter):
            y_predicted = self.net_input(X)
            errors = (y - y_predicted)

            self.w_ += self.lr * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.lr * 2.0 * errors.mean()

            loss = (errors**2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        return self.net_input(X)
    