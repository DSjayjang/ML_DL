import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        """
        X: numpy ndarray
        y: numpy ndarray
        """

        n_samples, n_features = X.shape # (800, 10)
        self._classes = np.unique(y) # [0 1]
        n_classes = len(self._classes) # 2

        # init mean, var, priors
        self._mean = np.zeros((n_classes, n_features), dtype = np.float64) # n_classes X n_features
        """
        array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        """

        self._var = np.zeros((n_classes, n_features), dtype = np.float64)
        """
        array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        """    

        self._priors = np.zeros(n_classes, dtype = np.float64)
        """
        array([0., 0.])
        """

        for c in self._classes:
            X_c = X[c == y]

            self._mean[c, :] = X_c.mean(axis = 0) # 각 class 0과 1에서, 각 feature들의 평균
            self._var[c, :] = X_c.var(axis = 0) # 각 class 0과 1에서, 각 feature들의 분산
            self._priors[c] = X_c.shape[0] / float(n_samples) # 각 class 0과 1에서, 전체 sample 중 차지하는 샘플 비율

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]

        return y_pred

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            """
            prior
            각 class별 비율
            log를 취하여 곱셈을 덧셈으로 변경
            """            
            prior = np.log(self._priors[idx])

            """
            조건부 확률 (log likelihood)
            """
            class_conditional = np.sum(np.log(self._pdf(idx, x)))

            """
            posterior = log prior + log likelihood
            """
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)] # 가장 높은 posterior를 가지는 class 출력

    # probability density function
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator
