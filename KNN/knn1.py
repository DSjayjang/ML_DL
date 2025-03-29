# KNN 구현
import numpy as np
from collections import Counter

from utils.distance import euclidean_distance

class KNN:
    def __init__(self, k = 3):
        self.k = k
    
    # Training
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    # Prediction
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]

        return np.array(predicted_labels)
    
    def _predict(self, x):
        # Compute Distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        """
        유클리디안 거리 계산
        [323.70000000000005,
         261.29999999999995,
         563.6999999999999,
         276.29999999999995,
         533.7,
         ...]
        """

        # Get k-nearest samples, and Labels
        k_indices = np.argsort(distances)[: self.k]
        """
        가장 가까운 거리에 있는 데이터 k개의 인덱스를 출력 
        e.g. array([48, 60, 74], dtype=int64)
        """
        
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        """
        가장 가까운 거리에 있는 k개의 target y 출력 
        e.g. [1, 1, 2]
        """

        # Majority vote, Most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        """
        가장 가까운 k개의 데이터에서 가장 많이 나온 label의 개수를 출력
        e.g. label 1이 2번
        e.g. [(1, 2)] 
        """

        return most_common[0][0]
        """
        그 때의 label을 출력
        """