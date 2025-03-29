# 유클리디안 거리 계산 함수
import numpy as np

def euclidean_distance(x1, x2):
    # e.g. 아래와 같은 거리를 계산함
    # x1 = X = array([5.1, 3.5, 1.4, 0.2])
    # x2 = X_train = array([6.2, 2.8, 4.8, 1.8])
    
    return np.sqrt(np.sum((x1-x2))**2)