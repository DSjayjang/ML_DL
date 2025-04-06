import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from Naive_Bayes.nb1 import NaiveBayes
from utils.metrics import accuracy

SEED = 123

# data
X, y = datasets.make_classification(n_samples = 1000, n_features = 10, n_classes = 2, random_state = SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# training
nb = NaiveBayes()
nb.fit(X_train, y_train)

# prediction
y_pred = nb.predict(X_test)
print(y_pred)

# evaluation
acc = accuracy(y_test, y_pred)
print('Naive Bayes classification accuracy', acc)

####
n_samples, n_features = X_train.shape # (800, 10)
classes = np.unique(y_train)
n_classes = len(classes)
n_features = X_train.shape[1]

mean = np.zeros((n_classes, n_features), dtype = np.float64) # n_classes X n_features
var = np.zeros((n_classes, n_features), dtype = np.float64)
priors = np.zeros(n_classes, dtype = np.float64)
priors

for c in classes:
    X_c = X_train[c == y_train]
    mean[c, :] = X_c.mean(axis = 0) # 각 class의 각 feature 평균
    var[c, :] = X_c.var(axis = 0)
    priors[c] = X_c.shape[0] / float(n_samples)
    print(priors)



classes
X_c = X_train[0 == y_train]
X_c.mean(axis = 0)

mean