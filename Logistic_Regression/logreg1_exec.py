import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from logreg1 import LogisticRegression
from utils.metrics import accuracy

# data load
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

# data spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)
print(X_train.shape)
print(X_train[:5])

print(y_train.shape)
print(y_train[:5])

# training
regressor = LogisticRegression(lr = 0.0001, n_iters = 1000)
regressor.fit(X_train, y_train)

# prediction
predictions = regressor.predict(X_test)

# evaluation
acc = accuracy(y_test, predictions)

print('LR classficication accuracy:', acc)