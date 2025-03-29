import numpy as np

from knn1 import KNN

# data load
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

print(X_train.shape)
print(X_train[0])

print(y_train.shape)
print(y_train[0])

# 시각화
import matplotlib.pyplot as plt
from matplotlib.colors import  ListedColormap
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

fig = plt.figure(figsize = (12,4))
fig.add_subplot(1,2,1)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = cmap, edgecolor = 'k', s = 30)
fig.add_subplot(1,2,2)
plt.scatter(X[:, 2], X[:, 3], c = y, cmap = cmap, edgecolor = 'k', s = 30)
plt.show()

# KNN 수행
k = 3
clf = KNN(k = k)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(y_test)

print(accuracy)