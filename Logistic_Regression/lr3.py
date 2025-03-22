# data load
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# data spliting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

# scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaling = scaler.transform(X_train)
X_test_scaling = scaler.transform(X_test)

X_combined_scaling = np.vstack((X_train_scaling, X_test_scaling))
y_combined = np.hstack((y_train, y_test))

# logistic regression
from sklearn.linear_model import LogisticRegression as LR

LR_model = LR(C = 100, solver = 'lbfgs', multi_class = 'multinomial')
LR_model.fit(X_train_scaling, y_train)

# visualization
import matplotlib.pyplot as plt
from Logistic_Regression.decision_regions import decision_regions

decision_regions(X_combined_scaling, y_combined,
                 classifier = LR_model,
                 test_idx = range(105, 150))

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')

plt.legend(loc = 'best')
plt.tight_layout()
plt.show()