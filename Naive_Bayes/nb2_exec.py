# wine classification

# data load
from sklearn import datasets

raw_wine = datasets.load_wine()

# feature, target
X = raw_wine.data
y = raw_wine.target

print(X.shape)
print(y.shape)

print(X)
print(y)

# data split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaling = scaler.transform(X_train)
X_test_scaling = scaler.transform(X_test)


# Naive Bayes
from sklearn.naive_bayes import GaussianNB

GNB_model = GaussianNB()
GNB_model.fit(X_train_scaling , y_train)

# prediction
y_pred = GNB_model.predict(X_test_scaling)
print(y_pred)

# evaluation
# recall
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred, average = 'macro')
print(recall)

# confusion matrix
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# class_report
from sklearn.metrics import classification_report

class_report = classification_report(y_test, y_pred)
print(class_report)