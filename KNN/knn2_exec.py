# data load
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# 데이터 확인
print(X[:3])
print(y[:3])

# 사이즈 확인
print(X.shape)
print(y.shape)

# class 확인
print('클래스 레이블:', np.unique(y))

# train / test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

# label count
print(np.bincount(y))
print(np.bincount(y_train))
print(np.bincount(y_test))

# scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaling = scaler.transform(X_train)
X_test_scaling = scaler.transform(X_test)

# 결정 경계를 그리기 위해 train, test 데이터를 combine
X_combined_scaling = np.vstack((X_train_scaling, X_test_scaling))
y_combined = np.hstack((y_train, y_test))

print(X_combined_scaling)
print(y_combined)

# 분류 전 시각화 해보기
import matplotlib.pyplot as plt

# x, y축 범위 지정하기 위한 최대, 최소값
x1_min, x1_max = X_combined_scaling[:, 0].min() - 1, X_combined_scaling[:, 0].max() + 1
x2_min, x2_max = X_combined_scaling[:, 1].min() - 1, X_combined_scaling[:, 1].max() + 1

# 마커와 색상
markers = ('o', 's', '^')
colors = ('red', 'blue', 'lightgreen')

# 시각화
plt.figure()
for idx, cl in enumerate(np.unique(y_combined)):
    plt.scatter(x = X_combined_scaling[y_combined == cl, 0], y = X_combined_scaling[y_combined == cl, 1],
                alpha = 0.8, c = colors[idx], marker = markers[idx], label = f'Class {cl}', edgecolor = 'black')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# 분류와 시각화를 함수로 설정하기
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02):

    # 마커, 컬러맵 설정
    markers = ('o', 's', '^')
    colors = ('red', 'blue', 'lightgreen')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    plt.contourf(xx1, xx2, lab, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 클래스 샘플그리기
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1],
                    alpha = 0.8, c = colors[idx], marker = markers[idx], label = f'Class {cl}', edgecolor='black')

    # 테스트 샘플 부각하기
    if test_idx:
        X_test = X[test_idx, :]
        
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c = 'none', edgecolor = 'black', alpha = 1.0, linewidth = 1, marker = 'o', s = 100, label = 'Test set')
        
# 분류 및 시각화
from sklearn.neighbors import KNeighborsClassifier as KNN

KNN_model = KNN(n_neighbors = 5, p = 2, metric = 'minkowski')
KNN_model.fit(X_train_scaling, y_train)

plot_decision_regions(X_combined_scaling, y_combined, classifier = KNN_model, test_idx = range(100, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# 정확도
accuracy = KNN_model.score(X_test_scaling, y_test)
print(f'accuracy: {accuracy * 100:.2f}%')

# confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = KNN_model.predict(X_test_scaling)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)