# Data Load
from sklearn import datasets

digits = datasets.load_digits()
print(digits.data)
print(digits.target)

# 데이터 사이즈 확인
print(digits.data.shape)
print(digits.target.shape)

# 샘플 데이터 시각화
import matplotlib.pyplot as plt

plt.imshow(digits.images[0], cmap = plt.cm.gray_r, interpolation = 'nearest')

# 이미지 크기 확인
print(digits.images.shape)
print(len(digits.images))

# 8x8 이미지가 1797개

# Flattening
n_samples = len(digits.images)

data = digits.images.reshape((n_samples, -1))

print(data)
print(data.shape)

# 트레이닝 / 테스트 분리
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size = 0.2)

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

# 분류 및 학습
from sklearn.neighbors import KNeighborsClassifier as KNN
KNN_model = KNN(n_neighbors = 6)

KNN_model.fit(X_train, y_train)

# 예측
y_pred = KNN_model.predict(X_test)
print(y_pred[:3])

# 평가
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)

# 시각화
# 이미지를 출력하기 위해서는 평탄화된 이미지를 다시 8x8로 만들어야 한다.
plt.imshow(X_test[5].reshape(8, 8), cmap = plt.cm.gray_r, interpolation = 'nearest')

y_pred = KNN_model.predict([X_test[5]]) # 항상 입력은 2차원
print(y_pred)