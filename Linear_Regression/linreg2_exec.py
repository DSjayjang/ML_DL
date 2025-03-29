import pandas as pd
import numpy as np

from linreg2 import LinearRegressionGD

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']

df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt',
                 sep = '\t', usecols = columns)

# 전처리
df['Central Air'] = df['Central Air'].map({'Y': 1, 'N': 0})

# NA handling
df.isna().sum()
df = df.dropna(axis = 0)

X = df[['Gr Liv Area']].values
y = df['SalePrice'].values

print(X.shape)
print(X)
print(y.shape)
print(y)

# scaling
from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaling = X_scaler.fit_transform(X)
y_scaling = y_scaler.fit_transform(y[:, np.newaxis]).flatten()
"""
데이터가 2차원 배열로 저장되어 있어야 하기 때문에,
np.newaxis로 새로운 차원을 추가 후 표준화를 진행,
그리고 다시 flatten을 통해 1차원 배열로 되돌림.
"""

print(X_scaling)
print(y_scaling)


# Training
regressor = LinearRegressionGD(lr = 0.1)
regressor.fit(X_scaling, y_scaling)


# Cost function
import matplotlib.pyplot as plt

plt.plot(range(1, regressor.n_iter+1), regressor.losses_)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.tight_layout()
plt.show()

from linreg_plot import lin_reg_plot

lin_reg_plot(X_scaling, y_scaling, regressor)
plt.xlabel('Living area above ground (standardized)')
plt.ylabel('Sale price (standardized)')
plt.show()


# 예측된 출력 값을 원래 스케일로 복원하기
# 2500일 때 주택 가격 예측
feature_std = X_scaler.transform(np.array([[2500]]))

# 예측
target_std = regressor.predict(feature_std)

# 스케일 복원
target_reverted = y_scaler.inverse_transform(target_std.reshape(-1, 1))

print(f'판매가격: ${target_reverted.flatten()[0]:.2f}')

print(f'기울기: {regressor.w_[0]:.3f}')
print(f'절편: {regressor.b_[0]:.3f}')