import pandas as pd
import numpy as np

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

from sklearn.linear_model import LinearRegression as LR

LR_model = LR()
LR_model.fit(X, y)

y_pred = LR_model.predict(X)
print(f'기울기: {LR_model.coef_[0]:.3f}')
print(f'절편: {LR_model.intercept_:.3f}')

# visualization
import matplotlib.pyplot as plt
from linreg_plot import lin_reg_plot

lin_reg_plot(X, y, LR_model)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.tight_layout()
plt.show()

# Normal equation(정규 방정식)으로 회귀계수 직접 계산하기
# w = (X'X)^{-1}X'y

Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(Xb.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))

print(f'기울기: {w[1]:.3f}')
print(f'절편: {w[0]:.3f}')

# QR decomposition으로 회귀계수 직접 계산하기
# w = (X'X)^{-1}X'y = X^{-1}y = (QR)^{-1}y = R^{-1}Q^{-1}y = R^{-1}Q'y

Xb = np.hstack((np.ones((X.shape[0], 1)), X))
Q, R = np.linalg.qr(Xb)
w = np.dot(np.linalg.inv(R), np.dot(Q.T, y))

print(f'기울기: {w[1]:.3f}')
print(f'절편: {w[0]:.3f}')