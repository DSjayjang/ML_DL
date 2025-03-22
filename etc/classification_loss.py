import numpy as np
import matplotlib.pyplot as plt

from utils.actv_func import sigmoid

"""
샘플이 하나인 경우, 시그모이드 함수에 대한 분류 손실
"""

def loss_1(z):
    return -np.log(sigmoid(z))

def loss_0(z):
    return -np.log(1-sigmoid(z))

z = np.arange(-10, 10, 0.1)
sigma_z = sigmoid(z)

c1 = [loss_1(x) for x in z]
plt.plot(sigma_z, c1, label = 'L(w, b) if y = 1')

c0 = [loss_0(x) for x in z]
plt.plot(sigma_z, c0, linestyle = '--', label = 'L(w, b) if y =0')

plt.xlim([0, 1])
plt.ylim(0.0, 5.1)

plt.xlabel('sigma(z)')
plt.ylabel('L(w, b)')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()