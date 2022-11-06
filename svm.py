import numpy as np
import matplotlib.pyplot as plt

from ds import *
#----

def HingeLoss(y, yh):
    return np.sum(np.where(1 - y*yh > 0, 1 - y*yh, 0)) / y.shape[0]

#----

w = np.random.uniform(-1, 1, (X.shape[1], 1))
b = np.random.uniform(-1, 1, (1,1))

L = []
epochs = int(1e2)
lr = 1e-1
#---
for iter in range(epochs):
    if iter % (epochs / 10) == 0:
        print(f"iter {iter} od {epochs}")

    yh = X @ w + b
    L.append(HingeLoss(y, yh))

    dyh = np.where(1 - y*yh > 0, -y / y.shape[0], 0)
    dw = np.transpose(X) @ dyh
    db = np.sum(dyh)

    w -= lr*dw
    b -= lr*db

plt.plot(L)
plt.show()