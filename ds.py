import numpy as np
import matplotlib.pyplot as plt

data1 = np.arange(0, 5)
data2 = np.arange(0, 5)

c1 = np.random.uniform(-1, 1, (500, 5))
c2 = np.random.uniform(-1, 1, (500, 5))

data = np.zeros((1000, 5))
data[:500] = data1 + c1
data[500:] = data2 + c2

X = (data - data.min(axis = 0)) / (data.max(axis = 0) - data.min(axis = 0))

y = np.ones((1000, 1))
y[:500] = -1

idxs = np.arange(X.shape[0])
np.random.shuffle(idxs)
X = X[idxs]
y = y[idxs]