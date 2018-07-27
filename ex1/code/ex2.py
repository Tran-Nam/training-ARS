import numpy as np
import matplotlib.pyplot as plt
# import os
# print(os.listdir('..'))

path = '../ex1data2.txt'
X = []
Y = []
with open(path) as f:
    # data = f.read().splitlines()
    for line in f:
        line = line.split(',')
        X.append([float(line[0]), float(line[1])])
        Y.append(float(line[2]))

m = len(Y)
X = np.array(X).reshape((m, 2))
Y = np.array(Y).reshape((m,1))
# print(X[0])
# print(Y)

X_max = np.array([[np.amax(X[:, column_id])
        for column_id in range(X.shape[1])]
        for _ in range(X.shape[0])])

X_min = np.array([[np.amin(X[:, column_id])
        for column_id in range(X.shape[1])]
        for _ in range(X.shape[0])])

X_mean = np.array([[np.mean(X[:, column_id])
        for column_id in range(X.shape[1])]
        for _ in range(X.shape[0])])
# print( X_max)
# print(X_min)
# print(X_mean)
X = (X - X_mean) / (X_max - X_min)
ones = np.array([[1] for _ in range(X.shape[0])])
X = np.column_stack((ones, X))
# print(X)

lr = .1
ite = 50
w_init = np.zeros((3,1))

def ComputeCost(w):
    return np.linalg.norm(X.dot(w) - Y)**2 / (2*m)

def grad(w):
    return (X.T).dot(X.dot(w) - Y) / m

def GD(w):
    cost = []
    for it in range(ite):
        w = w - lr*grad(w)
        # plt.plot(it, ComputeCost(w))
        cost.append(ComputeCost(w))
    return w, cost

w_op, cost = GD(w_init)
print(w_op)
it = range(50)
plt.plot(it, cost)
plt.show()
        



