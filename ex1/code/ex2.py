import numpy as np
import matplotlib.pyplot as plt
# import os
# print(os.listdir('..'))

# collect data
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
Y = np.array(Y).reshape((m, 1))
# print(X)

X_max = np.array([[np.amax(X[:, column_id])
            for column_id in range(X.shape[1])]
            for _ in range(X.shape[0])])

X_min = np.array([[np.amin(X[:, column_id])
            for column_id in range(X.shape[1])]
            for _ in range(X.shape[0])])

X_mean = np.array([[np.mean(X[:, column_id])
            for column_id in range(X.shape[1])]
            for _ in range(X.shape[0])])

# feature normalize
def normalize(x):
    

    x = (x - X_mean) / (X_max - X_min)

    # add ones
    ones = np.array([[1] for _ in range(x.shape[0])])
    x = np.column_stack((ones, x))
    return x
    # print(X)
X_nor = normalize(X)

# itilize parameters
# lrs = [.01, .03, .1, 3e-4]
lr = .1
ite = 100
w_init = np.zeros((3,1))

def ComputeCost(w):
    return np.linalg.norm(X_nor.dot(w) - Y)**2 / (2*m)

def grad(w):
    return (X_nor.T).dot(X_nor.dot(w) - Y) / m

def GD(w):
    cost = []
    for it in range(ite):
        w = w - lr*grad(w)
        # plt.plot(it, ComputeCost(w))
        cost.append(ComputeCost(w))
    return w, cost



# x = normalize(X)
# # for lr in lrs:
w_op, cost = GD(w_init)
print(w_op)
it = range(100)
plt.plot(it, cost) #, label='lr = %f' %(lr)
# plt.show()

# x_new = np.array([1650, 3])
# x_nor = normalize(x_new)
# y_pred = x_nor.dot(w_op)
# print(y_pred)





