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


# add ones
ones = np.array([[1] for _ in range(X.shape[0])])
X = np.column_stack((ones, X))

w = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
print(w)