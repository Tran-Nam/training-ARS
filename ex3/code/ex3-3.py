import scipy.io 
import numpy as np 
import matplotlib.pyplot as plt 

pathdata = '../datasets/ex3data1.mat'
mat = scipy.io.loadmat(pathdata)
X_tr = mat['X'] # 5000 * 400
y_tr = mat['y'] # 5000 * 1
m = X_tr.shape[0]
d = X_tr.shape[1]
# print(m,d)

pathweight = '../datasets/ex3weights.mat'
mat = scipy.io.loadmat(pathweight)
# print(mat.keys())
theta1 = mat['Theta1']
theta2 = mat['Theta2']
# print(theta1.shape) 25 * 401
# print(theta2.shape) 10 * 26
# print(theta1)

def add_one(x):
    m = x.shape[0]
    ones = np.ones((m, 1))
    x = np.concatenate((ones, x), axis=1)
    return x
# a = np.array([[1,2], [3, 2]])
# a = add_one(a)
# print(a)

X_nor = add_one(X_tr)
# print(X_nor.shape)
hidden1 = X_nor.dot(theta1.T)
# print(hidden1.shape)
hidden1_nor = add_one(hidden1)
hidden2 = hidden1_nor.dot(theta2.T)
# print(hidden2.shape)
hidden2_nor = np.argmax(hidden2, axis=1).reshape((y_tr.shape[0],1))
hidden2_nor += 1
# print(hidden2_nor.shape)
# print(hidden2_nor[2995:3005])
# print(y_tr[2995:3005])

compare = (hidden2_nor == y_tr)
ac = np.sum(compare == True) / (y_tr.shape[0])
print('Do chinh xac: %.2f '%(ac * 100))