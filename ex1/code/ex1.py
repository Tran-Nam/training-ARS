import numpy as np
import matplotlib.pyplot as plt

path = '../ex1data1.txt'
X = []
Y = []
with open(path) as f:
    # data = f.read().splitlines()
    for line in f:
        line = line.split(',')
        X.append([float(line[0])])
        Y.append([float(line[1])])
        # line[0][0].isdigit()
X = np.array(X)
Y = np.array(Y)       
# ex1data1
# print(data)
# print(type(X[0]))
# print(Y)

plt.plot(X, Y, 'r+')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

# X = [[_] for _ in X]
# Y = [[_] for _ in Y]

# print(Y)
# X = X.T

# add one
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
# print(X.shape)
m = X.shape[0]

# init par
theta = np.zeros((2,1))
# print(theta)
ite = 1500
alpha = .01
# print(theta.shape, X.shape, Y.shape)

def ComputeCost(theta):
    cost = np.sum((X.dot(theta) - Y)**2)
    return 1. * cost / (2 * m)

def ComputeCost2(theta1, theta2):
    cost = 0
    for t in range(X.shape[0]):
        cost += theta1 + X[t][1] * theta2 - Y[t]
    return cost / (2*m)
# a = ComputeCost(theta)
# print(a)

def daoham(theta):
    return (X.T).dot((X.dot(theta) - Y)) / m


def GD(theta):
    for it in range(ite):
        theta = theta - alpha * daoham(theta)
        # if it%100 == 0:
        #     print(it, theta)
    return theta

def GD(theta1_init, theta2_init, lr, ite):
    theta1 = theta1_init
    theta2 = theta2_init
    theta = np.array([[theat1], [theta2]])
    for it in range(ite):
        theta = np.array([[theta1], [theta2]])
        theta1 = theta1 - lr * np.sum(X.dot(theta) - Y) / m
        theta2 = theta2 - lr * np.sum(X.dot(theta) - Y) / m * 

theta_op = GD(theta)
print(theta_op)

plt.plot(X, X.dot(theta_op), 'g-')
plt.show()


theta1 = np.arange(-100, 100, .5)
theta2 = np.arange(-100, 100, .5)
theta1, theta2 = np.meshgrid(theta1, theta2)
z = ComputeCost2(theta1, theta2)
# theta
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.scatter3D(theta1, theta2, z, cmap="Greens")
# z = []
# for i in range(len(theta1)):
#     for j in range(len(theta2)):
#         t = np.array([[theta1[i]], [theta2[j]]])
#         z_new = ComputeCost(t)
#         z.append(z_new)
        # ax.scatter3D(theta1[i], theta2[j], z_new, c='r', cmap='Greens')

surf = ax.plot_surface(theta1, theta2, z, cmap='Greens')

# plt.show()