import numpy as np 
import matplotlib.pyplot as plt 

path = '../ex2data2.txt'
X = []
Y = []

with open(path) as f:
    for line in f.read().splitlines():
        line = line.split(',')
        X.append([float(line[0]), float(line[1])])
        Y.append([int(line[2])])

X = np.array(X)
Y = np.array(Y)



pos = np.where(Y == 1)[0]
neg = np.where(Y == 0)[0]

plt.plot(X[pos][:, 0], X[pos][:, 1], 'r^', label='y = 1')
plt.plot(X[neg][:, 0], X[neg][:, 1], 'bo', label='y = 0')
plt.legend()
plt.xlabel('Microchip test 1')
plt.ylabel('Microchip test 2')
# plt.show()


# print(X.shape)
# print(Y.shape)
m = X.shape[0]

x1 = X[:,0]
x2 = X[:,1]
def mapFeature(x1, x2):
    k = x1.shape[0]
    deg = 6
    X_train = np.ones((k, 1))
    for i in range(1, deg+1):
        for j in range(i+1):
            X_train = np.hstack((X_train, ((x1**(i-j))*(x2**j)).reshape(k,1)))
# print(X_train.shape)
    return X_train
X_train = mapFeature(x1, x2)
m = X_train.shape[0]
n = X_train.shape[1]
# print(m)
# print(n)

# print(X_train[:,0])
def predict(w):
    z = X_train.dot(w)
    return 1 / (1 + np.exp(-z))

def GD(w_init, lr, ite, alpha):
    w = w_init
    for it in range(ite):
        w[0] = w[0] - lr*X_train[:,0].dot(predict(w)- Y)
        for i in range(1, n):
            w[i] = w[i] - lr*(X_train[:, i].dot(predict(w) - Y)/m + alpha*w[i]/m)
        if it%1000 == 0:
            print(np.abs(predict(w) - Y).mean())
    return w


lr = .1
alpha = .5
ite = 10000
w_init = np.zeros((n,1)).reshape((n,1))
w_op = GD(w_init, lr, ite, alpha)
print(np.abs(predict(w_op) - Y).mean())
    
# print(X_train)
# print(X_train.shape)


u = np.arange(-1, 1.5, .1)
v = np.arange(-1, 1.5, .1)
xx, yy = np.meshgrid(u, v)
# z = np.zeros((len(u), len(v)))
# print(xx.shape)
xx1 = xx.ravel().reshape(xx.size, 1)
yy1 = yy.ravel().reshape(yy.size, 1)
x0 = mapFeature(xx1, yy1)
# print(x0.shape)
z0 = x0.dot(w_op)
z0 = z0.reshape(xx.shape)
print(z0.shape)
plt.contourf(xx, yy, z0, 200, cmap='jet', alpha=.1)
plt.show()
