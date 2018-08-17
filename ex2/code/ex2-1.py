import numpy as np 
import matplotlib.pyplot as plt 

path = '../ex2data1.txt'
X = []
Y = []
with open(path) as f:
    # X = [[float(line.split(',')[0]), float(line.split(',')[1])] 
    #     for line in f.read().splitlines()]
    for line in f.read().splitlines():
        line = line.split(',')
        X.append([float(line[0]), float(line[1])])
        Y.append([int(line[2])])
# print(X)
# print(Y)
X = np.array(X)
Y = np.array(Y)



# visualize data
pos = np.where(Y == 1)[0]
neg = np.where(Y == 0)[0]

plt.plot(X[pos][:, 0], X[pos][:, 1], 'k+', label='Admitted')
plt.plot(X[neg][:, 0], X[neg][:, 1], 'bo', label='Not admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
# plt.show()
# print(X_1)

m = X.shape[0]
ones = np.array([[1] for _ in range(m)])
X = np.column_stack((ones, X))
print(X.shape)
print(Y.shape)


# sigmoid function
# def sigm(z):
#     return 1/(1 + np.exp(-z))

# def scoreFunc(w):
#     return sigm(X.dot(w))

# def computeCost(w):
#     return np.sum(-Y.T.dot(np.log(scoreFunc(w))) - (1 - Y).T.dot(np.log(1 - scoreFunc(w))))

# def grad(w):
#     return X.T.dot(scoreFunc(w) - Y)/m 

# def GD(w_init, grad, lr):
#     cost = []
    
#     w_current = w_init
#     for it in range(ite):
#         w_new = w_current - lr * grad(w_current)
#         w_current = w_new
#         # cost.append(computeCost(w_new))
#         # print(w_new, cost[-1])
#         # print(w_current)
#     return w_new#, cost

# lr = .1
# w_init = np.array([[0], [0], [0]])
# ite = 1000000

# w_op = GD(w_init, grad, lr)
# print(w_op)

# x1 = np.arange(30, 100, .5)
# x2 = -(w_op[0][0] + w_op[1][0] * x1)/w_op[2][0]
# plt.plot(x1, x2, 'r-')
# plt.show()


def predict(w):
    y_pred = X.dot(w)
    return 1. / (1 + np.exp(-y_pred))

def grad(w):
    return 1. / m * (X.T.dot(predict(w) - Y))

def GD(w_init, lr, ite):
    w = w_init
    for it in range(ite):
        w = w - lr*grad(w)
    return w

w_init = np.array([[0], [0], [0]])
lr = .001
ite = 1000000

w_op = GD(w_init, lr, ite)
print(w_op)
    
x1 = np.arange(30, 100, .5)
x2 = -(w_op[0][0] + w_op[1][0] * x1)/w_op[2][0]
plt.plot(x1, x2, 'r-')
plt.show()

