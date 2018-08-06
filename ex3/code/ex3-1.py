import scipy.io 
import numpy as np 
import matplotlib.pyplot as plt 

path = '../datasets/ex3data1.mat'
mat = scipy.io.loadmat(path)
X_tr = mat['X'] # 5000 * 400
y_tr = mat['y'] # 5000 * 1
m = X_tr.shape[0]
d = X_tr.shape[1]
print(m,d)
# print(y_tr.shape)

def predict(w):
    z = X_tr.dot(w)
    return 1./(1 + np.exp(-z))

def grad(w, X_tr, y_tr_new):
    return X_tr.T.dot(predict(w) - y_tr_new) / m

def GD(w_init, lr, ite, grad):
    w = w_init

    for it in range(ite):
        w -= lr * grad(w, X_tr, y_tr_new)
    if it%100 == 0:
        print('%d: ' %(it))
        print(w)
    return w 

w_init = np.zeros((d,1)).reshape((d,1))
lr = .1
ite = 50000
# w_op = GD(w_init, lr, ite)
# print(w_op.shape)

pos0 = np.where(y_tr == 8)[0]
posnot0 = np.where(y_tr != 8)[0]
y_tr_new = y_tr
y_tr_new[pos0] = 1
y_tr_new[posnot0] = 0
# print(y_tr_new[498:502])
# print(posnot0)
w_op = GD(w_init, lr, ite, grad)
# # print(w_op.shape)
result = predict(w_op)
truepos = np.where(result >= .5)[0]
print(truepos)
# print(predict(w_op)[498:502])

# result = dict()
# for i in range(1, 11):
#     y_tr_new = y_tr
#     pos = np.where(y_tr_new == i)[0]
#     posnot = np.where(y_tr_new != i)[0]
#     y_tr_new[pos] = 1
#     y_tr_new[posnot] = 0
#     w_op = GD(w_init, lr, ite, grad)
#     pred = predict(w_op)
#     truepos = np.where(pred >= .5)[0]
#     print(truepos)
#     result['%d'%i] = truepos
# # print(result['2'])


