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
# print(X.shape)
# print(Y.shape)


# visualize data
pos = np.where(Y == 1)[0]
neg = np.where(Y == 0)[0]

plt.plot(X[pos][:, 0], X[pos][:, 1], 'k+', label='Admitted')
plt.plot(X[neg][:, 0], X[neg][:, 1], 'bo', label='Not admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()
# print(X_1)

# sigmoid function
def sigm(z):
    return 1./(1 + np.exp(-z))

