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

plt.plot(X[pos][:, 0], X[pos][:, 1], 'k+', label='y = 1')
plt.plot(X[neg][:, 0], X[neg][:, 1], 'bo', label='y = 0')
plt.legend()
plt.xlabel('Microchip test 1')
plt.ylabel('Microchip test 2')
plt.show()