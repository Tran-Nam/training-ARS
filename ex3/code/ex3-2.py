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