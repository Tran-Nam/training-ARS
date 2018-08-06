import scipy.io 
import numpy as np 
import matplotlib.pyplot as plt 

path = '../datasets/ex3weights.mat'
mat = scipy.io.loadmat(path)
# print(mat.keys())
theta1 = mat['Theta1']
theta2 = mat['Theta2']
print(theta1.shape)
print(theta2.shape)