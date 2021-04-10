import numpy as np 
from load_mnist import load_mnist 
from matplotlib import pyplot as plt 

x_train, y_train, x_test, y_test = load_mnist()

M = 10 
n = x_train.shape[0] # number of training examples - 60000 
p = x_train.shape[1] # number of input pixels - 784 (flattened 28x28 image) 

n_batch = 30 # batch size 

ind = np.arange(n) 
np.random.shuffle(ind) # shuffle indices 

xt_shuff = x_train[ind, :]
yt_shuff = y_train[ind, :] 

ytrue = np.argmax(y_train, axis = 1) 
ytrue_shuff = np.argmax(yt_shuff, axis = 1) 

mini_batch = np.random.randint(0, n, size = n_batch) # batch indices 

