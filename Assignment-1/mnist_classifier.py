import numpy as np 
from numba import jit 
from load_mnist import load_mnist 
from matplotlib import pyplot as plt 

x_train, y_train, x_test, y_test = load_mnist()

# shape x_train: 60000 x 784
# shape y_train: 60000 x 10 

M = 10 
n = x_train.shape[0] # number of training examples - 60000 
p = x_train.shape[1] # number of input pixels - 784 (flattened 28x28 image) 

def gradient_descent(xtrain, ytrain, lr, maxit): 

    w_mj = np.random.normal(size = (M, p)) # weight matrix                                                                                                                                                                                                                      
    b_m = np.zeros(shape = (1, M)) # offset vector 
    z_im = np.zeros(shape = (n, M)) # model in (n x M) 

    J = np.zeros(shape = (1, maxit)) # cost vector 

    it = 0 

    while it != maxit:
    
        z_im = xtrain@w_mj.T + b_m 

        dJdzim = 

        dJdbm = (1/n) * np.sum(dJdzim, axis = 0)
        dJdwmj = (1/n) * (dJzim.T@xtrain) 
        
        b_m = b_m - lr * dJdbm
        w_mj = w_mj - lr * dJdwmj
        
        J[it] = it 

        it += 1

    print(z_im.shape)
    print(dJdbm.shape)
    print(dJdwmj.shape)

    return J, it, w_mj, b_m , z_im

J, it, wmj, bm, zim = gradient_descent(x_train, y_train, 0.01, 100) 