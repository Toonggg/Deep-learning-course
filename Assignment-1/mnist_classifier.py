import numpy as np 
from numba import jit 
from load_mnist import load_mnist 


xt, yt, xtt, ytt = load_mnist()

M = 10 
n = xt.shape[0] # 60000 
p = xt.shape[1] # 784 

print(xt.shape)