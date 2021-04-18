import numpy as np 
from load_mnist import load_mnist 
from matplotlib import pyplot as plt 

x_train, y_train, x_test, y_test = load_mnist() 

def relu(x): 

    return np.maximum(0, x)

def sigmoid(x):

    return (np.exp(x)) / (1 + np.exp(x))

def relu_deriv(x):

    if x > 0:
        return x 

def sigmoid_deriv(x):

    return sigmoid(x) * (1-sigmoid(x))

def softmax(x):
    
    x_norm = x - np.max(x, axis = 1, keepdims = True)
    p_x = np.exp(x_norm) / np.sum(np.exp(x_norm), axis = 1, keepdims = True) 
    
    return p_x

def init_params(M, p, n_hidden):  

    W1 = np.random.normal(scale = 0.01, size = (n_hidden[0], p)) 
    W2 = np.random.normal(scale = 0.01, size = (n_hidden[1], n_hidden[0])) 
    W3 = np.random.normal(scale = 0.01, size = (n_hidden[2], n_hidden[1])) 
    W4 = np.random.normal(scale = 0.01, size = (M, n_hidden[2])) 

    b1 = np.zeros(shape = (n_hidden[0], 1)) 
    b2 = np.zeros(shape = (n_hidden[1], 1)) 
    b3 = np.zeros(shape = (n_hidden[2], 1)) 
    b4 = np.zeros(shape = (M, 1))

    return W1, b1, W2, b2, W3, b3, W4, b4

M = 10 # number of classes/ digits 
p = x_train.shape[1] # number of input pixels - 784 (flattened 28x28 image) 

n_train = x_train.shape[0] # number of training examples - 60000 
n_test = x_test.shape[0] # number of testing examples - 10000 

L = 4 # number of layers - input, h1, h2, h3, out 
n_hidden = np.array([392, 196, 98]) # hidden units per layer ---> L - 1 hidden layers 

n_batch = 100 # batch size 
epochs = 300 # number of epochs 

ytrue_test = np.argmax(y_test, axis = 1) 

w1,b1,w2,b2,w3,b3,w4,b4 = init_params(M, p , n_hidden)

z_1 = x_train @ w1.T + b1.T 
q_1 = relu(z_1) 
z_2 = q_1 @ w2.T + b2.T 
q_2 = relu(z_2) 
z_3 = q_2 @ w3.T + b3.T 
q_3 = relu(z_3) 
z = q_3 @ w4.T + b4.T 
softmax_z = softmax(z) 


