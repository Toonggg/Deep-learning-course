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

M = 10 # number of classes/ digits 
p = x_train.shape[1] # number of input pixels - 784 (flattened 28x28 image) 

n_train = x_train.shape[0] # number of training examples - 60000 
n_test = x_test.shape[0] # number of testing examples - 10000 

L = 4 # number of layers - input, h1, h2, h3, out 
n_hidden = np.array([392, 196, 98]) # hidden units per layer 

n_batch = 100 # batch size 
epochs = 300 # number of epochs 

ytrue_test = np.argmax(y_test, axis = 1) 

W = np.zeros(shape = (L - 1, ))

def init_params():

    W1 = np.random.normal(scale = 0.01, size = (n_hidden[0], p)) 
    W2 = np.random.normal(scale = 0.01, size = (n_hidden[1], n_hidden[0])) 
    W3 = np.random.normal(scale = 0.01, size = (n_hidden[2], n_hidden[1])) 
    W4 = np.random.normal(scale = 0.01, size = (M, n_hidden[2])) 



    w_mj = np.random.normal(scale = 0.01, size = (M, p)) # weight matrix                                                                                                                                                                                            
    b_m = np.zeros(shape = (1, M)) 
    z_im = np.zeros(shape = (n_train, M)) 

    Q = np.zeros(shape = (n_batch, ))

    dJdbm = np.zeros(shape = (1, M)) 
    dJdwmj = np.zeros(shape = (M, p)) 
    

    return None

def forward_prop():

    for l in np.arange(0, L, 1):
        h = 1

    return None

def compute_cost():



    return None 

def backward_prop():


    return None

def update_parameters():


    return None 

def full_nn(xtrain, ytrain, xtest, ytest, ep, nb, lr_init, tau, n_train, n_test, k): 

    e_p = 0 # epoch counter 
    lr0 = lr_init # initial learning rate 
    lrt = 0.01 * lr0 # final learning rate 
    t_tau = tau 

    tot_it = 0 

    while e_p != ep: 

        ### Shuffling indices 
        ind = np.arange(n_train) 
        np.random.shuffle(ind) # shuffle indices 

        ### Shuffling training data and labels
        xt = xtrain[ind, :] 
        yt = ytrain[ind, :] 

        ytrue_shuff = np.argmax(yt, axis = 1) 

        it = 0 
    
        #for j in range(n_train//nb): 
        while it != n_train//nb:

            lr = (1 - (it/t_tau)) * lr0 + (it/t_tau) * lrt

            mini_batch = np.random.randint(0, n_train, size = nb) # batch indices 

            z_im[mini_batch, :] = xt[mini_batch, :] @ w_mj.T + b_m 

            y_im = yt[mini_batch, :] 

            z_im_norm = z_im[mini_batch, :] - np.max(z_im[mini_batch, :], axis = 1, keepdims = True) 

            p_im = np.exp(z_im_norm) / np.sum(np.exp(z_im_norm), axis = 1, keepdims = True) 

            dJdzim = (1/nb) * (y_im * p_im - y_im) 
            dJdbm = np.sum(dJdzim, axis = 0) 
            dJdwmj = dJdzim.T @ xt[mini_batch, :] 

            b_m = b_m - lr * dJdbm 
            w_mj = w_mj - lr * dJdwmj 

            tot_it += 1
            it += 1 

            print("Epoch: (%s/%s), iteration: %s" % (e_p + 1, ep, it)) 

        e_p += 1 

    return None