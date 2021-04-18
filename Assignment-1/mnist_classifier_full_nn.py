import numpy as np 
from load_mnist import load_mnist 
from matplotlib import pyplot as plt 

x_train, y_train, x_test, y_test = load_mnist() 

def relu(x): 

    return np.maximum(0, x)

def sigmoid(x): 

    return (np.exp(x)) / (1 + np.exp(x))

def relu_deriv(x):

    x[x <= 0] = 0
    x[x > 0] = 1

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

def calc_cost(nb, mini_batch, y_L, z_L): 

    loss = np.sum(y_L[mini_batch, :] * np.log(np.sum(np.exp(z_L), axis = 1, keepdims = True)) - y_L[mini_batch, :] * z_L, axis = 1, keepdims = True)
    cost = (1/nb) * np.sum(loss, axis = 0, keepdims = True)

    dz_L = - y_L[mini_batch, :] + softmax(z_L)

    return cost, dz_L

def forward(xt, mb, w1, b1, w2, b2, w3, b3, w4, b4):

    z_1 = xt[mb, :] @ w1.T + b1.T 
    q_1 = relu(z_1) 
    z_2 = q_1 @ w2.T + b2.T 
    q_2 = relu(z_2) 
    z_3 = q_2 @ w3.T + b3.T 
    q_3 = relu(z_3) 
    z = q_3 @ w4.T + b4.T 
    softmax_z = softmax(z) 

    return z_1, q_1, z_2, q_2, z_3, q_3, z, softmax_z

def backward(q1,q2,q3, z1, z2, z3, dzl, w2, w3, w4, xt, mb):

    dz_3 = np.multiply(q3, relu_deriv(z3))
    dz_2 = np.multiply(q2, relu_deriv(z2))
    dz_1 = np.multiply(q1, relu_deriv(z1))

    dq_3 = dzl @ w4 
    dq_2 = dz_3 @ w3
    dq_1 = dz_2 @ w2

    dW_4 = (1/n_batch) * dzl.T @ q3
    dW_3 = (1/n_batch) * dz_3.T @ q2
    dW_2 = (1/n_batch) * dz_2.T @ q1  
    dW_1 = (1/n_batch) * dz_1.T @ xt[mb, :] 

    db_4 = (1/n_batch) * np.sum(dzl, axis = 0, keepdims = True)
    db_3 = (1/n_batch) * np.sum(dz_3, axis = 0, keepdims = True)
    db_2 = (1/n_batch) * np.sum(dz_2, axis = 0, keepdims = True)
    db_1 = (1/n_batch) * np.sum(dz_1, axis = 0, keepdims = True)

    return dW_1, db_1.T, dW_2, db_2.T, dW_3, db_3.T, dW_4, db_4.T

M = 10 # number of classes/ digits 
p = x_train.shape[1] # number of input pixels - 784 (flattened 28x28 image) 

n_train = x_train.shape[0] # number of training examples - 60000 
n_test = x_test.shape[0] # number of testing examples - 10000 

n_batch = 100 # batch size 
epochs = 10 # number of epochs 

def neural_network(epochs, nb, M, p, xtrain, ytrain, xtest, ytest): 

    ytrue_test = np.argmax(ytest, axis = 1) # labels for testing data

    e_p = 0 # epoch counter 
    lr0 = 0.01 # initial learning rate 
    lrt = 0.01 * lr0 # final learning rate 
    t_tau = 50 # iterations until learning rate is set to constant lrt value 

    tot_it = 0 # total iteration counter 

    n_hidden = np.array([392, 196, 98]) # hidden units per layer ---> L - 1 hidden layers 
    w1,b1,w2,b2,w3,b3,w4,b4 = init_params(M, p , n_hidden) 

    while e_p != epochs: 

        ### Shuffling indices 
        ind = np.arange(n_train) 
        np.random.shuffle(ind) 

        ### Shuffling training data and labels
        xt = xtrain[ind, :] 
        yt = ytrain[ind, :] 

        ytrue_train = np.argmax(yt, axis = 1) # labels for training data 

        it = 0 # iteration counter for an epoch 
    
        #for j in range(n_train//nb): 
        while it != n_train//nb: 

            lr = (1 - (it/t_tau)) * lr0 + (it/t_tau) * lrt 
            mini_batch = np.random.randint(0, n_train, size = nb) # batch indices 

            z1, q1, z2, q2, z3, q3, z, sz = forward(xt, mini_batch, w1,b1,w2,b2,w3,b3,w4,b4)
            cost, dzL = calc_cost(nb, mini_batch, yt, z) 
            dw1, db1, dw2, db2, dw3, db3, dw4, db4 = backward(q1, q2, q3, z1, z2, z3, dzL, w2, w3, w4, xt, mini_batch) 

            w1 = w1 - lr * dw1
            w2 = w2 - lr * dw2 
            w3 = w3 - lr * dw3
            w4 = w4 - lr * dw4 

            b1 = b1 - lr * db1
            b2 = b2 - lr * db2
            b3 = b3 - lr * db3
            b4 = b4 - lr * db4

            tot_it += 1 
            it += 1 

            print("Epoch: (%s/%s), iteration: %s" % (e_p + 1, epochs, it)) 

        e_p += 1 

    return w4 

w444 = neural_network(epochs, n_batch, M, p, x_train, y_train, x_test, y_test) 
