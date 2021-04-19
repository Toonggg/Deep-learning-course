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

def calc_cost(nb, mini_batch, batchBool, y_L, z_L, sz): 
    if batchBool == True:
        z_L_norm = z_L - np.max(z_L, axis = 1, keepdims = True) 
        loss = np.sum(y_L[mini_batch, :] * np.log(np.sum(np.exp(z_L_norm), axis = 1, keepdims = True)) - y_L[mini_batch, :] * z_L_norm, axis = 1, keepdims = True)
        cost = (1/nb) * np.sum(loss, axis = 0, keepdims = True) 

        dz_L = - y_L[mini_batch, :] + sz 
        return cost, dz_L
    else:
        z_L_norm = z_L - np.max(z_L, axis = 1, keepdims = True) 
        loss = np.sum(y_L * np.log(np.sum(np.exp(z_L_norm), axis = 1, keepdims = True)) - y_L * z_L_norm, axis = 1, keepdims = True)
        cost = (1/nb) * np.sum(loss, axis = 0, keepdims = True) 

        return cost
    
def forward(xt, mb, batchBool, w1, b1, w2, b2, w3, b3, w4, b4):

    if batchBool == True: 
        z_1 = xt[mb, :] @ w1.T + b1.T 
        q_1 = sigmoid(z_1) 
        z_2 = q_1 @ w2.T + b2.T 
        q_2 = sigmoid(z_2) 
        z_3 = q_2 @ w3.T + b3.T 
        q_3 = sigmoid(z_3) 
        z = q_3 @ w4.T + b4.T 
        softmax_z = softmax(z) 
    else:
        z_1 = xt @ w1.T + b1.T 
        q_1 = sigmoid(z_1) 
        z_2 = q_1 @ w2.T + b2.T 
        q_2 = sigmoid(z_2) 
        z_3 = q_2 @ w3.T + b3.T 
        q_3 = sigmoid(z_3) 
        z = q_3 @ w4.T + b4.T 
        softmax_z = softmax(z) 

    return z_1, q_1, z_2, q_2, z_3, q_3, z, softmax_z 

def backward(q1,q2,q3, z1, z2, z3, dzl, w2, w3, w4, xt, mb):
    
    dq_3 = dzl @ w4 

    dz_3 = np.multiply(dq_3, sigmoid_deriv(z3))
    dq_2 = dz_3 @ w3

    dz_2 = np.multiply(dq_2, sigmoid_deriv(z2)) 
    dq_1 = dz_2 @ w2
    
    dz_1 = np.multiply(dq_1, sigmoid_deriv(z1))   

    dW_4 = (1/n_batch) * dzl.T @ q3 
    dW_3 = (1/n_batch) * dz_3.T @ q2
    dW_2 = (1/n_batch) * dz_2.T @ q1  
    dW_1 = (1/n_batch) * dz_1.T @ xt[mb, :] 

    db_4 = (1/n_batch) * np.sum(dzl, axis = 0, keepdims = True)
    db_3 = (1/n_batch) * np.sum(dz_3, axis = 0, keepdims = True)
    db_2 = (1/n_batch) * np.sum(dz_2, axis = 0, keepdims = True)
    db_1 = (1/n_batch) * np.sum(dz_1, axis = 0, keepdims = True)

    return dW_1, db_1.T, dW_2, db_2.T, dW_3, db_3.T, dW_4, db_4.T

def init_params_2(M, p, n_hidden): 

    W1 = np.random.normal(scale = 0.01, size = (n_hidden[0], p)) 
    W2 = np.random.normal(scale = 0.01, size = (M, n_hidden[0])) 

    b1 = np.zeros(shape = (n_hidden[0], 1)) 
    b2 = np.zeros(shape = (M, 1)) 

    return W1, b1, W2, b2 

def backward_2(q1, z1, w1, w2, xt, mb, dzl):
    dq_1 = dzl @ w2
    dz_1 = np.multiply(dq_1, relu_deriv(z1))

    dW_2 = (1/n_batch) * dzl.T @ q1

    db_2 = (1/n_batch) * np.sum(dzl, axis = 0, keepdims = True)

    dW_1 = (1/n_batch) * dz_1.T @ xt[mb, :] 

    db_1 = (1/n_batch) * np.sum(dz_1, axis = 0, keepdims = True)

    return dW_1, db_1.T, dW_2, db_2.T

def forward_2(xt, mb, batchBool, w1, b1, w2, b2):

    if batchBool == True: 
        z_1 = xt[mb, :] @ w1.T + b1.T 
        q_1 = relu(z_1) 
        z = q_1 @ w2.T + b2.T 
        softmax_z = softmax(z) 
    else:
        z_1 = xt @ w1.T + b1.T 
        q_1 = relu(z_1) 
        z = q_1 @ w2.T + b2.T 
        softmax_z = softmax(z) 

    return z_1, q_1, z, softmax_z 

def neural_network(epochs, nb, M, p, k, xtrain, ytrain, xtest, ytest, ntrain, ntest): 

    ytrue_test = np.argmax(ytest, axis = 1) # labels for testing data

    e_p = 0 # epoch counter 
    lr0 = 1 # initial learning rate 
    lrt = 0.01 * lr0 # final learning rate 
    t_tau = 30 # iterations until learning rate is set to constant lrt value 

    tot_it = 0 # total iteration counter 
    it_k = 0 # k-th iteration counter 

    n_hidden_4 = np.array([100, 100, 100]) # hidden units per layer ---> L - 1 hidden layers 
    w1,b1,w2,b2,w3,b3,w4,b4 = init_params(M, p , n_hidden_4) 

    #n_hidden = np.array([100]) # hidden units per layer ---> L - 1 hidden layers 
    #w1,b1,w2,b2 = init_params_2(M, p , n_hidden) 

    acctrain = np.zeros(shape = (((n_train//nb)//k) * epochs,1)) 
    costtrain = np.zeros(shape = (((n_train//nb)//k) * epochs,1)) 

    acctest = np.zeros(shape = (((n_train//nb)//k) * epochs,1)) 
    costtest = np.zeros(shape = (((n_train//nb)//k) * epochs,1)) 

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

            #lr = (1 - (it/t_tau)) * lr0 + (it/t_tau) * lrt 
            lr = 0.5

            mini_batch = np.random.randint(0, n_train, size = nb) # batch indices 
 
            #z1, q1, z, softz = forward_2(xt, mini_batch, True, w1,b1,w2,b2) 
            #train_cost, dzL = calc_cost(nb, mini_batch, True, yt, z, softz) 
            #dw1, db1, dw2, db2 = backward_2(q1, z1, w1, w2, xt, mini_batch, dzL) 
            #_, _, z_test, softz_test = forward_2(xtest, mini_batch, False, w1,b1,w2,b2) 

            z1, q1, z2, q2, z3, q3, z, softz = forward(xt, mini_batch, True, w1,b1,w2,b2,w3,b3,w4,b4) 
            train_cost, dzL = calc_cost(nb, mini_batch, True, yt, z, softz) 
            dw1, db1, dw2, db2, dw3, db3, dw4, db4 = backward(q1, q2, q3, z1, z2, z3, dzL, w2, w3, w4, xt, mini_batch) 
            _, _, _, _, _, _, z_test, softz_test = forward(xtest, mini_batch, False, w1, b1, w2, b2, w3, b3, w4, b4)

            test_cost = calc_cost(n_test, mini_batch, False, ytest, z_test, softz_test) 

            w1 = w1 - lr * dw1 
            w2 = w2 - lr * dw2 
            b1 = b1 - lr * db1 
            b2 = b2 - lr * db2 

            w3 = w3 - lr * dw3 
            w4 = w4 - lr * dw4 
            b3 = b3 - lr * db3 
            b4 = b4 - lr * db4 

            if tot_it % k == 0: 
                y_predtrain = np.argmax(softz, axis = 1) 
                acctrain[it_k] = 100 * ((1/nb) * np.sum(y_predtrain == ytrue_train[mini_batch])) 
                costtrain[it_k] = train_cost 

                y_predtest = np.argmax(softz_test, axis = 1) 
                acctest[it_k] = 100 * ((1/n_test) * np.sum(y_predtest == ytrue_test)) 
                costtest[it_k] = test_cost 

                it_k += 1 

            tot_it += 1 
            it += 1 

            print("Epoch: (%s/%s), iteration: %s" % (e_p + 1, epochs, it)) 

        e_p += 1 

    return w1,w2,w3,w4,b1,b2,b3,b4, costtrain, acctrain, costtest, acctest
    #return w1,w2, softz, costtrain, acctrain, costtest, acctest 

M = 10 # number of classes/ digits 
p = x_train.shape[1] # number of input pixels - 784 (flattened 28x28 image) 

n_train = x_train.shape[0] # number of training examples - 60000 
n_test = x_test.shape[0] # number of testing examples - 10000 

n_batch = 1500 # batch size 
epochs = 30 # number of epochs 

k_acc = 5

#w1,w2,sz, costtrain, acctrain, costtest, acctest = neural_network(epochs, n_batch, M, p, k_acc, x_train, y_train, x_test, y_test, n_train, n_test) 
w1,w2,w3,w4,b1,b2,b3,b4, costtrain, acctrain, costtest, acctest = neural_network(epochs, n_batch, M, p, k_acc, x_train, y_train, x_test, y_test, n_train, n_test)

plt.figure(1) 
plt_Jtrain_it, = plt.plot(costtrain, 'r') 
plt_Jtest_it, = plt.plot(costtest, 'b') 
plt.legend([plt_Jtrain_it, plt_Jtest_it], ['Train cost', 'Test cost']) 
plt.annotate("Final train cost: %s" % (float(costtrain[-1])), xycoords = 'figure fraction', xy = (0.4,0.5))
plt.annotate("Final test cost: %s" % (float(costtest[-1])), xycoords = 'figure fraction', xy = (0.4,0.55))
plt.xlabel('Total number of iterations') 
plt.ylabel('Cost J') 
plt.title('Cost versus iterations') 
print("Final train cost: %s" % float(costtrain[-1])) 
print("Final test cost: %s" % float(costtest[-1])) 

plt.figure(2)
plt_acc_train_it, = plt.plot(acctrain, 'r') 
plt_acc_test_it, = plt.plot(acctest, 'b') 
plt.legend([plt_acc_train_it, plt_acc_test_it], ['Train accuracy', 'Test accuracy']) 
plt.annotate("Final train accuracy: %s%%" % (float(acctrain[-1])), xycoords = 'figure fraction', xy = (0.4,0.5))
plt.annotate("Final test accuracy: %s%%" % (float(acctest[-1])), xycoords = 'figure fraction', xy = (0.4,0.55))
plt.xlabel('Total number of iterations') 
plt.ylabel('Accuracy in %') 
plt.title('Accuracy versus iterations') 
print("Final train accuracy: %s%%" % float(acctrain[-1])) 
print("Final test accuracy: %s%%" % float(acctest[-1])) 

#plt.figure(2)
#plt.plot(np.array(db1ot).reshape(-1))

#plt.figure(3)
#plt.plot(np.array(db2ot).reshape(-1))

#plt.figure(4)
#plt.plot(np.array(db3ot).reshape(-1))

#plt.figure(5)
#plt.plot(np.array(db4ot).reshape(-1))

#plt.figure(6)
#plt.plot(np.array(dw1ot).reshape(-1))

#plt.figure(7)
#plt.plot(np.array(dw2ot).reshape(-1))

#plt.figure(8)
#plt.plot(np.array(dw3ot).reshape(-1))

#plt.figure(9)
#plt.plot(np.array(dw4ot).reshape(-1))

plt.show()