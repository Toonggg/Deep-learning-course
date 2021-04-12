import numpy as np 
from load_mnist import load_mnist 
from matplotlib import pyplot as plt 

x_train, y_train, x_test, y_test = load_mnist()

# shape x_train: 60000 x 784
# shape y_train: 60000 x 10 

ytrue = np.argmax(y_train, axis = 1) 

M = 10 
n = x_train.shape[0] # number of training examples - 60000 
p = x_train.shape[1] # number of input pixels - 784 (flattened 28x28 image) 

def predict_test(): 

    return None

def gradient_descent(xtrain, ytrain, lr, maxit): 

    #w_mj = np.random.normal(size = (M, p)) # weight matrix 
    w_mj = np.random.rand(M, p) # weight matrix                                                                                                                                                                                                                                    
    b_m = np.zeros(shape = (1, M)) # offset vector 
    z_im = np.zeros(shape = (n, M)) # model in (n x M) 
    dJdbm = np.zeros(shape = (1, M))
    dJdwmj = np.zeros(shape = (M, p)) 

    J = np.zeros(shape = (maxit, 1)) # cost vector 
    c_acc = np.zeros(shape = (maxit, 1)) # classifcation accuracy 

    it = 0 

    while it != maxit: 

        z_im = xtrain @ w_mj.T + b_m 

        y_im = ytrain 

        z_im_norm = z_im - np.max(z_im, axis = 1, keepdims = True) 

        p_im = np.exp(z_im_norm) / np.sum(np.exp(z_im_norm), axis = 1, keepdims = True) 

        dJdzim = (y_im * p_im - y_im) 

        dJdbm = np.sum(dJdzim, axis = 0) 
        dJdwmj = dJdzim.T @ xtrain 
                
        b_m = b_m - (1/n) * lr * dJdbm 
        w_mj = w_mj - (1/n) * lr * dJdwmj 
                
        L_i = np.sum(y_im * np.log(np.sum(np.exp(z_im_norm), axis = 1, keepdims = True)) - y_im * z_im_norm, axis = 1) 
        J[it] = (1/n) * np.sum(L_i) 

        ypred = np.argmax(p_im, axis = 1) 
        c_acc[it] = np.array([1 for i in range(0,n) if ypred[i] == ytrue[i]]).sum() 
                
        it += 1 

    return J, c_acc * (1/n), it, w_mj, b_m , z_im, p_im 

J, c_acc, it, wmj, bm, zim, pim = gradient_descent(x_train, y_train, 0.02, 500) 

### Cost, accuracy, and weights for training data 
plt.figure(1) 
plt.plot(J) 

plt.figure(2)
plt.plot(100 * c_acc) 

figw, axw = plt.subplots(2, 5, figsize = (10,10)) 
  
axw[0,0].imshow(wmj[0, :].reshape(28,28), cmap = 'gray') 

axw[0,1].imshow(wmj[1, :].reshape(28,28), cmap = 'gray') 

axw[0,2].imshow(wmj[2, :].reshape(28,28), cmap = 'gray') 

axw[0,3].imshow(wmj[3, :].reshape(28,28), cmap = 'gray') 

axw[0,4].imshow(wmj[4, :].reshape(28,28), cmap = 'gray') 

axw[1,0].imshow(wmj[5, :].reshape(28,28), cmap = 'gray') 

axw[1,1].imshow(wmj[6, :].reshape(28,28), cmap = 'gray') 

axw[1,2].imshow(wmj[7, :].reshape(28,28), cmap = 'gray') 

axw[1,3].imshow(wmj[8, :].reshape(28,28), cmap = 'gray') 

axw[1,4].imshow(wmj[9, :].reshape(28,28), cmap = 'gray') 

### Cost, accuracy, predicted results for test data 

plt.show()