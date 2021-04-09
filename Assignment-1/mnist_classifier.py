import numpy as np 
from load_mnist import load_mnist 
from matplotlib import pyplot as plt 

x_train, y_train, x_test, y_test = load_mnist()

# shape x_train: 60000 x 784
# shape y_train: 60000 x 10 

M = 10 
n = x_train.shape[0] # number of training examples - 60000 
p = x_train.shape[1] # number of input pixels - 784 (flattened 28x28 image) 

def gradient_descent(xtrain, ytrain, lr, maxit): 

    #w_mj = np.random.normal(size = (M, p)) # weight matrix 
    w_mj = np.random.rand(M, p) # weight matrix                                                                                                                                                                                                                                    
    b_m = np.zeros(shape = (1, M)) # offset vector 
    z_im = np.zeros(shape = (n, M)) # model in (n x M) 
    dJdbm = np.zeros(shape = (1, M))
    dJdwmj = np.zeros(shape = (M, p))

    J = np.zeros(shape = (maxit, 1)) # cost vector 

    it = 0 

    while it != maxit:
    
        z_im = xtrain @ w_mj.T + b_m 

        y_im = ytrain 

        z_im_norm = z_im - np.max(z_im, axis = 1, keepdims = True) 

        p_im = np.exp(z_im_norm) / np.sum(np.exp(z_im_norm), axis = 1, keepdims = True) 

        dJdzim = (1/n) * (y_im * p_im - y_im) 

        dJdbm = np.sum(dJdzim, axis = 0) 
        dJdwmj = dJdzim.T @ xtrain 
        
        b_m = b_m - lr * dJdbm 
        w_mj = w_mj - lr * dJdwmj 
        
        L_i = np.sum(y_im * np.log(np.sum(np.exp(z_im_norm), axis = 1, keepdims = True)) - y_im * z_im_norm, axis = 1) 
        J[it] = (1/n) * np.sum(L_i) 
        it += 1 

    return J, it, w_mj, b_m , z_im 

J, it, wmj, bm, zim = gradient_descent(x_train, y_train, 0.1, 300) 

plt.figure(1)
plt.plot(J) 

figw, axw = plt.subplots(2, 5, figsize = (10,10)) 

axw[0,0].imshow(wmj[0, :].reshape(28,28), vmin = 0, vmax = 1, cmap = 'gray') 

axw[0,1].imshow(wmj[1, :].reshape(28,28), vmin = 0, vmax = 1, cmap = 'gray') 

axw[0,2].imshow(wmj[2, :].reshape(28,28), vmin = 0, vmax = 1, cmap = 'gray') 

axw[0,3].imshow(wmj[3, :].reshape(28,28), vmin = 0, vmax = 1, cmap = 'gray') 

axw[0,4].imshow(wmj[4, :].reshape(28,28), vmin = 0, vmax = 1, cmap = 'gray') 

axw[1,0].imshow(wmj[5, :].reshape(28,28), vmin = 0, vmax = 1, cmap = 'gray') 

axw[1,1].imshow(wmj[6, :].reshape(28,28), vmin = 0, vmax = 1, cmap = 'gray') 

axw[1,2].imshow(wmj[7, :].reshape(28,28), vmin = 0, vmax = 1, cmap = 'gray') 

axw[1,3].imshow(wmj[8, :].reshape(28,28), vmin = 0, vmax = 1, cmap = 'gray') 

axw[1,4].imshow(wmj[9, :].reshape(28,28), vmin = 0, vmax = 1, cmap = 'gray') 

plt.show() 
