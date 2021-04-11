import numpy as np 
from load_mnist import load_mnist 
from matplotlib import pyplot as plt 

x_train, y_train, x_test, y_test = load_mnist()

M = 10 
n = x_train.shape[0] # number of training examples - 60000 
p = x_train.shape[1] # number of input pixels - 784 (flattened 28x28 image) 

n_batch = 30 # batch size 
epochs = 200 # epochs 

def gradient_descent(xtrain, ytrain): 

    #w_mj = np.random.normal(size = (M, p)) # weight matrix 
    w_mj = np.random.rand(M, p) # weight matrix                                                                                                                                                                                                                                    
    b_m = np.zeros(shape = (1, M)) # offset vector 
    z_im = np.zeros(shape = (n_batch, M)) # model in (n x M) 
    dJdbm = np.zeros(shape = (1, M)) 
    dJdwmj = np.zeros(shape = (M, p)) 

    #J = np.zeros(shape = (maxit, 1)) # cost vector 
    #c_acc = np.zeros(shape = (maxit, 1)) # classifcation accuracy 
    J = []
    c_acc = []

    lr0 = 0.1
    lrt = 0.05
    tau = 100  

    for e in range(0, epochs): 

        # Shuffling each epoch
        ### Shuffling indices 
        ind = np.arange(n)
        np.random.shuffle(ind) # shuffle indices 

        ### Shuffling training data and labels
        xt = xtrain[ind, :]

        yt = ytrain[ind, :] 

        ytrue_shuff = np.argmax(yt, axis = 1) 

        it = 0 
    
        for j in range(n//n_batch): 

                lr = (1 - (it/tau)) * lr0 + (it/tau) * lrt

                mini_batch = np.random.randint(0, n, size = n_batch) # batch indices 

                z_im = xt[mini_batch, :] @ w_mj.T + b_m 

                y_im = yt[mini_batch, :] 

                z_im_norm = z_im - np.max(z_im, axis = 1, keepdims = True) 

                p_im = np.exp(z_im_norm) / np.sum(np.exp(z_im_norm), axis = 1, keepdims = True) 

                dJdzim = (1/n_batch) * (y_im * p_im - y_im) 
                dJdbm = np.sum(dJdzim, axis = 0) 
                dJdwmj = dJdzim.T @ xt[mini_batch, :] 
                
                b_m = b_m - lr * dJdbm 
                w_mj = w_mj - lr * dJdwmj 
                
                L_i = np.sum(y_im * np.log(np.sum(np.exp(z_im_norm), axis = 1, keepdims = True)) - y_im * z_im_norm, axis = 1) 
                #J[it] = (1/n_batch) * np.sum(L_i) 
                J.append((1/n_batch) * np.sum(L_i))

                ypred = np.argmax(p_im, axis = 1) 
                #c_acc[it] = np.array([1 for i in range(0,n_batch) if ypred[i] == ytrue_shuff[j]]).sum() 
                c_acc.append(((1/n_batch) * np.array([1 for i in range(0,n_batch) if ypred[i] == ytrue_shuff[j]]).sum()))

                #print("sum p_im: ", np.sum(p_im, axis = 1))
                
                it += 1 

    return J, c_acc, it, w_mj, b_m , z_im, p_im 

J, c_acc, it, wmj, bm, zim, pim = gradient_descent(x_train, y_train) 

#plt.figure(1) 
#plt.plot(J) 

#plt.figure(2) 
#plt.plot(c_acc) 

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