import numpy as np 
from load_mnist import load_mnist 
from matplotlib import pyplot as plt 

x_train, y_train, x_test, y_test = load_mnist()

M = 10 
n_train = x_train.shape[0] # number of training examples - 60000 
p = x_train.shape[1] # number of input pixels - 784 (flattened 28x28 image) 

n_test = x_test.shape[0]

def predict_test(wmj, bm, xtest, ytest): 

    n_t = xtest.shape[0] 
    z_predict = xtest @ wmj.T + bm 

    z_im_norm = z_predict - np.max(z_predict, axis = 1, keepdims = True) 

    p_predict = np.exp(z_im_norm) / np.sum(np.exp(z_im_norm), axis = 1, keepdims = True) 

    y_predict = np.argmax(p_predict, axis = 1)  
    y_true = np.argmax(ytest, axis = 1) 
    
    acc_test = ((1/n_t) * (np.array([1 for i in range(0, n_t) if y_predict[i] == y_true[i]]).sum())) * 100 

    return z_predict, p_predict, acc_test 

def softmax_gd(xtrain, ytrain, xtest, ytest, n_train, n_test, lr, maxit): 

    w_mj = np.random.rand(M, p) # weight matrix                                                                                                                                                                                                                                    
    b_m = np.zeros(shape = (1, M)) # offset vector 
    z_im = np.zeros(shape = (n_train, M)) # model in (n x M) 
    dJdbm = np.zeros(shape = (1, M))
    dJdwmj = np.zeros(shape = (M, p)) 

    J = np.zeros(shape = (maxit, 1)) # cost vector 
    acc_train = np.zeros(shape = (maxit, 1)) # classifcation accuracy 

    J_test = np.zeros(shape = (maxit, 1)) # cost vector 
    acc_test = np.zeros(shape = (maxit, 1)) # classifcation accuracy 

    it = 0 

    while it != maxit: 

        z_im = xtrain @ w_mj.T + b_m 

        y_im = ytrain 

        z_im_norm = z_im - np.max(z_im, axis = 1, keepdims = True) 

        p_im = np.exp(z_im_norm) / np.sum(np.exp(z_im_norm), axis = 1, keepdims = True) 

        dJdzim = (y_im * p_im - y_im) 

        dJdbm = np.sum(dJdzim, axis = 0) 
        dJdwmj = dJdzim.T @ xtrain 
                
        b_m = b_m - (1/n_train) * lr * dJdbm 
        w_mj = w_mj - (1/n_train) * lr * dJdwmj

        # Cost and accuracy for training mini-batch 
        L_i = np.sum(y_im * np.log(np.sum(np.exp(z_im_norm), axis = 1, keepdims = True)) - y_im * z_im_norm, axis = 1) 
        J[it] = (1/n_train) * np.sum(L_i) 

        ypred_train = np.argmax(p_im, axis = 1) 
        ytrue_train = np.argmax(y_train, axis = 1) 
        acc_train[it] = np.array([1 for i in range(0,n_train) if ypred_train[i] == ytrue_train[i]]).sum() 

        # Cost and accuracy for testing data 
        z_test = xtest @ w_mj.T + b_m 
        z_test_norm = z_test - np.max(z_test, axis = 1, keepdims = True)
        L_i_test = np.sum(ytest * np.log(np.sum(np.exp(z_test_norm), axis = 1, keepdims = True)) - ytest * z_test_norm, axis = 1) 
        J_test[it] = (1/n_test) * np.sum(L_i_test) 

        p_test = np.exp(z_test_norm) / np.sum(np.exp(z_test_norm), axis = 1, keepdims = True) 
    
        ypred_test = np.argmax(p_test, axis = 1) 
        ytrue_test = np.argmax(y_test, axis = 1) 
        acc_test[it] = np.array([1 for i in range(0,n_test) if ypred_test[i] == ytrue_test[i]]).sum() 
  
        it += 1 
        print("Iteration: (%s/%s)" % (it, maxit))  

    return J, acc_train * (1/n_train) * 100, J_test, acc_test * (1/n_test) * 100, it, w_mj, b_m , z_im, p_im 

J, acc_train, J_test, acc_test, it, wmj, bm, zim, pim = softmax_gd(x_train, y_train, x_test, y_test, n_train, n_test, 0.02, 3000) 

#wmj /= np.max(wmj) # rescale weights by maximum for visualization 

plt.figure(1) 
plt_J, = plt.plot(J, 'r') 
plt_J_test, = plt.plot(J_test, 'b') 
plt.legend([plt_J, plt_J_test], ['Train cost', 'Test cost'])
plt.xlabel('Number of iterations') 
plt.ylabel('Cost J') 
plt.title('Cost versus iterations') 
print("Final train cost: %s" % float(J[-1]))
print("Final test cost: %s" % float(J_test[-1]))

plt.figure(2)
plt_acc_train, = plt.plot(acc_train, 'r') 
plt_acc_test, = plt.plot(acc_test, 'b') 
plt.legend([plt_acc_train, plt_acc_test], ['Train accuracy', 'Test accuracy']) 
plt.xlabel('Number of iterations') 
plt.ylabel('Accuracy in %%') 
plt.title('Accuracy versus iterations') 
print("Final train accuracy: %s%%" % float(acc_train[-1])) 
print("Final test accuracy: %s%%" % float(acc_test[-1])) 

figw, axw = plt.subplots(2, 5, figsize = (8,8)) 
axw[0,0].imshow(wmj[0, :].reshape(28,28), cmap = 'gray') 
axw[0,0].set_title('0') 
axw[0,1].imshow(wmj[1, :].reshape(28,28), cmap = 'gray') 
axw[0,1].set_title('1')
axw[0,2].imshow(wmj[2, :].reshape(28,28), cmap = 'gray') 
axw[0,2].set_title('2')
axw[0,3].imshow(wmj[3, :].reshape(28,28), cmap = 'gray') 
axw[0,3].set_title('3')
axw[0,4].imshow(wmj[4, :].reshape(28,28), cmap = 'gray') 
axw[0,4].set_title('4') 
axw[1,0].imshow(wmj[5, :].reshape(28,28), cmap = 'gray') 
axw[1,0].set_title('5')
axw[1,1].imshow(wmj[6, :].reshape(28,28), cmap = 'gray') 
axw[1,1].set_title('6')
axw[1,2].imshow(wmj[7, :].reshape(28,28), cmap = 'gray') 
axw[1,2].set_title('7')
axw[1,3].imshow(wmj[8, :].reshape(28,28), cmap = 'gray') 
axw[1,3].set_title('8')
axw[1,4].imshow(wmj[9, :].reshape(28,28), cmap = 'gray') 
axw[1,4].set_title('9') 

plt.show()