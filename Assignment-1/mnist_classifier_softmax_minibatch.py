import numpy as np 
from load_mnist import load_mnist 
from matplotlib import pyplot as plt 

batch_gen = np.random.default_rng()

x_train, y_train, x_test, y_test = load_mnist()

M = 10 
n_train = x_train.shape[0] # number of training examples - 60000 
p = x_train.shape[1] # number of input pixels - 784 (flattened 28x28 image) 

n_test = x_test.shape[0] # number of testing examples - 10000 
ytrue_test = np.argmax(y_test, axis = 1) 

n_batch = 2048 # batch size 
epochs = 1000 # epochs 

def softmax_gd_minibatch(xtrain, ytrain, xtest, ytest, nb, lr_init, tau, n_train, n_test, k): 

    w_mj = np.random.normal(scale = 0.01, size = (M, p)) # weight matrix                                                                                                                                                                                            
    b_m = np.zeros(shape = (1, M)) 
    z_im = np.zeros(shape = (n_train, M)) 
    dJdbm = np.zeros(shape = (1, M)) 
    dJdwmj = np.zeros(shape = (M, p)) 

    J_train = np.zeros(shape = (epochs, 1)) 
    acc_train = np.zeros(shape = (epochs, 1)) 

    J_test = np.zeros(shape = (epochs, 1)) 
    acc_test = np.zeros(shape = (epochs, 1)) 

    e_p = 0 # epoch counter 
    lr0 = lr_init # initial learning rate 
    lrt = 0.01 * lr0 # final learning rate 
    t_tau = tau 

    while e_p != epochs: 

        ### Shuffling indices 
        ind = np.arange(n_train) 
        np.random.shuffle(ind) # shuffle indices 

        ### Shuffling training data and labels
        xt = xtrain[ind, :] 
        yt = ytrain[ind, :] 

        ytrue_shuff = np.argmax(yt, axis = 1) 

        it = 0 

        Jtrainaccum = []
        acctrainaccum = []
        Jtestaccum = [] 
        acctestaccum = [] 
    
        for j in range(n_train//nb): 

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

            # Calculate the cost and accuracy every k-th iteration 
            if it % k == 0: 
                #Cost and accuracy for training data 
                L_i = np.sum(y_im * np.log(np.sum(np.exp(z_im_norm), axis = 1, keepdims = True)) - y_im * z_im_norm, axis = 1) 
                ypred = np.argmax(p_im, axis = 1) 

                Jtrainaccum.append((1/nb) * np.sum(L_i)) 
                acctrainaccum.append((1/nb) * np.sum(ypred == ytrue_shuff[mini_batch])) 

                # Cost and accuracy for testing data 
                z_test = xtest @ w_mj.T + b_m 
                z_test_norm = z_test - np.max(z_test, axis = 1, keepdims = True) 
                p_test = np.exp(z_test_norm) / np.sum(np.exp(z_test_norm), axis = 1, keepdims = True) 
                L_i_test = np.sum(y_test * np.log(np.sum(np.exp(z_test_norm), axis = 1, keepdims = True)) - y_test * z_test_norm, axis = 1) 
                ypred_test = np.argmax(p_test, axis = 1) 

                Jtestaccum.append((1/n_test) * np.sum(L_i_test)) 
                acctestaccum.append((1/n_test) * np.sum(ypred_test == ytrue_test)) 

            Jtrainaccum_av = np.mean(Jtrainaccum)
            acctrainaccum_av = np.mean(acctrainaccum) 
            Jtestaccum_av = np.mean(Jtestaccum)
            acctestaccum_av = np.mean(acctestaccum)

            it += 1 
            print("Epoch: (%s/%s), iteration: %s" % (e_p + 1, epochs, it)) 

        J_train[e_p] = Jtrainaccum_av 
        acc_train[e_p] = acctrainaccum_av
        J_test[e_p] = Jtestaccum_av 
        acc_test[e_p] = acctestaccum_av 

        e_p += 1 

    return J_train, 100 * acc_train, J_test, 100 * acc_test, it, w_mj, b_m , z_im, p_im 

Jtrain, acc_train, Jtest, acc_test, it, wmj, bm, zim, pim = softmax_gd_minibatch(x_train, y_train, x_test, y_test, n_batch, 0.001, 100, n_train, n_test, 2) 

plt.figure(1) 
plt_J, = plt.plot(Jtrain, 'r') 
plt_J_test, = plt.plot(Jtest, 'b') 
plt.legend([plt_J, plt_J_test], ['Train cost', 'Test cost'])
plt.annotate("Final train cost: %s and final test cost: %s" % (float(Jtrain[-1]), float(Jtest[-1])), xy = (epochs//2,1))
plt.xlabel('Number of epochs') 
plt.ylabel('Cost J') 
plt.title('Cost versus iterations') 
print("Final train cost: %s" % float(Jtrain[-1]))
print("Final test cost: %s" % float(Jtest[-1])) 

plt.figure(2)
plt_acc_train, = plt.plot(acc_train, 'r') 
plt_acc_test, = plt.plot(acc_test, 'b') 
plt.legend([plt_acc_train, plt_acc_test], ['Train accuracy', 'Test accuracy']) 
plt.annotate("Final train accuracy: %s and final test accuracy: %s" % (float(acc_train[-1]), float(acc_test[-1])), xy = (epochs//2,1))
plt.xlabel('Number of iterations') 
plt.ylabel('Accuracy in %') 
plt.title('Accuracy versus iterations') 
print("Final train accuracy: %s%%" % float(acc_train[-1])) 
print("Final test accuracy: %s%%" % float(acc_test[-1])) 

plot_hist = False 
if plot_hist == True: 
    plt.figure(4)
    plt.hist(wmj[0, :]) 
    plt.hist(wmj[1, :]) 
    plt.hist(wmj[2, :]) 
    plt.hist(wmj[3, :]) 
    plt.hist(wmj[4, :]) 
    plt.hist(wmj[5, :]) 
    plt.hist(wmj[6, :]) 
    plt.hist(wmj[7, :]) 
    plt.hist(wmj[8, :]) 
    plt.hist(wmj[9, :]) 

plot_weights = True  
if plot_weights == True: 
    figw, axw = plt.subplots(2, 5, figsize = (10,10)) 
    
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