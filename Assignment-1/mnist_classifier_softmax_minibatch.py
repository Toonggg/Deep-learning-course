import numpy as np 
from load_mnist import load_mnist 
from matplotlib import pyplot as plt 

x_train, y_train, x_test, y_test = load_mnist()

M = 10 
n_train = x_train.shape[0] # number of training examples - 60000 
p = x_train.shape[1] # number of input pixels - 784 (flattened 28x28 image) 

n_test = x_test.shape[0] # number of testing examples - 10000 
ytrue_test = np.argmax(y_test, axis = 1) 

def softmax_gd_minibatch(xtrain, ytrain, xtest, ytest, ep, nb, lr_init, tau, n_train, n_test, k): 

    w_mj = np.random.normal(scale = 0.01, size = (M, p)) # weight matrix                                                                                                                                                                                            
    b_m = np.zeros(shape = (1, M)) 
    z_im = np.zeros(shape = (n_train, M)) 
    dJdbm = np.zeros(shape = (1, M)) 
    dJdwmj = np.zeros(shape = (M, p)) 

    J_train = np.zeros(shape = (ep, 1)) 
    acc_train = np.zeros(shape = (ep, 1)) 

    J_test = np.zeros(shape = (ep, 1)) 
    acc_test = np.zeros(shape = (ep, 1)) 

    e_p = 0 # epoch counter 
    lr0 = lr_init # initial learning rate 
    lrt = 0.01 * lr0 # final learning rate 
    t_tau = tau 

    tot_it = 0 
    it_k = 0

    Jtrainiter = np.zeros(shape = ((((n_train//nb)//(k)) * ep), 1)) 
    Jtestniter = np.zeros(shape = ((((n_train//nb)//(k)) * ep), 1)) 

    acctrainiter = np.zeros(shape = ((((n_train//nb)//(k)) * ep), 1)) 
    acctestiter = np.zeros(shape = ((((n_train//nb)//(k)) * ep), 1)) 

    while e_p != ep: 

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

            # Calculate the cost and accuracy every k-th iteration for averaging per epoch 
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

             # Calculate the cost and accuracy every k-th iteration and accumulate over all epochs 
            if tot_it % k == 0: 
                #Cost and accuracy for training data 
                L_i = np.sum(y_im * np.log(np.sum(np.exp(z_im_norm), axis = 1, keepdims = True)) - y_im * z_im_norm, axis = 1) 
                ypred = np.argmax(p_im, axis = 1) 
                if nb == n_train:
                    Jtrainiter = []
                    acctrainiter = []
                    Jtrainiter.append(((1/nb) * np.sum(L_i)))
                    acctrainiter.append(((1/nb) * np.sum(ypred == ytrue_shuff[mini_batch])))
                else: 
                    Jtrainiter[it_k] = ((1/nb) * np.sum(L_i)) 
                    acctrainiter[it_k] = ((1/nb) * np.sum(ypred == ytrue_shuff[mini_batch])) 
                # Cost and accuracy for testing data 
                z_test = xtest @ w_mj.T + b_m 
                z_test_norm = z_test - np.max(z_test, axis = 1, keepdims = True) 
                p_test = np.exp(z_test_norm) / np.sum(np.exp(z_test_norm), axis = 1, keepdims = True) 
                L_i_test = np.sum(y_test * np.log(np.sum(np.exp(z_test_norm), axis = 1, keepdims = True)) - y_test * z_test_norm, axis = 1) 
                ypred_test = np.argmax(p_test, axis = 1) 
                if nb == n_train:
                    Jtestniter = []
                    acctestiter = []
                    Jtestniter.append(((1/n_test) * np.sum(L_i_test)))
                    acctestiter.append(((1/n_test) * np.sum(ypred_test == ytrue_test)))
                else: 
                    Jtestniter[it_k] = ((1/n_test) * np.sum(L_i_test)) 
                    acctestiter[it_k] = ((1/n_test) * np.sum(ypred_test == ytrue_test)) 
                it_k += 1

            Jtrainaccum_av = np.mean(Jtrainaccum) 
            acctrainaccum_av = np.mean(acctrainaccum) 
            Jtestaccum_av = np.mean(Jtestaccum) 
            acctestaccum_av = np.mean(acctestaccum)

            it += 1 
            tot_it += 1 

            print("Epoch: (%s/%s), iteration: %s" % (e_p + 1, ep, it)) 

        J_train[e_p] = Jtrainaccum_av 
        acc_train[e_p] = acctrainaccum_av 
        J_test[e_p] = Jtestaccum_av 
        acc_test[e_p] = acctestaccum_av 

        e_p += 1 

    return J_train, 100 * acc_train, J_test, 100 * acc_test, Jtrainiter, Jtestniter, 100 * acctrainiter, 100 * acctestiter, it, w_mj, b_m , z_im, p_im 

n_batch = 1500 # batch size ---> 30 iterations per epoch 
epochs = 300 # epochs 
lr0 = 0.01 # initial learning rate 
tau_it = (n_train//n_batch) - 5 # decay 
k_plot = 5 # storing accuracy/cost values each k-th iteration 

Jtrain, acc_train, Jtest, acc_test, J_trainiter, J_testniter, acc_trainiter, acc_testiter, it, wmj, bm, zim, pim = softmax_gd_minibatch(x_train, y_train, x_test, y_test, epochs, n_batch, lr0, tau_it, n_train, n_test, k_plot) 

# Costs and accuracies averaged over k-th iteration per epoch 
plt.figure(1) 
plt_J, = plt.plot(Jtrain, 'r') 
plt_J_test, = plt.plot(Jtest, 'b') 
plt.legend([plt_J, plt_J_test], ['Train cost', 'Test cost']) 
plt.annotate("Final train cost: %s" % (float(Jtrain[-1])), xycoords = 'figure fraction', xy = (0.4,0.5))
plt.annotate("Final test cost: %s" % (float(Jtest[-1])), xycoords = 'figure fraction', xy = (0.4,0.55))
plt.xlabel('Number of epochs') 
plt.ylabel('Cost J') 
plt.title('Average cost versus epochs') 
print("Final train cost: %s" % float(Jtrain[-1])) 
print("Final test cost: %s" % float(Jtest[-1]))     

plt.figure(2)
plt_acc_train, = plt.plot(acc_train, 'r') 
plt_acc_test, = plt.plot(acc_test, 'b') 
plt.legend([plt_acc_train, plt_acc_test], ['Train accuracy', 'Test accuracy']) 
plt.annotate("Final train accuracy: %s%%" % (float(acc_train[-1])), xycoords = 'figure fraction', xy = (0.4,0.5))
plt.annotate("Final test accuracy: %s%%" % (float(acc_test[-1])), xycoords = 'figure fraction', xy = (0.4,0.55))
plt.xlabel('Number of epochs') 
plt.ylabel('Accuracy in %') 
plt.title('Average accuracy versus epochs') 
print("Final train accuracy: %s%%" % float(acc_train[-1])) 
print("Final test accuracy: %s%%" % float(acc_test[-1])) 

### Costs and accuracies per k-th iteration 
plt.figure(3) 
plt_Jtrain_it, = plt.plot(J_trainiter, 'r') 
plt_Jtest_it, = plt.plot(J_testniter, 'b') 
plt.legend([plt_Jtrain_it, plt_Jtest_it], ['Train cost', 'Test cost']) 
plt.annotate("Final train cost: %s" % (float(J_trainiter[-1])), xycoords = 'figure fraction', xy = (0.4,0.5))
plt.annotate("Final test cost: %s" % (float(J_testniter[-1])), xycoords = 'figure fraction', xy = (0.4,0.55))
plt.xlabel('Total number of iterations') 
plt.ylabel('Cost J') 
plt.title('Cost versus iterations') 
print("Final train cost: %s" % float(J_trainiter[-1])) 
print("Final test cost: %s" % float(J_testniter[-1]))   

plt.figure(4)
plt_acc_train_it, = plt.plot(acc_trainiter, 'r') 
plt_acc_test_it, = plt.plot(acc_testiter, 'b') 
plt.legend([plt_acc_train_it, plt_acc_test_it], ['Train accuracy', 'Test accuracy']) 
plt.annotate("Final train accuracy: %s%%" % (float(acc_trainiter[-1])), xycoords = 'figure fraction', xy = (0.4,0.5))
plt.annotate("Final test accuracy: %s%%" % (float(acc_testiter[-1])), xycoords = 'figure fraction', xy = (0.4,0.55))
plt.xlabel('Total number of iterations') 
plt.ylabel('Accuracy in %') 
plt.title('Accuracy versus iterations') 
print("Final train accuracy: %s%%" % float(acc_trainiter[-1])) 
print("Final test accuracy: %s%%" % float(acc_testiter[-1])) 

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