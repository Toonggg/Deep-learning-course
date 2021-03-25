from load_mnist import load_mnist
import matplotlib.pyplot as plt

xt, yt, xtt, ytt = load_mnist()
print(xt.shape)
print(yt.shape)
print(xtt.shape)
print(ytt.shape)