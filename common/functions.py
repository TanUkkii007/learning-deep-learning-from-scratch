# coding: utf-8
import numpy as np

def identity_function(x):
    return x

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    t = np.max(x)
    xx = x - t
    return np.exp(xx) / np.sum(np.exp(xx))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # convert one-hot-vector to answer-label-index vector
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return - np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
