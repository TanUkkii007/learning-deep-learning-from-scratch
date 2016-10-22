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
    t = np.max(x)
    xx = x - t
    return np.exp(xx) / np.sum(np.exp(xx))

def cross_entropy_error(y, t):
    delta = 1e-7
    return - np.sum(t * np.log(y + delta))
