import numpy as np

# Activation functions:

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return x * (x > 0)

def linear(x):
    return x

def tanh(x):
    return np.tanh(x)

# R to R gen functions:
