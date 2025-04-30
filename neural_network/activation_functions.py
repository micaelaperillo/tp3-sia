import numpy as np

def step(x:float)->float:
    return 1 if x > 0 else -1

def identity(x:float)->float:
    return x

def prime_identity(x:float)->float:
    return 1

def tanh(x:float, beta:float)->float:
    return np.tanh(beta * x)

def prime_tanh(x:float, beta:float)->float:
    return beta * (1 - (tanh(x,beta) ** 2))

def logistic(x:float, beta:float)->float:
    return 1 / (1 + np.exp(-2 * beta * x))

def prime_logistic(x:float, beta:float)->float:
    return 2 * beta * logistic(x, beta) * (1 - logistic(x, beta))