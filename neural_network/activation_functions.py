from typing import Callable
import numpy as np

ActivationFunctionType = Callable[[float, float], float]

def step(x:float, beta:float=1.0)->float:
    return 1 if x > 0 else -1

def identity(x:float, beta:float=1.0)->float:
    return x

def prime_identity(x:float, beta:float=1.0)->float:
    return 1

def tanh(x:float, beta:float)->float:
    return np.tanh(beta * x)

def prime_tanh(x:float, beta:float)->float:
    return beta * (1 - (tanh(x,beta) ** 2))

def logistic(x:float, beta:float)->float:
    return 1 / (1 + np.exp(-2 * beta * x))

def prime_logistic(x:float, beta:float)->float:
    return 2 * beta * logistic(x, beta) * (1 - logistic(x, beta))

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0