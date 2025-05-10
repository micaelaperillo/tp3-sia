from typing import List, Callable, Union, Tuple
from neural_network.activation_functions import ActivationFunctionType
import numpy as np

OptimizerFunctionType = Union[Callable[[float, float, List[float]], List[float]],Callable[[float, float, float, ActivationFunctionType, float, float], List[float]]] 

def rosenblatt_optimizer(learning_rate:float, basic_error:float, input_data:List[float]):
    return - learning_rate * basic_error * input_data

def gradient_descent_optimizer(learning_rate:float, basic_error:float, input_data:float, prime_activation_function:ActivationFunctionType, h_supra_mu:float, beta:float):
    return  learning_rate * basic_error * input_data * prime_activation_function(h_supra_mu, beta)

def gradient_descent_optimizer_with_delta(learning_rate:float, delta:float, input_data:float, alpha:float = 0.0):
    return - learning_rate * delta * input_data

def momentum_gradient_descent_optimizer_with_delta(learning_rate:float, delta:float, input_data:float, alpha:float):
    gradient_descent = - gradient_descent_optimizer_with_delta(learning_rate, delta, input_data)/learning_rate 
    return - learning_rate * gradient_descent + alpha * gradient_descent

def adam_optimizer_with_delta(learning_rate:float, delta:List[float], input_data:List[float], alpha: float, beta1: float, beta2: float, epsilon: float, epoch:int) -> List[float]:
    m = 0
    v = 0
    t = epoch
    g = gradient_descent_optimizer_with_delta(learning_rate, delta, input_data, alpha)

    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    return alpha * m_hat / (np.sqrt(v_hat) + epsilon)
