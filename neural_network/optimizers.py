from typing import List, Callable, Union
from neural_network.activation_functions import ActivationFunctionType
import numpy as np

OptimizerFunctionType = Union[Callable[[float, float, List[float]], List[float]],Callable[[float, float, float, ActivationFunctionType, float, float], List[float]]] 

def rosenblatt_optimizer(learning_rate:float, basic_error:float, data_with_bias:List[float]):
    return learning_rate * basic_error * data_with_bias

def gradient_descent_optimizer(learning_rate:float, basic_error:float, data_with_bias:float, prime_activation_function:ActivationFunctionType, h_supra_mu:float, beta:float):
    return learning_rate * basic_error * data_with_bias * prime_activation_function(h_supra_mu, beta)

def gradient_descent_optimizer_with_delta(learning_rate:float, delta:List[float], data_with_bias:List[float]):
    return learning_rate * np.matmul(delta,data_with_bias)