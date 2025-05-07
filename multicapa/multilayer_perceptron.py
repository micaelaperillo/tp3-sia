import numpy as np
from typing import List
from neural_network.activation_functions import ActivationFunctionType
from neural_network.error_functions import ErrorFunctionType
from neural_network.error_functions import squared_error, mean_error

class Perceptron:
    def __init__(self, input_vector_size:int, activation_function:ActivationFunctionType, prime_activation_funcion=None, weights:List[float] = None, seed_value:int = 43):
        self.input_vector_size = input_vector_size
        self.activation_function = activation_function
        self.prime_activation_function = prime_activation_funcion
        self.seed = seed_value
        np.random.seed(seed_value)
        # we used weights[0] as bias b
        # we multiply by 0.1 to get even smaller numbers
        self.weights:List[float] = np.random.rand(input_vector_size + 1) * 0.1 if weights == None else weights

    def predict(self, input_set:List[int], beta:float = 1.0):
            h_supra_mu = np.dot(self.weights, input_set) 
            return self.activation_function(h_supra_mu, beta), h_supra_mu