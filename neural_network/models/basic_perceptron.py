import numpy as np
from typing import List
from neural_network.activation_functions import ActivationFunctionType
from neural_network.error_functions import ErrorFunctionType

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

    
    # to test for generalization
    def predict(self, input_set:List[int], beta:float = 1.0):
        h_supra_mu = np.dot(self.weights, input_set) 
        return self.activation_function(h_supra_mu, beta), h_supra_mu

    def train(self, training_set:List[List[int]], labels:List[int], learning_rate:float, epochs:int, optimizer, error_function:ErrorFunctionType, max_acceptable_error:float,file, method:str, is_activation_derivable=False, beta:float= 1.0, partition:int=0):
        for epoch in range(epochs):
            for data_instance, label in zip(training_set, labels):
                data_with_bias = np.insert(data_instance, 0, 1)
                prediction, h_supra_mu = self.predict(data_with_bias, beta)

                basic_error = (label - prediction) 
                
                # we update bias and weights altogether
                if basic_error != 0:
                    if (is_activation_derivable):
                        delta_w = optimizer(learning_rate, basic_error, data_with_bias, self.prime_activation_function, h_supra_mu, beta)
                    else:
                        delta_w = optimizer(learning_rate, basic_error, data_with_bias)
                    self.weights += delta_w

            errors = []
            for x_value, expected_value in zip(training_set, labels):
                x_value = np.insert(x_value, 0, 1)
                prediction, h_supra_mu = self.predict(x_value)
                error = (expected_value - prediction) 
                errors.append(error)

            perceptron_error = error_function(np.array(errors))
            file.write(f"{self.seed},{partition},{self.weights},{method},{beta},{learning_rate},{epochs},{epoch+1},{perceptron_error}\n")
            if perceptron_error < max_acceptable_error:
                return epoch + 1, perceptron_error
        
        return epochs, perceptron_error
            