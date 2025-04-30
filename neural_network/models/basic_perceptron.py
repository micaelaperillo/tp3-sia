import numpy as np
from typing import List, Callable, Union

ActivationFunctionType = Union[Callable[[float], int], Callable[[float, float], float]]
ErrorFunctionType = Callable[[List[float]], float]

def apply_step_activation(x:float) -> int:
    return 1 if x > 0 else -1

class Perceptron:
    def __init__(self, input_vector_size:int, learning_rate:float, epochs:int, activation_function:ActivationFunctionType= apply_step_activation, prime_activation_funcion=None, weights:List[float] = None, seed_value:int = 43):
        self.input_vector_size = input_vector_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function = activation_function
        self.prime_activation_function = prime_activation_funcion
        self.seed = seed_value
        np.random.seed(seed_value)
        # we used weights[0] as bias b
        # we multiply by 0.1 to get even smaller numbers
        self.weights:List[float] = np.random.rand(input_vector_size + 1) * 0.1 if weights == None else weights

    
    # to test for generalization
    def predict(self, input_set:List[int]):
        h_supra_mu = np.dot(self.weights, input_set) 
        return self.activation_function(h_supra_mu), h_supra_mu

    def train(self, training_set:List[List[int]], labels:List[int], error_function:ErrorFunctionType, max_acceptable_error:float,file, method:str, is_activation_derivable=False):
        for epoch in range(self.epochs):
            errors = []
            for data_instance, label in zip(training_set, labels):
                data_with_bias = np.insert(data_instance, 0, 1)
                prediction, h_supra_mu = self.predict(data_with_bias)

                basic_error = (label - prediction) 
                errors.append(basic_error)
                
                # we update bias and weights altogether
                if basic_error != 0:
                    delta_w = self.learning_rate * basic_error * data_with_bias
                    if (is_activation_derivable):
                        delta_w *= self.prime_activation_function(h_supra_mu)
                    self.weights += delta_w

            perceptron_error = error_function(np.array(errors))
            file.write(f"{self.seed},{method},{self.learning_rate},{epoch},{perceptron_error}\n")
            if perceptron_error < max_acceptable_error:
                return epoch + 1, perceptron_error
        
        return self.epochs, perceptron_error
            