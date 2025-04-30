import numpy as np
from typing import List, Callable

ActivationFunctionType = Callable[[float], int]
def apply_step_activation(x:float) -> int:
    return 1 if x >= 0 else -1

class Perceptron:
    def __init__(self, input_vector_size:int, learning_rate:float , epochs:int, weights:List[float] = None, activation_function:ActivationFunctionType= apply_step_activation, seed_value:int = 43):
        self.input_vector_size = input_vector_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function = activation_function
        self.seed = seed_value
        np.random.seed(seed_value)
        # we used weights[0] as bias b
        # we multiply by 0.1 to get even smaller numbers
        self.weights:List[float] = np.random.rand(input_vector_size + 1) * 0.1 if weights == None else weights

    
    # to test for generalization
    def predict(self, input_set:List[int]):
        h_star = np.dot(self.weights, input_set) 
        return self.activation_function(h_star)

    def linear_training(self, training_set:List[List[int]], labels:List[int], file, method:str, max_acceptable_error:float = 0.9):
        for epoch in range(self.epochs):
            errors = []
            for data_instance, label in zip(training_set, labels):
                data_with_bias = np.insert(data_instance, 0, 1)
                prediction = self.predict(data_with_bias)

                basic_error = (label - prediction) 
                errors.append(basic_error)
                
                # we update bias and weights altogether
                if basic_error != 0:
                    delta_w = self.learning_rate * basic_error * data_with_bias
                    self.weights += delta_w

            squared_errors = np.array(errors) ** 2
            perceptron_error = 0.5 * np.sum(squared_errors)
            file.write(f"{self.seed},{method},{self.learning_rate},{epoch},{perceptron_error}\n")
            if perceptron_error < max_acceptable_error:
                return epoch + 1, perceptron_error
        
        return self.epochs, perceptron_error
            