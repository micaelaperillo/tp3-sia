import numpy as np
from typing import List
from neural_network.models.layer import Layer
from neural_network.activation_functions import ActivationFunctionType
from neural_network.error_functions import ErrorFunctionType
from neural_network.optimizers import OptimizerFunctionType

class NeuralNetwork:
    # hidden_layers_amount: [35,15, 10] -> last layer has to be related to the problem
    # x_values = all the numbers used for clasification
    def __init__(self, x_values:List[List[int]], y_values:List[int], hidden_layers_neuron_amounts:List[int], activation_function:ActivationFunctionType, prime_activation_function:ActivationFunctionType, seed:int):
        self.layers = [ Layer(len(x_values[0]), hidden_layers_neuron_amounts[(current_layer_index - 1)], current_layer_neuron_amount, activation_function, prime_activation_function, seed) for current_layer_index, current_layer_neuron_amount in enumerate(hidden_layers_neuron_amounts[1:])]
        self.x_values = x_values
        self.y_values = y_values

    def predict_using_feed_forward_pass(self, input_values:List[int]):
        a_j_vector = input_values
        for layer in self.layers:
            a_j_vector = layer.forward(a_j_vector, 1.0)
        return a_j_vector
    
    def backpropagate(self, input_values:List[List[int]], y_values:List[List[int]], learning_rate:float, epochs:int, optimizer:OptimizerFunctionType, error_function:ErrorFunctionType, beta:float):
        # vector with a_n values
        for input_vector, y_value in zip(input_values, y_values):
            for epoch in range(epochs):
                prediction = self.predict_using_feed_forward_pass(input_vector)

                # [0.0 1.0 0.0 0.0 0.0] - [0 0.3 -0.3 -0.4 0 0 0]
                basic_error = (y_value - prediction)

                if basic_error != 0:
                    #updating the weights and biases using backpropagation
                    reverse_layers = self.layers[::-1]
                    for layer_index, layer in enumerate(reverse_layers):
                        delta_w_array = optimizer(learning_rate, basic_error, reverse_layers[layer_index-1].a_j_values if layer_index > 0 else input_vector, layer.prime_activation_function, prediction, beta)
                        layer.weights_matrix += delta_w_array
    