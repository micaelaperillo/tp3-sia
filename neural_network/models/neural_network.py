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
        self.layers = [ Layer(len(x_values[0]), hidden_layers_neuron_amounts[current_layer_index-1] if current_layer_index > 0 else len(x_values[0]), current_layer_neuron_amount, activation_function, prime_activation_function, seed) for current_layer_index, current_layer_neuron_amount in enumerate(hidden_layers_neuron_amounts)]
        self.x_values = x_values
        self.y_values = y_values

    def predict_using_feed_forward_pass(self, input_values:List[int]):
        a_j_vector = input_values
        for layer in self.layers:
            a_j_vector = layer.forward(a_j_vector, 1.0)
        return a_j_vector
    
    def backpropagate(self, input_values:List[List[int]], y_values:List[List[int]], learning_rate:float, epochs:int, optimizer:OptimizerFunctionType, error_function:ErrorFunctionType, beta:float):
        for epoch in range(epochs):
            for input_vector, y_value in zip(input_values, y_values):
                prediction = self.predict_using_feed_forward_pass(input_vector)

                basic_error = y_value - prediction
                
                #if (basic_error > 0.01).any():
                reverse_layers = self.layers[::-1]

                output_layer = reverse_layers[0]
                output_delta = basic_error * output_layer.prime_activation_function(output_layer.h_j_values, beta)
                layer_deltas = [output_delta]

                # we calculate all deltas before updating weights
                for layer_idx in range(1, len(reverse_layers)):
                    current_layer = reverse_layers[layer_idx]
                    next_layer = reverse_layers[layer_idx-1]

                    # first we try doing it without bias
                    next_weights = next_layer.weights_matrix[:, 1:]  

                    propagated_error = np.dot(layer_deltas[0], next_weights)  
                    current_delta = current_layer.prime_activation_function(current_layer.h_j_values, beta) * propagated_error

                    layer_deltas.insert(0, current_delta)

            #batch
            for layer_index, layer in enumerate(self.layers):
                delta = layer_deltas[layer_index]
                if (layer_index == 0):
                    input_to_layer = input_vector
                else:
                    input_to_layer = self.layers[layer_index-1].a_j_values

                for j in range(len(delta)):
                    for i in range(len(input_to_layer)):
                        delta_w = learning_rate * delta[j] * input_to_layer[i]
                        layer.weights_matrix[j][i+1] += delta_w 
            
            #error for error
            errors = []
            for input_vector, y_value in zip(input_values, y_values):
                prediction = self.predict_using_feed_forward_pass(input_vector)
                basic_error = y_value - prediction
                errors.append(basic_error)
            
            network_error = error_function(np.array(errors))

            if network_error < 0.01:
                return epoch+1, network_error
            
        return epochs, network_error