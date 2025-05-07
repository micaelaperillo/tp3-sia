import numpy as np
from .multilayer_perceptron import Perceptron

class Layer:
    def __init__(self, num_inputs: int, num_neurons: int, activation_function, prime_activation_function=None, seed: int = 43):
        self.activation_function = activation_function  
        self.prime_activation_function = prime_activation_function
        self.neurons: list[Perceptron] = []
        for i in range(num_neurons):
            perceptron = Perceptron(num_inputs, activation_function, prime_activation_function, seed_value=seed + i)
            self.neurons.append(perceptron)

    def forward(self, inputs, beta=1.0):
        activated_outputs = []
        z_values = []
        input_with_bias = np.insert(inputs, 0, 1)  # añado el bias como x0 = 1

        for neuron in self.neurons:
            a_j, z_j = neuron.predict(input_with_bias, beta)
            z_values.append(z_j)
            activated_outputs.append(a_j)

        return np.array(activated_outputs), np.array(z_values)

   