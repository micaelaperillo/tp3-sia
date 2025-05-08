import numpy as np
from neural_network.models.neural_network import NeuralNetwork
from neural_network.activation_functions import relu, logistic, prime_logistic, relu_derivative
from neural_network.optimizers import rosenblatt_optimizer, gradient_descent_optimizer
from neural_network.error_functions import mean_error, squared_error

if __name__ == '__main__':

    with open("input_data/TP3-ej3-digitos.txt", "r") as file:
        lines = [line.strip() for line in file if line.strip()]

    digits_vectors = []
    for i in range(0, len(lines), 7):
        block = lines[i:i+7]
        flat_list = [int(x) for row in block for x in row.split()]
        digits_vectors.append(flat_list)

    seed:int = 43

    # Discriminacion de paridad:
    # impar: [0, 1], par: [1, 0]
    y_values = [[1, 0], [0, 1],[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]
    hidden_layers_amounts = [[35, 16, 2]]
    activation_functions = [(relu, relu_derivative), (logistic, prime_logistic)]
    optimizers = [gradient_descent_optimizer, rosenblatt_optimizer]
    error_functions = [squared_error, mean_error]
    epochs = [200]
    learning_rates = [0.0001, 0.05]
    beta_values = [0.01, 0.05, 0.1]


    for layer_amount in hidden_layers_amounts:
        for activation_function in activation_functions:
            for optimizer in optimizers:
                for error_function in error_functions:
                    for total_epochs in epochs:
                        for learning_rate in learning_rate:
                            for beta in beta_values:
                                neural_network = NeuralNetwork(digits_vectors, y_values, layer_amount, activation_function[0], activation_function[1], seed)
                                training_errors = neural_network.backpropagate(digits_vectors, y_values, learning_rate, total_epochs, optimizer, error_function, beta)



    # Discriminacion de digito:
    # Por cada digito n crea una lista donde todos los valores son 0 excepto por la posicion n
    # Ejemplo: 1 = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    y_values = np.eye(10, dtype=float).tolist()
    hidden_layers_amount = [[35, 16, 10]]
    activation_functions = [(relu, relu_derivative), (logistic, prime_logistic)]
    optimizers = [gradient_descent_optimizer, rosenblatt_optimizer]
    error_functions = [squared_error, mean_error]
    epochs = [200]
    learning_rates = [0.0001, 0.05]
    beta_values = [0.01, 0.05, 0.1]

    for layer_amount in hidden_layers_amounts:
        for activation_function in activation_functions:
            for optimizer in optimizers:
                for error_function in error_functions:
                    for total_epochs in epochs:
                        for learning_rate in learning_rate:
                            for beta in beta_values:
                                neural_network = NeuralNetwork(digits_vectors, y_values, layer_amount, activation_function[0], activation_function[1], seed)
                                training_errors = neural_network.backpropagate(digits_vectors, y_values, learning_rate, total_epochs, optimizer, error_function, beta)

