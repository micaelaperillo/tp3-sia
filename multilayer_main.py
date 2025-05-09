import numpy as np
import os       
from typing import List
from neural_network.models.neural_network import NeuralNetwork
from neural_network.activation_functions import relu, logistic, prime_logistic, relu_derivative
from neural_network.optimizers import rosenblatt_optimizer, gradient_descent_optimizer
from neural_network.error_functions import mean_error, squared_error
from neural_network.partition_methods import k_cross_validation

if __name__ == '__main__':

    results_data_dir_name = "output_data"
    if not os.path.exists(results_data_dir_name):
        os.makedirs(results_data_dir_name)

    results_files:List[str] = ["ej3_parity_data.csv", "ej3_digits_data.csv"]

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

    training_errors = []
    training_data_prediction_errors = []
    testing_data_prediction_errors = []

    k = 5
    training_testing_pairs = k_cross_validation(k, digits_vectors, y_values)
    parity_results_file = open(os.path.join(results_data_dir_name, results_files[0]), "w", newline='')
    parity_results_file.write(f"seed,activation_function,neurons_per_layer,beta,learning_rate,epochs,error_method,error\n")

    for layer_amount in hidden_layers_amounts:
        for activation_function in activation_functions:
            for optimizer in optimizers:
                for error_function in error_functions:
                    for total_epochs in epochs:
                        for learning_rate in learning_rate:
                            for beta in beta_values:
                                for partition_index, configuration in enumerate(training_testing_pairs):

                                    training_set = configuration[0]
                                    testing_set = configuration[1]

                                    neural_network = NeuralNetwork(training_set[0], training_set[1], layer_amount, activation_function[0], activation_function[1], seed)
                                    error = neural_network.backpropagate(training_set[0], training_set[1], learning_rate, total_epochs, optimizer, error_function, beta)
                                    training_errors.append(error)
                                    
                                training_mean_error = np.mean(training_errors)
                                training_error_std = np.std(training_errors)
                                parity_results_file.write(f"{seed},{activation_function.__name__},{str(layer_amount)},{beta},{learning_rate},{total_epochs},mean_error,{training_mean_error}\n")




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

    training_errors = []
    training_data_prediction_errors = []
    testing_data_prediction_errors = []

    k = 5
    training_testing_pairs = k_cross_validation(k, digits_vectors, y_values)
    digits_results_file = open(os.path.join(results_data_dir_name, results_files[1]), "w", newline='')
    digits_results_file.write(f"seed,activation_function,neurons_per_layer,beta,learning_rate,epochs,error_method,error\n")


    for layer_amount in hidden_layers_amounts:
        for activation_function in activation_functions:
            for optimizer in optimizers:
                for error_function in error_functions:
                    for total_epochs in epochs:
                        for learning_rate in learning_rate:
                            for beta in beta_values:
                                for partition_index, configuration in enumerate(training_testing_pairs):
                                    training_set = configuration[0]
                                    testing_set = configuration[1]

                                    neural_network = NeuralNetwork(training_set[0], training_set[1], layer_amount, activation_function[0], activation_function[1], seed)
                                    error = neural_network.backpropagate(training_set[0], training_set[1], learning_rate, total_epochs, optimizer, error_function, beta)
                                    training_errors.append(error)
                                    
                                training_mean_error = np.mean(training_errors)
                                training_error_std = np.std(training_errors)
                                digits_results_file.write(f"{seed},{activation_function.__name__},{str(layer_amount)},{beta},{learning_rate},{total_epochs},mean_error,{training_mean_error}\n")
