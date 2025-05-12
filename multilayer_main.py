import os
import numpy as np
import json
from typing import List
from neural_network.models.neural_network import NeuralNetwork
from neural_network.activation_functions import relu, logistic, prime_logistic, relu_derivative, prime_tanh, tanh
from neural_network.optimizers import rosenblatt_optimizer, gradient_descent_optimizer_with_delta, momentum_gradient_descent_optimizer_with_delta, adam_optimizer_with_delta
from neural_network.error_functions import mean_error, squared_error
from neural_network.partition_methods import k_cross_validation
from metric_functions import get_prediction_error_for_neural_network, parity_calculate_accuracy, digits_calculate_accuracy

if __name__ == '__main__':

    results_data_dir_name = "output_data"
    if not os.path.exists(results_data_dir_name):
        os.makedirs(results_data_dir_name)

    results_files:List[str] = ["ej3_parity_data.csv", "ej3_digits_data.csv"]
    errors_results_files:List[str] = ["ej3_parity_data_errors.csv", "ej3_digits_data_errors.csv"]

    with open("input_data/TP3-ej3-digitos.txt", "r") as file:
        lines = [line.strip() for line in file if line.strip()]

    digits_vectors = []
    for i in range(0, len(lines), 7):
        block = lines[i:i+7]
        flat_list = [int(x) for row in block for x in row.split()]
        digits_vectors.append(flat_list)

    with open("config.json") as f:
        config = json.load(f)

    activation_functions_map = {
        "relu": (relu, relu_derivative),
        "logistic": (logistic, prime_logistic),
        "tanh": (tanh, prime_tanh)
    }

    error_functions_map = {
        "squared_error": squared_error,
        "mean_error": mean_error
    }

    def write_header_if_needed(file_path, header):
        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
            with open(file_path, "w", newline='') as file:
                file.write(header)

    write_header_if_needed("output_data/ej3_accuracy.csv", f"ej,activation_function,optimizer,partitions,partition,neurons_per_layer,learning_rate,total_epochs,training_accuracy,testing_accuracy\n")
    accuracy_file = open("output_data/ej3_accuracy.csv", "a", newline='')

    seed:int = 43


    ######################################################## PARIDAD ##################################################################

    # Discriminacion de paridad:
    # impar: [0.0, 1.0], par: [1.0, 0.0]
    x_values = np.array(digits_vectors)
    y_values = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    # y_values = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    k = 5
    training_testing_pairs = k_cross_validation(k, x_values, y_values)

    parity_config = config['parity']
    network_configurations = parity_config['network_configurations']
    activation_functions = [activation_functions_map[name] for name in parity_config['activation_functions']]
    error_functions = [error_functions_map[name] for name in parity_config['error_functions']]
    epochs = parity_config['epochs']
    learning_rates = parity_config['learning_rates']

    training_errors = []
    training_data_prediction_errors = []
    testing_data_prediction_errors = []

    parity_path = os.path.join(results_data_dir_name, results_files[0])
    errors_parity_path = os.path.join(results_data_dir_name, errors_results_files[0])

    write_header_if_needed(parity_path, f"seed,activation_function,optimizer,partition,neurons_per_layer,beta,learning_rate,alpha,total_epochs,epoch,error_function,error\n")
    write_header_if_needed(errors_parity_path, f"seed,activation_function,optimizer,partitions,neurons_per_layer,beta,learning_rate,alpha,total_epochs,error_function,training_mean_error,training_std_error,testing_mean_error,testing_std_error\n")
    parity_results_file = open(parity_path, "a", newline='')
    errors_parity_results_file = open(errors_parity_path, "a", newline='')


    def train_and_evaluate_parity_network(optimizer):

        if(optimizer == gradient_descent_optimizer_with_delta):
            max_error = 0.01
            for network_configuration in network_configurations:
                for activation_function in activation_functions:
                        for error_function in error_functions:
                            for learning_rate in learning_rates:
                                for total_epochs in epochs:
                                    training_errors = []
                                    testing_data_prediction_errors = []

                                    for partition_index, configuration in enumerate(training_testing_pairs):
                                        training_set = configuration[0]
                                        testing_set = configuration[1]

                                        neurons_per_layer_str = f"[{'-'.join(map(str, network_configuration))}]"
                                        neural_network = NeuralNetwork(training_set[0], training_set[1], network_configuration, activation_function[0], activation_function[1], seed)
                                        breaking_epoch, training_error = neural_network.backpropagate(training_set[0], training_set[1], learning_rate, total_epochs, optimizer, error_function, max_error, parity_results_file, is_adam_optimizer= False, partition= partition_index, neurons_per_layer= neurons_per_layer_str, activation_function= activation_function[0].__name__, activation_beta=1.0)
                                        training_errors.append(training_error)

                                        testing_data_prediction_error = get_prediction_error_for_neural_network(neural_network, testing_set[0], testing_set[1], mean_error)
                                        testing_data_prediction_errors.append(testing_data_prediction_error)

                                        training_accuracy = parity_calculate_accuracy(neural_network, training_set[0], training_set[1])
                                        testing_accuracy = parity_calculate_accuracy(neural_network, testing_set[0], testing_set[1])
                                        accuracy_file.write(f"parity,{activation_function[0].__name__},{optimizer.__name__},{k},{partition_index},{neurons_per_layer_str},{learning_rate},{total_epochs},{training_accuracy},{testing_accuracy}\n")

                                    training_mean_error = np.mean(training_errors)
                                    training_error_std = np.std(training_errors)                            
                                    testing_data_mean_prediction_error = np.mean(testing_data_prediction_errors)
                                    testing_data_prediction_error_std = np.std(testing_data_prediction_errors)    
                                    errors_parity_results_file.write(f"{seed},{activation_function[0].__name__},{optimizer.__name__},{k},{neurons_per_layer_str},{1.0},{learning_rate},{0.0},{total_epochs},{error_function.__name__},{training_mean_error},{training_error_std},{testing_data_mean_prediction_error},{testing_data_prediction_error_std}\n")

        if(optimizer == momentum_gradient_descent_optimizer_with_delta):
            max_error = 1.0
            alpha = 0.9
            for network_configuration in network_configurations:
                for activation_function in activation_functions:
                        for error_function in error_functions:
                            for learning_rate in learning_rates:
                                for total_epochs in epochs:
                                    training_errors = []
                                    testing_data_prediction_errors = []
                                    for partition_index, configuration in enumerate(training_testing_pairs):
                                        training_set = configuration[0]
                                        testing_set = configuration[1]

                                        neurons_per_layer_str = f"[{'-'.join(map(str, network_configuration))}]"
                                        neural_network = NeuralNetwork(training_set[0], training_set[1], network_configuration, activation_function[0], activation_function[1], seed)
                                        breaking_epoch, training_error = neural_network.backpropagate(digits_vectors, y_values, learning_rate, total_epochs, optimizer, error_function, max_error, parity_results_file, is_adam_optimizer= False, partition= partition_index, neurons_per_layer= neurons_per_layer_str, activation_function= activation_function[0].__name__, activation_beta= 1.0, alpha= alpha)
                                        
                                        testing_data_prediction_error = get_prediction_error_for_neural_network(neural_network, testing_set[0], testing_set[1], mean_error)
                                        testing_data_prediction_errors.append(testing_data_prediction_error)

                                        training_accuracy = parity_calculate_accuracy(neural_network, training_set[0], training_set[1])
                                        testing_accuracy = parity_calculate_accuracy(neural_network, testing_set[0], testing_set[1])
                                        accuracy_file.write(f"parity,{activation_function[0].__name__},{optimizer.__name__},{k},{partition_index},{neurons_per_layer_str},{learning_rate},{total_epochs},{training_accuracy},{testing_accuracy}\n")

                                    training_mean_error = np.mean(training_errors)
                                    training_error_std = np.std(training_errors)                            
                                    testing_data_mean_prediction_error = np.mean(testing_data_prediction_errors)
                                    testing_data_prediction_error_std = np.std(testing_data_prediction_errors)
                                    errors_parity_results_file.write(f"{seed},{activation_function[0].__name__},{optimizer.__name__},{k},{neurons_per_layer_str},{1.0},{learning_rate},{alpha},{total_epochs},{training_mean_error},{training_error_std},{testing_data_mean_prediction_error},{testing_data_prediction_error_std}\n")

        if(optimizer == adam_optimizer_with_delta):
            max_error = 1.0
            for network_configuration in network_configurations:
                for activation_function in activation_functions:
                        for error_function in error_functions:
                            for learning_rate in learning_rates:
                                for total_epochs in epochs:
                                    training_errors = []
                                    testing_data_prediction_errors = []
                                    for partition_index, configuration in enumerate(training_testing_pairs):
                                        training_set = configuration[0]
                                        testing_set = configuration[1]

                                        neurons_per_layer_str = f"[{'-'.join(map(str, network_configuration))}]"
                                        neural_network = NeuralNetwork(training_set[0], training_set[1], network_configuration, activation_function[0], activation_function[1], seed)
                                        breaking_epoch, training_error = neural_network.backpropagate(digits_vectors, y_values, learning_rate, total_epochs, optimizer, error_function, max_error, parity_results_file, is_adam_optimizer= False, partition= partition_index, neurons_per_layer= neurons_per_layer_str, activation_function= activation_function[0].__name__, activation_beta= 1.0, alpha= alpha)

                                        testing_data_prediction_error = get_prediction_error_for_neural_network(neural_network, testing_set[0], testing_set[1], mean_error)
                                        testing_data_prediction_errors.append(testing_data_prediction_error)

                                        training_accuracy = parity_calculate_accuracy(neural_network, training_set[0], training_set[1])
                                        testing_accuracy = parity_calculate_accuracy(neural_network, testing_set[0], testing_set[1])
                                        accuracy_file.write(f"parity,{activation_function[0].__name__},{optimizer.__name__},{k},{partition_index},{neurons_per_layer_str},{learning_rate},{total_epochs},{training_accuracy},{testing_accuracy}\n")

                                    training_mean_error = np.mean(training_errors)
                                    training_error_std = np.std(training_errors)                            
                                    testing_data_mean_prediction_error = np.mean(testing_data_prediction_errors)
                                    testing_data_prediction_error_std = np.std(testing_data_prediction_errors)
                                    errors_parity_results_file.write(f"{seed},{activation_function[0].__name__},{optimizer.__name__},{k},{neurons_per_layer_str},{1.0},{learning_rate},{alpha},{total_epochs},{training_mean_error},{training_error_std},{testing_data_mean_prediction_error},{testing_data_prediction_error_std}\n")


####################################################### RUN PARITY ##################################################################

    train_and_evaluate_parity_network(gradient_descent_optimizer_with_delta)
    #train_and_evaluate_parity_network(momentum_gradient_descent_optimizer_with_delta)
    #train_and_evaluate_parity_network(adam_optimizer_with_delta)



######################################################### DIGITOS ##################################################################
   
    # Discriminacion de digito:
    # Por cada digito n crea una lista donde todos los valores son 0 excepto por la posicion n
    # Ejemplo: 1 = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    y_values = np.eye(10, dtype=float)
    x_values = np.array(digits_vectors)

    k = 5
    training_testing_pairs = k_cross_validation(k, x_values, y_values)

    digits_config = config['digits']
    network_configurations = digits_config['network_configurations']
    activation_functions = [activation_functions_map[name] for name in digits_config['activation_functions']]
    error_functions = [error_functions_map[name] for name in digits_config['error_functions']]
    epochs = digits_config['epochs']
    learning_rates = digits_config['learning_rates']

    training_errors = []
    training_data_prediction_errors = []
    testing_data_prediction_errors = []

    digits_path = os.path.join(results_data_dir_name, results_files[1])
    errors_digits_path = os.path.join(results_data_dir_name, errors_results_files[1])

    write_header_if_needed(digits_path, f"seed,activation_function,optimizer,partition,neurons_per_layer,beta,learning_rate,alpha,total_epochs,epoch,error_function,error\n")
    write_header_if_needed(errors_digits_path, f"seed,activation_function,optimizer,partitions,neurons_per_layer,beta,learning_rate,alpha,total_epochs,error_function,training_mean_error,training_std_error,testing_mean_error,testing_std_error\n")
    parity_results_file = open(digits_path, "a", newline='')
    errors_parity_results_file = open(errors_digits_path, "a", newline='')

    def train_and_evaluate_digits_network(optimizer):

        if(optimizer == gradient_descent_optimizer_with_delta):
            max_error = 0.01
            for network_configuration in network_configurations:
                for activation_function in activation_functions:
                        for error_function in error_functions:
                            for learning_rate in learning_rates:
                                for total_epochs in epochs:
                                    training_errors = []
                                    testing_data_prediction_errors = []

                                    for partition_index, configuration in enumerate(training_testing_pairs):
                                        training_set = configuration[0]
                                        testing_set = configuration[1]

                                        neurons_per_layer_str = f"[{'-'.join(map(str, network_configuration))}]"
                                        neural_network = NeuralNetwork(training_set[0], training_set[1], network_configuration, activation_function[0], activation_function[1], seed)
                                        breaking_epoch, training_error = neural_network.backpropagate(training_set[0], training_set[1], learning_rate, total_epochs, optimizer, error_function, max_error, parity_results_file, is_adam_optimizer= False, partition= partition_index, neurons_per_layer= neurons_per_layer_str, activation_function= activation_function[0].__name__, activation_beta=1.0)
                                        training_errors.append(training_error)

                                        testing_data_prediction_error = get_prediction_error_for_neural_network(neural_network, testing_set[0], testing_set[1], mean_error)
                                        testing_data_prediction_errors.append(testing_data_prediction_error)

                                        training_accuracy = digits_calculate_accuracy(neural_network, training_set[0], training_set[1])
                                        testing_accuracy = digits_calculate_accuracy(neural_network, testing_set[0], testing_set[1])
                                        accuracy_file.write(f"digits,{activation_function[0].__name__},{optimizer.__name__},{k},{partition_index},{neurons_per_layer_str},{learning_rate},{total_epochs},{training_accuracy},{testing_accuracy}\n")

                                    training_mean_error = np.mean(training_errors)
                                    training_error_std = np.std(training_errors)                            
                                    testing_data_mean_prediction_error = np.mean(testing_data_prediction_errors)
                                    testing_data_prediction_error_std = np.std(testing_data_prediction_errors)    
                                    errors_parity_results_file.write(f"{seed},{activation_function[0].__name__},{optimizer.__name__},{k},{neurons_per_layer_str},{1.0},{learning_rate},{0.0},{total_epochs},{error_function.__name__},{training_mean_error},{training_error_std},{testing_data_mean_prediction_error},{testing_data_prediction_error_std}\n")

        if(optimizer == momentum_gradient_descent_optimizer_with_delta):
            max_error = 1.0
            alpha = 0.9
            for network_configuration in network_configurations:
                for activation_function in activation_functions:
                        for error_function in error_functions:
                            for learning_rate in learning_rates:
                                for total_epochs in epochs:
                                    training_errors = []
                                    testing_data_prediction_errors = []
                                    for partition_index, configuration in enumerate(training_testing_pairs):
                                        training_set = configuration[0]
                                        testing_set = configuration[1]

                                        neurons_per_layer_str = f"[{'-'.join(map(str, network_configuration))}]"
                                        neural_network = NeuralNetwork(training_set[0], training_set[1], network_configuration, activation_function[0], activation_function[1], seed)
                                        breaking_epoch, training_error = neural_network.backpropagate(digits_vectors, y_values, learning_rate, total_epochs, optimizer, error_function, max_error, parity_results_file, is_adam_optimizer= False, partition= partition_index, neurons_per_layer= neurons_per_layer_str, activation_function= activation_function[0].__name__, activation_beta= 1.0, alpha= alpha)
                                        
                                        testing_data_prediction_error = get_prediction_error_for_neural_network(neural_network, testing_set[0], testing_set[1], mean_error)
                                        testing_data_prediction_errors.append(testing_data_prediction_error)

                                        training_accuracy = digits_calculate_accuracy(neural_network, training_set[0], training_set[1])
                                        testing_accuracy = digits_calculate_accuracy(neural_network, testing_set[0], testing_set[1])
                                        accuracy_file.write(f"digits,{activation_function[0].__name__},{optimizer.__name__},{k},{partition_index},{neurons_per_layer_str},{learning_rate},{total_epochs},{training_accuracy},{testing_accuracy}\n")

                                    training_mean_error = np.mean(training_errors)
                                    training_error_std = np.std(training_errors)                            
                                    testing_data_mean_prediction_error = np.mean(testing_data_prediction_errors)
                                    testing_data_prediction_error_std = np.std(testing_data_prediction_errors)
                                    errors_parity_results_file.write(f"{seed},{activation_function[0].__name__},{optimizer.__name__},{k},{neurons_per_layer_str},{1.0},{learning_rate},{alpha},{total_epochs},{training_mean_error},{training_error_std},{testing_data_mean_prediction_error},{testing_data_prediction_error_std}\n")

        if(optimizer == adam_optimizer_with_delta):
            max_error = 1.0
            for network_configuration in network_configurations:
                for activation_function in activation_functions:
                        for error_function in error_functions:
                            for learning_rate in learning_rates:
                                for total_epochs in epochs:
                                    training_errors = []
                                    testing_data_prediction_errors = []
                                    for partition_index, configuration in enumerate(training_testing_pairs):
                                        training_set = configuration[0]
                                        testing_set = configuration[1]

                                        neurons_per_layer_str = f"[{'-'.join(map(str, network_configuration))}]"
                                        neural_network = NeuralNetwork(training_set[0], training_set[1], network_configuration, activation_function[0], activation_function[1], seed)
                                        breaking_epoch, training_error = neural_network.backpropagate(digits_vectors, y_values, learning_rate, total_epochs, optimizer, error_function, max_error, parity_results_file, is_adam_optimizer= False, partition= partition_index, neurons_per_layer= neurons_per_layer_str, activation_function= activation_function[0].__name__, activation_beta= 1.0, alpha= alpha)

                                        testing_data_prediction_error = get_prediction_error_for_neural_network(neural_network, testing_set[0], testing_set[1], mean_error)
                                        testing_data_prediction_errors.append(testing_data_prediction_error)

                                        training_accuracy = digits_calculate_accuracy(neural_network, training_set[0], training_set[1])
                                        testing_accuracy = digits_calculate_accuracy(neural_network, testing_set[0], testing_set[1])
                                        accuracy_file.write(f"digits,{activation_function[0].__name__},{optimizer.__name__},{k},{partition_index},{neurons_per_layer_str},{learning_rate},{total_epochs},{training_accuracy},{testing_accuracy}\n")

                                    training_mean_error = np.mean(training_errors)
                                    training_error_std = np.std(training_errors)                            
                                    testing_data_mean_prediction_error = np.mean(testing_data_prediction_errors)
                                    testing_data_prediction_error_std = np.std(testing_data_prediction_errors)
                                    errors_parity_results_file.write(f"{seed},{activation_function[0].__name__},{optimizer.__name__},{k},{neurons_per_layer_str},{1.0},{learning_rate},{alpha},{total_epochs},{training_mean_error},{training_error_std},{testing_data_mean_prediction_error},{testing_data_prediction_error_std}\n")



###################################################### RUN DIGITS ##################################################################

    train_and_evaluate_digits_network(gradient_descent_optimizer_with_delta)
    #train_and_evaluate_digits_network(momentum_gradient_descent_optimizer_with_delta)
    #train_and_evaluate_digits_network(adam_optimizer_with_delta)
