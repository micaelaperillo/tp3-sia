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
from digits_utils import add_noise_to_digits, plot_digit_images

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

    plot_digit_images(digits_vectors, cols= 10, title="Original Digits")
    digits_with_noise = add_noise_to_digits(digits_vectors, noise_level=0.05)
    plot_digit_images(digits_with_noise, cols= 10, title="Digits with Noise")
