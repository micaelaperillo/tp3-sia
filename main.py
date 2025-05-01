import os
import numpy as np
import pandas as pd
from typing import List
from neural_network.models.basic_perceptron import Perceptron
from stats import plots_for_exercise_1
from neural_network.activation_functions import step, identity, prime_identity, tanh, prime_tanh, logistic, prime_logistic
from neural_network.error_functions import squared_error
from partition_methods import k_cross_validation
from metric_functions import get_prediction_error

if __name__ == '__main__':
    results_data_dir_name = "output_data"
    if not os.path.exists(results_data_dir_name):
        os.makedirs(results_data_dir_name)

    results_files:List[str] = ["ej1_data.csv", "ej2_data.csv", "ej3_data.csv", "ej4_data.csv"]

    # exercise 1
    first_exercise_results_file = open(os.path.join(results_data_dir_name, results_files[0]), "w", newline='')
    and_x_values:List[List[int]] = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    and_y_values:List[int] = np.array([-1, -1, -1, 1])

    xor_x_values:List[List[int]] = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    xor_y_values:List[int] = np.array([1, 1, -1, -1])

    seed:int = 43
    learning_rates:List[float] = [0.1, 0.05, 0.01]
    epochs:List[int] = [200]

    # by storing the seed we get which weights are being used initially
    first_exercise_results_file.write(f"seed,method,learning_rate,epochs,error\n")

    for learning_rate in learning_rates:
        for epoch_amount in epochs:
            and_simple_perceptron = Perceptron(len(and_x_values[0]), learning_rate, epoch_amount, step)
            and_breaking_epoch, and_training_error = and_simple_perceptron.train(and_x_values, and_y_values, squared_error, 1.0, first_exercise_results_file, "and")

            xor_simple_perceptron = Perceptron(len(xor_x_values[0]), learning_rate, epoch_amount, step)            
            xor_breaking_epoch, xor_training_error = xor_simple_perceptron.train(xor_x_values, xor_y_values, squared_error, 1.0, first_exercise_results_file, "xor")

    plots_for_exercise_1(os.path.join(results_data_dir_name, results_files[0]))

    # exercise 2
    input_data_dir_name = "input_data"
    exercise_2_input_data_filename = "TP3-ej2-conjunto.csv"

    exercise_2_input_data_path= os.path.join(input_data_dir_name, exercise_2_input_data_filename)
    df = pd.read_csv(exercise_2_input_data_path)
    x_values = df[['x1', 'x2', 'x3']].to_numpy()
    y_values = df['y'].to_numpy()

    k = 5
    training_testing_pairs = k_cross_validation(k, x_values, y_values)
    seed:int = 43
    learning_rates:List[float] = [0.1, 0.05, 0.01]
    epochs:List[int] = [200]

    second_exercise_training_results_file = open(os.path.join(results_data_dir_name, results_files[1]), "w", newline='')
    second_exercise_training_results_file.write(f"seed,activation_function,learning_rate,epochs,error\n")
    second_exercise_results_file = open(os.path.join(results_data_dir_name, "ej2_data_errors.csv"), "w", newline='')
    second_exercise_results_file.write(f"seed,activation_function,beta,learning_rate,epochs,error_method,training_mean_error,training_error_std,training_data_mean_prediction_error,training_data_prediction_error_std,testing_data_mean_prediction_error,testing_data_prediction_error_std\n")

    # linear perceptron
    # descendant gradient
    # using identity
    training_errors = []
    training_data_prediction_errors = []
    testing_data_prediction_errors = []
    for learning_rate in learning_rates:
        for epoch_amount in epochs:
            for configuration in training_testing_pairs:
                training_set = configuration[0]
                testing_set = configuration[1]

                perceptron = Perceptron(len(x_values[0]), learning_rate, epoch_amount, identity, prime_identity)
                breaking_epoch, training_error = perceptron.train(training_set[0], training_set[1], squared_error, 0.9, second_exercise_results_file, "identity", True)
                training_errors.append(training_error)

                training_data_prediction_error = get_prediction_error(perceptron, training_set[0], training_set[1], squared_error)
                training_data_prediction_errors.append(training_data_prediction_error)

                testing_data_prediction_error = get_prediction_error(perceptron, testing_set[0], testing_set[1], squared_error)
                testing_data_prediction_errors.append(testing_data_prediction_error)

            training_mean_error = np.mean(training_errors)
            training_error_std = np.std(training_errors)
            training_data_mean_prediction_error = np.mean(training_data_prediction_errors)
            training_data_prediction_error_std = np.std(training_data_prediction_errors)
            testing_data_mean_prediction_error = np.mean(testing_data_prediction_errors)
            testing_data_prediction_error_std = np.std(testing_data_prediction_errors)
            second_exercise_results_file.write(f"{seed},identity,{learning_rate},{1.0},{breaking_epoch},square_error,{training_mean_error},{training_error_std},{training_data_mean_prediction_error},{training_data_prediction_error_std},{testing_data_mean_prediction_error},{testing_data_prediction_error_std}\n")

    # using tanh(x) with b around [0.01, 0.1] to have a valid aproximation to x
    beta_values_for_linear = [0.01, 0.05, 0.1]

    training_errors = []
    training_data_prediction_errors = []
    testing_data_prediction_errors = []
    for learning_rate in learning_rates:
        for epoch_amount in epochs:
            for beta in beta_values_for_linear:
                for configuration in training_testing_pairs:
                    training_set = configuration[0]
                    testing_set = configuration[1]
                    perceptron = Perceptron(len(x_values[0]), learning_rate, epoch_amount, tanh, prime_tanh)
                    breaking_epoch, training_error = perceptron.train(training_set[0], training_set[1], squared_error, 0.9, second_exercise_results_file, f"tanh_linear_b_{beta}", True, beta)
                    training_errors.append(training_error)

                    training_data_prediction_error = get_prediction_error(perceptron, training_set[0], training_set[1], squared_error)
                    training_data_prediction_errors.append(training_data_prediction_error)

                    testing_data_prediction_error = get_prediction_error(perceptron, testing_set[0], testing_set[1], squared_error)
                    testing_data_prediction_errors.append(testing_data_prediction_error)

                training_mean_error = np.mean(training_errors)
                training_error_std = np.std(training_errors)
                training_data_mean_prediction_error = np.mean(training_data_prediction_errors)
                training_data_prediction_error_std = np.std(training_data_prediction_errors)
                testing_data_mean_prediction_error = np.mean(testing_data_prediction_errors)
                testing_data_prediction_error_std = np.std(testing_data_prediction_errors)
                second_exercise_results_file.write(f"{seed},tanh_linear,{beta},{learning_rate},{breaking_epoch},square_error,{training_mean_error},{training_error_std},{training_data_mean_prediction_error},{training_data_prediction_error_std},{testing_data_mean_prediction_error},{testing_data_prediction_error_std}\n")

    #non-linear perceptron
    beta_values_for_non_linear = [1.0]
    training_errors = []
    training_data_prediction_errors = []
    testing_data_prediction_errors = []
    for learning_rate in learning_rates:
        for epoch_amount in epochs:
            for beta in beta_values_for_linear:
                for configuration in training_testing_pairs:
                    training_set = configuration[0]
                    testing_set = configuration[1]
                    perceptron = Perceptron(len(x_values[0]), learning_rate, epoch_amount, tanh, prime_tanh)
                    breaking_epoch, training_error = perceptron.train(training_set[0], training_set[1], squared_error, 0.9, second_exercise_results_file, "tanh_non_linear", True, beta)
                    training_errors.append(training_error)

                    training_data_prediction_error = get_prediction_error(perceptron, training_set[0], training_set[1], squared_error)
                    training_data_prediction_errors.append(training_data_prediction_error)

                    testing_data_prediction_error = get_prediction_error(perceptron, testing_set[0], testing_set[1], squared_error)
                    testing_data_prediction_errors.append(testing_data_prediction_error)

                training_mean_error = np.mean(training_errors)
                training_error_std = np.std(training_errors)
                training_data_mean_prediction_error = np.mean(training_data_prediction_errors)
                training_data_prediction_error_std = np.std(training_data_prediction_errors)
                testing_data_mean_prediction_error = np.mean(testing_data_prediction_errors)
                testing_data_prediction_error_std = np.std(testing_data_prediction_errors)
                second_exercise_results_file.write(f"{seed},tanh_non_linear,{beta},{learning_rate},{breaking_epoch},square_error,{training_mean_error},{training_error_std},{training_data_mean_prediction_error},{training_data_prediction_error_std},{testing_data_mean_prediction_error},{testing_data_prediction_error_std}\n")
        
    #for logistic function
    training_errors = []
    training_data_prediction_errors = []
    testing_data_prediction_errors = []
    for learning_rate in learning_rates:
        for epoch_amount in epochs:
            for beta in beta_values_for_linear:
                for configuration in training_testing_pairs:
                    training_set = configuration[0]
                    testing_set = configuration[1]
                    perceptron = Perceptron(len(x_values[0]), learning_rate, epoch_amount, logistic, prime_logistic)
                    breaking_epoch, training_error = perceptron.train(training_set[0], training_set[1], squared_error, 0.9, second_exercise_results_file, "logistic", True, beta)
                    training_errors.append(training_error)

                    training_data_prediction_error = get_prediction_error(perceptron, training_set[0], training_set[1], squared_error)
                    training_data_prediction_errors.append(training_data_prediction_error)

                    testing_data_prediction_error = get_prediction_error(perceptron, testing_set[0], testing_set[1], squared_error)
                    testing_data_prediction_errors.append(testing_data_prediction_error)

                training_mean_error = np.mean(training_errors)
                training_error_std = np.std(training_errors)
                training_data_mean_prediction_error = np.mean(training_data_prediction_errors)
                training_data_prediction_error_std = np.std(training_data_prediction_errors)
                testing_data_mean_prediction_error = np.mean(testing_data_prediction_errors)
                testing_data_prediction_error_std = np.std(testing_data_prediction_errors)
                second_exercise_results_file.write(f"{seed},logistic_non_linear,{beta},{learning_rate},{breaking_epoch},square_error,{training_mean_error},{training_error_std},{training_data_mean_prediction_error},{training_data_prediction_error_std},{testing_data_mean_prediction_error},{testing_data_prediction_error_std}\n")
