import os
import numpy as np
import pandas as pd
from typing import List
from neural_network.models.basic_perceptron import Perceptron
from metrics_and_stats.stats import plots_for_exercise_1
from neural_network.activation_functions import step, identity, prime_identity, tanh, prime_tanh, logistic, prime_logistic
from neural_network.error_functions import squared_error, mean_error
from neural_network.partition_methods import k_cross_validation,stratified_k_cross_validation
from metric_functions import get_prediction_error_for_perceptron
from neural_network.scale_functions import ScaleFunctions
from neural_network.optimizers import rosenblatt_optimizer, gradient_descent_optimizer

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
    learning_rates:List[float] = [0.0001, 0.00005, 0.00001]
    epochs:List[int] = [200]

    # by storing the seed we get which weights are being used initially
    first_exercise_results_file.write(f"seed,partition,weights,method,beta,learning_rate,total_epochs,epoch,error\n")

    for learning_rate in learning_rates:
        for epoch_amount in epochs:
            and_simple_perceptron = Perceptron(len(and_x_values[0]), step)
            and_breaking_epoch, and_training_error = and_simple_perceptron.train(and_x_values, and_y_values, learning_rate, epoch_amount, rosenblatt_optimizer, squared_error, 1.0, first_exercise_results_file, "and")

            xor_simple_perceptron = Perceptron(len(xor_x_values[0]), step)            
            xor_breaking_epoch, xor_training_error = xor_simple_perceptron.train(xor_x_values, xor_y_values, learning_rate, epoch_amount, rosenblatt_optimizer, squared_error, 1.0, first_exercise_results_file, "xor")

    plots_for_exercise_1(os.path.join(results_data_dir_name, results_files[0]), learning_rates)

    # exercise 2
    input_data_dir_name = "input_data"
    exercise_2_input_data_filename = "TP3-ej2-conjunto.csv"

    exercise_2_input_data_path= os.path.join(input_data_dir_name, exercise_2_input_data_filename)
    df = pd.read_csv(exercise_2_input_data_path)
    x_values = df[['x1', 'x2', 'x3']].to_numpy()
    y_values = df['y'].to_numpy()

    y_min = np.min(y_values)
    y_max = np.max(y_values)
    scale_functions = ScaleFunctions(y_min, y_max)

    # Escalo al rango [0, 1] para la función logística y al rango [-1, 1] para la función tanh
    y_values_scaled_logistic = scale_functions.scale_logistic(y_values)
    y_values_scaled_tanh = scale_functions.scale_tanh(y_values)
    x_values_scaled_logistic = scale_functions.scale_logistic(x_values)
    x_values_scaled_tanh = scale_functions.scale_tanh(x_values)

    k = 5
    training_testing_pairs_lineal = k_cross_validation(k, x_values_scaled_logistic, y_values_scaled_logistic)
    training_testing_pairs_logistic = k_cross_validation(k, x_values_scaled_logistic, y_values_scaled_logistic)
    training_testing_pairs_tanh = k_cross_validation(k, x_values_scaled_tanh, y_values_scaled_tanh)
    seed:int = 43
    learning_rates:List[float] = [1,0.5,0.01,0.0001, 0.00005, 0.00001, 0.000001]
    epochs:List[int] = [1000]

    second_exercise_training_results_file = open(os.path.join(results_data_dir_name, results_files[1]), "w", newline='')
    second_exercise_training_results_file.write(f"seed,partition,weights,activation_function,beta,learning_rate,total_epochs,epoch,error\n")
    second_exercise_results_file = open(os.path.join(results_data_dir_name, "ej2_data_errors.csv"), "w", newline='')
    second_exercise_results_file.write(f"seed,activation_function,beta,learning_rate,epochs,error_method,training_mean_error,training_error_std,training_data_mean_prediction_error,training_data_prediction_error_std,testing_data_mean_prediction_error,testing_data_prediction_error_std\n")

    # linear perceptron
    # descendant gradient
    # using identity
    for learning_rate in learning_rates:
        for epoch_amount in epochs:
            training_errors = []
            training_data_prediction_errors = []
            testing_data_prediction_errors = []
            for partition_index, configuration in enumerate(training_testing_pairs_lineal):
                training_set = configuration[0]
                testing_set = configuration[1]

                perceptron = Perceptron(len(x_values[0]), identity, prime_identity)
                breaking_epoch, training_error = perceptron.train(training_set[0], training_set[1], learning_rate, epoch_amount,gradient_descent_optimizer,mean_error, 0.1, second_exercise_training_results_file, "identity", True,1.0,partition_index+1,descale_fun = scale_functions.descale_logistic)
                training_errors.append(training_error)

                training_data_prediction_error = get_prediction_error_for_perceptron(perceptron, training_set[0], training_set[1], mean_error,1, descale_fun = scale_functions.descale_logistic)
                training_data_prediction_errors.append(training_data_prediction_error)

                testing_data_prediction_error = get_prediction_error_for_perceptron(perceptron, testing_set[0], testing_set[1], mean_error,1, descale_fun = scale_functions.descale_logistic)
                testing_data_prediction_errors.append(testing_data_prediction_error)

            training_mean_error = np.mean(training_errors)
            training_error_std = np.std(training_errors)
            training_data_mean_prediction_error = np.mean(training_data_prediction_errors)
            training_data_prediction_error_std = np.std(training_data_prediction_errors)
            testing_data_mean_prediction_error = np.mean(testing_data_prediction_errors)
            testing_data_prediction_error_std = np.std(testing_data_prediction_errors)
            second_exercise_results_file.write(f"{seed},identity,{1.0},{learning_rate},{breaking_epoch},mean_error,{training_mean_error},{training_error_std},{training_data_mean_prediction_error},{training_data_prediction_error_std},{testing_data_mean_prediction_error},{testing_data_prediction_error_std}\n")

    # using tanh(x) with b around [0.01, 0.1] to have a valid aproximation to x
    beta_values = [0.01, 0.05, 0.1,1,5,10,50]

    for learning_rate in learning_rates:
        for epoch_amount in epochs:
            for beta in beta_values:
                training_errors = []
                training_data_prediction_errors = []
                testing_data_prediction_errors = []
                for partition_index,configuration in enumerate(training_testing_pairs_tanh):
                    training_set = configuration[0]
                    testing_set = configuration[1]
                    perceptron = Perceptron(len(x_values[0]), tanh, prime_tanh)
                    breaking_epoch, training_error = perceptron.train(training_set[0], training_set[1], learning_rate, epoch_amount, gradient_descent_optimizer,mean_error, 0.1, second_exercise_training_results_file, f"tanh", True, beta, partition_index+1,descale_fun=scale_functions.descale_tanh)
                    training_errors.append(training_error)

                    training_data_prediction_error = get_prediction_error_for_perceptron(perceptron, training_set[0], training_set[1], mean_error, descale_fun = scale_functions.descale_tanh)
                    training_data_prediction_errors.append(training_data_prediction_error)

                    testing_data_prediction_error = get_prediction_error_for_perceptron(perceptron, testing_set[0], testing_set[1], mean_error, descale_fun = scale_functions.descale_tanh)
                    testing_data_prediction_errors.append(testing_data_prediction_error)

                training_mean_error = np.mean(training_errors)
                training_error_std = np.std(training_errors)
                training_data_mean_prediction_error = np.mean(training_data_prediction_errors)
                training_data_prediction_error_std = np.std(training_data_prediction_errors)
                testing_data_mean_prediction_error = np.mean(testing_data_prediction_errors)
                testing_data_prediction_error_std = np.std(testing_data_prediction_errors)
                second_exercise_results_file.write(f"{seed},tanh,{beta},{learning_rate},{breaking_epoch},mean_error,{training_mean_error},{training_error_std},{training_data_mean_prediction_error},{training_data_prediction_error_std},{testing_data_mean_prediction_error},{testing_data_prediction_error_std}\n")

    ##non-linear perceptron
    #beta_values_for_non_linear = []
    #for learning_rate in learning_rates:
    #    for epoch_amount in epochs:
    #        for beta in beta_values_for_non_linear:
    #            training_errors = []
    #            training_data_prediction_errors = []
    #            testing_data_prediction_errors = []
    #            for partition_index,configuration in enumerate(training_testing_pairs_tanh):
    #                training_set = configuration[0]
    #                testing_set = configuration[1]
    #                perceptron = Perceptron(len(x_values[0]), tanh, prime_tanh)
    #                breaking_epoch, training_error = perceptron.train(training_set[0], training_set[1], learning_rate, epoch_amount, gradient_descent_optimizer, mean_error, 0.1, second_exercise_training_results_file, "tanh_non_linear", True, beta, partition_index+1)
    #                training_errors.append(training_error)
    #
    #                training_data_prediction_error = get_prediction_error_for_perceptron(perceptron, training_set[0], training_set[1], mean_error, descale_fun = scale_functions.descale_tanh)
    #                training_data_prediction_errors.append(training_data_prediction_error)
    #
    #                testing_data_prediction_error = get_prediction_error_for_perceptron(perceptron, testing_set[0], testing_set[1], mean_error, descale_fun = scale_functions.descale_tanh)
    #                testing_data_prediction_errors.append(testing_data_prediction_error)
    #
    #            training_mean_error = np.mean(training_errors)
    #            training_error_std = np.std(training_errors)
    #            training_data_mean_prediction_error = np.mean(training_data_prediction_errors)
    #            training_data_prediction_error_std = np.std(training_data_prediction_errors)
    #            testing_data_mean_prediction_error = np.mean(testing_data_prediction_errors)
    #            testing_data_prediction_error_std = np.std(testing_data_prediction_errors)
    #            second_exercise_results_file.write(f"{seed},tanh_non_linear,{beta},{learning_rate},{breaking_epoch},square_error,{training_mean_error},{training_error_std},{training_data_mean_prediction_error},{training_data_prediction_error_std},{testing_data_mean_prediction_error},{testing_data_prediction_error_std}\n")
        
    #for logistic function
    for learning_rate in learning_rates:
        for epoch_amount in epochs:
            for beta in beta_values:
                training_errors = []
                training_data_prediction_errors = []
                testing_data_prediction_errors = []
                for partition_index,configuration in enumerate(training_testing_pairs_logistic):
                    training_set = configuration[0]
                    testing_set = configuration[1]
                    perceptron = Perceptron(len(x_values[0]), logistic, prime_logistic)
                    breaking_epoch, training_error = perceptron.train(training_set[0], training_set[1], learning_rate, epoch_amount, gradient_descent_optimizer,mean_error, 0.1, second_exercise_training_results_file, "logistic", True, beta, partition_index+1,descale_fun=scale_functions.descale_logistic)
                    training_errors.append(training_error)

                    training_data_prediction_error = get_prediction_error_for_perceptron(perceptron, training_set[0], training_set[1], mean_error, descale_fun = scale_functions.descale_logistic)
                    training_data_prediction_errors.append(training_data_prediction_error)

                    testing_data_prediction_error = get_prediction_error_for_perceptron(perceptron, testing_set[0], testing_set[1], mean_error, descale_fun = scale_functions.descale_logistic)
                    testing_data_prediction_errors.append(testing_data_prediction_error)

                training_mean_error = np.mean(training_errors)
                training_error_std = np.std(training_errors)
                training_data_mean_prediction_error = np.mean(training_data_prediction_errors)
                training_data_prediction_error_std = np.std(training_data_prediction_errors)
                testing_data_mean_prediction_error = np.mean(testing_data_prediction_errors)
                testing_data_prediction_error_std = np.std(testing_data_prediction_errors)
                second_exercise_results_file.write(f"{seed},logistic,{beta},{learning_rate},{breaking_epoch},mean_error,{training_mean_error},{training_error_std},{training_data_mean_prediction_error},{training_data_prediction_error_std},{testing_data_mean_prediction_error},{testing_data_prediction_error_std}\n")