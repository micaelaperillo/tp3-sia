import os
import numpy as np
import pandas as pd
from typing import List
from neural_network.models.basic_perceptron import Perceptron
from stats import plots_for_exercise_1
from neural_network.activation_functions import step
from neural_network.error_functions import squared_error

if __name__ == '__main__':
    results_data_dir_name = "output_data"
    if not os.path.exists(results_data_dir_name):
        os.makedirs(results_data_dir_name)

    results_files:List[str] = ["ej1_data.csv", "ej2_data.csv", "ej3_data.csv", "ej4_data.csv"]

    # exercise 1
    first_exercise_results_file = open(os.path.join(results_data_dir_name, results_files[0]), "w", newline='')
    and_data:List[List[int]] = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    and_labels:List[int] = np.array([-1, -1, -1, 1])

    xor_data:List[List[int]] = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    xor_labels:List[int] = np.array([1, 1, -1, -1])

    seed:int = 43
    learning_rates:List[float] = [0.1, 0.05, 0.01]
    epochs:List[int] = [200]

    # by storing the seed we get which weights are being used initially
    first_exercise_results_file.write(f"seed,method,learning_rate,epochs,error\n")

    for learning_rate in learning_rates:
        for epoch_amount in epochs:
            and_simple_perceptron = Perceptron(len(and_data[0]), learning_rate, epoch_amount, step)
            and_breaking_epoch, and_training_error = and_simple_perceptron.train(and_data, and_labels, squared_error, 1.0, first_exercise_results_file, "and")

            xor_simple_perceptron = Perceptron(len(xor_data[0]), learning_rate, epoch_amount, step)            
            xor_breaking_epoch, xor_training_error = xor_simple_perceptron.train(xor_data, xor_labels, squared_error, 1.0, first_exercise_results_file, "xor")

    plots_for_exercise_1(os.path.join(results_data_dir_name, results_files[0]))

    # exercise 2
    input_data_dir_name = "input_data"
    exercise_2_input_data_filename = "TP3-ej2-conjunto.csv"

    exercise_2_input_data_path= os.path.join(input_data_dir_name, exercise_2_input_data_filename)
    df = pd.read_csv(exercise_2_input_data_path)
    x_values = df[['x1', 'x2', 'x3']].to_numpy()
    y_values = df['y'].to_numpy()

