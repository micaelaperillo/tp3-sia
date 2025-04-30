import os
import numpy as np
from typing import List
from neural_network.models.basic_perceptron import Perceptron
from stats import plots_for_exercise_1

if __name__ == '__main__':
    if not os.path.exists("data"):
        os.makedirs("data")

    results_files:List[str] = ["ej1_data.csv", "ej2_data.csv", "ej3_data.csv", "ej4_data.csv"]

    first_exercise_results_file = open(os.path.join("data", results_files[0]), "w", newline='')
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
            and_simple_perceptron = Perceptron(len(and_data[0]), learning_rate, epoch_amount)
            and_breaking_epoch, and_training_error = and_simple_perceptron.linear_training(and_data, and_labels, first_exercise_results_file, "and")

            xor_simple_perceptron = Perceptron(len(xor_data[0]), learning_rate, epoch_amount)            
            xor_breaking_epoch, xor_training_error = xor_simple_perceptron.linear_training(xor_data, xor_labels, first_exercise_results_file, "xor")

    plots_for_exercise_1(os.path.join("data", results_files[0]))
