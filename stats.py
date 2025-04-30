import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List

if not os.path.exists("graphs"):
    os.makedirs("graphs")

def plots_for_exercise_1(results_file:str):
    df_for_exercise_1 = pd.read_csv(results_file)
    learning_rates:List[float] = [0.1, 0.05, 0.01]
    for learning_rate in learning_rates: 
        plot_training_error_vs_epoch_for_each_method(df_for_exercise_1, 43, learning_rate)


def plot_training_error_vs_epoch_for_each_method(df, seed:int, learning_rate:float):
    filtered_by_seed = df[df['seed'] == seed]
    filtered_by_learning_rate = filtered_by_seed[filtered_by_seed['learning_rate'] == learning_rate]

    and_training = filtered_by_learning_rate[filtered_by_learning_rate['method'] == 'and']
    xor_training = filtered_by_learning_rate[filtered_by_seed['method'] == 'xor']

    plt.plot(and_training['epochs'], and_training['error'])
    plt.title("Variación del error por épocas")
    plt.xlabel("Época")
    plt.ylabel("Error cuadrático")
    plt.grid(True)
    plt.savefig(f"graphs/and_method_error_vs_epochs_s_{seed}_eta_{learning_rate}.png")
    plt.clf()

    plt.plot(xor_training['epochs'], xor_training['error'])
    plt.title("Variación del error por épocas")
    plt.xlabel("Época")
    plt.ylabel("Error cuadrático")
    plt.grid(True)
    plt.savefig(f"graphs/xor_method_error_vs_epochs_s_{seed}_eta_{learning_rate}.png")
    plt.clf()

if __name__ == '__main__':
    results_files:List[str] = ["ej1_data.csv", "ej2_data.csv", "ej3_data.csv", "ej4_data.csv"]
    plots_for_exercise_1(os.path.join("data", results_files[0]))