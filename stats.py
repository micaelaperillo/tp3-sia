import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List
import numpy as np
from matplotlib.lines import Line2D


if not os.path.exists("graphs"):
    os.makedirs("graphs")

def plots_for_exercise_1(results_file:str, learning_rates:List[float]):
    df_for_exercise_1 = pd.read_csv(results_file)
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


def graph_decision_boundary(x, y, weights, title = "Frontera de decisión", save_name="decision_boundary"):
    
    colors = ['red' if label == -1 else 'blue' for label in y]

    x1_vals = np.array([xi[0] for xi in x])
    x2_vals = np.array([xi[1] for xi in x])

    bias = weights[0]
    w1 = weights[1]
    w2 = weights[2]

    # Calculo la frontera de decisión: x2 = -(w1/w2)*x1 - (bias/w2)
    x_line = np.linspace(min(x1_vals) - 1, max(x1_vals) + 1, 100)
    if w2 != 0:
        y_line = -(w1 / w2) * x_line - (bias / w2)
    else:
        y_line = np.zeros_like(x_line)

    plt.figure(figsize=(6,6))
    plt.scatter(x1_vals, x2_vals, c=colors, s=100, edgecolors='k')
    plt.plot(x_line, y_line, 'k-')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='True', markerfacecolor='blue', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='False', markerfacecolor='red', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], color='black', label='Frontera de decisión')
    ]

    plt.legend(handles=legend_elements)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('Valor 1')
    plt.ylabel('Valor 2')
    plt.title(title)
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.8)
    plt.xticks(np.arange(-2, 3, 1))
    plt.yticks(np.arange(-2, 3, 1))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.savefig("graphs/" + save_name + ".png")


if __name__ == '__main__':
    results_files:List[str] = ["ej1_data.csv", "ej2_data.csv", "ej3_data.csv", "ej4_data.csv"]
    plots_for_exercise_1(os.path.join("data", results_files[0]))