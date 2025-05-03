import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from stats_utils import load_ej1_weights_from_csv, load_ej1_weights_from_csv, load_ej2_weights_from_csv


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
    plt.savefig("graphs/decision boundaries/" + save_name + ".png")


def plot_regression_plane(x, y, weights, title = "Plano de regresión", save_name="regression_plane"):
    x1 = np.array([xi[0] for xi in x])
    x2 = np.array([xi[1] for xi in x])
    x3 = np.array([xi[2] for xi in x])
    bias = weights[0]
    w1 = weights[1]
    w2 = weights[2]
    w3 = weights[3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x1, x2, x3, c=y, cmap='viridis', marker='o', label='Datos reales')

    x1_range = np.linspace(min(x1), max(x1), 10)
    x2_range = np.linspace(min(x2), max(x2), 10)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    x3_grid = -(w1 * x1_grid + w2 * x2_grid + bias) / w3
    ax.plot_surface(x1_grid, x2_grid, x3_grid, alpha=0.5, color='orange', label='Plano de regresión')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.title(title)
    plt.savefig("graphs/decision boundaries/" + save_name + ".png")


def animate_decision_boundary_2D(x, y, method, learning_rate, title = "Animación frontera de decisión", save_name="animated_decision_boundary"):
    weights_list = load_ej1_weights_from_csv("output_data/ej1_data.csv", method, learning_rate)

    fig, ax = plt.subplots()

    x_vals = np.linspace(x[:, 0].min(), x[:, 0].max(), 200)
    line, = ax.plot([], [], 'k-')
    
    def update(frame):
        w = weights_list[frame]
        if w[2] == 0:
            line.set_data([], [])  
        else:
            y_vals = -(w[1] * x_vals + w[0]) / w[2]
            line.set_data(x_vals, y_vals)
        return line,

    ax.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    plt.xlabel('Valor 1')
    plt.ylabel('Valor 2')
    plt.title(title)
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.8)
    plt.xticks(np.arange(-2, 3, 1))
    plt.yticks(np.arange(-2, 3, 1))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    ani = animation.FuncAnimation(fig, update, frames=len(weights_list), interval=700, repeat=False)
    ani.save("graphs/animations/" + save_name + ".gif", writer="pillow", fps=2)


def animate_regression_plane_3D(activation_function, learning_rate, beta, partition, title="Animación plano de regresión", save_name="animated_regression_plane"):

    input_data_dir_name = "input_data"
    exercise_2_input_data_filename = "TP3-ej2-conjunto.csv"

    exercise_2_input_data_path= os.path.join(input_data_dir_name, exercise_2_input_data_filename)
    df = pd.read_csv(exercise_2_input_data_path)
    x = df[['x1', 'x2', 'x3']].to_numpy()
    y = df['y'].to_numpy()

    weights_list = load_ej2_weights_from_csv("output_data/ej2_data.csv", activation_function, learning_rate, beta, partition)
    
    x1 = np.array([xi[0] for xi in x])
    x2 = np.array([xi[1] for xi in x])
    x3 = np.array([xi[2] for xi in x])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x1, x2, x3, c=y, cmap='viridis', marker='o')

    x1_range = np.linspace(min(x1), max(x1), 20)
    x2_range = np.linspace(min(x2), max(x2), 20)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    plane = [ax.plot_surface(x1_grid, x2_grid, np.zeros_like(x1_grid), alpha=0.5, color='orange')]

    ax.set_xlim(min(x1), max(x1))
    ax.set_ylim(min(x2), max(x2))
    ax.set_zlim(min(x3), max(x3))

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.title(title)

    def update(frame):
        plane[0].remove()

        weights = weights_list[frame]
        bias, w1, w2, w3 = weights

        if w3 == 0:
            z = np.zeros_like(x1_grid)
        else:
            z = -(w1 * x1_grid + w2 * x2_grid + bias) / w3

        plane[0] = ax.plot_surface(x1_grid, x2_grid, z, alpha=0.5, color='orange')
        return plane[0],

    ani = animation.FuncAnimation(fig, update, frames=len(weights_list), interval=500, repeat=False)
    ani.save("graphs/animations/" + save_name + ".gif", writer='pillow', fps=2)



if __name__ == '__main__':
    #results_files:List[str] = ["ej1_data.csv", "ej2_data.csv", "ej3_data.csv", "ej4_data.csv"]
    #plots_for_exercise_1(os.path.join("data", results_files[0]))
    
    #animate_decision_boundary_2D(np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]]), np.array([-1, -1, -1, 1]), "and", 0.0001, "Frontera de decisión AND", "animated_and_decision_boundary")
    #animate_decision_boundary_2D(np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]]), np.array([1, 1, -1, -1]), "xor", 0.0001, "Frontera de decisión XOR", "animated_xor_decision_boundary")
    animate_regression_plane_3D("identity", 0.0001, 1.0, 1, "Plano de regresión con función identidad", "animated_identity_regression_plane_1partition")

