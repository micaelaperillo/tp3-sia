import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from metrics_and_stats.stats_utils import load_ej1_animation_weights_from_csv, load_ej2_animation_weights_from_csv, get_ej1_data_xy, get_ej2_data_xy, load_ej1_last_weights_from_csv, load_ej2_last_weights_from_csv, get_save_name, get_title


if not os.path.exists("graphs"):
    os.makedirs("graphs")

def plots_for_exercise_1(results_file:str, learning_rates:List[float]):
    df_for_exercise_1 = pd.read_csv(results_file)
    for learning_rate in learning_rates: 
        plot_training_error_vs_epoch_for_each_method(df_for_exercise_1, 43, learning_rate, "and", "Error Cuadrático")
        plot_training_error_vs_epoch_for_each_method(df_for_exercise_1, 43, learning_rate, "xor", "Error cuadrático")


def plot_train_vs_test_error(df, seed: int, learning_rate: float, method: str,
                             error_function_name: str, beta: float = 1.0):
    # Filtrado por parámetros
    df_filt = df[(df['seed'] == seed) &
                 (df['learning_rate'] == learning_rate) &
                 (df['beta'] == beta) &
                 (df['activation_function'] == method)]

    if df_filt.empty:
        print(f"[WARN] No hay datos para seed={seed}, eta={learning_rate}, beta={beta}, method={method}")
        return

    # Agrupado por época
    df_grouped = df_filt.groupby('epoch').agg(
        train_mean=('error', 'mean'),
        test_mean=('test_error', 'mean'),
        train_std=('error', 'std'),
        test_std=('test_error', 'std')
    ).reset_index()

    # Filtrar épocas para las que se quiere mostrar barras de error (cada 1000)
    df_errorbars = df_grouped[df_grouped['epoch'] % 1000 == 0]

    plt.figure()
    
    # Graficar líneas principales
    plt.plot(df_grouped['epoch'], df_grouped['train_mean'], label='Training Error', color='blue')
    plt.plot(df_grouped['epoch'], df_grouped['test_mean'], label='Testing Error', color='orange')

    # Agregar barras de error solo cada 1000 épocas
    plt.errorbar(df_errorbars['epoch'], df_errorbars['train_mean'], yerr=df_errorbars['train_std'],
                 fmt='o', color='blue', capsize=3)
    plt.errorbar(df_errorbars['epoch'], df_errorbars['test_mean'], yerr=df_errorbars['test_std'],
                 fmt='s', color='orange', capsize=3)

    plt.title("Error de entrenamiento vs test (promediado por época)")
    plt.xlabel("Época")
    plt.ylabel(error_function_name)
    plt.grid(True)
    plt.legend()

    out_dir = os.path.join("graphs", "errors", "train_vs_test")
    os.makedirs(out_dir, exist_ok=True)

    filename = f"{method}_train_vs_test_s_{seed}_eta_{learning_rate}_beta{beta}_{error_function_name}.png"
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

def plot_training_error_vs_epoch_for_each_method(df, seed: int, learning_rate: float, method: str, error_function_name: str, beta: float = 1.0):
    df_filt = df[(df['seed'] == seed) &
                 (df['learning_rate'] == learning_rate) &
                 (df['beta'] == beta)]

    if method in ("and", "xor"):
        df_filt = df_filt[df_filt['method'] == method]
    else:
        df_filt = df_filt[df_filt['activation_function'] == method]

    df_filt['epoch_group'] = (df_filt['epoch'] ) 
    df_grouped = df_filt.groupby('epoch_group').agg(
        error_mean=('error', 'mean'),
        error_std=('error', 'std')
    ).reset_index()

    plt.figure()
    plt.errorbar(df_grouped['epoch_group'], df_grouped['error_mean'],
                 yerr=df_grouped['error_std'], capsize=3, label='Error medio')
    plt.title("Error por época")
    plt.xlabel("Época")
    plt.ylabel(error_function_name)
    plt.grid(True)
    #plt.legend()

    if method not in ("and", "xor"):
        # Agregar etiqueta con el último valor (última fila del DataFrame agrupado)
        last_row = df_grouped.iloc[-1]
        label_text = f"Error: {last_row['error_mean']:.4f} ± {last_row['error_std']:.4f}"
        plt.text(0.98, 0.95, label_text, transform=plt.gca().transAxes,
                 fontsize=9, ha='right', va='top',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    out_dir = os.path.join("graphs", "errors")
    os.makedirs(out_dir, exist_ok=True)

    filename = f"{method}_error_vs_epochs_s_{seed}_eta_{learning_rate}_beta{beta}_{error_function_name}_std.png"
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def plot_errors_vs_epochs_multiple_configs(
    df, seed: int, error_function_name: str,
    config_list: list
):
    """
    Grafica curvas de error medio vs época para múltiples configuraciones de
    (activation_function, learning_rate, beta).

    Parámetros:
    - df: DataFrame con los datos
    - seed: valor de semilla fijo para filtrar
    - error_function_name: nombre para el eje Y
    - config_list: lista de tuplas (activation_function, learning_rate, beta)
    """

    plt.figure()

    for activation, lr, beta in config_list:
        df_filt = df[
            (df['seed'] == seed) &
            (df['learning_rate'] == lr) &
            (df['beta'] == beta) &
            (df['activation_function'] == activation)
        ].copy()

        if 'epoch' not in df_filt.columns or 'error' not in df_filt.columns:
            raise ValueError("El DataFrame debe contener columnas 'epoch' y 'error'")

        df_filt['epoch_group'] = (df_filt['epoch'] // 500) * 500
        df_grouped = df_filt.groupby('epoch_group').agg(
            error_mean=('error', 'mean'),
            error_std=('error', 'std')
        ).reset_index()

        label = f"{activation} (η={lr}, β={beta})"

        plt.plot(df_grouped['epoch_group'], df_grouped['error_mean'],
                 label=label)

    plt.title(f"Comparación de errores por época")
    plt.xlabel("Época")
    plt.ylabel(error_function_name)
    plt.grid(True)
    plt.legend()

    out_dir = os.path.join("graphs", "errors", "comparison")
    os.makedirs(out_dir, exist_ok=True)

    config_labels = "_".join([f"{a}_eta{lr}_b{b}" for a, lr, b in config_list])
    filename = f"compare_errors_seed{seed}_{config_labels}_{error_function_name}.png"
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def heat_map(df, method: str):
    # Elegir función de activación que te interese visualizar
    activation = method

    # Filtrar por función de activación
    df_filt = df[df["activation_function"] == activation]

    # Crear pivot table: filas = beta, columnas = learning_rate, valores = training_mean_error
    heatmap_data = df_filt.pivot_table(
        index="beta",
        columns="learning_rate",
        values="training_mean_error"
    )

    # Ordenar ejes para que el gráfico quede prolijo
    heatmap_data = heatmap_data.sort_index().sort_index(axis=1)

    # Graficar heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={'label': 'Error Medio'},
        vmin=0,        
        vmax=60        
    )
    plt.title(f"HeatMap del error medio para {activation}")
    plt.xlabel("Tasa de Aprendizaje")
    plt.ylabel("Beta")
    plt.tight_layout()

    out_dir = os.path.join("graphs", "errors", "heatmap")
    os.makedirs(out_dir, exist_ok=True)

    filename = f"{method}_heatmap_error_vs_lr_beta.png"
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight')
    plt.close()


def plot_training_error_curves_by_learning_rate(df, seed: int, method: str, error_function_name: str, beta: float = 1.0):
    # Filtrar según seed, método y beta
    df_filt = df[(df['seed'] == seed) & (df['beta'] == beta)]

    if method in ("and", "xor"):
        df_filt = df_filt[df_filt['method'] == method]
    else:
        df_filt = df_filt[df_filt['activation_function'] == method]

    if 'epoch' not in df_filt.columns:
        raise ValueError("La columna 'epoch' no está presente en el DataFrame.")

    df_filt['epoch_group'] = (df_filt['epoch'] // 100) * 100
    learning_rates = sorted(df_filt['learning_rate'].unique())

    plt.figure(figsize=(10, 6))  # Aumentar tamaño para dejar espacio a la derecha

    for lr in learning_rates:
        df_lr = df_filt[df_filt['learning_rate'] == lr]
        df_grouped = df_lr.groupby('epoch_group').agg(
            error_mean=('error', 'mean')
        ).reset_index()

        plt.plot(df_grouped['epoch_group'], df_grouped['error_mean'],
                 label=f"η = {lr}", alpha=0.9, linewidth=2)

    if method == "identity":
            plt.title(f"Error en función de las épocas - Método: {method}")
    else:
        plt.title(f"Error en función de las épocas - Método: {method}, β={beta}")
    plt.xlabel("Época")
    plt.ylabel(error_function_name)
    plt.grid(True)

    # Mover leyenda fuera del gráfico
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ajustar layout para dejar espacio

    out_dir = os.path.join("graphs", "errors", "comparative")
    os.makedirs(out_dir, exist_ok=True)

    filename = f"multi_lr_{method}_s{seed}_b{beta}_{error_function_name}.png"
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight')  # Guardar sin recortar la leyenda
    plt.close()

def plots_for_exercise_2(results_file:str, error_file:str):
    df_results = pd.read_csv(results_file)
    df_errors = pd.read_csv(error_file)
    learning_rates:List[float] = [1,0.5,0.01,0.001,0.0001, 0.00005, 0.00001, 0.000001]
    beta_values = [0.01, 0.05, 0.1,1,5,10,50]
    plot_linear_perceptron_errors_for_activation_function(df_errors, 43,"identity")
    plot_linear_perceptron_errors_for_activation_function(df_errors, 43,"tanh")
    plot_linear_perceptron_errors_for_activation_function(df_errors, 43,"logistic")

    plot_training_error_curves_by_learning_rate(df_results, 43, "identity", "Error promedio")

    heat_map(df_errors, "tanh")
    heat_map(df_errors, "logistic")

    plot_errors_vs_epochs_multiple_configs(
    df=df_results,
    seed=43,
    error_function_name="mean_error",
    config_list=[
        ("identity", 0.5, 1.0),
        ("tanh", 0.01, 5),
        ("logistic", 0.001, 50.0)
        ]
    )

    plot_train_vs_test_error(df_results, 43, 0.01, "tanh", "Error promedio",5)
    plot_train_vs_test_error(df_results, 43, 0.001, "logistic", "Error promedio",50)
    plot_train_vs_test_error(df_results, 43, 0.5, "identity", "Error promedio")

    for beta in beta_values:
        plot_percepton_errors_by_learning_rate(df_errors, beta, 43, "tanh")
        plot_percepton_errors_by_learning_rate(df_errors, beta, 43, "logistic")

    for learning_rate in learning_rates:
        plot_linear_perceptron_errors_for_different_beta_by_learning_rate(df_errors, learning_rate, 43,"tanh")
        plot_linear_perceptron_errors_for_different_beta_by_learning_rate(df_errors, learning_rate, 43,"logistic")

        plot_training_error_vs_epoch_for_each_method(df_results, 43, learning_rate, "identity", "Error promedio",1.0)
        #plot_training_error_vs_epoch_for_each_method(df_results, 43, learning_rate, "tanh_linear_b_0.1", "Error promedio")
        #plot_training_error_vs_epoch_for_each_method(df_results, 43, learning_rate, "tanh_linear_b_0.01", "Error promedio")
        #plot_training_error_vs_epoch_for_each_method(df_results, 43, learning_rate, "tanh_linear_b_0.05", "Error promedio")
        plot_training_error_vs_epoch_for_each_method(df_results, 43, learning_rate, "tanh", "Error promedio",5.0)
        plot_training_error_vs_epoch_for_each_method(df_results, 43, learning_rate, "logistic", "Error promedio",50.0)

def plot_linear_perceptron_errors_for_activation_function(df, seed:int, activation_function:str):
    filtered = df[(df['seed'] == seed) & (df['activation_function'] == activation_function)]

    # Agrupar por learning rate para promediar y evitar múltiples barras por valor
    grouped = filtered.groupby('learning_rate', as_index=False)[[
    'training_mean_error',
    'training_error_std',
    'training_data_mean_prediction_error',
    'training_data_prediction_error_std',
    'testing_data_mean_prediction_error',
    'testing_data_prediction_error_std'
    ]].mean()

    out_dir = os.path.join("graphs", "errors")
    os.makedirs(out_dir, exist_ok=True)


    plt.bar(grouped['learning_rate'].astype(str), grouped['training_mean_error'], 
            yerr=grouped['training_error_std'], capsize=5, color='skyblue', alpha=0.8)
    plt.xlabel("Tasa de aprendizaje")
    plt.ylabel("Error medio de entrenamiento")
    plt.title(f"Error medio para diferentes tasas de aprendizaje")
    plt.grid(True, axis='y', linestyle="--", alpha=0.7)
    plt.savefig(f"graphs/{activation_function}_training_errors_for_learning_rates.png")
    plt.clf()

    plt.bar(grouped['learning_rate'].astype(str), grouped['training_data_mean_prediction_error'], 
            yerr=grouped["training_data_prediction_error_std"], capsize=5, color='red', alpha=0.8)
    plt.xlabel("Tasa de aprendizaje")
    plt.ylabel("Error medio de predicción de datos de entrenamiento")
    plt.title(f"Error medio para diferentes tasas de aprendizaje")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"graphs/{activation_function}_training_data_prediction_errors_for_learning_rates.png")
    plt.clf()

    plt.bar(grouped['learning_rate'].astype(str), grouped['testing_data_mean_prediction_error'], 
            yerr=grouped["testing_data_prediction_error_std"], capsize=5, color='green', alpha=0.8)
    plt.xlabel("Tasa de aprendizaje")
    plt.ylabel("Error medio de predicción de datos de prueba")
    plt.title(f"Error medio para diferentes tasas de aprendizaje")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"graphs/{activation_function}_testing_data_prediction_errors_for_learning_rates.png")
    plt.clf()



def get_test_errors_by_partition(csv_path, k):
     # Leer el CSV con precauciones
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["seed"] != "seed"]

    for col in ["seed", "total_partitions", "partition", "total_epochs", "epoch"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["beta", "learning_rate", "error", "test_error"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filtrar por el valor de k deseado
    df_k = df[df["total_partitions"] == k]

    # Para cada partición, quedarnos solo con la fila de la última época
    last_epoch_df = df_k.sort_values("epoch").groupby("partition").tail(1)

    # Seleccionar columnas relevantes
    result = last_epoch_df[["partition", "test_error"]].copy()

    # Agregar los folds usados como entrenamiento
    all_folds = set(range(1, k + 1))
    result["train_folds"] = result["partition"].apply(lambda p: sorted(all_folds - {p}))

    #print(result)
    return result

def plot_generalization_vs_partitions():
    csv_path = "output_data/ej2_data-2025-05-13-09-00-33.csv"

    # Cargar el CSV evitando problemas de tipo
    df = pd.read_csv(csv_path, low_memory=False)

    # Eliminar encabezados repetidos si se colaron como filas
    df = df[df["seed"] != "seed"]

    # Asegurar tipos correctos
    for col in ["seed", "total_partitions", "partition", "total_epochs", "epoch"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["beta", "learning_rate", "error", "test_error"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parsear los pesos de string a array de float
    df["weights"] = df["weights"].apply(lambda w: np.fromstring(str(w).strip("[]"), sep=" "))

    # Obtener todos los valores únicos de total_partitions (k)
    ks = df['total_partitions'].dropna().unique()

    for k in ks:
        df_k = df[df['total_partitions'] == k]

        plt.figure(figsize=(12, 8))  # Sin constrained_layout para evitar warning

        # Promedio y desviación estándar sobre todas las particiones para este k
        df_mean = df_k.groupby("epoch")[["error", "test_error"]].mean().reset_index()
        df_std = df_k.groupby("epoch")[["error", "test_error"]].std().reset_index()

        # Graficar solo el promedio de los errores de train y test
        plt.plot(df_mean["epoch"], df_mean["error"], label="Train (avg)", color="blue", linewidth=2, linestyle="--")
        plt.plot(df_mean["epoch"], df_mean["test_error"], label="Test (avg)", color="orange", linewidth=2)

        # Graficar la desviación estándar
        plt.fill_between(df_mean["epoch"], df_mean["error"] - df_std["error"], df_mean["error"] + df_std["error"], color="black", alpha=0.2)
        plt.fill_between(df_mean["epoch"], df_mean["test_error"] - df_std["test_error"], df_mean["test_error"] + df_std["test_error"], color="red", alpha=0.2)

        # Filtrar las épocas cada 500 para evitar que se plotee demasiada información
        plt.xticks(df_mean["epoch"][::500])

        out_dir = os.path.join("graphs", "errors", "generalization")
        os.makedirs(out_dir, exist_ok=True)

        plt.title(f'Evolución del error promedio y desviación estándar para k = {k}')
        plt.xlabel('Épocas')
        plt.ylabel('Error')
        plt.legend(fontsize="small")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"generalization_for_best_perceptron_with_k_{k}_avg_and_std.png"))
        plt.close()


def plot_generalization_all_ks():
    csv_path = "output_data/ej2_data-2025-05-13-09-00-33.csv"

    # Cargar el CSV y limpiar
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["seed"] != "seed"]

    # Conversión de tipos
    for col in ["seed", "total_partitions", "partition", "total_epochs", "epoch"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["beta", "learning_rate", "error", "test_error"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parsear pesos si están presentes
    if "weights" in df.columns:
        df["weights"] = df["weights"].apply(lambda w: np.fromstring(str(w).strip("[]"), sep=" "))

    ks = sorted(df['total_partitions'].dropna().unique())

    plt.figure(figsize=(14, 8))

    # Obtener lista de colores, uno por k
    colors = plt.cm.get_cmap("tab10", len(ks))  # tab10 soporta hasta 10 colores bien diferenciados

    for i, k in enumerate(ks):
        df_k = df[df['total_partitions'] == k]
        df_mean = df_k.groupby("epoch")[["error", "test_error"]].mean().reset_index()

        color = colors(i)

        # Etiqueta especial para k=1
        label_prefix = "split" if k == 1 else f"k={k}"

        # Train: línea punteada
        plt.plot(df_mean["epoch"], df_mean["error"], linestyle="--", label=f"Train ({label_prefix})", color=color, alpha=0.9)
        # Test: línea sólida
        plt.plot(df_mean["epoch"], df_mean["test_error"], linestyle="-", label=f"Test ({label_prefix})", color=color, alpha=0.9)

    plt.title("Error promedio por época para distintos conjuntos de entrenamiento (Train vs Test)")
    plt.xlabel("Épocas")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()

    out_dir = os.path.join("graphs", "errors", "generalization")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "generalization_comparison_all_k_avg.png"))
    plt.close()



def plot_percepton_errors_by_learning_rate(df, beta:float, seed:int, activation_function:str):
    filtered = df[
    (df['seed'] == seed) &
    (df['activation_function'] == activation_function) &
    (df['beta'] == beta)
    ]

    out_dir = os.path.join("graphs", "ej2")
    os.makedirs(out_dir, exist_ok=True)

    plt.bar(filtered['learning_rate'].astype(str), filtered['training_mean_error'], yerr=filtered["training_error_std"], capsize=5, color='skyblue', alpha=0.8)
    plt.xlabel("Tasa de aprendizaje")
    plt.ylabel("Error medio de entrenamiento")
    plt.title(f"Error medio de entrenamiento con {activation_function}")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"graphs/ej2/{activation_function}_betas_training_errors_for_beta_{beta}_by_learning_rate.png")
    plt.clf()


def plot_linear_perceptron_errors_for_different_beta_by_learning_rate(df, learning_rate:float, seed:int, activation_function:str):
    filtered = df[
    (df['seed'] == seed) &
    (df['activation_function'] == activation_function) &
    (df['learning_rate'] == learning_rate)
    ]

    out_dir = os.path.join("graphs", "errors")
    os.makedirs(out_dir, exist_ok=True)
    
    plt.bar(filtered['beta'].astype(str), filtered['training_mean_error'], yerr=filtered["training_error_std"], capsize=5, color='skyblue', alpha=0.8)
    plt.xlabel("Beta")
    plt.ylabel("Error medio de entrenamiento")
    plt.title(f"Error medio con tasa de aprendizaje:{learning_rate}")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"graphs/{activation_function}_betas_training_errors_for_learning_rate_{learning_rate}.png")
    plt.clf()

    plt.bar(filtered['beta'].astype(str), filtered['training_data_mean_prediction_error'], yerr=filtered["training_data_prediction_error_std"], capsize=5, color='red', alpha=0.8)
    plt.xlabel("Beta")
    plt.ylabel("Error medio de predicción de datos de entrenamiento")
    plt.title(f"Error medio con tasa de aprendizaje: {learning_rate}")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"graphs/{activation_function}_betas_training_data_prediction_errors_for_learning_rate_{learning_rate}.png")
    plt.clf()

    plt.bar(filtered['beta'].astype(str), filtered['testing_data_mean_prediction_error'], yerr=filtered["testing_data_prediction_error_std"], capsize=5, color='green', alpha=0.8)
    plt.xlabel("Beta")
    plt.ylabel("Error medio de predicción de datos de prueba")
    plt.title(f"Error medio con tasa de aprendizaje: {learning_rate}")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"graphs/{activation_function}_betas_testing_data_prediction_errors_for_learning_rate_{learning_rate}.png")
    plt.clf()



def graph_decision_boundary(method, learning_rate, epochs):
    
    x, y = get_ej1_data_xy(method)
    weights = load_ej1_last_weights_from_csv("output_data/ej1_data.csv", method, learning_rate, epochs)

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

    title = get_title("Frontera de decisión", method, learning_rate, epochs)
    save_name = get_save_name("decision_boundary", method, learning_rate, epochs)

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


def plot_regression_plane(activation_function, learning_rate, epochs, beta, partition):
    
    x, y = get_ej2_data_xy()
    weights = load_ej2_last_weights_from_csv("output_data/ej2_data.csv", activation_function, learning_rate, epochs, beta, partition)

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

    title = get_title("Plano de regresión", activation_function, learning_rate, epochs, True, beta, partition)
    save_name = get_save_name("regression_plane", activation_function, learning_rate, epochs, True, beta, partition)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.title(title)
    plt.savefig("graphs/decision boundaries/" + save_name + ".png")


def animate_decision_boundary_2D(method, learning_rate, epochs):
    
    x, y = get_ej1_data_xy(method)
    weights_list = load_ej1_animation_weights_from_csv("output_data/ej1_data.csv", method, learning_rate, epochs)

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

    title = get_title("Frontera de decisión", method, learning_rate, epochs)
    save_name = get_save_name("animated_decision_boundary", method, learning_rate, epochs)

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


def animate_regression_plane_3D(activation_function, learning_rate, epochs, beta, partition):
    
    x, y = get_ej2_data_xy()
    weights_list = load_ej2_animation_weights_from_csv("output_data/ej2_data.csv", activation_function, learning_rate, epochs, beta, partition)
    
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

    title = get_title("Plano de regresión", activation_function, learning_rate, epochs, True, beta, partition)
    save_name = get_save_name("animated_regression_plane", activation_function, learning_rate, epochs, True, beta, partition)

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


def general_error_plots_for_exercise_2():
    df = pd.read_csv("output_data/ej2_data_errors-2025-05-13-06-36-13.csv")

    filtered_df = df[(df['beta'] == 0.1) & (df['learning_rate'] == 0.0001)]
    grouped = filtered_df.groupby('activation_function').mean(numeric_only=True)

    activation_functions = grouped.index.tolist()

    # GRAFICO 1: training_data_mean_prediction_error vs testing_data_mean_prediction_error
    plt.figure(figsize=(8,5))
    x = range(len(activation_functions))
    plt.bar(x, grouped['training_data_mean_prediction_error'], width=0.4, label='Training', align='center')
    plt.bar([i + 0.4 for i in x], grouped['testing_data_mean_prediction_error'], width=0.4, label='Testing', align='center')
    plt.xticks([i + 0.2 for i in x], activation_functions, rotation=45)
    plt.ylabel("Error medio de predicción")
    plt.title("Error medio de predicción (Training vs Testing)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/errors/training_vs_testing_data_mean_prediction_error.png")

    # GRAFICO 2: training_data_prediction_error_std vs testing_data_prediction_error_std
    plt.figure(figsize=(8,5))
    plt.bar(x, grouped['training_data_prediction_error_std'], width=0.4, label='Training STD', align='center')
    plt.bar([i + 0.4 for i in x], grouped['testing_data_prediction_error_std'], width=0.4, label='Testing STD', align='center')
    plt.xticks([i + 0.2 for i in x], activation_functions, rotation=45)
    plt.ylabel("Error STD de predicción")
    plt.title("Error STD de predicción (Training vs Testing)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/errors/training_vs_testing_data_prediction_error_std.png")



if __name__ == '__main__':
    #results_files:List[str] = ["ej1_data.csv", "ej2_data.csv", "ej3_data.csv", "ej4_data.csv"]
    #plots_for_exercise_1(os.path.join("output_data", results_files[0]))
    
    plots_for_exercise_2(os.path.join("output_data", "ej2_data-2025-05-13-06-36-13.csv"), os.path.join("output_data", "ej2_data_errors-2025-05-13-06-36-13.csv"))
    #graph_decision_boundary("and", learning_rate=0.0001, epochs=200)
    #graph_decision_boundary("xor", learning_rate=0.0001, epochs=200)
    #plot_regression_plane("identity", learning_rate=0.0001, epochs=200, beta=1.0, partition=1)
    #animate_decision_boundary_2D("and", learning_rate=0.0001, epochs=200)
    #animate_decision_boundary_2D("xor", learning_rate=0.0001, epochs=200)
    #animate_regression_plane_3D("identity", learning_rate=0.0001, epochs=200, beta=1.0, partition=1)
    #general_error_plots_for_exercise_2()

    #plot_generalization_vs_partitions()

    csv_path = "output_data/ej2_data-2025-05-13-07-43-15.csv"
    
    #for k in [2, 3, 15, 5, 28]:
    #    tabla = get_test_errors_by_partition(csv_path, k)
    #    print(tabla)

