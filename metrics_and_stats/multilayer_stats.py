import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_epochs_evolution_per_partition(ej_type= 'parity', activation_function='relu', optimizer='gradient_descent_optimizer_with_delta', learning_rate=5e-5, total_epochs=10000, error_function='squared_error'):

    df = pd.read_csv(f"output_data/ej3_{ej_type}_data.csv")

    df_filtered = df[
        (df['activation_function'] == activation_function) & 
        (df['optimizer'] == optimizer) & 
        (df['total_epochs'] == total_epochs) & 
        (df['learning_rate'] == learning_rate) &
        (df['error_function'] == error_function) &
        (df['epoch'] > 1)
    ]

    plt.figure(figsize=(10, 6))

    for partition in df_filtered['partition'].unique():
        partition_data = df_filtered[df_filtered['partition'] == partition]

        plt.plot(partition_data['epoch'], partition_data['error'], label=f'Partition {partition}')

    plt.title("Evolución del error a lo largo de las épocas")
    plt.xlabel("Épocas")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_epochs_evolution(ej_type= 'parity', activation_function='relu', optimizer='gradient_descent_optimizer_with_delta', learning_rate=1e-5, total_epochs=1000, error_function='mean_error', beta=1.0):

    df = pd.read_csv(f"output_data/ej3_{ej_type}_data.csv")

    df_filtered = df[
        (df['activation_function'] == activation_function) & 
        (df['optimizer'] == optimizer) & 
        (df['total_epochs'] == total_epochs) & 
        (df['learning_rate'] == learning_rate) &
        (df['error_function'] == error_function) &
        (df['beta'] == beta) &
        (df['epoch'] > 1)
    ]

    grouped = df_filtered.groupby("epoch").agg({
        "error": ["mean", "std"]
    }).reset_index()

    grouped.columns = ["epoch", "mean_error", "std_error"]

    plt.figure(figsize=(10, 6))
    plt.plot(grouped["epoch"], grouped["mean_error"], label="Mean Error", color="blue")
    plt.fill_between(grouped["epoch"],
                     grouped["mean_error"] - grouped["std_error"],
                     grouped["mean_error"] + grouped["std_error"],
                     alpha=0.3, color="blue", label="±1 std")

    plt.title("Evolución promedio del error a lo largo de las épocas")
    plt.xlabel("Épocas")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_epochs_comparing_optimizers(ej_type='parity', activation_function='relu', learning_rate=1e-5, total_epochs=1000, error_function='mean_error', beta=1.0):

    df = pd.read_csv(f"output_data/ej3_{ej_type}_data.csv")

    df_filtered = df[
        (df['activation_function'] == activation_function) & 
        (df['total_epochs'] == total_epochs) & 
        (df['learning_rate'] == learning_rate) &
        (df['error_function'] == error_function) &
        (df['beta'] == beta) &
        (df['epoch'] > 0)
    ]

    grouped = df_filtered.groupby(['optimizer', 'epoch']).agg(
        mean_error=('error', 'mean'),
        std_error=('error', 'std')
    ).reset_index()

    plt.figure(figsize=(12, 7))

    for optimizer in grouped['optimizer'].unique():
        optimizer_data = grouped[grouped['optimizer'] == optimizer]
        plt.plot(optimizer_data['epoch'], optimizer_data['mean_error'], label=f"{optimizer}")
        plt.fill_between(optimizer_data['epoch'],
                         optimizer_data['mean_error'] - optimizer_data['std_error'],
                         optimizer_data['mean_error'] + optimizer_data['std_error'],
                         alpha=0.2)

    plt.title("Comparación de optimizadores - Evolución del error a lo largo de las épocas")
    plt.xlabel("Épocas")
    plt.ylabel("Error")
    plt.legend(title="Optimizador")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_learning_rates_bars(ej_type='parity', activation_function='relu', optimizer='gradient_descent_optimizer_with_delta', total_epochs=1000):
    df = pd.read_csv(f"output_data/ej3_{ej_type}_data_errors.csv")

    df = df[
        (df['activation_function'] == activation_function) &
        (df['optimizer'] == optimizer) &
        (df['total_epochs'] == total_epochs) 
    ]

    grouped = df.groupby("learning_rate").agg({
        "training_mean_error": "mean",
        "training_std_error": "mean"
    }).reset_index()

    learning_rates = grouped["learning_rate"]
    mean_errors = grouped["training_mean_error"]
    std_errors = grouped["training_std_error"]

    # Gráfico 1: Mean Error por learning rate
    plt.figure(figsize=(8, 6))
    plt.bar(learning_rates.astype(str), mean_errors, yerr=std_errors, capsize=5, color='skyblue', edgecolor='black')
    plt.xlabel("Learning Rate")
    plt.ylabel("Training Mean Error")
    plt.title("Training Mean Error por Learning Rate")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Gráfico 2: Std Error por learning rate
    plt.figure(figsize=(8, 6))
    plt.bar(learning_rates.astype(str), std_errors, color='skyblue')
    plt.xlabel("Learning Rate")
    plt.ylabel("Training Std Error")
    plt.title("Training Std Error por Learning Rate")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def plot_error_by_learning_rates_epochs_evolution(ej_type='parity', activation_function='relu', optimizer='gradient_descent_optimizer_with_delta', total_epochs=5000, error_function='squared_error', beta=1.0):
    df = pd.read_csv(f"output_data/ej3_{ej_type}_data.csv")

    df_filtered = df[
        (df['activation_function'] == activation_function) &
        (df['optimizer'] == optimizer) &
        (df['total_epochs'] == total_epochs) &
        (df['error_function'] == error_function) &
        (df['beta'] == beta) &
        (df['epoch'] > 1)
    ]

    grouped = df_filtered.groupby(["learning_rate", "epoch"]).agg({
        "error": "mean"
    }).reset_index()

    plt.figure(figsize=(10, 6))

    for lr, group in grouped.groupby("learning_rate"):
        plt.plot(group["epoch"], group["error"], label=f"lr={lr:.0e}")

    plt.title("Evolución del error por época según learning rate para " + activation_function)
    plt.xlabel("Épocas")
    plt.ylabel("Error cuadrático")
    plt.legend(title="Learning rate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_error_per_epoch_for_optimizers(ej_type='parity', activation_function='logistic', learning_rate=0.01, total_epochs=5000, error_function='squared_error'):
    df = pd.read_csv(f"output_data/ej3_{ej_type}_data.csv")
    
    df_filtered = df[
        (df['activation_function'] == activation_function) &
        (df['total_epochs'] == total_epochs) &
        (df['error_function'] == error_function) &
        (df['learning_rate'] == learning_rate) &
        (df['epoch'] > 1)
    ]
    
    plt.figure(figsize=(10, 6))
    optimizers = df_filtered['optimizer'].unique()
    
    colors = ['blue', 'red', 'green']
    optimizer_labels = {'gradient_descent_optimizer_with_delta': 'Gradiente descendente', 'momentum_gradient_descent_optimizer_with_delta': 'Momentum', 'adam_optimizer_with_delta': 'Adam'}
    
    for i, optimizer in enumerate(optimizers):
        optimizer_data = df_filtered[df_filtered['optimizer'] == optimizer]

        epoch_data = optimizer_data.groupby('epoch')['error'].mean().reset_index()
        
        plt.plot(epoch_data['epoch'], epoch_data['error'], 
                 color=colors[i % len(colors)], 
                 label=optimizer_labels[optimizer])
    
    plt.xlabel("Épocas")
    plt.ylabel("Error")
    plt.title(f"Evolución del error para diferentes optimizadores\n"
              f"(Función de activación: {activation_function}, Learning rate: {learning_rate})")
    
    plt.legend(title="Optimizador")
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"neural_network_graphs/{ej_type}_errors_per_epoch_for_optimizers.png")

def plot_error_by_activation_function_epochs_evolution(ej_type='parity', optimizer='gradient_descent_optimizer_with_delta', total_epochs=5000, error_function='squared_error', beta=1.0):

    df = pd.read_csv(f"output_data/ej3_{ej_type}_data.csv")

    # Diccionario con learning rates personalizados para cada función de activación
    # Permite que cada función de activación tenga el learning rate con el que converja mejor
    learning_rates = {
        'logistic': 0.1,
        'tanh': 0.01,
        'relu': 0.01,
    }

    dfs = []

    for activation_function, lr in learning_rates.items():
        subset = df[
            (df['activation_function'] == activation_function) &
            (df['optimizer'] == optimizer) &
            (df['total_epochs'] == total_epochs) &
            (df['error_function'] == error_function) &
            (df['beta'] == beta) &
            (df['learning_rate'] == lr) &
            (df['epoch'] > 1)
        ]
        subset["label"] = f"{activation_function} (tasa={lr:.0e})"
        dfs.append(subset)

    df_filtered = pd.concat(dfs)

    grouped = df_filtered.groupby(["label", "epoch"]).agg({
        "error": "mean"
    }).reset_index()

    plt.figure(figsize=(10, 6))

    for label, group in grouped.groupby("label"):
        plt.plot(group["epoch"], group["error"], label=label)

    plt.title("Evolución del error por época según función de activación")
    plt.xlabel("Épocas")
    plt.ylabel("Error cuadrático")
    plt.legend(title="Función de activación")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_accuracy(ej_type = 'parity'):
    df = pd.read_csv("output_data/ej3_accuracy.csv")

    df = df[(df['ej'] == ej_type) ]

    grouped = df.groupby("partitions").agg(
        train_acc_mean=('training_accuracy', 'mean'),
        train_acc_std=('training_accuracy', 'std'),
        test_acc_mean=('testing_accuracy', 'mean'),
        test_acc_std=('testing_accuracy', 'std')
    ).reset_index()

    # Limitar las barras de error al rango [0, 1]
    grouped["train_err_up"] = np.minimum(grouped["train_acc_std"], 1 - grouped["train_acc_mean"])
    grouped["train_err_down"] = np.minimum(grouped["train_acc_std"], grouped["train_acc_mean"])
    grouped["test_err_up"] = np.minimum(grouped["test_acc_std"], 1 - grouped["test_acc_mean"])
    grouped["test_err_down"] = np.minimum(grouped["test_acc_std"], grouped["test_acc_mean"])

    plt.figure(figsize=(10, 6))
    plt.errorbar(grouped["partitions"], grouped["train_acc_mean"],
                 yerr=[grouped["train_err_down"], grouped["train_err_up"]],
                 label="Training Accuracy", fmt='-o', capsize=5)
    plt.errorbar(grouped["partitions"], grouped["test_acc_mean"],
                 yerr=[grouped["test_err_down"], grouped["test_err_up"]],
                 label="Testing Accuracy", fmt='-s', capsize=5)

    plt.xlabel("Número de particiones (k)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy con barras de error por k-fold cross-validation")
    plt.xticks(grouped["partitions"])
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #plot_parity_epochs_evolution(ej_type= 'digits', total_epochs=200, learning_rate=0.001, activation_function='tanh', error_function='mean_error', beta=1.0)
    #plot_parity_learning_rates()
    #plot_parity_epochs_evolution_per_partition()
    #plot_accuracy()
    #plot_parity_epochs_comparing_optimizers(activation_function='relu', learning_rate=1e-5, total_epochs=1000, error_function='mean_error', beta=1.0)
    #plot_error_by_learning_rates_epochs_evolution(ej_type='digits', activation_function='tanh')
    plot_error_by_activation_function_epochs_evolution(ej_type='digits', optimizer='gradient_descent_optimizer_with_delta', total_epochs=5000, error_function='squared_error', beta=1.0)
