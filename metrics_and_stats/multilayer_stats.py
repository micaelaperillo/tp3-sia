import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_parity_epochs_evolution_per_partition(activation_function='relu', optimizer='gradient_descent_optimizer_with_delta', learning_rate=5e-5, total_epochs=10000, error_function='squared_error'):

    df = pd.read_csv("output_data/ej3_parity_data.csv")

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


def plot_parity_epochs_evolution(activation_function='relu', optimizer='gradient_descent_optimizer_with_delta', learning_rate=1e-5, total_epochs=1000, error_function='mean_error', beta=1.0):

    df = pd.read_csv("output_data/ej3_parity_data.csv")

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


def plot_parity_learning_rates(activation_function='relu', optimizer='gradient_descent_optimizer_with_delta', total_epochs=1000):
    df = pd.read_csv("output_data/ej3_parity_data_errors.csv")

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



def plot_accuracy():
    df = pd.read_csv("output_data/ej3_accuracy.csv")

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
    #plot_parity_epochs_evolution(total_epochs=1000, learning_rate=0.01, activation_function='tanh', error_function='mean_error', beta=1.0)
    #plot_parity_learning_rates()
    #plot_parity_epochs_evolution_per_partition()
    plot_accuracy()
