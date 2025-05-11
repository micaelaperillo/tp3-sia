import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_parity_epochs_evolution(learning_rate=1e-5, total_epochs=1000):

    df = pd.read_csv("output_data/ej3_parity_data.csv")

    df_filtered = df[
        (df['activation_function'] == 'relu') & 
        (df['optimizer'] == 'gradient_descent_optimizer_with_delta') & 
        (df['total_epochs'] == total_epochs) & 
        (df['learning_rate'] == learning_rate)
    ]

    plt.figure(figsize=(10, 6))

    for partition in df_filtered['partition'].unique():
        partition_data = df_filtered[df_filtered['partition'] == partition]
        partition_data = partition_data[
            (partition_data['epoch'] > 1)
        ]

        plt.plot(partition_data['epoch'], partition_data['error'], label=f'Partition {partition}')

    plt.title("Evolución del error a lo largo de las épocas")
    plt.xlabel("Épocas")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_parity_learning_rates():
    df = pd.read_csv("output_data/ej3_parity_data_errors.csv")

    df = df[
        (df['activation_function'] == 'relu') &
        (df['optimizer'] == 'gradient_descent_optimizer_with_delta') &
        (df['total_epochs'] == 1000) 
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


if __name__ == "__main__":
    plot_parity_epochs_evolution()
    plot_parity_learning_rates()