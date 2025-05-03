import pandas as pd
import numpy as np

def parse_weights(weights_str):
    weights_str = weights_str.strip()[1:-1]
    return np.array([float(w) for w in weights_str.split()])


def load_ej1_weights_from_csv(filepath: str, method: str, learning_rate: float):
    df = pd.read_csv(filepath)

    filtered = df[(df["method"] == method) & (df["learning_rate"] == learning_rate)]
    filtered = filtered.sort_values(by="epochs").reset_index(drop=True)

    indices = np.linspace(0, len(filtered) - 1, num=20, dtype=int)
    sampled_weights = filtered.loc[indices, "weights"].apply(parse_weights).to_list()
    return sampled_weights


def load_ej2_weights_from_csv(filepath: str, activation_function: str, learning_rate: float, beta: float, partition: int):
    df = pd.read_csv(filepath)

    filtered = df[(df["activation_function"] == activation_function) & (df["learning_rate"] == learning_rate) & (df["beta"] == beta) & (df["partition"] == partition)]
    filtered = filtered.sort_values(by="epochs").reset_index(drop=True)

    indices = np.linspace(0, len(filtered) - 1, num=20, dtype=int)
    sampled_weights = filtered.loc[indices, "weights"].apply(parse_weights).to_list()
    return sampled_weights