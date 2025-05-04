import pandas as pd
import numpy as np
import os

def get_title(init_string: str, fun: str, learning_rate: float, epochs: int, extra_params:bool = False, beta: float = 0, partition: int = 0):
    title = f"{init_string} {fun.upper()}: tasa={learning_rate}, épocas={epochs}"
    if extra_params:
        title = title + f", beta={beta}, particiones={partition}"
    return title

def get_save_name(init_string: str, fun: str, learning_rate: float, epochs: int, extra_params: bool = False, beta: float = 0, partition: int = 0):
    save_name = f"{fun}_{init_string}_tasa_{learning_rate}_epochs_{epochs}"
    if extra_params:
        save_name = save_name + f"_beta_{beta}_part_{partition}"
    return save_name
        
def get_ej1_data_xy(method: str):
    if method == "and":
        return np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]]), np.array([-1, -1, -1, 1])
    else:
        return np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]]), np.array([1, 1, -1, -1])
    

def get_ej2_data_xy():
    input_data_dir_name = "input_data"
    exercise_2_input_data_filename = "TP3-ej2-conjunto.csv"
    exercise_2_input_data_path= os.path.join(input_data_dir_name, exercise_2_input_data_filename)
    df = pd.read_csv(exercise_2_input_data_path)
    x = df[['x1', 'x2', 'x3']].to_numpy()
    y = df['y'].to_numpy()
    return x, y


def parse_weights(weights_str):
    weights_str = weights_str.strip()[1:-1]
    return np.array([float(w) for w in weights_str.split()])


def load_ej1_animation_weights_from_csv(filepath: str, method: str, learning_rate: float, epochs: int):
    df = pd.read_csv(filepath)

    filtered = df[(df["method"] == method) & (df["learning_rate"] == learning_rate) & (df["total_epochs"] == epochs)]
    filtered = filtered.sort_values(by="epoch").reset_index(drop=True)
    check_filters_existance(filtered)

    indices = np.linspace(0, len(filtered) - 1, num=20, dtype=int)
    sampled_weights = filtered.loc[indices, "weights"].apply(parse_weights).to_list()
    return sampled_weights


def load_ej2_animation_weights_from_csv(filepath: str, activation_function: str, learning_rate: float, epochs: int, beta: float, partition: int):
    df = pd.read_csv(filepath)

    filtered = df[(df["activation_function"] == activation_function) & (df["learning_rate"] == learning_rate) & (df["total_epochs"] == epochs) & (df["beta"] == beta) & (df["partition"] == partition)]
    filtered = filtered.sort_values(by="epoch").reset_index(drop=True)
    check_filters_existance(filtered)

    indices = np.linspace(0, len(filtered) - 1, num=20, dtype=int)
    sampled_weights = filtered.loc[indices, "weights"].apply(parse_weights).to_list()
    return sampled_weights


def load_ej1_last_weights_from_csv(filepath: str, method: str, learning_rate: float, epochs: int):
    df = pd.read_csv(filepath)

    filtered = df[(df["method"] == method) & (df["learning_rate"] == learning_rate) & (df["total_epochs"] == epochs)]
    filtered = filtered.sort_values(by="epoch").reset_index(drop=True)
    check_filters_existance(filtered)

    last_row = filtered.iloc[-1]
    weights = parse_weights(last_row["weights"])
    return weights


def load_ej2_last_weights_from_csv(filepath: str, activation_function: str, learning_rate: float, epochs: int, beta: float, partition: int):
    df = pd.read_csv(filepath)

    filtered = df[(df["activation_function"] == activation_function) & (df["learning_rate"] == learning_rate) & (df["total_epochs"] == epochs) & (df["beta"] == beta) & (df["partition"] == partition)]
    filtered = filtered.sort_values(by="epoch").reset_index(drop=True)
    check_filters_existance(filtered)

    last_row = filtered.iloc[-1]
    weights = parse_weights(last_row["weights"])
    return weights


def check_filters_existance(filtered_data: pd.DataFrame):
    if filtered_data.empty:
        raise ValueError("No data found for the given filters.")
