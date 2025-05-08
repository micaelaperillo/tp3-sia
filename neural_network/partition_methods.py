from typing import List 
import numpy as np
from sklearn.model_selection import StratifiedKFold

def initial_partition_by_testing_percentage(testing_percentage:float, x_values:List[float], y_values:List[float]):
    testing_partition_len = np.ceil(len(x_values) * testing_percentage)
    testing_x_values = x_values[:testing_partition_len].copy()
    testing_y_values = y_values[:testing_partition_len].copy()
    testing = np.array([testing_x_values, testing_y_values])

    training_x_values = x_values[testing_partition_len:].copy()
    training_y_values = y_values[testing_partition_len:].copy()
    training = np.array([training_x_values, training_y_values])

    return np.array([training, testing]) 

def k_cross_validation(k:int, x_values:List[float], y_values:List[float], seed:int=43):
    # returns an array with all k possible configurations
    # a configuration is defined as:
    # configuration = [training_set, test_set]
    # where training_set and test_set are defined as:
    # set = [x_values, y_values]
    results = []
    n = len(x_values)
    np.random.seed(seed)
    indices = np.random.permutation(n)
    
    fold_indices = np.array_split(indices, k)

    for i in range(k):
        test_indices = fold_indices[i]
        train_indices = np.concatenate([fold_indices[j] for j in range(k) if j != i])

        train_x = x_values[train_indices]
        train_y = y_values[train_indices]
        test_x = x_values[test_indices]
        test_y = y_values[test_indices]

        training_set = [train_x, train_y]
        testing_set = [test_x, test_y]
        configuration = [training_set, testing_set]

        results.append(configuration)

    return results


def stratified_k_cross_validation(k: int, x_values: List[float], y_values: List[int], seed: int = 43):
    results = []
    x_array = np.array(x_values)
    y_array = np.array(y_values)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for train_index, test_index in skf.split(x_array, y_array):
        train_x = x_array[train_index]
        train_y = y_array[train_index]
        test_x = x_array[test_index]
        test_y = y_array[test_index]

        training_set = [train_x, train_y]
        testing_set = [test_x, test_y]
        configuration = [training_set, testing_set]

        results.append(configuration)

    return results