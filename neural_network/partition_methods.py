from typing import List 
import numpy as np
from collections import defaultdict

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

class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        np.random.seed(random_state)

    def split(self, x_array, y_array):

        # Handle splits
        if self.shuffle:
            indices = np.arange(len(x_array))
            np.random.shuffle(indices)
            x_array = x_array[indices]
            y_array = y_array[indices]
        else:
            indices = np.arange(len(x_array))

        # Group by class
        class_indices = defaultdict(list)
        for idx, label in zip(indices, y_array):
            class_indices[label].append(idx)

        folds = [[] for _ in range(self.n_splits)]
        for cls, cls_indices in class_indices.items():
            if self.shuffle:
                np.random.shuffle(cls_indices)
            cls_folds = np.array_split(cls_indices, self.n_splits)
            for i in range(self.n_splits):
                folds[i].extend(cls_folds[i])

        for i in range(self.n_splits):
            val_idx = np.array(folds[i])
            train_idx = np.array([idx for j, fold in enumerate(folds) if j != i for idx in fold])
            yield train_idx, val_idx