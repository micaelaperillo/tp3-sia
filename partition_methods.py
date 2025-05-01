from typing import List 
import numpy as np

def initial_partition_by_testing_percentage(testing_percentage:float, x_values:List[float], y_values:List[float]):
    testing_partition_len = np.ceil(len(x_values) * testing_percentage)
    testing_x_values = x_values[:testing_partition_len].copy()
    testing_y_values = y_values[:testing_partition_len].copy()
    testing = np.array([testing_x_values, testing_y_values])

    training_x_values = x_values[testing_partition_len:].copy()
    training_y_values = y_values[testing_partition_len:].copy()
    training = np.array([training_x_values, training_y_values])

    return np.array([training, testing]) 

def k_cross_validation(k:int, x_values:List[float], y_values:List[float]):
    # returns an array with all k possible configurations
    # a configuration is defined as:
    # configuration = [training_set, test_set]
    # where training_set and test_set are defined as:
    # set = [x_values, y_values]
    results = []
    n = len(x_values)
    
    testing_partition_len = int(np.ceil(n / k))
    for testing_partition_index in range(k):
        test_start = testing_partition_index * testing_partition_len
        test_end = min(test_start + testing_partition_len, n)  
        
        test_indices = np.arange(test_start, test_end)
        train_indices = np.array([i for i in range(n) if i not in test_indices])
        
        test_x = x_values[test_indices]
        test_y = y_values[test_indices]
        train_x = x_values[train_indices]
        train_y = y_values[train_indices]
        
        training_set = [train_x, train_y]
        testing_set = [test_x, test_y]
        configuration = [training_set, testing_set]
        
        results.append(configuration)

    return results