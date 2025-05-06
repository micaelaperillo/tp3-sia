import numpy as np
from typing import List
from neural_network.models.basic_perceptron import Perceptron, ErrorFunctionType

def get_prediction_error(perceptron:Perceptron, x_values:List[List[float]], y_values:List[float], error_function:ErrorFunctionType, beta:float=1.0)->float:
    errors = []
    for x_value, y_value in zip(x_values, y_values):
        x_with_bias = np.insert(x_value, 0, 1)
        prediction = perceptron.predict(x_with_bias, beta)
        basic_error = (y_value - prediction)
        errors.append(basic_error)

    return error_function(np.array(errors), y_values)
        

