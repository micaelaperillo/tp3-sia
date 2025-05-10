import numpy as np
from typing import List
from neural_network.models.basic_perceptron import Perceptron, ErrorFunctionType
from neural_network.models.neural_network import NeuralNetwork

def get_prediction_error_for_perceptron(perceptron:Perceptron, x_values:List[List[float]], y_values:List[float], error_function:ErrorFunctionType, beta:float=1.0, descale_fun = None)->float:
    errors = []
    for x_value, y_value in zip(x_values, y_values):
        x_with_bias = np.insert(x_value, 0, 1)
        prediction = perceptron.predict(x_with_bias, beta)
        prediction = prediction[0]
        if descale_fun is not None:
            prediction = descale_fun(prediction)
            y_value = descale_fun(y_value)
        basic_error = (y_value - prediction)
        errors.append(basic_error)

    return error_function(np.array(errors))
        
def get_prediction_error_for_neural_network(neural_network:NeuralNetwork, x_values:List[List[float]], y_values:List[float], error_function:ErrorFunctionType, beta:float=1.0, descale_fun = None)->float:
    errors = []
    for x_value, y_value in zip(x_values, y_values):
        prediction = neural_network.predict(x_value, beta)
        if descale_fun is not None:
            prediction = descale_fun(prediction)
            y_value = descale_fun(y_value)
        basic_error = (y_value - prediction)
        errors.append(basic_error)

    return error_function(np.array(errors))