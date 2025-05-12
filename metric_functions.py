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


def parity_calculate_accuracy(neural_network: NeuralNetwork, x:List[List[float]], y:List[float], beta:float=1.0):
    tp = tn = fp = fn = 0
    for x_value, y_value in zip(x, y):
        pred = neural_network.predict(x_value, beta)
        if pred[0] > pred[1]:
            pred_class = 0 # es par
        else:
            pred_class = 1 # es impar

        true_class = int(y_value[1])

        if pred_class == 1 and true_class == 1:
            tp += 1
        elif pred_class == 0 and true_class == 0:
            tn += 1
        elif pred_class == 1 and true_class == 0:
            fp += 1
        elif pred_class == 0 and true_class == 1:
            fn += 1
    return (tp + tn) / (tp + tn + fp + fn)