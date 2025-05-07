from typing import List, Union, Callable
import numpy as np
ErrorFunctionType = Callable[[List[float]], float]

def squared_error(errors:List[float])->Union[int, float]:
    # REVISAR!!!!
    return (0.5 * np.sum(errors ** 2)) / len(errors)

def mean_error(errors:List[float], y_values:List[float])->float:
    return np.abs(np.sum(errors)) / len(y_values)
