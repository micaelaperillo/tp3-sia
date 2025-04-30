from typing import List, Union
import numpy as np

def squared_error(errors:List[float])->Union[int, float]:
    return 0.5 * np.sum(errors ** 2)