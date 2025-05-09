import numpy as np

class ConfusionMatrix:

    def __init__(self, y_true, y_pred, labels=None): 
        # labels = None for binary classification
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels

        if (labels is None):
            shape = (2, 2)
        else:
            shape = (len(labels), len(labels))

        self.confusion_matrix = np.zeros(shape, dtype=int)

        if labels is not None:
            label_to_index = {label: idx for idx, label in enumerate(labels)}

        for i in range(len(y_true)):
            if (labels is None):
                self.confusion_matrix[y_true[i], y_pred[i]] += 1
            else:
                self.confusion_matrix[label_to_index[y_true[i]], label_to_index[y_pred[i]]] += 1

    def display(self):
        pass

    def plot(self):
        pass