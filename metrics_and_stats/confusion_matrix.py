import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

        plt.figure(figsize=(6, 5))
        
        if (self.labels is None):
            # Binary confusion matrix
            plt.title('Matriz de Confusión Binaria')
            sns.heatmap(self.confusion_matrix, annot=True, cmap='Blues', fmt='d', cbar=True,
                        xticklabels=self.labels, yticklabels=self.labels, square=True)
        else:
            # Multiclass confusion matrix
            plt.title('Matriz de Confusión Multiclase')
            sns.heatmap(self.confusion_matrix, annot=True, cmap='Blues', fmt='d', cbar=True, square=True)

        plt.tight_layout()
        plt.show()