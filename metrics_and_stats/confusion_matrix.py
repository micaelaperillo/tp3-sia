import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ConfusionMatrix:
    def __init__(self, y_true, y_pred, labels=None): 
        self.y_true = y_true
        self.y_pred = y_pred

        if labels is None:
            # Binary classification
            self.labels = ["Negativo", "Positivo"]
            label_to_index = {0: 0, 1: 1} 
            shape = (2, 2)
        else:
            # Multiclass
            self.labels = labels
            label_to_index = {key: idx for idx, key in enumerate(labels)}  
            shape = (len(labels), len(labels))

        self.confusion_matrix = np.zeros(shape, dtype=int)

        for i in range(len(y_true)):
            true_idx = y_true[i]  
            pred_idx = y_pred[i] 
            
            if true_idx is not None and pred_idx is not None:
                self.confusion_matrix[true_idx, pred_idx] += 1
            else:
                print(f"Warning: Label mismatch at index {i}: true label {y_true[i]}, predicted label {y_pred[i]}")

    def display(self):
        plt.figure(figsize=(6, 5))
        title = 'Matriz de Confusión Binaria' if len(self.labels) == 2 else 'Matriz de Confusión Multiclase'
        plt.title(title)
        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            cmap='Blues',
            fmt='d',
            cbar=True,
            xticklabels=self.labels,
            yticklabels=self.labels,
            square=True
        )
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        plt.show()
