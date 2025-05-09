import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from confusion_matrix import ConfusionMatrix

def plot_binary_confusion_matrix(y_true, y_pred, labels=["Negative", "Positive"], title="Binary Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    plt.show()


def plot_multiclass_confusion_matrix(y_true, y_pred, labels=None, title="Multiclass Confusion Matrix"):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    if isinstance(labels, dict):
        # Si es un diccionario: map class -> label
        labels = [labels[c] for c in classes]
    else:
        labels = [f"Class {i}" for i in classes]
        
    
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Oranges, values_format='d')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()


if __name__ == "__main__":
    y_true = [0, 1, 1, 0, 1, 0, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1, 1, 0]
    # plot_binary_confusion_matrix(y_true, y_pred)
    # plot_multiclass_confusion_matrix(y_true, y_pred, labels={0: "Neg ", 1: "Pos"})

    # cm = ConfusionMatrix(y_true, y_pred)
    cm = ConfusionMatrix(y_true, y_pred, {0: "Neg ", 1: "Pos"})
    cm.display()