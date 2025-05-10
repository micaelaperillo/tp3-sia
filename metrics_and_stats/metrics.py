import matplotlib.pyplot as plt
import numpy as np
from confusion_matrix import ConfusionMatrix

def plot_binary_confusion_matrix(y_true, y_pred, labels=["Negative", "Positive"]):
    cm = ConfusionMatrix(y_true, y_pred)
    cm.display()

def plot_multiclass_confusion_matrix(y_true, y_pred, labels=None):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    if isinstance(labels, dict):
        labels = [labels[c] for c in classes]
    else:
        labels = [f"Class {i}" for i in classes]
        
    cm = ConfusionMatrix(y_true, y_pred, labels=labels)
    cm.display()

if __name__ == "__main__":
    y_true = [0, 1, 1, 0, 1, 0, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1, 1, 0]
    plot_binary_confusion_matrix(y_true, y_pred)
    plot_multiclass_confusion_matrix(y_true, y_pred, labels={0: "Neg ", 1: "Pos"})