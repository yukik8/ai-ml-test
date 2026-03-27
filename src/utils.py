import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def print_classification_report(y_true, y_pred, class_names):
    print(classification_report(y_true, y_pred, target_names=class_names))
