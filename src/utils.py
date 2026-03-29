"""
Shared utilities for evaluation output.

    ensure_dir()               — create output directories if missing
    save_confusion_matrix()    — plot and save confusion matrix as PNG
    print_classification_report() — print precision/recall/F1 to stdout
    save_misclassified_report() — write misclassified samples to CSV
"""

import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def ensure_dir(path):
    """Create directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot a confusion matrix and save it as a PNG file.

    Rows represent the true label; columns represent the predicted label.
    Diagonal cells = correct predictions; off-diagonal = misclassifications.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def print_classification_report(y_true, y_pred, class_names):
    """Print per-class precision, recall, F1, and support to stdout."""
    print(classification_report(y_true, y_pred, target_names=class_names))


def save_misclassified_report(misclassified, save_path):
    """Write misclassified samples to a CSV for manual inspection.

    Fieldnames are inferred from the first row, so callers can include
    extra columns (e.g. anomaly_score) without changing this function.
    """
    if not misclassified:
        print(f"No misclassified samples → {save_path}")
        return
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(misclassified[0].keys()))
        writer.writeheader()
        writer.writerows(misclassified)
    print(f"Misclassified report: {len(misclassified)} samples → {save_path}")
