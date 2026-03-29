"""
Train PatchCore and calibrate the ensemble thresholds.

Workflow
--------
1. Build a PatchCore memory bank from good-only training images.
2. Score the validation set with PatchCore and find a threshold that
   achieves >= 95% bad recall (missing a defect is more costly than a
   false alarm in QC applications).
3. Load the pre-trained supervised classifier (best_model.pth) and score
   the same validation set to find a complementary threshold that catches
   whatever PatchCore misses.
4. Apply a veto: if the classifier is very confident the image is good
   (p_bad < veto_threshold), override a PatchCore "bad" flag to "good"
   — this suppresses PatchCore false positives on hard good images.
5. Save all thresholds and metadata to models/patchcore_config.json for
   use by inference_anomaly.py.

Prerequisites
-------------
    Run train.py first to produce models/best_model.pth.
"""

import json
import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

from anomaly import PatchCore
from dataset import get_anomaly_dataloaders, get_dataloaders
from model import get_model
from utils import ensure_dir, save_confusion_matrix, save_misclassified_report


def score_validation_set(model, val_loader):
    """Return arrays of PatchCore anomaly scores and true labels for every validation image."""
    scores, labels = [], []
    for image, label in tqdm(val_loader, desc="Scoring validation set"):
        scores.append(model.score(image.squeeze(0)))
        labels.append(label.item())
    return np.array(scores), np.array(labels)


def score_with_classifier(clf_model, clf_val_loader, class_names):
    """Return P(bad) scores for every validation sample."""
    bad_idx = class_names.index("bad")
    clf_model.eval()
    clf_scores = []
    with torch.no_grad():
        for images, _ in tqdm(clf_val_loader, desc="Classifier scoring"):
            probs = torch.softmax(clf_model(images), dim=1)
            clf_scores.extend(probs[:, bad_idx].tolist())
    return np.array(clf_scores)


def find_clf_threshold(patchcore_scores, clf_scores, labels, patchcore_thresh, bad_idx):
    """Find the minimum P(bad) threshold that catches what PatchCore misses.

    Sets the threshold just below the lowest classifier score among PatchCore's
    false negatives, guaranteeing those samples are caught with minimal extra
    false positives.
    """
    good_idx = 1 - bad_idx
    patchcore_preds = np.where(patchcore_scores > patchcore_thresh, bad_idx, good_idx)
    missed_mask = (patchcore_preds == good_idx) & (labels == bad_idx)

    if not missed_mask.any():
        print("PatchCore catches all bad samples — classifier not needed.")
        return 1.1  # effectively disabled

    clf_threshold = float(clf_scores[missed_mask].min()) - 1e-4
    n_extra_fp = int(((clf_scores > clf_threshold) & (labels == good_idx)).sum())
    print(f"Classifier threshold: {clf_threshold:.4f}  "
          f"(catches {missed_mask.sum()} PatchCore misses, "
          f"{n_extra_fp} extra false positives)")
    return clf_threshold


def find_threshold(scores, labels, bad_idx, min_bad_recall=0.95):
    """Find the highest threshold that still guarantees min_bad_recall.

    For QC use cases a missed bad sample is far more costly than a false alarm,
    so we maximise the threshold (= minimise false positives) subject to the
    constraint that bad recall >= min_bad_recall.
    """
    good_idx = 1 - bad_idx
    n_bad = int((labels == bad_idx).sum())

    best_thresh = scores.min()  # fallback: flag everything
    for t in np.percentile(scores, np.linspace(0, 100, 300)):
        preds = np.where(scores > t, bad_idx, good_idx)
        tp = int(((preds == bad_idx) & (labels == bad_idx)).sum())
        recall = tp / (n_bad + 1e-8)
        if recall >= min_bad_recall:
            best_thresh = t  # keep raising threshold while recall holds

    actual_recall = int(((np.where(scores > best_thresh, bad_idx, good_idx) == bad_idx)
                         & (labels == bad_idx)).sum()) / n_bad
    print(f"Threshold set to {best_thresh:.4f} "
          f"(bad recall = {actual_recall:.2%}, target ≥ {min_bad_recall:.0%})")
    return best_thresh


def evaluate(scores, clf_scores, labels, threshold, clf_threshold, veto_threshold, class_names, val_loader):
    bad_idx = class_names.index("bad")
    good_idx = 1 - bad_idx

    ensemble_bad = ((scores > threshold) & (clf_scores > veto_threshold)) \
                 | (clf_scores > clf_threshold)
    preds = np.where(ensemble_bad, bad_idx, good_idx)

    print(f"\nEnsemble — PatchCore threshold: {threshold:.4f} | "
          f"Veto threshold: {veto_threshold:.4f} | "
          f"Classifier threshold: {clf_threshold:.4f}")
    print(classification_report(labels, preds, target_names=class_names))

    auc = roc_auc_score((labels == bad_idx).astype(int), scores)
    print(f"ROC-AUC (PatchCore): {auc:.4f}")

    save_confusion_matrix(labels, preds, class_names,
                          "outputs/anomaly_confusion_matrix.png")

    val_dataset = val_loader.dataset
    misclassified = []
    for i, (score, clf_score, label, pred) in enumerate(
            zip(scores, clf_scores, labels, preds)):
        if pred != label:
            global_idx = val_dataset.indices[i]
            filepath, _ = val_dataset.dataset.samples[global_idx]
            misclassified.append({
                "filepath": filepath,
                "true_label": class_names[label],
                "predicted_label": class_names[pred],
                "anomaly_score": f"{score:.4f}",
                "p_bad": f"{clf_score:.4f}",
            })
    save_misclassified_report(misclassified, "outputs/anomaly_misclassified.csv")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clf-threshold", type=float, default=None,
                        help="Override classifier threshold instead of auto-calibrating.")
    parser.add_argument("--veto-threshold", type=float, default=0.05,
                        help="PatchCore flag is ignored if p_bad < this value (default 0.05).")
    args = parser.parse_args()

    ensure_dir("outputs")
    ensure_dir("models")

    # PatchCore uses a higher resolution (384) than the classifier (320) to
    # preserve fine patch detail in the memory bank.
    image_size = 384
    good_loader, val_loader, class_names = get_anomaly_dataloaders(
        data_dir="data/raw",
        batch_size=16,
        image_size=image_size,
    )

    patchcore = PatchCore(k=3, subsample_ratio=0.1)
    patchcore.fit(good_loader)
    patchcore.save("models/patchcore_memory_bank.npy")

    scores, labels = score_validation_set(patchcore, val_loader)

    bad_idx = class_names.index("bad")
    threshold = find_threshold(scores, labels, bad_idx, min_bad_recall=0.95)

    # Ensemble: load supervised classifier to catch what PatchCore misses.
    with open("models/class_names.json") as f:
        clf_config = json.load(f)
    clf_image_size = clf_config["image_size"]
    clf_model = get_model(num_classes=2)
    clf_model.load_state_dict(torch.load("models/best_model.pth", weights_only=True))
    clf_model.eval()
    _, clf_val_loader, _ = get_dataloaders(
        data_dir="data/raw", batch_size=16, image_size=clf_image_size, val_ratio=0.2)
    clf_scores = score_with_classifier(clf_model, clf_val_loader, class_names)
    if args.clf_threshold is not None:
        clf_threshold = args.clf_threshold
        print(f"Classifier threshold overridden to {clf_threshold}")
    else:
        clf_threshold = find_clf_threshold(scores, clf_scores, labels, threshold, bad_idx)

    veto_threshold = args.veto_threshold

    with open("models/patchcore_config.json", "w") as f:
        json.dump({"threshold": threshold, "class_names": class_names,
                   "image_size": image_size, "clf_threshold": clf_threshold,
                   "clf_image_size": clf_image_size,
                   "veto_threshold": veto_threshold}, f)
    print(f"Config saved → models/patchcore_config.json")

    evaluate(scores, clf_scores, labels, threshold, clf_threshold, veto_threshold, class_names, val_loader)


if __name__ == "__main__":
    main()
