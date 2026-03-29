"""
Ensemble anomaly inference — PatchCore OR supervised classifier.

An image is flagged as bad if either:
  - PatchCore anomaly score  > threshold      (from patchcore_config.json)
  - Classifier P(bad)        > clf_threshold  (from patchcore_config.json)

Usage:
    python inference_anomaly.py --input path/to/folder
    python inference_anomaly.py --input path/to/folder --output results.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from anomaly import PatchCore
from model import get_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def collect_images(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser(description="Ensemble anomaly inference")
    parser.add_argument("--input", required=True,
                        help="Image file or directory")
    parser.add_argument("--memory-bank", default="models/patchcore_memory_bank.npy",
                        help="Path to PatchCore memory bank")
    parser.add_argument("--classifier", default="models/best_model.pth",
                        help="Path to supervised classifier weights")
    parser.add_argument("--config", default="models/patchcore_config.json",
                        help="Path to config JSON")
    parser.add_argument("--output", default=None,
                        help="Optional CSV file to write results to")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    threshold = config["threshold"]
    clf_threshold = config["clf_threshold"]
    veto_threshold = config.get("veto_threshold", 0.0)
    class_names = config["class_names"]
    image_size = config["image_size"]
    clf_image_size = config["clf_image_size"]
    bad_label = class_names[class_names.index("bad")]
    good_label = class_names[class_names.index("good")]

    patchcore = PatchCore()
    patchcore.load(args.memory_bank)
    pc_transform = get_transform(image_size)

    clf_model = get_model(num_classes=len(class_names))
    clf_model.load_state_dict(torch.load(args.classifier, weights_only=True))
    clf_model.eval()
    clf_transform = get_transform(clf_image_size)
    bad_idx = class_names.index("bad")

    image_paths = collect_images(args.input)
    if not image_paths:
        print("No images found.")
        return

    results = []
    misclassified = []
    t_start = time.perf_counter()

    for path in image_paths:
        t_img = time.perf_counter()
        image = Image.open(path).convert("RGB")

        anomaly_score = patchcore.score(pc_transform(image))

        with torch.no_grad():
            probs = torch.softmax(
                clf_model(clf_transform(image).unsqueeze(0)), dim=1)[0]
        p_bad = float(probs[bad_idx])

        elapsed_img = time.perf_counter() - t_img

        prediction = bad_label if (
            (anomaly_score > threshold and p_bad > veto_threshold)
            or p_bad > clf_threshold
        ) else good_label

        true_label = path.parent.name if path.parent.name in class_names else None
        wrong = true_label is not None and prediction != true_label
        marker = "  ✗ WRONG" if wrong else ""
        print(f"{path.name:<40}  {prediction}"
              f"  (anomaly: {anomaly_score:.3f}, P(bad): {p_bad:.3f})"
              f"  [{elapsed_img:.2f}s]{marker}")

        row = {
            "filepath": str(path),
            "prediction": prediction,
            "anomaly_score": f"{anomaly_score:.4f}",
            "p_bad": f"{p_bad:.4f}",
        }
        if true_label:
            row["true_label"] = true_label
        results.append(row)
        if wrong:
            misclassified.append(row)

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTotal: {len(results)} images in {total_elapsed:.1f}s "
          f"({total_elapsed / len(results):.2f}s/image)")

    if misclassified:
        print(f"\nMisclassified: {len(misclassified)} / {len(results)}")
        for r in misclassified:
            print(f"  {r['filepath']}  "
                  f"(true={r['true_label']}, pred={r['prediction']})")

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
