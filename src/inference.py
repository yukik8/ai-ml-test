"""
Classifier inference — runs the trained ResNet model on new images.

Usage:
    # Single image
    python inference.py --input path/to/image.jpg

    # Directory of images
    python inference.py --input path/to/folder --output results.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image

from dataset import get_transforms
from model import get_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_model(model_path, num_classes):
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def predict(model, image_path, transform, class_names, bad_threshold=0.5):
    """Classify image. bad_threshold: classify as bad if P(bad) >= this value.
    Lower values catch more bad samples at the cost of more false positives."""
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    bad_idx = class_names.index("bad")
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    if float(probs[bad_idx]) >= bad_threshold:
        pred = bad_idx
    else:
        pred = 1 - bad_idx
    return class_names[pred], float(probs[bad_idx])


def collect_images(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser(description="Classifier inference")
    parser.add_argument("--input", required=True,
                        help="Image file or directory")
    parser.add_argument("--model", default="models/best_model.pth",
                        help="Path to saved model weights")
    parser.add_argument("--class-names", default="models/class_names.json",
                        help="Path to class names JSON saved during training")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--output", default=None,
                        help="Optional CSV file to write results to")
    parser.add_argument("--bad-threshold", type=float, default=0.3,
                        help="Classify as bad if P(bad) >= this value (default 0.3). "
                             "Lower = higher bad recall, more false positives.")
    args = parser.parse_args()

    with open(args.class_names) as f:
        clf_config = json.load(f)
    class_names = clf_config["class_names"]
    image_size = clf_config["image_size"]

    model = load_model(args.model, num_classes=len(class_names))
    _, transform = get_transforms(image_size)  # val transform — no augmentation

    image_paths = collect_images(args.input)
    if not image_paths:
        print("No images found.")
        return

    results = []
    misclassified = []
    t_start = time.perf_counter()

    for path in image_paths:
        t_img = time.perf_counter()
        prediction, confidence = predict(model, path, transform, class_names,
                                         bad_threshold=args.bad_threshold)
        elapsed_img = time.perf_counter() - t_img

        # Derive true label from parent folder name if it matches a known class.
        true_label = path.parent.name if path.parent.name in class_names else None
        wrong = true_label is not None and prediction != true_label
        marker = "  ✗ WRONG" if wrong else ""
        print(f"{path.name:<40}  {prediction}  (P(bad): {confidence:.3f})"
              f"  [{elapsed_img:.2f}s]{marker}")
        row = {
            "filepath": str(path),
            "prediction": prediction,
            "p_bad": f"{confidence:.4f}",
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
            print(f"  {r['filepath']}  (true={r['true_label']}, pred={r['prediction']})")

    if args.output:
        fieldnames = list(results[0].keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
