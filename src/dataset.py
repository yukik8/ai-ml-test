"""
Data loading and preprocessing for the logo contamination classifier.

Directory layout expected under data_dir:
    data/raw/
        good/   ← normal product images
        bad/    ← defective product images  (contamination on logo)

Two loaders are provided:
    get_dataloaders()        — for supervised classifier training (train.py)
    get_anomaly_dataloaders() — for PatchCore training (train_anomaly.py)
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_transforms(image_size=224):
    """Return (train_transform, val_transform) for the given resolution.

    Training augmentations are designed for logo images where rotation has
    no effect on the label (a dirty logo rotated 180° is still dirty):
        - RandomRotation(180)          : full rotation invariance
        - RandomAffine(scale=0.95-1.05): slight zoom variation
        - ColorJitter                  : robustness to lighting changes
        - RandomAdjustSharpness        : helps detect fine contamination textures

    Validation uses only resize + normalise — no augmentation — so metrics
    reflect true model performance.
    ImageNet mean/std normalisation is used because the backbone is pre-trained
    on ImageNet.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # Full rotation: logo label is rotation-invariant
        transforms.RandomRotation(180),
        # Small zoom to handle slight distance variation in camera setup
        transforms.RandomAffine(
            degrees=0,
            scale=(0.95, 1.05)
        ),
        # Mild colour jitter for lighting robustness
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15
        ),
        # Occasionally sharpen to emphasise contamination edges
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform, val_transform


def get_dataloaders(data_dir="data/raw", batch_size=16, image_size=224, val_ratio=0.2):
    """Return (train_loader, val_loader, class_names) for classifier training.

    The dataset is split 80/20 train/val with a fixed seed (42) so results are
    reproducible. The same split is reused in train_anomaly.py so PatchCore
    and the classifier are evaluated on identical validation images.
    """
    train_transform, val_transform = get_transforms(image_size)

    base_dataset = datasets.ImageFolder(data_dir)
    class_names = base_dataset.classes  # alphabetical: ['bad', 'good']

    # Deterministic shuffle so the split is stable across runs
    indices = np.arange(len(base_dataset))
    np.random.seed(42)
    np.random.shuffle(indices)

    val_size = int(len(indices) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Two separate ImageFolder instances are needed because train and val
    # use different transforms but share the same underlying files.
    train_dataset_full = datasets.ImageFolder(
        data_dir, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(data_dir, transform=val_transform)

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, class_names


def get_anomaly_dataloaders(data_dir="data/raw", batch_size=16, image_size=320, val_ratio=0.2):
    """Return (good_train_loader, val_loader, class_names) for PatchCore.

    good_train_loader: only 'good' images from the train split, no augmentation.
    val_loader:        all images from the val split (batch_size=1 for per-image scoring).
    """
    _, val_transform = get_transforms(image_size)

    base_dataset = datasets.ImageFolder(data_dir)
    class_names = base_dataset.classes          # alphabetical: ['bad', 'good']
    good_idx = class_names.index("good")

    indices = np.arange(len(base_dataset))
    np.random.seed(42)
    np.random.shuffle(indices)

    val_size = int(len(indices) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    full = datasets.ImageFolder(data_dir, transform=val_transform)

    good_train_indices = [
        i for i in train_indices if full.targets[i] == good_idx]
    good_train_loader = DataLoader(
        Subset(full, good_train_indices), batch_size=batch_size,
        shuffle=False, num_workers=0)

    val_loader = DataLoader(
        Subset(full, val_indices), batch_size=1,
        shuffle=False, num_workers=0)

    return good_train_loader, val_loader, class_names
