from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # Rotation is irrelevant to the label, so augmenting with full rotation
        # forces the model to ignore orientation and focus on contaminants instead.
        transforms.RandomRotation(degrees=360),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # Simulate lighting variation so contaminant features generalise better.
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
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
    train_transform, val_transform = get_transforms(image_size)

    base_dataset = datasets.ImageFolder(data_dir)
    class_names = base_dataset.classes

    indices = np.arange(len(base_dataset))
    np.random.seed(42)
    np.random.shuffle(indices)

    val_size = int(len(indices) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

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
