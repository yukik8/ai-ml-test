import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloaders
from model import get_model
from utils import ensure_dir, save_confusion_matrix, print_classification_report


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return running_loss / total, correct / total, all_labels, all_preds


def main():
    ensure_dir("outputs")
    ensure_dir("models")

    device = torch.device("cpu")

    train_loader, val_loader, class_names = get_dataloaders(
        data_dir="data/raw",
        batch_size=16,
        image_size=320,
        val_ratio=0.2
    )

    model = get_model(num_classes=2).to(device)

    # Increased bad weight (2.5x vs previous ~1.93x) to penalise missing
    # contaminated samples more heavily and push bad-class recall higher.
    class_counts = {"bad": 350, "good": 1000}
    weights = torch.tensor([
        2.5,
        1350 / (2 * class_counts["good"])
    ], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_acc = 0.0
    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = validate(
            model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/best_model.pth")
            save_confusion_matrix(
                y_true, y_pred, class_names, "outputs/confusion_matrix.png")
            print_classification_report(y_true, y_pred, class_names)
            print("Best model saved.")

    print(f"Training finished. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
