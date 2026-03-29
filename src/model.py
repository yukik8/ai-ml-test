"""
Model definition — ResNet-18 for binary image classification.

ResNet-18 pre-trained on ImageNet is used as a feature extractor.
Only the final fully-connected layer is replaced to output `num_classes` logits.
Fine-tuning all layers (not just the head) lets the network adapt its low-level
features to the specific texture of logo contamination.
"""

import torch.nn as nn
from torchvision import models


def get_model(num_classes=2):
    """Return a ResNet-18 with its classification head replaced.

    All layers remain trainable so the backbone can adapt to the domain.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace the 1000-class ImageNet head with a 2-class (good/bad) head.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
