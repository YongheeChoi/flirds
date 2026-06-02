"""CNN models for Phase 0 (no LoRA, full-param updates).

LeNet5 for MNIST/FMNIST; FedSVCNN (the FedSV/ComFedSV CIFAR CNN) for CIFAR-10.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, in_ch=1, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FedSVCNN(nn.Module):
    """Two 5x5 conv (32, 64) + FC512 — the FedSV/ComFedSV CIFAR-10 CNN."""

    def __init__(self, in_ch=3, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
