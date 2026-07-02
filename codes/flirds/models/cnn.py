"""CNN models for Phase 0 (no LoRA, full-param updates).

LeNet5 for MNIST/FMNIST; FedSVCNN (the FedSV/ComFedSV CIFAR CNN) for CIFAR-10.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, in_ch=1, n_classes=10, width=1.0):
        super().__init__()
        c1, c2, f1, f2 = (max(1, round(w * width)) for w in (6, 16, 120, 84))
        self.conv1 = nn.Conv2d(in_ch, c1, 5, padding=2)
        self.conv2 = nn.Conv2d(c1, c2, 5)
        self.fc1 = nn.Linear(c2 * 5 * 5, f1)
        self.fc2 = nn.Linear(f1, f2)
        self.fc3 = nn.Linear(f2, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FedSVCNN(nn.Module):
    """Two 5x5 conv (32, 64) + FC512 — the FedSV/ComFedSV CIFAR-10 CNN."""

    def __init__(self, in_ch=3, n_classes=10, width=1.0):
        super().__init__()
        c1, c2, f1 = (max(1, round(w * width)) for w in (32, 64, 512))
        self.conv1 = nn.Conv2d(in_ch, c1, 5, padding=2)
        self.conv2 = nn.Conv2d(c1, c2, 5, padding=2)
        self.fc1 = nn.Linear(c2 * 8 * 8, f1)
        self.fc2 = nn.Linear(f1, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
