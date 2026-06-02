"""CNN datasets + per-client loaders for Phase 0."""
from __future__ import annotations

import os

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fmnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


def get_dataset(name, root="~/data", train=True):
    name = name.lower()
    mean, std = _STATS[name]
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    root = os.path.expanduser(root)
    if name == "mnist":
        return datasets.MNIST(root, train=train, download=True, transform=tf)
    if name == "fmnist":
        return datasets.FashionMNIST(root, train=train, download=True, transform=tf)
    if name == "cifar10":
        return datasets.CIFAR10(root, train=train, download=True, transform=tf)
    raise ValueError(f"unknown dataset: {name}")


def get_labels(dataset):
    return np.asarray(dataset.targets)


def client_loaders(dataset, client_idx, batch_size=64, shuffle=True):
    return [
        DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=shuffle)
        for idx in client_idx
    ]
