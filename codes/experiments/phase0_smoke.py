"""Phase 0 step 1 smoke test: vanilla FedAvg converges on MNIST / CIFAR-10.

Verify gate: MNIST IID FedAvg -> ~97% test acc; CIFAR-10 -> ~77%.

Run from codes/:  PYTHONPATH=. python experiments/phase0_smoke.py --dataset mnist
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from flirds.data.cnn import client_loaders, get_dataset, get_labels
from flirds.fl.partition import iid_partition
from flirds.fl.server import fedavg
from flirds.models.cnn import FedSVCNN, LeNet5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fmnist", "cifar10"])
    ap.add_argument("--n_clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--sample_frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = get_dataset(args.dataset, train=True)
    test = get_dataset(args.dataset, train=False)
    idx = iid_partition(get_labels(train), args.n_clients, seed=args.seed)
    loaders = client_loaders(train, idx, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=256)

    if args.dataset in ("mnist", "fmnist"):
        model_fn = lambda: LeNet5(in_ch=1)
    else:
        model_fn = lambda: FedSVCNN(in_ch=3)

    _, hist = fedavg(model_fn, loaders, test_loader, args.rounds,
                     args.local_epochs, args.lr, args.sample_frac, device, args.seed)
    for r, acc in hist:
        print(f"round {r:3d}  test_acc {acc:.4f}")
    print(f"FINAL {args.dataset} acc={hist[-1][1]:.4f}")


if __name__ == "__main__":
    main()
