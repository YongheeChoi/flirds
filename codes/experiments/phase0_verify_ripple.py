"""Verify full Ripple Shapley (drop + ripple) via noisy-client detection.

Ripple has NO ground-truth SV (paper rejects MSE/correlation), so we use the
task-driven check: label-shuffled (noisy) clients should receive LOW phi, clean
clients HIGH phi -> high detection AUROC.

Run from codes/:  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase0_verify_ripple.py
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from flirds.baselines.ripple import ripple_shapley
from flirds.data.cnn import get_dataset, get_labels
from flirds.fl.partition import dirichlet_partition
from flirds.models.cnn import LeNet5


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, rounds, E, lr, seed = 8, 8, 2, 0.01, 0
    noisy = {6, 7}
    train = get_dataset("mnist")
    test = get_dataset("mnist", train=False)
    idx = dirichlet_partition(get_labels(train), N, alpha=100.0, seed=seed)  # ~IID
    idx = [i[:300] for i in idx]
    loaders = []
    for c in range(N):
        xs = torch.stack([train[i][0] for i in idx[c]])
        ys = torch.tensor([train[i][1] for i in idx[c]])
        if c in noisy:
            g = torch.Generator().manual_seed(100 + c)
            ys = ys[torch.randperm(len(ys), generator=g)]
        loaders.append(DataLoader(TensorDataset(xs, ys), batch_size=32))
    vx = torch.stack([test[i][0] for i in range(512)]).to(device)
    vy = torch.tensor([test[i][1] for i in range(512)]).to(device)

    phi = ripple_shapley(LeNet5, loaders, rounds, E, lr, vx, vy, device,
                         seed=seed, k=5, m=20, R=5)  # full: drop + ripple
    y = [1 if c in noisy else 0 for c in range(N)]
    auc = roc_auc_score(y, -np.asarray(phi))  # noisy -> low phi -> high -phi
    print("phi:", np.round(phi, 4))
    print(f"noisy detection AUROC={auc:.4f}", "PASS" if auc > 0.75 else "CHECK")


if __name__ == "__main__":
    main()
