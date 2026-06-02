"""Verify ComFedSV (partial participation + low-rank completion) vs full-obs GT.

Same trajectory + same MC permutations. GT observes ALL prefix coalitions (no
completion); ComFedSV observes only prefixes within each round's K-of-N cohort
and completes the matrix low-rank. ComFedSV's claim: completion recovers the
full-matrix Shapley ranking that FedSV (zero-credit for unselected) cannot.

Run from codes/:  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase0_verify_comfedsv.py
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

from flirds.baselines.comfedsv import comfedsv_from_logs, comfedsv_train
from flirds.data.cnn import client_loaders, get_dataset, get_labels
from flirds.fl.partition import dirichlet_partition
from flirds.models.cnn import LeNet5


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, rounds, E, lr, seed = 8, 15, 1, 0.1, 0
    train = get_dataset("mnist")
    test = get_dataset("mnist", train=False)
    idx = dirichlet_partition(get_labels(train), N, alpha=0.1, seed=seed)
    idx = [ix[:100 * (k + 1)] for k, ix in enumerate(idx)]
    loaders = client_loaders(train, idx, batch_size=64)
    test_loader = DataLoader(Subset(test, range(2000)), batch_size=512)

    model, logs = comfedsv_train(LeNet5, loaders, test_loader, rounds, E, lr,
                                 device, seed, sample_frac=0.375)  # K=3 of 8
    gt = comfedsv_from_logs(logs, model, N, test_loader, device, seed, partial=False)
    com = comfedsv_from_logs(logs, model, N, test_loader, device, seed, rank=5,
                             partial=True)
    cos = float(np.dot(gt, com) / (np.linalg.norm(gt) * np.linalg.norm(com) + 1e-12))
    rho = spearmanr(gt, com).correlation
    print("GT (full obs):", np.round(gt, 4))
    print("ComFedSV     :", np.round(com, 4))
    print(f"cosine={cos:.4f}  spearman={rho:.4f}", "PASS" if rho > 0.9 else "CHECK")


if __name__ == "__main__":
    main()
