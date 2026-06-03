"""Phase 0.5 sanity gates for the Flirds estimator (core/flirds_estimator).

Gate E=1 (Taylor residual): one full-batch GD step per round.  The estimator must
match the exact (b) oracle to a small residual.  NOTE: as lr -> 0 the oracle's
loss-difference U(S) cancels in fp32 (protocol 1 noise floor ~3e-3), so relL2
plateaus rather than -> 0; the 2nd-order advantage is visible only at moderate lr
(e.g. lr=0.2) where the step is above the noise floor and within the Taylor radius.
The broad trend (2nd-order helps with curvature / within radius) lives in the
regime sweep, not this gate.

Gate N=2 (singleton): at N=2 the estimator must match the exact 2-coalition
in-run Shapley at small step.

Run from codes/:  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase05_sanity.py
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from flirds.backends.cnn import make_cnn_loss
from flirds.core.flirds_estimator import flirds_values
from flirds.data.cnn import get_dataset, get_labels
from flirds.fl.partition import dirichlet_partition
from flirds.fl.server import run_fedavg_logs
from flirds.models.cnn import LeNet5
from flirds.oracle.in_run_sv import in_run_shapley
from flirds.repro import seed_everything


def make(N, n_per, seed, full_batch=True):
    train = get_dataset("mnist")
    idx = dirichlet_partition(get_labels(train), N, alpha=100.0, seed=seed)
    idx = [i[:n_per] for i in idx]
    bs = n_per if full_batch else 32
    loaders = []
    for c in range(N):
        xs = torch.stack([train[i][0] for i in idx[c]])
        ys = torch.tensor([train[i][1] for i in idx[c]])
        loaders.append(DataLoader(TensorDataset(xs, ys), batch_size=bs))
    return loaders


def relL2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(0)
    test = get_dataset("mnist", train=False)
    vx = torch.stack([test[i][0] for i in range(512)]).to(device)
    vy = torch.tensor([test[i][1] for i in range(512)]).to(device)
    tl = DataLoader(test, batch_size=256)
    loss_fn, pkeys = make_cnn_loss(LeNet5, vx, vy, device)

    print("Gate E=1 (single full-batch GD step/round; relL2 vs (b) oracle):")
    print(f"  {'lr':>7} {'1st-only':>10} {'1st+2nd':>10}")
    N, rounds = 6, 3
    loaders = make(N, 200, seed=0, full_batch=True)
    for lr in [0.2, 0.05, 0.01, 0.002]:
        _, logs = run_fedavg_logs(LeNet5, loaders, tl, rounds, 1, lr, device=device, seed=0)
        oracle, _ = in_run_shapley(logs, N, loss_fn, pkeys, device)
        p1, _ = flirds_values(logs, loss_fn, pkeys, device, second_order=False)
        p2, _ = flirds_values(logs, loss_fn, pkeys, device, second_order=True)
        print(f"  {lr:>7} {relL2(p1, oracle):>10.5f} {relL2(p2, oracle):>10.5f}")

    print("Gate N=2 (singleton; small step):")
    loaders = make(2, 200, seed=1, full_batch=True)
    _, logs = run_fedavg_logs(LeNet5, loaders, tl, 3, 1, 0.01, device=device, seed=1)
    oracle, _ = in_run_shapley(logs, 2, loss_fn, pkeys, device)
    p2, _ = flirds_values(logs, loss_fn, pkeys, device, second_order=True)
    print("  oracle :", np.round(oracle, 6))
    print("  1st+2nd:", np.round(p2, 6))
    print("  relL2  :", round(relL2(p2, oracle), 5))


if __name__ == "__main__":
    main()
