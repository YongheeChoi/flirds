"""Phase 0.5: Flirds estimator vs (b) in-run SV oracle on CNN.

Validates the 1st+2nd Taylor estimator (core/flirds_estimator) against the exact
in-run Shapley (oracle/in_run_sv) on a shared frozen FedAvg trajectory, and shows
the 2nd-order term tightens the match.  Noisy-client detection AUROC as a sanity.

Run from codes/:  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase05_flirds_oracle.py
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from flirds.core.flirds_estimator import flirds_values
from flirds.data.cnn import get_dataset, get_labels
from flirds.fl.partition import dirichlet_partition
from flirds.fl.server import run_fedavg_logs
from flirds.models.cnn import LeNet5
from flirds.oracle.in_run_sv import in_run_shapley
from flirds.repro import seed_everything


def build(N, noisy, n_per=300, seed=0):
    train = get_dataset("mnist")
    test = get_dataset("mnist", train=False)
    idx = dirichlet_partition(get_labels(train), N, alpha=100.0, seed=seed)
    idx = [i[:n_per] for i in idx]
    loaders = []
    for c in range(N):
        xs = torch.stack([train[i][0] for i in idx[c]])
        ys = torch.tensor([train[i][1] for i in idx[c]])
        if c in noisy:
            g = torch.Generator().manual_seed(100 + c)
            ys = ys[torch.randperm(len(ys), generator=g)]
        loaders.append(DataLoader(TensorDataset(xs, ys), batch_size=32, shuffle=True))
    return loaders, test


def report(name, est, oracle):
    rho = spearmanr(est, oracle).correlation
    cos = float(np.dot(est, oracle) / (np.linalg.norm(est) * np.linalg.norm(oracle)))
    rel = float(np.linalg.norm(est - oracle) / np.linalg.norm(oracle))
    print(f"  {name:9s}: spearman={rho:.4f}  cosine={cos:.4f}  relL2={rel:.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(0)
    N, rounds, E, lr, seed = 8, 5, 2, 0.05, 0
    noisy = {6, 7}
    loaders, test = build(N, noisy, seed=seed)
    test_loader = DataLoader(test, batch_size=256)
    vx = torch.stack([test[i][0] for i in range(512)]).to(device)
    vy = torch.tensor([test[i][1] for i in range(512)]).to(device)

    _, logs = run_fedavg_logs(LeNet5, loaders, test_loader, rounds, E, lr,
                              device=device, seed=seed)

    print(f"(b) in-run oracle: exact Shapley over 2^{N}={2**N} coalitions ...")
    oracle, _ = in_run_shapley(logs, N, LeNet5, vx, vy, device)
    phi1, _ = flirds_values(logs, LeNet5, vx, vy, device, second_order=False)
    phi2, _ = flirds_values(logs, LeNet5, vx, vy, device, second_order=True)

    print("Flirds estimator vs (b) oracle:")
    report("1st-only", phi1, oracle)
    report("1st+2nd", phi2, oracle)

    y = [1 if c in noisy else 0 for c in range(N)]
    print("noisy detection AUROC (higher phi = more val-loss = noisy):")
    for nm, ph in [("oracle", oracle), ("1st-only", phi1), ("1st+2nd", phi2)]:
        print(f"  {nm:9s}: AUROC={roc_auc_score(y, ph):.4f}")
    print("oracle  phi:", np.round(oracle, 5))
    print("1st+2nd phi:", np.round(phi2, 5))


if __name__ == "__main__":
    main()
