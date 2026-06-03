"""Phase 0.5 close: dual-oracle cross-check + Shapley-axiom & reproducibility gates.

(a) retrain SV  (oracle/exact_sv : U = FedAvg-retrained-on-S test acc)  counterfactual
(b) in-run SV   (oracle/in_run_sv: U = frozen-trajectory val-loss change)
estimator       (core/flirds_estimator: 1st & 1st+2nd Taylor of (b))

Values oriented good->high (loss-based ones negated) for comparison.  Gates:
  - dual-oracle agreement: Spearman / noisy-AUROC across (a), (b), estimator
  - (b) Shapley efficiency:  sum_k phi_k == U(grand) - U(empty)
  - (b) Shapley symmetry:    two identical clients get equal phi
  - estimator vs (b):        3-seed Spearman mean+-std (the headline)
  - reproducibility:         same config+seed -> identical estimator output

Run from codes/:  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase05_dual_oracle.py
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
from flirds.oracle.exact_sv import exact_shapley, subset_utility
from flirds.oracle.in_run_sv import in_run_shapley, in_run_utility
from flirds.repro import seed_everything


def build(N, noisy, n_per, seed, bs=32, dup=None):
    train = get_dataset("mnist")
    idx = dirichlet_partition(get_labels(train), N, alpha=100.0, seed=seed)
    idx = [i[:n_per] for i in idx]
    if dup is not None:                      # client dup[1] := copy of client dup[0]
        idx[dup[1]] = idx[dup[0]]
    loaders = []
    for c in range(N):
        xs = torch.stack([train[i][0] for i in idx[c]])
        ys = torch.tensor([train[i][1] for i in idx[c]])
        if c in noisy:
            g = torch.Generator().manual_seed(100 + c)
            ys = ys[torch.randperm(len(ys), generator=g)]
        loaders.append(DataLoader(TensorDataset(xs, ys), batch_size=bs, shuffle=(bs < n_per)))
    return loaders


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(0)   # cudnn-deterministic + seeds (fl.server.fedavg re-seeds per run)
    N, rounds, E, lr, n_per = 6, 3, 2, 0.05, 300
    noisy = {4, 5}
    test = get_dataset("mnist", train=False)
    vx = torch.stack([test[i][0] for i in range(512)]).to(device)
    vy = torch.tensor([test[i][1] for i in range(512)]).to(device)
    tl = DataLoader(test, batch_size=256)

    # ---- dual-oracle cross-check (seed 0) ----
    loaders = build(N, noisy, n_per, seed=0)
    _, logs = run_fedavg_logs(LeNet5, loaders, tl, rounds, E, lr, device=device, seed=0)
    phi_b, p = in_run_shapley(logs, N, LeNet5, vx, vy, device)
    phi_e2, _ = flirds_values(logs, LeNet5, vx, vy, device, second_order=True)
    phi_a = exact_shapley(
        N, lambda S: subset_utility(LeNet5, loaders, tl, S, rounds, E, lr, device=device, seed=0))

    va, vb, ve = phi_a, -phi_b, -phi_e2     # orient good -> high
    y = [1 if c in noisy else 0 for c in range(N)]
    print("== dual-oracle cross-check (N=6, good->high) ==")
    print(f"  (a) retrain : {np.round(va, 4)}")
    print(f"  (b) in-run  : {np.round(vb, 4)}")
    print(f"  estimator   : {np.round(ve, 4)}")
    print(f"  Spearman  (a)~(b)={spearmanr(va, vb).correlation:.3f}  "
          f"(a)~est={spearmanr(va, ve).correlation:.3f}  (b)~est={spearmanr(vb, ve).correlation:.3f}")
    print(f"  noisy AUROC  (a)={roc_auc_score(y, -va):.3f}  (b)={roc_auc_score(y, -vb):.3f}  "
          f"est={roc_auc_score(y, -ve):.3f}")

    # ---- gate: (b) Shapley efficiency ----
    ug = in_run_utility(logs, range(N), LeNet5, vx, vy, {k: p[k] for k in range(N)}, device)
    print("== gate: (b) Shapley efficiency  sum(phi)==U(grand)-U(empty) ==")
    print(f"  sum phi_b={phi_b.sum():.6f}  U(grand)={ug:.6f}  |diff|={abs(phi_b.sum() - ug):.2e}")

    # ---- gate: (b) Shapley symmetry (clients 0,1 identical, deterministic full-batch) ----
    ld_s = build(N, set(), n_per, seed=1, bs=n_per, dup=(0, 1))
    _, logs_s = run_fedavg_logs(LeNet5, ld_s, tl, rounds, 1, lr, device=device, seed=1)
    phi_s, _ = in_run_shapley(logs_s, N, LeNet5, vx, vy, device)
    print("== gate: (b) Shapley symmetry  phi_0==phi_1 (identical clients) ==")
    print(f"  phi_0={phi_s[0]:.6f}  phi_1={phi_s[1]:.6f}  |diff|={abs(phi_s[0] - phi_s[1]):.2e}")

    # ---- headline: estimator vs (b), 3 seeds ----
    print("== estimator vs (b) oracle: 3-seed Spearman (good->high) ==")
    s1, s2 = [], []
    for sd in [42, 123, 2024]:
        ld = build(N, noisy, n_per, seed=sd)
        _, lg = run_fedavg_logs(LeNet5, ld, tl, rounds, E, lr, device=device, seed=sd)
        b, _ = in_run_shapley(lg, N, LeNet5, vx, vy, device)
        e1, _ = flirds_values(lg, LeNet5, vx, vy, device, second_order=False)
        e2, _ = flirds_values(lg, LeNet5, vx, vy, device, second_order=True)
        s1.append(spearmanr(-e1, -b).correlation)
        s2.append(spearmanr(-e2, -b).correlation)
    print(f"  1st-only : {np.mean(s1):.3f} +- {np.std(s1):.3f}   (seeds {np.round(s1, 3)})")
    print(f"  1st+2nd  : {np.mean(s2):.3f} +- {np.std(s2):.3f}   (seeds {np.round(s2, 3)})")

    # ---- gate: reproducibility (fresh loaders, same seed) ----
    _, g1 = run_fedavg_logs(LeNet5, build(N, noisy, n_per, seed=7), tl, rounds, E, lr,
                            device=device, seed=7)
    _, g2 = run_fedavg_logs(LeNet5, build(N, noisy, n_per, seed=7), tl, rounds, E, lr,
                            device=device, seed=7)
    r1, _ = flirds_values(g1, LeNet5, vx, vy, device, second_order=True)
    r2, _ = flirds_values(g2, LeNet5, vx, vy, device, second_order=True)
    print("== gate: reproducibility (same config+seed, cudnn-deterministic) ==")
    print(f"  max|phi_run1 - phi_run2| = {np.max(np.abs(r1 - r2)):.2e}")


if __name__ == "__main__":
    main()
