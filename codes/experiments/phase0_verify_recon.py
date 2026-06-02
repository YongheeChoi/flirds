"""Verify recon-based FL Shapley baselines (GTG, FedSV) vs exact reconstruction SV.

All baselines value the SAME shared FedAvg trajectory (one run -> logs). Each is
compared to the exact 2^N enumeration under ITS OWN per-round convention:
  - FedSV = raw per-round Shapley (efficiency is automatic)        -> exact_raw
  - GTG   = efficiency-normalized round Shapley (cyyever design)   -> exact_norm
Small N for tractable ground truth. Spearman checks ranking; cosine checks values.

Run from codes/:  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase0_verify_recon.py
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

from flirds.baselines.fedsv import fedsv_from_logs
from flirds.baselines.gtg import _aggregate_subset, _normalize, gtg_from_logs
from flirds.data.cnn import client_loaders, get_dataset, get_labels
from flirds.fl.partition import dirichlet_partition
from flirds.fl.server import evaluate, run_fedavg_logs
from flirds.models.cnn import LeNet5
from flirds.oracle.exact_sv import exact_shapley


def exact_recon_from_logs(logs, model, n_clients, test_loader, device, normalize=False):
    """Exact reconstruction Shapley per round (raw, or efficiency-normalized)."""
    phi = np.zeros(n_clients)
    for gb, dm in logs:
        players = sorted(dm.keys())

        def util(sub_idx, gb=gb, dm=dm, players=players):
            st = _aggregate_subset(gb, dm, [players[i] for i in sub_idx], device)
            return evaluate(model, st, test_loader, device)

        raw = exact_shapley(len(players), util)
        if normalize:
            raw = _normalize(raw, util(tuple(range(len(players)))) - util(()))
        for i, p in enumerate(players):
            phi[p] += raw[i]
    return phi


def _report(name, est, exact):
    cos = float(np.dot(exact, est) /
                (np.linalg.norm(exact) * np.linalg.norm(est) + 1e-12))
    rho = spearmanr(exact, est).correlation
    flag = "PASS" if cos > 0.99 else "CHECK"
    print(f"{name}: {np.round(est, 4)}  cosine={cos:.4f} spearman={rho:.4f} {flag}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, rounds, E, lr, seed = 8, 6, 1, 0.1, 0
    train = get_dataset("mnist")
    test = get_dataset("mnist", train=False)
    idx = dirichlet_partition(get_labels(train), N, alpha=0.1, seed=seed)
    idx = [ix[:100 * (k + 1)] for k, ix in enumerate(idx)]  # graded 100..500
    loaders = client_loaders(train, idx, batch_size=64)
    test_loader = DataLoader(Subset(test, range(2000)), batch_size=512)

    model, logs = run_fedavg_logs(LeNet5, loaders, test_loader, rounds, E, lr,
                                  device, seed)
    exact_raw = exact_recon_from_logs(logs, model, N, test_loader, device, False)
    exact_norm = exact_recon_from_logs(logs, model, N, test_loader, device, True)
    print("exact raw :", np.round(exact_raw, 4))
    print("exact norm:", np.round(exact_norm, 4))
    _report("FedSV(raw) ",
            fedsv_from_logs(logs, model, N, test_loader, device, seed,
                            n_perm=300, trunc_eps=0.0), exact_raw)
    _report("GTG(trunc) ",
            gtg_from_logs(logs, model, N, test_loader, device, seed,
                          normalize=True), exact_norm)
    _report("GTG(exact) ",
            gtg_from_logs(logs, model, N, test_loader, device, seed,
                          normalize=True, eps=0.0), exact_norm)


if __name__ == "__main__":
    main()
