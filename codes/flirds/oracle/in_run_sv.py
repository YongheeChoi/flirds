"""(b) IRDS-定 in-run Shapley oracle (dual-oracle path (b)).

U_(b)(S) = sum_r [ ell(w^r + sum_{k in S} p_k Δw_k, z^val) - ell(w^r, z^val) ]
on the FROZEN FedAvg trajectory (logs from fl.server.run_fedavg_logs).  Exact
Shapley over 2^N coalitions.  p_k = n_k / sum_j n_j (fixed FedAvg weights, all
clients participate cross-silo).

SEPARATE code path from (a) oracle/exact_sv retrain (protocol 4.3): the two
utilities are DIFFERENT functions and share no utility-computation code
  U_(a)(S) = test-acc of FedAvg retrained on S
  U_(b)(S) = trajectory-frozen val-loss change from S's weighted deltas.
fp32 forward (protocol 1; CNN params/data are fp32 by default).
"""
from __future__ import annotations

import itertools
from math import factorial

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def _val_loss(model, state, val_x, val_y, device):
    model.load_state_dict(state)
    model.to(device).eval()
    return float(F.cross_entropy(model(val_x), val_y))


def _perturbed(w_r, dm, subset, p, pkeys):
    """w^r + sum_{k in subset} p_k Δw_k  (trainable params only; buffers from w^r)."""
    pert = {kk: v.clone() for kk, v in w_r.items()}
    for k in subset:
        dwk = dm[k][0]
        for name in pkeys:
            pert[name] = pert[name] + p[k] * dwk[name].to(pert[name].device)
    return pert


def in_run_utility(logs, subset, model_fn, val_x, val_y, p, device):
    """U_(b)(S) on the frozen trajectory.  p = {client: weight}."""
    model = model_fn().to(device)
    pkeys = [n for n, _ in model.named_parameters()]
    subset = tuple(subset)
    total = 0.0
    for w_r, dm in logs:
        base = _val_loss(model, w_r, val_x, val_y, device)
        if subset:
            total += _val_loss(model, _perturbed(w_r, dm, subset, p, pkeys),
                               val_x, val_y, device) - base
    return total


def in_run_shapley(logs, n_clients, model_fn, val_x, val_y, device):
    """Exact (b) in-run Shapley values over `logs`.  Returns (phi[n], p[n])."""
    model = model_fn().to(device)
    pkeys = [n for n, _ in model.named_parameters()]
    ns = {k: logs[0][1][k][1] for k in range(n_clients)}
    tot = sum(ns.values())
    p = {k: ns[k] / tot for k in ns}
    base = [_val_loss(model, w_r, val_x, val_y, device) for w_r, _ in logs]

    def utility(S):
        if not S:
            return 0.0
        t = 0.0
        for (w_r, dm), b in zip(logs, base):
            t += _val_loss(model, _perturbed(w_r, dm, S, p, pkeys), val_x, val_y, device) - b
        return t

    clients = list(range(n_clients))
    U = {S: utility(S)
         for r in range(n_clients + 1)
         for S in itertools.combinations(clients, r)}
    phi = np.zeros(n_clients)
    for k in clients:
        others = [c for c in clients if c != k]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n_clients - r - 1) / factorial(n_clients)
            for S in itertools.combinations(others, r):
                phi[k] += w * (U[tuple(sorted(S + (k,)))] - U[S])
    return phi, np.array([p[k] for k in clients])
