"""(b) IRDS-定 in-run Shapley oracle (dual-oracle path (b)).

U_(b)(S) = sum_r [ ell(w^r + sum_{k in S∩P_r} p_k^r Δw_k) - ell(w^r) ]  (ell = loss_fn)
on the FROZEN FedAvg trajectory (logs from fl.server.run_fedavg_logs).  Exact
Shapley over 2^N coalitions.  P_r = round-r participants (deltas_map.keys());
p_k^r = n_k / Σ_{j∈P_r} n_j is the FedAvg participant-normalized weight for round
r -- matching fl.server.fedavg's aggregate, so the perturbation equals the
realized update when S ⊇ P_r.  Reduces to a fixed p_k = n_k/Σn under full
participation (cross-silo); correct under partial participation (cross-device).

Backend-agnostic like the estimator: the val loss is loss_fn(params, buffers)
(see flirds.backends.*), pkeys lists the trainable param names.  Forward only,
under @no_grad (the same loss_fn the estimator differentiates).

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


def _round_weight(dm):
    """FedAvg participant-normalized weight p_k^r = n_k / Σ_{j∈P_r} n_j for round r."""
    tot = sum(n for _, n in dm.values())
    return {k: n / tot for k, (_, n) in dm.items()}


def _split(w_r, pkeys, device):
    """Split a logged state dict into (params, buffers), moved to device."""
    params = {n: w_r[n].to(device) for n in pkeys}
    buffers = {n: w_r[n].to(device) for n in w_r if n not in pkeys}
    return params, buffers


def _perturbed_params(base_params, dm, subset, pr, pkeys):
    """base_params + sum_{k in subset} pr[k] Δw_k  (params only; subset ⊆ P_r)."""
    params = {n: base_params[n].clone() for n in pkeys}
    for k in subset:
        dwk = dm[k][0]
        for name in pkeys:
            params[name] = params[name] + pr[k] * dwk[name].to(params[name].device)
    return params


@torch.no_grad()
def in_run_utility(logs, subset, loss_fn, pkeys, device):
    """U_(b)(S) on the frozen trajectory.  Per-round weights are read from `logs`."""
    sset = set(subset)
    total = 0.0
    for w_r, dm in logs:
        players = [k for k in dm if k in sset]
        if not players:
            continue
        pr = _round_weight(dm)
        base_params, buffers = _split(w_r, pkeys, device)
        base = float(loss_fn(base_params, buffers))
        pert = _perturbed_params(base_params, dm, players, pr, pkeys)
        total += float(loss_fn(pert, buffers)) - base
    return total


@torch.no_grad()
def in_run_shapley(logs, n_clients, loss_fn, pkeys, device):
    """Exact (b) in-run Shapley values over `logs`.  Returns (phi[n], p[n]).

    p is the global weight n_k/Σn (exact under full participation), returned for
    reference; the utility uses per-round participant weights internally."""
    client_n = {}
    for _, dm in logs:
        for k, (_, n) in dm.items():
            client_n.setdefault(k, n)
    tot = sum(client_n.values())
    p = np.array([client_n.get(k, 0.0) / tot for k in range(n_clients)])

    split = [_split(w_r, pkeys, device) for w_r, _ in logs]
    base = [float(loss_fn(bp, bf)) for bp, bf in split]
    round_w = [_round_weight(dm) for _, dm in logs]

    def utility(S):
        if not S:
            return 0.0
        sset = set(S)
        t = 0.0
        for (w_r, dm), (bp, bf), b, pr in zip(logs, split, base, round_w):
            players = [k for k in dm if k in sset]
            if players:
                pert = _perturbed_params(bp, dm, players, pr, pkeys)
                t += float(loss_fn(pert, bf)) - b
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
    return phi, p
