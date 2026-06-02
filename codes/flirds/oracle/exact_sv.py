"""(a) Exact retrain Shapley value oracle (dual-oracle path (a)).

U(S) = FedAvg trained on clients in S only, evaluated as test utility.
phi_k = exact Shapley over all 2^N coalitions. Feasible for N<=10 on CNN.

SEPARATE code path from (b) in-run SV oracle — see protocol 4.3:
U_(a)(S) = FL-trained-on-S  vs  U_(b)(S) = sum_r [ell(w^r + sum_{k in S} p_k Δw_k) - ell(w^r)].
Do NOT share utility-computation code between (a) and (b).
"""
from __future__ import annotations

import itertools
from math import factorial

import numpy as np

from ..fl.server import evaluate, fedavg


def subset_utility(model_fn, client_loaders, test_loader, subset, rounds,
                   local_epochs, lr, device="cuda", seed=0, empty_value=None):
    """U(S): FedAvg on `subset` clients only -> final test accuracy."""
    if len(subset) == 0:
        if empty_value is not None:
            return empty_value
        model = model_fn().to(device)
        return evaluate(model, model.state_dict(), test_loader, device)
    loaders = [client_loaders[c] for c in subset]
    _, hist = fedavg(model_fn, loaders, test_loader, rounds, local_epochs, lr,
                     sample_frac=1.0, device=device, seed=seed)
    return hist[-1][1]


def exact_shapley(n_clients, utility_fn):
    """Exact Shapley over 2^n coalitions.

    utility_fn(tuple_of_client_ids) -> float. U(S) is cached (computed once).
    Returns phi array of length n_clients.
    """
    clients = list(range(n_clients))
    u = {
        S: utility_fn(S)
        for r in range(n_clients + 1)
        for S in itertools.combinations(clients, r)
    }
    phi = np.zeros(n_clients)
    for k in clients:
        others = [c for c in clients if c != k]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n_clients - r - 1) / factorial(n_clients)
            for S in itertools.combinations(others, r):
                phi[k] += w * (u[tuple(sorted(S + (k,)))] - u[S])
    return phi
