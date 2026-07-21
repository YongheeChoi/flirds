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
import torch
import torch.nn.functional as F

from ..fl.server import evaluate, fedavg
from ..repro import seed_everything


def subset_utility(model_fn, client_loaders, test_loader, subset, rounds,
                   local_epochs, lr, device="cuda", seed=0, empty_value=None):
    """U(S): FedAvg on `subset` clients only -> final test accuracy."""
    if len(subset) == 0:
        if empty_value is not None:
            return empty_value
        seed_everything(seed, cudnn_deterministic=True)   # deterministic init (match subset_utility_valloss)
        model = model_fn().to(device)
        return evaluate(model, model.state_dict(), test_loader, device)
    loaders = [client_loaders[c] for c in subset]
    _, hist = fedavg(model_fn, loaders, test_loader, rounds, local_epochs, lr,
                     sample_frac=1.0, device=device, seed=seed)
    return hist[-1][1]


@torch.no_grad()
def _val_loss(model, state, loader, device):
    """Mean cross-entropy of `state` over `loader` (fp32)."""
    model.load_state_dict(state)
    model.to(device).eval()
    tot = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        tot += F.cross_entropy(model(x), y, reduction="sum").item()
        n += y.size(0)
    return tot / n


def subset_utility_valloss(model_fn, client_loaders, val_loader, subset, rounds,
                           local_epochs, lr, device="cuda", seed=0, empty_value=None):
    """U_(a)(S) = -val-loss of the FedAvg-on-S final model (Track C1 PRIMARY (a)
    utility; good -> high).  Same game as the (b) oracle / estimator -- the task-6
    lesson: (a) validates the Shapley computation only when it plays the utility the
    estimator targets.  `subset_utility` (test-acc) stays as the secondary/appendix
    utility.  Only the FINAL state is scored on the server-held val split
    (eval_every > rounds skips fedavg's per-round test eval -- the 2^N sweep cost).
    Empty S -> the seed-deterministic init model's score (the trajectory's w_0),
    or `empty_value` if given.
    """
    if len(subset) == 0:
        if empty_value is not None:
            return empty_value
        seed_everything(seed, cudnn_deterministic=True)   # match fedavg's init RNG
        model = model_fn()
        return -_val_loss(model, model.state_dict(), val_loader, device)
    loaders = [client_loaders[c] for c in subset]
    final, _ = fedavg(model_fn, loaders, None, rounds, local_epochs, lr,
                      sample_frac=1.0, device=device, seed=seed,
                      eval_every=rounds + 1)
    return -_val_loss(model_fn(), final, val_loader, device)


def exact_shapley(n_clients, utility_fn, return_u=False):
    """Exact Shapley over 2^n coalitions.

    utility_fn(tuple_of_client_ids) -> float. U(S) is cached (computed once).
    Returns phi array of length n_clients.

    return_u=True -> (phi, u): u maps every coalition tuple (sorted client ids) to
    its cached utility U(S).  Lets a caller derive removal/selection curves by lookup
    from the same 2^N retrains -- no extra retraining (Exp A1 removal-curve reuse).
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
    return (phi, u) if return_u else phi
