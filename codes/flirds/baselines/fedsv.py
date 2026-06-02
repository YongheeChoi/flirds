"""FedSV (Wang et al. 2020, "A Principled Approach to Data Valuation for FL").

The origin of the federated Shapley value. No official code -> self-build from
the paper (Alg.2 permutation Monte-Carlo). Per-round canonical Shapley over the
participating cohort, summed over rounds. Same sub-model reconstruction utility
U(S) as GTG (re-uses _aggregate_subset); difference is the SV estimator:
plain permutation-MC (+ optional TMC truncation) and an optional per-round
norm-normalization (the paper's "normalized FedSV", better for detection).
"""
from __future__ import annotations

import numpy as np

from ..fl.server import evaluate, run_fedavg_logs
from .gtg import _aggregate_subset


def _round_permutation_sv(n_players, last_metric, full_metric, metric_fun,
                          n_perm, rng, trunc_eps=0.001):
    """Permutation-MC Shapley for one round. metric_fun(subset_idx_tuple)->U."""
    sv = np.zeros(n_players)
    cache = {(): last_metric}
    for _ in range(n_perm):
        perm = rng.permutation(n_players)
        prev = last_metric
        for j in range(n_players):
            # TMC truncation: negligible remaining gain -> carry value forward
            if trunc_eps > 0 and abs(full_metric - prev) < trunc_eps:
                cur = prev
            else:
                sub = tuple(sorted(perm[: j + 1].tolist()))
                if sub not in cache:
                    cache[sub] = metric_fun(sub)
                cur = cache[sub]
            sv[perm[j]] += cur - prev
            prev = cur
    return sv / n_perm


def fedsv_from_logs(logs, model, n_clients, test_loader, device, seed=0,
                    n_perm=None, normalized=False, trunc_eps=0.001):
    """FedSV per-round permutation-MC Shapley from a shared FedAvg trajectory."""
    rng = np.random.default_rng(seed)
    phi = np.zeros(n_clients)
    for gb, dm in logs:
        players = sorted(dm.keys())
        m = n_perm or max(30, 2 * len(players))
        last_m = evaluate(model, gb, test_loader, device)
        full_m = evaluate(model, _aggregate_subset(gb, dm, players, device),
                          test_loader, device)

        def metric_fun(sub_idx, gb=gb, dm=dm, players=players):
            st = _aggregate_subset(gb, dm, [players[i] for i in sub_idx], device)
            return evaluate(model, st, test_loader, device)

        rsv = _round_permutation_sv(len(players), last_m, full_m, metric_fun, m, rng,
                                    trunc_eps=trunc_eps)
        if normalized:
            rsv = rsv / (np.linalg.norm(rsv) or 1.0)
        for i, p in enumerate(players):
            phi[p] += rsv[i]
    return phi


def fedsv_shapley(model_fn, client_loaders, test_loader, rounds, local_epochs, lr,
                  device="cuda", seed=0, n_perm=None, normalized=False,
                  trunc_eps=0.001):
    """Convenience: run FedAvg then FedSV. Returns total phi over rounds."""
    model, logs = run_fedavg_logs(model_fn, client_loaders, test_loader, rounds,
                                  local_epochs, lr, device, seed)
    return fedsv_from_logs(logs, model, len(client_loaders), test_loader, device,
                           seed, n_perm, normalized, trunc_eps)
