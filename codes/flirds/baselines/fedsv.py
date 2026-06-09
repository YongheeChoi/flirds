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

from .gtg import _round_metrics


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
                    n_perm=None, normalized=False, trunc_eps=0.001,
                    loss_fn=None, pkeys=None):
    """FedSV per-round permutation-MC Shapley from a shared FedAvg trajectory.

    Backend-agnostic like gtg_from_logs: (model, test_loader) for the CNN accuracy
    metric (default), or (loss_fn, pkeys) for the LLM val-loss metric (model/
    test_loader then None).  trunc_eps is in the metric's units."""
    rng = np.random.default_rng(seed)
    phi = np.zeros(n_clients)
    for gb, dm in logs:
        players = sorted(dm.keys())
        m = n_perm or max(30, 2 * len(players))
        last_m, full_m, metric_fun = _round_metrics(
            gb, dm, players, model, test_loader, device, loss_fn, pkeys)
        rsv = _round_permutation_sv(len(players), last_m, full_m, metric_fun, m, rng,
                                    trunc_eps=trunc_eps)
        if normalized:
            rsv = rsv / (np.linalg.norm(rsv) or 1.0)
        for i, p in enumerate(players):
            phi[p] += rsv[i]
    return phi
