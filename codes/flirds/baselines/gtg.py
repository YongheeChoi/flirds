"""GTG-Shapley (Liu et al., ACM TIST 2022) — self-build on our FL core.

Reference algorithm: cyyever/torch_algorithm shapley_value/gtg_shapley_value.py
(the only real implementation; liuzelei13/GTG-Shapley is an empty shell).

Guided truncation Monte-Carlo Shapley, computed per round then summed.
Sub-model utility U(S) = eval(global_before + FedAvg-agg of subset's deltas) —
trajectory-faithful reconstruction, NO retraining (distinct from (a) retrain SV).
"""
from __future__ import annotations

import numpy as np

from ..fl.server import evaluate, run_fedavg_logs


def _aggregate_subset(global_before, deltas_map, subset, device):
    """FedAvg aggregation restricted to clients in `subset` (re-normalized)."""
    state = {k: v.clone().to(device) for k, v in global_before.items()}
    if not subset:
        return state
    tot = sum(deltas_map[c][1] for c in subset)
    for c in subset:
        d, n = deltas_map[c]
        for k in state:
            state[k] = state[k] + (n / tot) * d[k].to(device)
    return state


def _normalize(sv, marginal_gain):
    """Efficiency normalization: scale SV so its same-sign mass = marginal_gain.

    If there is no same-sign mass to scale, return SV unchanged (avoid the
    1/1e-9 blow-up that would sign-flip and explode the values).
    """
    s = sum(x for x in sv if (x >= 0) == (marginal_gain >= 0))
    if abs(s) < 1e-12:
        return np.asarray(sv, dtype=float)
    return np.array([marginal_gain * x / s for x in sv])


class _RoundGTG:
    """Guided truncation MC Shapley for a single round's sub-model utilities.

    metric_fun(subset_indices) -> utility, where indices are into `players`.
    """

    def __init__(self, n_players, last_metric, this_metric, metric_fun,
                 eps=0.001, converge_criteria=0.05, last_k=10):
        self.n = n_players
        self.last_metric = last_metric
        self.this_metric = this_metric
        self.metric_fun = metric_fun
        self.eps = eps
        self.converge_criteria = converge_criteria
        self.last_k = last_k
        self.converge_min = max(30, n_players)
        self.max_number = min(2 ** n_players,
                              max(self.converge_min, int(0.8 * 2 ** n_players)))

    def compute(self, rng, normalize=True):
        cache = {(): self.last_metric}
        records = []
        index = 0
        while not records or self._not_convergent(index, records):
            for p in range(self.n):
                index += 1
                perm = np.concatenate(
                    ([p], rng.permutation([i for i in range(self.n) if i != p]))
                ).astype(int)
                v = [self.last_metric] + [0.0] * self.n
                marg = [0.0] * self.n
                for j in range(self.n):
                    sub = tuple(sorted(perm[: j + 1].tolist()))
                    if abs(self.this_metric - v[j]) >= self.eps:
                        if sub not in cache:
                            cache[sub] = self.metric_fun(sub)
                        v[j + 1] = cache[sub]
                    else:
                        v[j + 1] = v[j]
                    marg[perm[j]] = v[j + 1] - v[j]
                records.append(marg)
        sv = np.mean(records, axis=0)
        if normalize:
            return _normalize(sv, self.this_metric - self.last_metric)
        return sv

    def _not_convergent(self, index, records):
        if index >= self.max_number:
            return False
        if index <= self.converge_min:
            return True
        run = np.cumsum(records, 0) / np.arange(1, len(records) + 1)[:, None]
        tail = run[-self.last_k:]
        err = np.mean(np.abs(tail - run[-1:]) / (np.abs(run[-1:]) + 1e-12), axis=1)
        return np.max(err) > self.converge_criteria


def gtg_from_logs(logs, model, n_clients, test_loader, device, seed=0,
                  round_trunc=0.001, normalize=True, eps=0.001):
    """GTG round-Shapley from a shared FedAvg trajectory; total phi over rounds."""
    rng = np.random.default_rng(seed)
    phi = np.zeros(n_clients)
    for gb, dm in logs:
        players = sorted(dm.keys())
        last_m = evaluate(model, gb, test_loader, device)
        this_m = evaluate(model, _aggregate_subset(gb, dm, players, device),
                          test_loader, device)
        if abs(this_m - last_m) <= round_trunc:
            continue

        def metric_fun(sub_idx, gb=gb, dm=dm, players=players):
            st = _aggregate_subset(gb, dm, [players[i] for i in sub_idx], device)
            return evaluate(model, st, test_loader, device)

        rsv = _RoundGTG(len(players), last_m, this_m, metric_fun, eps=eps).compute(
            rng, normalize)
        for i, p in enumerate(players):
            phi[p] += rsv[i]
    return phi


def gtg_shapley(model_fn, client_loaders, test_loader, rounds, local_epochs, lr,
                device="cuda", seed=0, round_trunc=0.001, normalize=True):
    """Convenience: run FedAvg then GTG. Returns total phi over rounds."""
    model, logs = run_fedavg_logs(model_fn, client_loaders, test_loader, rounds,
                                  local_epochs, lr, device, seed)
    return gtg_from_logs(logs, model, len(client_loaders), test_loader, device,
                         seed, round_trunc, normalize)
