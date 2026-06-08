"""ComFedSV (Fan et al. 2022, "Improving Fairness for Data Valuation in
Horizontal FL") — completed federated Shapley via low-rank matrix completion.

Reference: paper Alg.1 + Eq.6/10/12/13. Official code = Huawei AI Gallery + LIBMF;
we self-build the completion with numpy ALS to avoid the external LIBMF dependency.

Idea: under partial participation (K of N per round) many coalition utilities are
unobserved. ComFedSV builds the reduced T x C utility matrix over M permutations'
prefix coalitions, observes only prefixes that are subsets of the round cohort I_t,
completes it low-rank (U ~= W H^T), and reads Shapley marginals off the completed
matrix (Eq.12). Differences from GTG/FedSV:
  - UNIFORM subset model  w_S = mean_{k in S} w_k  (paper Eq.), not n_k-weighted.
  - utility = per-round test-LOSS decrease  u_t(S) = loss(w^t) - loss(w_S)  (Eq.6).
  - Assumption 1: round 0 selects all clients so every client is observed.

LLM port (loss_fn/pkeys, same backend-agnostic switch as GTG/FedSV): the per-round
metric is the val-loss over the uniform sub-model PARAMS; logs are the standard
(w_r, deltas_map) 2-tuples (cohort = deltas_map.keys()) -> partial-completion only
(no non-cohort deltas at cross-device scale, which is exactly ComFedSV's setting).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..fl.client import local_train


def _uniform_subset_model(gb, deltas, subset, device):
    """w_S = global_before + mean_{k in S} delta_k  (uniform average)."""
    state = {k: v.clone().to(device) for k, v in gb.items()}
    if not subset:
        return state
    for c in subset:
        d, _ = deltas[c]
        for k in state:
            state[k] = state[k] + d[k].to(device) / len(subset)
    return state


def _uniform_subset_params(w_r, deltas, subset, pkeys, device):
    """LLM analog of _uniform_subset_model: params only (pkeys), w_S = w_r +
    mean_{k in S} delta_k (uniform).  The frozen base lives inside loss_fn's model,
    so this returns params only; loss_fn's buffers arg is then empty."""
    params = {n: w_r[n].detach().float().to(device) for n in pkeys}
    if not subset:
        return params
    for c in subset:
        d = deltas[c][0]
        for k in pkeys:
            params[k] = params[k] + d[k].float().to(device) / len(subset)
    return params


@torch.no_grad()
def _llm_util(w_r, deltas, subset, pkeys, loss_fn, device):
    """val-loss of the uniform sub-model (value-only forward; the LLM _test_loss)."""
    return float(loss_fn(_uniform_subset_params(w_r, deltas, subset, pkeys, device), {}))


@torch.no_grad()
def _test_loss(model, state, loader, device):
    model.load_state_dict(state)
    model.to(device).eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        tot += F.cross_entropy(model(x), y, reduction="sum").item()
        n += y.size(0)
    return tot / n


def _complete_lowrank(U, mask, rank, n_iter=60, lam=0.1, seed=0):
    """Low-rank completion U ~= W H^T via ALS over observed entries (mask=True)."""
    rng = np.random.default_rng(seed)
    T, C = U.shape
    W = 0.1 * rng.standard_normal((T, rank))
    H = 0.1 * rng.standard_normal((C, rank))
    eye = lam * np.eye(rank)
    for _ in range(n_iter):
        for t in range(T):
            cols = np.where(mask[t])[0]
            if len(cols):
                Hc = H[cols]
                W[t] = np.linalg.solve(Hc.T @ Hc + eye, Hc.T @ U[t, cols])
        for c in range(C):
            rows = np.where(mask[:, c])[0]
            if len(rows):
                Wr = W[rows]
                H[c] = np.linalg.solve(Wr.T @ Wr + eye, Wr.T @ U[rows, c])
    return W, H


def comfedsv_train(model_fn, client_loaders, test_loader, rounds, local_epochs, lr,
                   device="cuda", seed=0, sample_frac=0.3):
    """Train the global model with K-of-N selection (round 0 = all, Assumption 1),
    logging ALL clients' deltas each round (needed for the ground-truth matrix).

    Returns (eval_model, logs[(global_before, all_deltas, cohort_set)]).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = model_fn().to(device)
    gstate = {k: v.detach().clone() for k, v in model.state_dict().items()}
    n = len(client_loaders)
    k = max(1, round(sample_frac * n))
    logs = []
    for r in range(rounds):
        gb = {key: v.clone() for key, v in gstate.items()}
        all_deltas = {}
        for c in range(n):
            d, ncount = local_train(model, gstate, client_loaders[c],
                                    local_epochs, lr, device)
            all_deltas[c] = (d, ncount)
        cohort = (set(range(n)) if r == 0
                  else set(rng.choice(n, size=k, replace=False).tolist()))
        tot = sum(all_deltas[c][1] for c in cohort)
        for key in gstate:
            gstate[key] = gstate[key] + sum(
                (all_deltas[c][1] / tot) * all_deltas[c][0][key].to(device)
                for c in cohort)
        logs.append((gb, all_deltas, cohort))
    return model, logs


def comfedsv_from_logs(logs, model, n, test_loader, device, seed=0, n_mc=None,
                       rank=5, partial=True, loss_fn=None, pkeys=None):
    """ComFedSV from a logged trajectory.

    partial=True  -> observe only prefixes subset of cohort I_t, then complete.
    partial=False -> observe all prefixes (ground truth, no completion).
    Backend-agnostic (like GTG/FedSV): pass (model, test_loader) for the CNN
    test-loss metric (default), or (loss_fn, pkeys) for the LLM val-loss metric
    (model/test_loader unused).  CNN logs are (gb, all_deltas, cohort) 3-tuples;
    LLM logs are the standard (w_r, deltas_map) 2-tuples (cohort = participants).
    """
    rng = np.random.default_rng(seed)
    M = n_mc or max(10, int(np.ceil(n * np.log(n))))
    perms = [rng.permutation(n) for _ in range(M)]

    coalitions = {}  # sorted prefix tuple -> column index
    for perm in perms:
        for i in range(n):
            key = tuple(sorted(perm[: i + 1].tolist()))
            coalitions.setdefault(key, len(coalitions))

    is_llm = loss_fn is not None
    T, C = len(logs), len(coalitions)
    U = np.zeros((T, C))
    mask = np.zeros((T, C), dtype=bool)
    for t, entry in enumerate(logs):
        if is_llm:                                   # LLM: (w_r, deltas_map); cohort = participants
            gb, deltas = entry
            cohort = set(deltas.keys())
            base = _llm_util(gb, deltas, [], pkeys, loss_fn, device)
        else:                                        # CNN: (gb, all_deltas, cohort)
            gb, deltas, cohort = entry
            base = _test_loss(model, gb, test_loader, device)
        for key, col in coalitions.items():
            if (not partial) or set(key) <= cohort:
                sub = (_llm_util(gb, deltas, list(key), pkeys, loss_fn, device) if is_llm
                       else _test_loss(model, _uniform_subset_model(gb, deltas, list(key), device),
                                       test_loader, device))
                U[t, col] = base - sub
                mask[t, col] = True

    if partial:
        W, H = _complete_lowrank(U, mask, rank, seed=seed)
        colsum = (W @ H.T).sum(axis=0)
    else:
        colsum = U.sum(axis=0)

    s = np.zeros(n)
    for perm in perms:
        for pos, c in enumerate(perm):
            cw = coalitions[tuple(sorted(perm[: pos + 1].tolist()))]
            before = (colsum[coalitions[tuple(sorted(perm[:pos].tolist()))]]
                      if pos > 0 else 0.0)
            s[c] += colsum[cw] - before
    return s / M


def comfedsv(model_fn, client_loaders, test_loader, rounds, local_epochs, lr,
             device="cuda", seed=0, sample_frac=0.3, n_mc=None, rank=5):
    """Convenience: partial-participation training + completed federated Shapley."""
    model, logs = comfedsv_train(model_fn, client_loaders, test_loader, rounds,
                                 local_epochs, lr, device, seed, sample_frac)
    return comfedsv_from_logs(logs, model, len(client_loaders), test_loader,
                              device, seed, n_mc, rank, partial=True)
