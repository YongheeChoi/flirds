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
def in_run_loo(logs, n_clients, loss_fn, pkeys, device):
    """(b)-game leave-one-out marginals  phi_i = U_(b)(N) - U_(b)(N\\{i})  on the frozen
    trajectory -- the field-standard low-cost Shapley proxy (the FedSV / GTG / Data-Shapley
    baseline; an ADDITIVITY anchor, NOT a discriminator).

    DISTINCT from loss-heur's singleton U({i}): LOO is the marginal of REMOVING i from the
    grand coalition, U({i}) is i's standalone utility.  By the same per-round additivity that
    in_run_shapley_perround exploits (U_(b) is additive over the frozen rounds and round r's
    term depends only on P_r), the global LOO decomposes to
      phi_i = sum_{r: i in P_r} [ u_r(P_r) - u_r(P_r\\{i}) ]
    so it costs O(sum_r |P_r|) loss evals (== loss-heur's cost class, ~2+|P_r| per round),
    NOT O(N * trajectory).  Same orientation as in_run_shapley (good client -> LOW): a helpful
    client lowers the grand-coalition loss, so its marginal is negative.  Returns phi[n_clients]."""
    phi = np.zeros(n_clients)
    for w_r, dm in logs:
        players = sorted(dm.keys())
        if not players:
            continue
        pr = _round_weight(dm)
        base_params, buffers = _split(w_r, pkeys, device)
        base = float(loss_fn(base_params, buffers))
        u_full = float(loss_fn(_perturbed_params(base_params, dm, players, pr, pkeys), buffers)) - base
        for i in players:
            others = [k for k in players if k != i]
            u_wo = (float(loss_fn(_perturbed_params(base_params, dm, others, pr, pkeys), buffers)) - base
                    if others else 0.0)              # U_(b)(emptyset) = 0
            phi[i] += u_full - u_wo
    return phi


@torch.no_grad()
def in_run_singletons(logs, n_clients, loss_fn, pkeys, device):
    """All N singleton (b)-utilities U_(b)({k}) on the frozen trajectory -- the loss-heuristic
    floor -- with the per-round base loss ell(w^r) computed ONCE and reused across clients.

    EXACTLY equal to [in_run_utility(logs, [k], ...) for k in range(n_clients)], but that
    N-call form recomputes base (and re-splits w^r to device) once PER client, doing 2|P_r|
    forwards/round; this caches base per round for 1+|P_r| forwards/round -- a 2|P_r|/(1+|P_r|)
    reduction (~1.7x at |P_r|=5, ->2x for large cohorts).  Same per-round base-caching that
    in_run_loo and _coalition_utilities already use.  good->low (helpful client lowers the
    round loss -> negative singleton utility); free-rider(zero delta) -> exactly 0.  Returns
    phi[n_clients] with phi[k] = U_(b)({k})."""
    phi = np.zeros(n_clients)
    for w_r, dm in logs:
        players = sorted(dm.keys())
        if not players:
            continue
        pr = _round_weight(dm)
        base_params, buffers = _split(w_r, pkeys, device)
        base = float(loss_fn(base_params, buffers))          # ONCE per round (was: once per client)
        for k in players:
            pert = _perturbed_params(base_params, dm, [k], pr, pkeys)
            phi[k] += float(loss_fn(pert, buffers)) - base
    return phi


@torch.no_grad()
def _coalition_utilities(logs, n_clients, loss_fn, pkeys, device):
    """All 2^N coalition utilities U_(b)(S) on the frozen trajectory + global weight p.

    Shared by the exact in-run semivalues: Shapley (in_run_shapley) and Banzhaf
    (baselines.banzhaf) differ ONLY in how this same U(S) dict is reweighted into
    phi.  U(emptyset)=0; p is n_k/Σn (returned for reference)."""
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
    return U, p


@torch.no_grad()
def in_run_shapley(logs, n_clients, loss_fn, pkeys, device):
    """Exact (b) in-run Shapley values over `logs`.  Returns (phi[n], p[n]).

    p is the global weight n_k/Σn (exact under full participation), returned for
    reference; the utility uses per-round participant weights internally."""
    U, p = _coalition_utilities(logs, n_clients, loss_fn, pkeys, device)
    clients = list(range(n_clients))
    phi = np.zeros(n_clients)
    for k in clients:
        others = [c for c in clients if c != k]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n_clients - r - 1) / factorial(n_clients)
            for S in itertools.combinations(others, r):
                phi[k] += w * (U[tuple(sorted(S + (k,)))] - U[S])
    return phi, p


@torch.no_grad()
def in_run_shapley_perround(logs, n_clients, loss_fn, pkeys, device):
    """(b) in-run Shapley via per-round decomposition -- the cross-device (large-N) path.

    U_(b) is additive over the FROZEN rounds and round r's term depends only on its
    participants P_r (the weights p_k^r are over the full P_r, fixed w.r.t. S), so by
    Shapley linearity + the null-player property the client value decomposes to
      phi_i = sum_{r: i in P_r} (exact 2^{|P_r|} Shapley of round r's val-loss-change
              sub-game over P_r).
    This returns EXACTLY in_run_shapley's 2^N value (proven equal; see
    phase2_crossdevice_oracle_smoke) but at cost sum_r 2^{|P_r|} forwards instead of
    2^N -- feasible at N>>K (e.g. N=100, K=10 -> 200*1024, not 2^100), and round-
    independent so it shards across rounds.  Returns (phi[n_clients], p[n_clients]);
    p = n_k/Σn for reference.
    """
    client_n = {}
    for _, dm in logs:
        for k, (_, n) in dm.items():
            client_n.setdefault(k, n)
    tot = sum(client_n.values())
    p = np.array([client_n.get(k, 0.0) / tot for k in range(n_clients)])

    phi = np.zeros(n_clients)
    for w_r, dm in logs:
        players = sorted(dm.keys())
        K = len(players)
        pr = _round_weight(dm)
        base_params, buffers = _split(w_r, pkeys, device)
        base = float(loss_fn(base_params, buffers))
        u = {(): 0.0}                                # round-r sub-game utility u_r(S)
        for r in range(1, K + 1):
            for S in itertools.combinations(players, r):
                pert = _perturbed_params(base_params, dm, S, pr, pkeys)
                u[S] = float(loss_fn(pert, buffers)) - base
        for k in players:                            # exact Shapley within the K-player round
            others = [c for c in players if c != k]
            for r in range(len(others) + 1):
                w = factorial(r) * factorial(K - r - 1) / factorial(K)
                for S in itertools.combinations(others, r):
                    phi[k] += w * (u[tuple(sorted(S + (k,)))] - u[S])
    return phi, p
