"""ShapleyFL / AFedSV (Sun et al., KDD 2023) surrogate federated Shapley value.

Reference algorithm: ZJU-DIVER/ShapleyFL-Robust-Federated-Learning-Based-on-Shapley
-Value (image-classification CNN code); reference-guided self-build on our FL core,
adapted to LoRA deltas + from-logs trajectory.

The surrogate federated SV (the paper's client VALUE) over a frozen FedAvg trajectory:
  - per-round utility  U_F(S) = Phi(D_0, Psi(x^t, S))   (Def 4.1), where the submodel
    is the **uniform average** of the subset's updates Psi(x^t,S) = x^t + (1/|S|) Σ_{i∈S} Δ_i
    (eta_g=1 for plain FedAvg; uniform, NOT n_k-weighted -- distinct from the (b) oracle's
    n_k/Σ_{P_r} weight and from GTG/FedSV's within-subset n_k renorm; same family as ComFedSV).
    Phi = higher-is-better: accuracy (CNN) / -val_loss (LLM).
  - per-round EXACT Shapley SV_i^t (Eq 1; affordable at N<=10), then **min-max normalized**
    per round NSV_i^t ∈ [0,1] (Def 4.2), then **EMA across rounds** (Def 4.3):
        SSV_i^t = β·SSV_i^{t-1} + (1-β)·NSV_i^t   (participants; others carried forward).

The min-max + EMA + uniform utility make SSV genuinely != the (b) in-run Shapley oracle
(so it is NOT degenerate by Shapley linearity, unlike an un-normalized plain-sum surrogate).

SSV is good->HIGH (a helpful client gets a high value/weight); the comparison negates it
to the good->low convention (like Ripple).  The paper's DMC difference-estimator (§5.2)
is a cheap *estimator of this same SSV* for large N -- deferred to cross-device (N=100,
plan task 7), where the full per-round 2^N sweep is infeasible (cf. ComFedSV defer).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from ..fl.server import evaluate
from ..oracle.exact_sv import exact_shapley

# EMA rate beta (Def 4.3; paper value 0.3).  SINGLE source of truth: beta used to be a
# source literal repeated across 7 call sites, so it never reached any runner's `config`
# -- the rundir guard could not see a 0.5 -> 0.3 change and silently overwrote the old
# canonical run (protocol §1.7 prescription 2).  Runners log this value and put it in
# their rundir identity, so changing it now fails at launch instead.
BETA = float(os.environ.get("SFL_BETA", "0.3"))


def _uniform_submodel_cnn(gb, dm, subset, device):
    """x^t + (1/|S|) Σ_{c∈S} Δ_c  (uniform average; CNN full state dict)."""
    state = {k: v.clone().to(device) for k, v in gb.items()}
    if not subset:
        return state
    for c in subset:
        d, _ = dm[c]
        for k in state:
            state[k] = state[k] + (1.0 / len(subset)) * d[k].to(device)
    return state


def _uniform_submodel_llm(w_r, dm, subset, pkeys, device):
    """x^t + (1/|S|) Σ_{c∈S} Δ_c  (uniform average; LoRA params only, buffers empty)."""
    params = {n: w_r[n].detach().float().to(device) for n in pkeys}
    if not subset:
        return params
    for c in subset:
        d, _ = dm[c]
        for k in pkeys:
            params[k] = params[k] + (1.0 / len(subset)) * d[k].float().to(device)
    return params


def _minmax(sv):
    """Def 4.2 min-max normalize to [0,1]; flat round (no spread) -> zeros."""
    lo, hi = float(np.min(sv)), float(np.max(sv))
    if hi - lo < 1e-12:
        return np.zeros_like(sv)
    return (sv - lo) / (hi - lo)


def _ema_aggregate(round_nsvs, round_players, n_clients, beta):
    """Def 4.3 EMA over per-round normalized partial FSVs.  SSV^0=0; a participant is
    updated SSV_i = β·SSV_i + (1-β)·NSV_i, a non-participant carries forward (cross-device)."""
    ssv = np.zeros(n_clients)
    for nsv, players in zip(round_nsvs, round_players):
        for i, p in enumerate(players):
            ssv[p] = beta * ssv[p] + (1.0 - beta) * nsv[i]
    return ssv


def shapleyfl_round_raw(gb, dm, players, model, test_loader, device, loss_fn, pkeys):
    """One round's RAW (pre-min-max) exact per-round Shapley over the uniform-average
    submodel utility Phi (good->HIGH; CNN accuracy if loss_fn is None, else -val_loss),
    aligned with `players`.  Shared by `_round_nsv` (post-hoc) and the C2 online
    intervention (fl.intervene), so both value rounds identically."""
    if loss_fn is None:
        def util(sub):
            st = _uniform_submodel_cnn(gb, dm, [players[i] for i in sub], device)
            return evaluate(model, st, test_loader, device)          # accuracy, higher=better
    else:
        @torch.no_grad()
        def util(sub):
            ps = _uniform_submodel_llm(gb, dm, [players[i] for i in sub], pkeys, device)
            return -float(loss_fn(ps, {}))                           # -val_loss, higher=better

    return exact_shapley(len(players), util)


def _round_nsv(gb, dm, players, model, test_loader, device, loss_fn, pkeys):
    """Normalized partial federated SV for one round (raw Shapley, min-max [0,1])."""
    return _minmax(shapleyfl_round_raw(gb, dm, players, model, test_loader, device,
                                       loss_fn, pkeys))


def shapleyfl_from_logs(logs, model, n_clients, test_loader, device, beta=BETA,
                        loss_fn=None, pkeys=None):
    """Surrogate federated SV (SSV_i^T) from a shared FedAvg trajectory.  Returns phi[n].

    Backend-agnostic like GTG/FedSV: pass (model, test_loader) for the CNN accuracy
    metric, or (loss_fn, pkeys) for the LLM -val_loss metric (model/test_loader unused
    -> None).  beta = EMA rate (Def 4.3; the paper's training hyperparam, ablatable).
    SSV is good->high; negate for the good->low comparison convention."""
    round_nsvs, round_players = [], []
    for gb, dm in logs:
        players = sorted(dm.keys())
        round_nsvs.append(_round_nsv(gb, dm, players, model, test_loader, device, loss_fn, pkeys))
        round_players.append(players)
    return _ema_aggregate(round_nsvs, round_players, n_clients, beta)
