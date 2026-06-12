"""FedIF (Tang et al., 2025) -- influence-function client valuation.

Reference: Guojun Tang et al., "Lightweight and Robust Federated Data Valuation"
(arXiv:2509.25560; code github.com/guojuntang/FedIF).  Reference-guided self-build
on our `logs` contract: a VALUATION method (per-client influence value -> Spearman
+ AUROC), the closest published competitor to Flirds (gradient-influence valuation,
~450x cheaper than Shapley, noise-robust -- the same pitch).

Per round r over participants P_r (reference utils.influence + algo.fedif):
  Phi_k = < (w_{r} - w_k) / ||w_{r} - w_k|| , g^r >        (Eq 6, cal_influence)
        = - <g^r, Δw_k> / ||Δw_k||                          (Δw_k = w_k - w_r = our delta)
  Psi_k = (Phi_k - min Phi) / (max Phi - min Phi)           (Eq 7, per-round min-max -> [0,1])
  Ω_k   = (1-γ)·Ω_k + γ·Psi_k   (k∈P_r)  else  Ω_k          (Eq 8, EMA; Ω^0 = 0)
g^r = ∇_val(w_r) (loss_fn/pkeys; loss_chunks micro-batches the eager LLM val grad,
as the estimator/FLTrust do).  The reference strips BatchNorm buffers before the
diff -- our pkeys (trainable params only; buffers excluded) does this for free.

The final client value Ω is good->HIGH (a helpful client descends val loss ->
<g,Δw> < 0 -> Phi > 0 -> high influence/weight); the comparison NEGATES it to the
good->low convention (like baselines.shapleyfl / ripple).

Kinship (for the writeup, cf. fltrust.py's note): the per-round raw influence is
  Phi_k = -||g^r|| · cos(Δw_k, g^r) = -||g^r|| · (FLTrust per-round cosine),
i.e. FLTrust's signed cosine, sign-flipped (good->HIGH) and val-grad-norm scaled.
It is also the unit-normalized, sign-flipped Flirds FIRST-order term <g^r, Δw_k>
(no p_k^r weight, no 2nd-order curvature).  What makes FedIF a DISTINCT estimator
(not degenerate with FLTrust/Flirds-1st) is the per-round MIN-MAX + cross-round EMA
(γ) post-processing -- same family as ShapleyFL's min-max+EMA, and most divergent
under partial participation (device100), where the EMA carries non-participants'
stale Ω rather than averaging only participated rounds.

Cost = 1 val gradient per round (no Hessian) -- the cheapest valuation tier, with
FLTrust and Flirds-1st.
"""
from __future__ import annotations

import numpy as np
from torch.func import grad

from ..core.flirds_estimator import _chunked


def _minmax(infl):
    """Eq 7: per-round min-max to [0,1]; a flat round (no spread) -> zeros (no
    differentiation that round; matches baselines.shapleyfl._minmax)."""
    lo, hi = float(np.min(infl)), float(np.max(infl))
    if hi - lo < 1e-12:
        return np.zeros_like(infl)
    return (infl - lo) / (hi - lo)


def fedif_update(omega, players, infl, gamma):
    """FedIF round step (Eq 7 min-max + Eq 8 EMA) -- the core SHARED with the online
    aggregation reproduction (experiments/phase2_fedif_repro), so both paths use the
    identical FedIF post-processing.  Mirrors the reference
    utils.influence.update_global_scores: min-max this round's per-participant
    influences `infl`, then EMA them into the running global influence `omega` for
    `players` (Ω^0 = 0; non-participants carry forward).  In place; good->HIGH."""
    psi = _minmax(np.asarray(infl, dtype=float))
    for i, k in enumerate(players):
        omega[k] = (1.0 - gamma) * omega[k] + gamma * float(psi[i])


def fedif_round_raw(w_r, dm, players, loss_fn, pkeys, device, loss_chunks=None):
    """One round's raw FedIF influence per participant (Eq 6, good->HIGH, PRE
    min-max), aligned with `players`.  Shared by `fedif_from_logs` (post-hoc) and
    the C2 online intervention (fl.intervene), so both value rounds identically."""
    params = {n: w_r[n].detach().float().to(device) for n in pkeys}
    buffers = {n: w_r[n].detach().to(device) for n in w_r if n not in pkeys}

    def vloss(pp):
        return loss_fn(pp, buffers)

    g = (grad(vloss)(params) if loss_chunks is None
         else _chunked(loss_chunks, buffers, params, None, pkeys)[0])
    infl = np.empty(len(players))
    for i, k in enumerate(players):
        d = {n: dm[k][0][n].float().to(device) for n in pkeys}
        dot = sum(float((g[n] * d[n]).sum()) for n in pkeys)
        d_norm = sum(float((d[n] ** 2).sum()) for n in pkeys) ** 0.5
        infl[i] = -dot / (d_norm + 1e-12)                  # Eq 6
    return infl


def fedif_from_logs(logs, n_clients, loss_fn, pkeys, device, gamma=0.3, loss_chunks=None):
    """FedIF influence value per client over a frozen FedAvg trajectory.  Returns
    omega[n_clients], good->HIGH (negate for the good->low comparison convention).

    g^r = ∇_val(w_r) per round (loss_fn/pkeys; loss_chunks micro-batches the eager
    LLM val grad).  Per round: Phi_k = -<g^r, Δw_k>/||Δw_k|| (Eq 6), then
    fedif_update applies the min-max (Eq 7) + EMA (Eq 8).  gamma = EMA smoothing rate
    (the paper's hyperparam: 0.3 CIFAR / 0.4 Fashion-MNIST; ablatable, like
    shapleyfl's beta).  Rounds are processed in order (the EMA is order-dependent)."""
    omega = np.zeros(n_clients)
    for w_r, dm in logs:
        players = sorted(dm.keys())
        infl = fedif_round_raw(w_r, dm, players, loss_fn, pkeys, device, loss_chunks)
        fedif_update(omega, players, infl, gamma)
    return omega
