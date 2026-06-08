"""FLDetector (Zhang et al., KDD 2022) -- server-side from-logs malicious/noisy detector.

Reference algorithm: zaixizhang/FLDetector (the official code's `lbfgs` + detection loop);
reference-guided self-build on our `logs` contract.  A DETECTION score, NOT a valuation
phi -- so it joins the AUROC table only (no Spearman / no marginal-contribution value).

Temporal model-update consistency: for a benign client the Cauchy mean-value theorem gives
    g_i^t = g_i^{t-1} + H^t (w^t - w^{t-1}),
so the server predicts each client's update from its OWN previous update plus a single
shared Hessian-vector product, and flags clients whose RECEIVED update is repeatedly far
from the prediction.  The integrated Hessian H^t is approximated by L-BFGS (Byrd-Nocedal
compact form) over a window of (model-difference, global-update-difference) secant pairs
read off the global trajectory.  Per-round distances ||g_hat_i - g_i||_2 are L1-normalized
across clients; the suspicious score is each client's mean normalized distance over the
window.  Higher score = more inconsistent = more suspicious -- matching the
`eval.metrics.detection_auroc` convention (corrupt clients score high).

Pure server-side + MODEL-FREE: it needs ONLY the logged update vectors, so unlike
GTG/FedSV/ShapleyFL it takes no model / loss_fn / test_loader (a different KIND of
baseline -> the minimal signature).  Mapping our `logs` to the paper:
  - g_i^t   = the client delta dm[c][0] (raw, n_c-unweighted -- the received update);
  - w^t-w^{t-1} = the n_c-WEIGHTED aggregate of the round's deltas.  FedAvg folds the
    server step into the local delta (w^{t} = w^{t-1} + Σ_k p_k Δw_k), so the aggregated
    delta IS the model difference, and the "global gradient" sequence whose differences
    form the L-BFGS Y_k is that same aggregate -- hence Y_k = a^t - a^{t-1}, S_k = a^{t-1}.
  - w_r (the logged global state) is UNUSED (the trajectory is reconstructed from deltas).

Threat match: FLDetector flags CRAFTED-UPDATE attackers (its model-consistency test),
so its matched threat here is POISONING / model-replacement backdoors (task 7e), not
the honest-but-noisy answer_swap client (a threat mismatch -- see the redesigned
detector suite).  Run on the poisoning trajectory it needs no change.

Adaptations from the paper (offline from-logs):
  - the unsupervised Gap-statistic + 2-means CLUSTERING is skipped; only the continuous
    suspicious score is kept (for AUROC).  2-means on N=5 scores is degenerate;
  - the paper gates scoring on a FULL window (iteration > N); our R can be < N=10, so we
    score as soon as >=1 secant pair exists and average over the available <= N rounds;
  - CROSS-DEVICE partial participation: a client's previous update is from its last
    participation t' < r-1, so the prediction integrates the Hessian over the gap
    w^r - w^{t'} (one cached HVP per distinct gap); full participation recovers the
    original per-round prediction bit-identically (see fldetector_from_logs).
"""
from __future__ import annotations

import numpy as np
import torch

from .ripple import _flat


def _lbfgs_hvp(S, Y, v):
    """Byrd-Nocedal compact-form L-BFGS Hessian-vector product  B v ~= H v.

    S, Y: (P, k) most-recent secant pairs (model diffs S_k / global-update diffs Y_k),
    columns oldest..newest.  v: (P,) the current model difference.  Returns (P,).
    Mirrors the official FLDetector `lbfgs`:
        sigma = (y_last . s_last) / (s_last . s_last);
        B = sigma I - [sigma S, Y] M^{-1} [sigma S, Y]^T,
        M = [[sigma S^T S, L], [L^T, -D]],  L = strictly-lower(S^T Y),  D = diag(S^T Y).
    The (2k x 2k) solve is done in float64 for conditioning (k <= window is tiny)."""
    SY = S.T @ Y                                       # (k, k)
    SS = S.T @ S                                       # (k, k)
    L = torch.tril(SY, -1)                             # strictly-lower part
    D = torch.diag(torch.diag(SY))                     # diagonal part
    s_last, y_last = S[:, -1], Y[:, -1]
    sigma = (y_last @ s_last) / (s_last @ s_last)
    M = torch.cat([torch.cat([sigma * SS, L], dim=1),
                   torch.cat([L.T, -D], dim=1)], dim=0)            # (2k, 2k)
    p = torch.cat([sigma * (S.T @ v), Y.T @ v])                    # (2k,)
    coef = torch.linalg.solve(M.double(), p.double()).to(v.dtype)
    return sigma * v - torch.cat([sigma * S, Y], dim=1) @ coef     # (P,)


def fldetector_from_logs(logs, n_clients, window=10, device="cpu"):
    """FLDetector suspicious score per client over a frozen FedAvg trajectory.

    Returns score[n_clients] (higher = more suspicious; non-negative, averages of
    L1-normalized per-round prediction distances).  Model-free -> no loss_fn/model/
    test_loader.  `window` (paper N=10) bounds the L-BFGS pair count and the score
    average.  Full participation (cross-silo) AND partial participation (cross-device)
    both supported: each client's update is predicted from its OWN previous
    participation round t' (not necessarily r-1) plus a single integrated Hessian-
    vector product over the GAP model change w^r - w^{t'} (the Cauchy-MVT extended
    across the sparse-participation gap; one HVP per distinct gap, cached).  Reduces
    BIT-IDENTICALLY to the per-round prediction under full participation (t' == r-1
    for every client every round)."""
    keys = sorted(next(iter(logs[0][1].values()))[0].keys())      # delta param keys, sorted
    G, A = [], []                                                 # per-client updates; weighted aggregate
    for _, dm in logs:
        gs = {c: _flat(dm[c][0], keys).to(device).float() for c in dm}
        tot = sum(n for _, n in dm.values())
        G.append(gs)
        A.append(sum((dm[c][1] / tot) * gs[c] for c in dm))       # a^t = w^{t+1} - w^t

    S_pairs, Y_pairs, round_dicts = [], [], []
    last_seen = {}                                               # client -> (t', g_c^{t'}) last participation
    for r in range(len(logs)):
        rd = None
        if S_pairs:                                               # >=1 secant pair -> predict
            S = torch.stack(S_pairs[-window:], dim=1)            # (P, k)
            Y = torch.stack(Y_pairs[-window:], dim=1)
            hvp_cache, rd = {}, {}                                # t' -> H (w^r - w^{t'})  (one HVP per gap)
            for c in G[r]:
                if c in last_seen:
                    tprime, g_prev = last_seen[c]
                    if tprime not in hvp_cache:
                        gap = A[tprime] if tprime == r - 1 else sum(A[s] for s in range(tprime, r))
                        hvp_cache[tprime] = _lbfgs_hvp(S, Y, gap)   # H (w^r - w^{t'})
                    rd[c] = float(torch.linalg.norm(g_prev + hvp_cache[tprime] - G[r][c]))
            if rd:
                tot_d = sum(rd.values()) + 1e-12
                rd = {c: d / tot_d for c, d in rd.items()}        # L1-normalize across predicted clients
        round_dicts.append(rd)
        if r >= 1:                                                # record GLOBAL (model diff, update diff)
            S_pairs.append(A[r - 1])
            Y_pairs.append(A[r] - A[r - 1])
        for c in G[r]:
            last_seen[c] = (r, G[r][c])                           # reference (no copy); G already holds it

    sum_dist, cnt = np.zeros(n_clients), np.zeros(n_clients)
    for rd in [d for d in round_dicts if d][-window:]:            # last `window` predicted rounds
        for c, v in rd.items():
            sum_dist[c] += v
            cnt[c] += 1
    return np.divide(sum_dist, cnt, out=np.zeros(n_clients), where=cnt > 0)   # per-client mean (0 if unseen)
