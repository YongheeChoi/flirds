"""FLTrust (Cao et al., NDSS 2021) -- server-side trusted-direction detector.

Reference: Xiaoyu Cao, Minghong Fang, Jia Liu, Neil Gong, "FLTrust: Byzantine-
robust Federated Learning via Trust Bootstrapping" (arXiv:2012.13995).  Reference-
guided self-build on our `logs` contract: a DETECTION score (no Shapley / no
Spearman -> AUROC table only).

FLTrust holds a small clean ROOT dataset on the server, each round fine-tunes the
global model on it for a server update g0, and scores every client update by
ReLU-clipped cosine to g0 (magnitude-normalized) for a trust-weighted aggregate.
Our root = the server VALIDATION set, and we take
    g0 = -∇_val(w_r)        (one server gradient step; cosine is scale-free so the
                             step size/count only fixes the direction).
So the trust signal is
    cos(Δw_i, -∇_val) = -<∇_val, Δw_i> / (||∇_val|| ||Δw_i||),
the NORMALIZED Flirds first-order term (Flirds-1st = <∇_val, Δw_k>) -- which is
exactly why the plan files FLTrust as the AUXILIARY free-rider/poisoning baseline
(cosine ~= Flirds-1st with the SAME server direction).

NOT model-free: it needs the val gradient per round, via the loss_fn/pkeys
backend-agnostic switch (same as GTG/FedSV/ComFedSV; pass loss_chunks to micro-
batch the eager LLM val grad, as the estimator does).  This is the contrast with
FLDetector / STD-DAGMM, which read only the update vectors.

As a DETECTOR we return the SIGNED cosine-to-gradient
    score_i = mean_r cos(Δw_i^r, ∇_val(w_r)) = mean_r <g^r, Δw_i^r>/(||g^r|| ||Δw_i^r||),
oriented for eval.metrics.detection_auroc (corrupt scores HIGH): a benign update
descends val loss -> cos(Δw, ∇_val) < 0 (low); a poison update ascends -> > 0
(high); a free-rider (zero/random) is ~orthogonal -> ~0 (above benign).  We do NOT
apply FLTrust's ReLU to the detection score: ReLU(cos) clips benign (<0) AND
free-rider (~0) both to 0, erasing the sign that separates them -> the signed
cosine is required for the free-rider regime.  (The ReLU + magnitude-norm are
FLTrust's robust-AGGREGATION gates; cosine is already scale-free, so neither
changes the per-client detection ranking.)

Any-N, full or partial participation: the per-round cosine is averaged over the
rounds each client participated; unseen clients (never sampled) get the minimum
score -- a free-rider must participate to be one (matching baselines.std_dagmm).
"""
from __future__ import annotations

import numpy as np
from torch.func import grad

from ..core.flirds_estimator import _chunked


def fltrust_from_logs(logs, n_clients, loss_fn, pkeys, device, loss_chunks=None):
    """FLTrust cosine-misalignment suspicious score per client (corrupt = HIGH).

    g^r = ∇_val(w_r) per round (loss_fn/pkeys; loss_chunks micro-batches the eager
    LLM val grad).  Returns score[n_clients] = mean signed cos(Δw_i, ∇_val) over the
    rounds each client participated."""
    sum_cos = np.zeros(n_clients)
    cnt = np.zeros(n_clients)
    for w_r, dm in logs:
        params = {n: w_r[n].detach().float().to(device) for n in pkeys}
        buffers = {n: w_r[n].detach().to(device) for n in w_r if n not in pkeys}

        def vloss(pp):
            return loss_fn(pp, buffers)

        g = (grad(vloss)(params) if loss_chunks is None
             else _chunked(loss_chunks, buffers, params, None, pkeys)[0])
        g_norm = sum(float((g[n] ** 2).sum()) for n in pkeys) ** 0.5
        for k in dm:
            d = {n: dm[k][0][n].float().to(device) for n in pkeys}
            dot = sum(float((g[n] * d[n]).sum()) for n in pkeys)
            d_norm = sum(float((d[n] ** 2).sum()) for n in pkeys) ** 0.5
            sum_cos[k] += dot / (g_norm * d_norm + 1e-12)
            cnt[k] += 1
    score = np.where(cnt > 0, sum_cos / np.maximum(cnt, 1), np.nan)
    unseen = np.isnan(score)
    if unseen.any():                                       # never sampled -> least suspicious
        score[unseen] = np.nanmin(score)
    return score
