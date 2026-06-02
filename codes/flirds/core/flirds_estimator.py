"""Flirds estimator: client-level in-run SV via 1st + 2nd order Taylor of val loss.

Closed-form approximation of the (b) in-run Shapley (oracle in oracle/in_run_sv)
on the frozen FedAvg trajectory (logs from fl.server.run_fedavg_logs):

  phi_k = sum_r p_k [ <g^r, Δw_k> + 1/2 <Δw_k, u^r> ],   u^r = H^r ΔW^r

  g^r  = ∇ val-loss at w^r ;  H^r = val-loss Hessian at w^r (true Hessian, as IRDS)
  ΔW^r = Σ_j p_j Δw_j (full round aggregate) ;  p_k = n_k / Σ_j n_j

The quadratic-Shapley of the 2nd-order term collapses to ONE HVP per round
(u^r = H^r ΔW^r) plus N dot products <Δw_k, u^r>.  Same sign as the (b) oracle
(val-loss CHANGE attribution: a client that reduces val loss -> phi_k < 0).

Param-only (named_parameters); buffers held fixed.  fp32 (protocol 1).
"""
from __future__ import annotations

import numpy as np
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp


def flirds_values(logs, model_fn, val_x, val_y, device, second_order=True):
    """Per-client Flirds in-run SV over the frozen trajectory `logs`.

    Returns (phi[n_clients], p[n_clients]).  second_order=False -> 1st-order only
    (the IRDS-1st self-ablation baseline).
    """
    model = model_fn().to(device)
    pkeys = [n for n, _ in model.named_parameters()]
    n_clients = len(logs[0][1])
    ns = np.array([logs[0][1][k][1] for k in range(n_clients)], dtype=float)
    p = ns / ns.sum()

    phi = np.zeros(n_clients)
    for w_r, dm in logs:
        params = {n: w_r[n].detach().float().to(device) for n in pkeys}
        buffers = {n: w_r[n].detach().to(device) for n in w_r if n not in pkeys}
        dw = [{n: dm[k][0][n].float().to(device) for n in pkeys}
              for k in range(n_clients)]

        def vloss(pp):
            return F.cross_entropy(functional_call(model, (pp, buffers), (val_x,)), val_y)

        if second_order:
            dW = {n: sum(p[j] * dw[j][n] for j in range(n_clients)) for n in pkeys}
            g, u = jvp(grad(vloss), (params,), (dW,))      # g^r and u^r = H^r ΔW^r (1 HVP)
        else:
            g, u = grad(vloss)(params), None

        for k in range(n_clients):
            val = p[k] * float(sum((g[n] * dw[k][n]).sum() for n in pkeys))
            if second_order:
                val += 0.5 * p[k] * float(sum((dw[k][n] * u[n]).sum() for n in pkeys))
            phi[k] += val
    return phi, p
