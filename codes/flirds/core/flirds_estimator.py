"""Flirds estimator: client-level in-run SV via 1st + 2nd order Taylor of val loss.

Backend-agnostic.  The per-round validation loss is supplied as
  loss_fn(params, buffers) -> scalar
(see flirds.backends.* for the CNN / LLM builders); `pkeys` lists the trainable
param names so each logged state w_r splits into params (Taylor variables) and
buffers (held fixed).  The estimator never sees the model, the val data, or the
task — only loss_fn + pkeys.

Closed-form approximation of the (b) in-run Shapley (oracle/in_run_sv) on the
frozen FedAvg trajectory (logs from fl.server.run_fedavg_logs):

  phi_k = sum_r p_k^r [ <g^r, Δw_k> + 1/2 <Δw_k, u^r> ],   u^r = H^r ΔW^r

  g^r   = ∇ loss_fn at w^r ;  H^r = loss_fn Hessian at w^r (true Hessian, as IRDS)
  ΔW^r  = Σ_{j∈P_r} p_j^r Δw_j  (round aggregate over participants P_r)
  p_k^r = n_k / Σ_{j∈P_r} n_j   (FedAvg participant-normalized weight for round r)

Weights are read PER ROUND from the participating cohort P_r = deltas_map.keys(),
matching fl.server.fedavg's aggregate exactly -> correct under partial
participation (cross-device); reduces to the fixed global n_k/Σn under full
participation (cross-silo).  Client-level Shapley sums over participated rounds
with NO participation normalization (locked decision: value scales with
participation; rank-within-tier recovers quality).

The quadratic-Shapley of the 2nd-order term collapses to ONE HVP per round
(u^r = H^r ΔW^r) plus |P_r| dot products <Δw_k, u^r>.  Same sign as the (b)
oracle (val-loss CHANGE attribution: a client that reduces val loss -> phi_k < 0).

per_layer=True additionally returns the per-(client, param) dot-product
components (Σ over params == phi_k, bit-identical) as an OBSERVATION-ONLY
diagnostic for the ②/Q2/#17 analyses -- the spine never reweights them.

fp32 (protocol 1); params cast to float, buffers kept as-is.
"""
from __future__ import annotations

import numpy as np
from torch.func import grad, jvp


def flirds_values(logs, loss_fn, pkeys, device, second_order=True,
                  n_clients=None, per_layer=False):
    """Per-client Flirds in-run SV over the frozen trajectory `logs`.

    loss_fn(params, buffers) -> scalar val loss (backend builder, e.g.
      flirds.backends.cnn.make_cnn_loss).  pkeys = trainable param names.
    second_order=False -> 1st-order only (the IRDS-1st self-ablation baseline).
    n_clients: total client count (inferred from the logs union if None); pass
      explicitly under partial participation if a client never appears in `logs`.
    per_layer=False -> returns (phi[n_clients], p[n_clients]).
    per_layer=True  -> returns (phi, p, components), components[k][name] the
      per-param contributions summing to phi[k] (observation-only diagnostic).

    p[k] = n_k / Σ_j n_j is the global weight (exact under full participation),
    returned for reference; the estimator uses per-round participant weights.
    """
    client_n = {}                                   # first-seen n per client (partial-safe)
    for _, dm in logs:
        for k, (_, n) in dm.items():
            client_n.setdefault(k, n)
    if n_clients is None:
        n_clients = 1 + max(client_n)
    ns = np.array([client_n.get(k, 0.0) for k in range(n_clients)], dtype=float)
    p = ns / ns.sum()

    phi = np.zeros(n_clients)
    comp = {k: {} for k in range(n_clients)} if per_layer else None
    for w_r, dm in logs:
        players = sorted(dm.keys())
        nr = np.array([dm[k][1] for k in players], dtype=float)
        pr = {k: nr[i] / nr.sum() for i, k in enumerate(players)}   # per-round FedAvg weight
        params = {n: w_r[n].detach().float().to(device) for n in pkeys}
        buffers = {n: w_r[n].detach().to(device) for n in w_r if n not in pkeys}
        dw = {k: {n: dm[k][0][n].float().to(device) for n in pkeys} for k in players}

        def vloss(pp):
            return loss_fn(pp, buffers)

        if second_order:
            dW = {n: sum(pr[k] * dw[k][n] for k in players) for n in pkeys}
            g, u = jvp(grad(vloss), (params,), (dW,))      # g^r and u^r = H^r ΔW^r (1 HVP)
        else:
            g, u = grad(vloss)(params), None

        for k in players:
            if per_layer:
                for n in pkeys:
                    c = pr[k] * float((g[n] * dw[k][n]).sum())
                    if second_order:
                        c += 0.5 * pr[k] * float((dw[k][n] * u[n]).sum())
                    phi[k] += c
                    comp[k][n] = comp[k].get(n, 0.0) + c
            else:
                v = pr[k] * float(sum((g[n] * dw[k][n]).sum() for n in pkeys))
                if second_order:
                    v += 0.5 * pr[k] * float(sum((dw[k][n] * u[n]).sum() for n in pkeys))
                phi[k] += v
    return (phi, p, comp) if per_layer else (phi, p)
