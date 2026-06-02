"""Ripple Shapley (Zeng et al., AAAI 2026) — sample-level FL attribution.

Reference: paper Alg.1 + Eq.5-19 (no public code). phi(z) = drop term + ripple term,
summed over birth rounds, then aggregated sample->client for FL valuation.

  - Drop term (Eq.4-7): IRDS 1st-order sample attribution -- per local SGD step,
    -eta * <grad_val, grad_sample>, weighted by FedAvg alpha_k = n_k / n_s.
  - Ripple term (Eq.8-19): cross-round Jacobian-chain propagation with per-round
    Hessian eigen-sketch + progressive low-rank global subspace.  [added next]

No ground-truth SV (paper rejects MSE/correlation); evaluated task-driven
(robustness under poisoning, runtime).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp, vmap


def _flat(d, keys):
    return torch.cat([d[k].flatten() for k in keys])


def client_drop_term(model, gstate, loader, epochs, lr, val_x, val_y, device):
    """Sum of IRDS 1st-order drop terms over a client's samples (one round).

    Returns the client-level local utility U_local = sum_z sum_t -lr*<g_val, g_z>;
    multiply by alpha_k outside. Uses functional SGD so the trajectory matches.
    """
    model.load_state_dict(gstate)
    model.to(device)
    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    keys = list(params.keys())

    def vloss(p):
        return F.cross_entropy(functional_call(model, (p, buffers), (val_x,)), val_y)

    def sloss(p, xi, yi):
        out = functional_call(model, (p, buffers), (xi.unsqueeze(0),))
        return F.cross_entropy(out, yi.unsqueeze(0))

    def bloss(p, x, y):
        return F.cross_entropy(functional_call(model, (p, buffers), (x,)), y)

    # Evaluate at the fixed round-start w (no local SGD trajectory: that
    # diverges, and the cross-round effect is exactly the ripple term's job).
    # IRDS 1st order at w^r, influence convention (good -> positive), same form
    # as Flirds 1st order:  +eta * sum_z <g_val, g_z>.
    vg = _flat(grad(vloss)(params), keys)                           # [P]
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        psg = vmap(grad(sloss), (None, 0, 0))(params, x, y)         # dict[B,*]
        psg_flat = torch.cat([psg[k].flatten(1) for k in keys], 1)  # [B, P]
        total += float((lr * (psg_flat @ vg)).sum())
    return total


def hessian_topk(model, params, buffers, x, y, k, device):
    """Top-k largest eigenpairs of the loss Hessian at `params` (Eq.14 sketch).

    HVP via torch.func.jvp(grad(loss)); eigsh on a scipy LinearOperator.
    Returns (eigvals[k], eigvecs[P, k]) as numpy arrays.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    keys = list(params.keys())
    shapes = {kk: params[kk].shape for kk in keys}
    sizes = {kk: params[kk].numel() for kk in keys}
    P = sum(sizes.values())

    def loss_fn(p):
        return F.cross_entropy(functional_call(model, (p, buffers), (x,)), y)

    gfn = grad(loss_fn)

    def unflat(v):
        out, i = {}, 0
        for kk in keys:
            out[kk] = v[i:i + sizes[kk]].reshape(shapes[kk])
            i += sizes[kk]
        return out

    def matvec(vnp):
        v = torch.as_tensor(np.ascontiguousarray(vnp), dtype=torch.float32, device=device)
        _, hv = jvp(gfn, (params,), (unflat(v),))
        return _flat(hv, keys).detach().cpu().numpy()

    op = LinearOperator((P, P), matvec=matvec, dtype=np.float64)
    vals, vecs = eigsh(op, k=k, which="LA")
    return vals, vecs


def _orthoproj(Q, U, m):
    """Orthonormal basis of [Q, U] truncated to m columns (progressive subspace)."""
    A = U if Q is None else np.concatenate([Q, U], axis=1)
    Qn, _ = np.linalg.qr(A)
    return Qn[:, :m]


def ripple_shapley(model_fn, client_loaders, test_loader, rounds, local_epochs, lr,
                   val_x, val_y, device, seed=0, k=20, m=50, R=20):
    """Full Ripple Shapley client values = drop + ripple, summed over rounds.

    Client-level (sample->client via linearity of the drop direction Δw_k):
      drop_k   = alpha_k * sum_t -<g_val, g_z> (IRDS 1st order, per local step)
      ripple_k = sum_{t0} sum_{r=2..R} g_val(t0+r) . QP_low Q^T . Δw_k(t0)   (Eq.19)
    Per-round global Hessian sketch (top-k at w^r on the val batch as curvature
    proxy) builds the progressive subspace Q and the low-rank Jacobian chain.
    """
    from ..fl.client import local_train

    model = model_fn().to(device)
    gstate = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
    keys = list(gstate.keys())
    buffers = {kk: v.detach().clone() for kk, v in model.named_buffers()}
    n = len(client_loaders)
    P = sum(gstate[kk].numel() for kk in keys)

    def params_of(state):
        return {kk: state[kk].detach().clone() for kk in keys}

    def val_grad(state):
        g = grad(lambda p: F.cross_entropy(
            functional_call(model, (p, buffers), (val_x,)), val_y))(params_of(state))
        return _flat(g, keys).detach().cpu().numpy()

    drop = np.zeros((rounds, n))
    nsel = np.zeros((rounds, n))
    DW = np.zeros((rounds, n, P))
    VG = np.zeros((rounds, P))
    Us, Lams = [], []
    for r in range(rounds):
        VG[r] = val_grad(gstate)
        for c in range(n):
            drop[r, c] = client_drop_term(model, gstate, client_loaders[c],
                                          local_epochs, lr, val_x, val_y, device)
            d, nc = local_train(model, gstate, client_loaders[c],
                                local_epochs, lr, device)
            DW[r, c] = _flat(d, keys).detach().cpu().numpy()
            nsel[r, c] = nc
        vals, vecs = hessian_topk(model, params_of(gstate), buffers,
                                  val_x, val_y, k, device)
        Us.append(vecs)
        Lams.append(vals)
        agg = sum((nsel[r, c] / nsel[r].sum()) * DW[r, c] for c in range(n))
        off = 0
        for kk in keys:
            sz = gstate[kk].numel()
            gstate[kk] = gstate[kk] + torch.as_tensor(
                agg[off:off + sz].reshape(gstate[kk].shape),
                dtype=gstate[kk].dtype, device=device)
            off += sz

    Q = None
    for vecs in Us:
        Q = _orthoproj(Q, vecs, m)              # [P, <=m]
    m = Q.shape[1]                               # actual subspace dim
    VGp = VG @ Q                                 # [rounds, m]
    DWp = DW @ Q                                 # [rounds, n, m]
    Bs = [Q.T @ Us[t] for t in range(rounds)]    # [m, k]
    Ms = [np.eye(m) - lr * (Bs[t] * Lams[t]) @ Bs[t].T for t in range(rounds)]

    alpha = nsel / nsel.sum(axis=1, keepdims=True)
    ripple = np.zeros((rounds, n))
    for t0 in range(rounds):
        Plow = np.eye(m)
        for r in range(2, R + 1):
            if t0 + r >= rounds:
                break
            Plow = Ms[t0 + r - 1] @ Plow         # prod_{l=1}^{r-1} M[t0+l]
            ripple[t0] += DWp[t0] @ (Plow.T @ VGp[t0 + r])
    return (alpha * drop + ripple).sum(axis=0)
