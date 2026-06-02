"""Ripple Shapley (Zeng et al., AAAI 2026) — sample/client-level FL attribution.

Faithful to Algorithm 1 + Eq 5-19 (no public code). phi_k = drop + ripple, summed
over rounds; client-level via linearity of the per-round update direction.

  - Drop term (Eq 4-7): IRDS 1st-order along the REALIZED local SGD trajectory.
    Each local step adds  lr * <g_val(w_k^t), g_batch(w_k^t)>  at the current local
    model (value = val-loss reduction; clean client -> positive), then weighted by
    alpha_k = n_k / n_s.  (g_batch is the mean-reduction batch gradient, matching
    fl.client.local_train, so drop and the realized Delta w_k share units.)
  - Ripple term (Eq 8-19): cross-round propagation of that update direction. Each
    client sketches the top-k eigenpairs of its LOCAL Hessian at w^r (Alg.1 L6);
    the per-round sketches build a progressive global subspace Q (Eq 15); the
    low-rank Jacobian chain (Eq 16-18) runs in the m-dim subspace; influence path
    is Eq 19.  Sign and alpha-weighting match the drop term so the two reinforce.

All curvature/Jacobian work stays in the m-dim subspace (the paper's efficiency
claim). Param-only (named_parameters); buffers held fixed -> BatchNorm-safe.

No ground-truth SV (paper rejects MSE/correlation); evaluated task-driven
(noisy/poisoned-client detection, runtime).

LLM-scale note: the per-round full-parameter DW/VG/U arrays below are a CNN-scale
convenience. At LLM (LoRA) scale, project onto Q on the fly over protocol-logged
deltas (build Q in one pass, stream-project in a second) instead of materializing
(rounds, n, P).  [deferred to the LLM port]
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp


def _flat(d, keys):
    return torch.cat([d[k].flatten() for k in keys])


def _loader_xy(loader, device):
    xs = torch.cat([x for x, _ in loader]).to(device)
    ys = torch.cat([y for _, y in loader]).to(device)
    return xs, ys


def client_drop_and_delta(model, gstate, loader, epochs, lr, val_x, val_y, device,
                          momentum=0.9):
    """Run local SGD (identical to fl.client.local_train) while accumulating the
    IRDS drop term along the realized local trajectory.

    Returns (drop, delta):
      drop  = sum over local steps of  lr * <g_val(w_t), g_batch(w_t)>  (Eq 5-7,
              client-level, value convention clean->+),
      delta = w_local - w_global over trainable params only (named_parameters).
    Val forward uses the round-start buffers (BN-safe; eval-mode is moot for the
    BN-free CNN track).
    """
    model.load_state_dict(gstate)
    model.to(device).train()
    pkeys = [n for n, _ in model.named_parameters()]
    w0 = {n: gstate[n].detach().clone().to(device) for n in pkeys}
    buffers = {n: b.detach() for n, b in model.named_buffers()}

    def vloss(p):
        return F.cross_entropy(functional_call(model, (p, buffers), (val_x,)), val_y)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    drop = 0.0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()                  # realized batch grad
            gb = {n: p.grad.detach() for n, p in model.named_parameters()}
            gv = grad(vloss)({n: p.detach() for n, p in model.named_parameters()})
            drop += lr * float(sum((gv[n] * gb[n]).sum() for n in pkeys))
            opt.step()
    delta = {n: (p.detach() - w0[n]) for n, p in model.named_parameters()}
    return drop, delta


def local_hessian_topk(model, gstate, x, y, k, device):
    """Top-k eigenpairs of the LOCAL-data loss Hessian at w^r (Eq 14 sketch).

    Per-client curvature on the client's own data (Alg.1 L6) -- not the global/val
    Hessian.  HVP via jvp(grad(loss)); eigsh on a scipy LinearOperator.
    Returns (eigvals[k], eigvecs[P, k]) over trainable params.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    model.load_state_dict(gstate)
    model.to(device)
    pkeys = [n for n, _ in model.named_parameters()]
    params = {n: gstate[n].detach().to(device) for n in pkeys}
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    sizes = {n: params[n].numel() for n in pkeys}
    shapes = {n: params[n].shape for n in pkeys}
    P = sum(sizes.values())

    def loss_fn(p):
        return F.cross_entropy(functional_call(model, (p, buffers), (x,)), y)
    gfn = grad(loss_fn)

    def unflat(v):
        out, i = {}, 0
        for n in pkeys:
            out[n] = v[i:i + sizes[n]].reshape(shapes[n])
            i += sizes[n]
        return out

    def matvec(vnp):
        v = torch.as_tensor(np.ascontiguousarray(vnp), dtype=torch.float32, device=device)
        _, hv = jvp(gfn, (params,), (unflat(v),))
        return _flat(hv, pkeys).detach().cpu().numpy()

    op = LinearOperator((P, P), matvec=matvec, dtype=np.float64)
    # TODO(deferred): eigsh convergence fallback -- set maxiter/tol, retry with a
    # larger ncv on ArpackNoConvergence, and assert ||H v - lam v|| is small.
    vals, vecs = eigsh(op, k=k, which="LA")
    return vals, vecs


def _orthoproj(Q, U, m):
    """Orthonormal basis of [Q, U] truncated to m columns (progressive subspace, Eq 15)."""
    A = U if Q is None else np.concatenate([Q, U], axis=1)
    Qn, _ = np.linalg.qr(A)
    return Qn[:, :m]


def ripple_shapley(model_fn, client_loaders, rounds, local_epochs, lr,
                   val_x, val_y, device, seed=0, k=20, m=50, R=20):
    """Client-level Ripple Shapley values = (drop + ripple), summed over rounds."""
    torch.manual_seed(seed)
    model = model_fn().to(device)
    gstate = {n: v.detach().clone() for n, v in model.state_dict().items()}
    pkeys = [n for n, _ in model.named_parameters()]
    buffers0 = {n: b.detach().clone() for n, b in model.named_buffers()}
    n = len(client_loaders)
    P = sum(p.numel() for _, p in model.named_parameters())
    client_xy = [_loader_xy(ld, device) for ld in client_loaders]

    def val_grad_flat(state):
        params = {nn: state[nn].detach().to(device) for nn in pkeys}
        g = grad(lambda p: F.cross_entropy(
            functional_call(model, (p, buffers0), (val_x,)), val_y))(params)
        return _flat(g, pkeys).detach().cpu().numpy()

    drop = np.zeros((rounds, n))
    nsel = np.zeros((rounds, n))
    DW = np.zeros((rounds, n, P), dtype=np.float32)
    VG = np.zeros((rounds, P), dtype=np.float32)
    Us = []                                            # (U_r [P, n*k], Lam_r [n*k]) per round
    Q = None
    for r in range(rounds):
        VG[r] = val_grad_flat(gstate)
        vecs_r, vals_r = [], []
        for c in range(n):
            d_drop, delta = client_drop_and_delta(
                model, gstate, client_loaders[c], local_epochs, lr, val_x, val_y, device)
            drop[r, c] = d_drop
            DW[r, c] = _flat(delta, pkeys).cpu().numpy()
            nsel[r, c] = len(client_loaders[c].dataset)
            xc, yc = client_xy[c]
            vv, ee = local_hessian_topk(model, gstate, xc, yc, k, device)
            vecs_r.append(ee)
            vals_r.append(vv)
        Ur = np.concatenate(vecs_r, axis=1)            # [P, n*k]
        Us.append((Ur, np.concatenate(vals_r)))
        Q = _orthoproj(Q, Ur, m)                       # progressive global subspace (Eq 15)
        # FedAvg over trainable params (buffers held fixed -- param-only valuation)
        agg = sum((nsel[r, c] / nsel[r].sum()) * DW[r, c] for c in range(n))
        off = 0
        for key in pkeys:
            sz = gstate[key].numel()
            gstate[key] = gstate[key] + torch.as_tensor(
                agg[off:off + sz].reshape(gstate[key].shape),
                dtype=gstate[key].dtype, device=device)
            off += sz

    m = Q.shape[1]
    VGp = VG @ Q                                        # [rounds, m]
    DWp = DW @ Q                                        # [rounds, n, m]
    Bs = [Q.T @ Ur for Ur, _ in Us]                    # [m, n*k] per round (Eq 16)
    Ls = [lam for _, lam in Us]
    Ms = [np.eye(m) - lr * (Bs[t] * Ls[t]) @ Bs[t].T for t in range(rounds)]  # Eq 18 factor

    alpha = nsel / nsel.sum(axis=1, keepdims=True)
    dphi = alpha * drop                                 # Eq 7
    rphi = np.zeros((rounds, n))
    for t0 in range(rounds):
        Plow = np.eye(m)
        for r in range(2, R + 1):
            if t0 + r >= rounds:
                break
            Plow = Ms[t0 + r - 1] @ Plow                # prod_{l=1}^{r-1} M[t0+l] (Eq 18)
            # value = -<alpha*Delta w^{t0}, Q Plow Q^T g_val^{t0+r}>  (Eq 19; sign and
            # alpha match the drop term so drop and ripple reinforce, clean -> +).
            rphi[t0] += -((alpha[t0][:, None] * DWp[t0]) @ (Plow.T @ VGp[t0 + r]))
    return (dphi + rphi).sum(axis=0)
