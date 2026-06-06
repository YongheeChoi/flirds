"""Ripple Shapley (Zeng et al., AAAI 2026) -- LLM/LoRA port of baselines/ripple.py.

Mirrors the CNN `ripple_shapley` (drop + ripple, Eq 5-19) for a PEFT-LoRA causal-LM.
Like the CNN version it runs its OWN FedAvg trajectory (full participation): Ripple
has no ground-truth-SV metric (the paper rejects MSE/correlation) -> evaluated
task-driven (noisy / free-rider detection AUROC + runtime), so it needs no shared
`logs`.  Client-level via linearity of the per-round update direction; drop and
ripple share sign + alpha so clean clients -> HIGH phi (noisy/free-rider -> LOW).

Backend-specific vs the CNN code (baselines/ripple.py):
  - local-train loss / val loss = completion-only CE via functional_call over the
    data.llm `build_val_batch` tokenization (prompt masked), NOT F.cross_entropy(x,y);
  - params = LoRA only (named_parameters, requires_grad); buffers empty (frozen base
    lives inside the captured model, exactly as backends.llm.make_llm_loss);
  - the per-drop-step val grad is CHUNKED (core.flirds_estimator._chunked) and the
    local-Hessian HVP runs over EAGER attention (jvp.grad forward-AD), like the
    estimator; eigsh gets a FIXED v0 (reproducible) + a no-convergence fallback;
  - free-rider clients fabricate (zero/random delta, no drop, no Hessian sketch).

The (rounds, n, P) arrays are MATERIALIZED here (P ~ 12M LoRA params -> ~20GB at
N=5, ~40GB at N=10; trivially within RAM).  The on-the-fly stream-projection that
baselines/ripple.py defers is only needed at cross-device N=100 (a separate loader,
not this cross-silo setup).  The pure subspace helpers (_flat / _orthoproj) are
imported from ripple.py; the Eq 16-19 chain is re-stated below (to avoid editing the
Phase-0-verified CNN code) and is identical math.

Memory: BOTH eager terms are bounded by small batches -- the drop term's train-batch
grad is single-AD (~0.01*batch_size*seq GB; a full batch_size=16/seq=768 grad OOMs a
178GB GPU) and the local-Hessian HVP is double-AD (~0.021*hess_bs*seq GB); batch_size
and hess_bs default to 4 so each fits beside the model.  Cost is dominated by the HVP
(eigsh ~O(k) matvecs * n * rounds) -- runtime is itself Ripple's metric.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
from torch.func import functional_call, grad, jvp

from ..backends.llm import make_llm_loss
from ..core.flirds_estimator import _chunked
from ..data.corruptors import free_rider
from ..data.llm import build_val_batch, build_val_batches
from ..repro import seed_everything
from .ripple import _flat, _orthoproj


def _val_grad_flat(loss_chunks, params, pkeys):
    """Flattened val-loss gradient at `params` (chunked weighted-sum == the
    estimator's g^r) -> numpy P-vector."""
    g, _ = _chunked(loss_chunks, {}, params, None, pkeys)
    return _flat(g, pkeys).detach().cpu().numpy()


def _client_drop_delta(model, gstate, train_batches, steps, lr, loss_chunks, pkeys, device):
    """Realized local plain-SGD (momentum 0, completion-only CE) accumulating the IRDS
    drop term  sum_t lr*<g_val(w_t), g_batch(w_t)>; returns (drop, delta_cpu).

    Pure-functional like the estimator: the model is only the function (functional_call
    overrides params), so no model mutation / no standard-autograd interplay with the
    functorch val grad.  g_batch = grad(train-batch loss) = the realized SGD step
    direction (manual step `w -= lr*g_batch` == SGD momentum 0); g_val = the chunked
    val grad at the current local params.  drop>0 <=> the step reduced val loss (clean
    client -> +).  Mirrors ripple.client_drop_and_delta."""
    params = {n: gstate[n].detach().float().to(device) for n in pkeys}
    w0 = {n: params[n].clone() for n in pkeys}
    drop = 0.0
    for t in range(steps):
        batch = train_batches[t % len(train_batches)]

        def bloss(p):
            return functional_call(model, (p, {}), kwargs=batch).loss

        gb = grad(bloss)(params)                                # realized minibatch grad (step dir)
        gv, _ = _chunked(loss_chunks, {}, params, None, pkeys)  # val grad at w_t (chunked)
        drop += lr * float(sum((gv[n] * gb[n]).sum() for n in pkeys))
        params = {n: params[n] - lr * gb[n] for n in pkeys}     # plain SGD step (momentum 0)
    delta = {n: (params[n] - w0[n]).detach().cpu() for n in pkeys}
    return drop, delta


def _local_hessian_topk(model, gstate, hess_batch, k, pkeys, device):
    """Top-k eigenpairs of the client's LOCAL-data loss Hessian at w^r (Eq 14 sketch),
    LoRA params only; HVP via jvp(grad(loss)) over eager attention.  Mirrors
    ripple.local_hessian_topk with a fixed v0 + a no-convergence fallback (the
    deferred eigsh-robustness TODO).  Returns (vals[k], vecs[P, k]); zero-padded if
    eigsh returns fewer than k."""
    params = {n: gstate[n].detach().float().to(device) for n in pkeys}
    sizes = {n: params[n].numel() for n in pkeys}
    shapes = {n: params[n].shape for n in pkeys}
    P = sum(sizes.values())

    def loss_fn_local(p):
        return functional_call(model, (p, {}), kwargs=hess_batch).loss
    gfn = grad(loss_fn_local)

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
    v0 = np.ones(P) / np.sqrt(P)                                # fixed v0 -> reproducible eigvecs
    ncv = min(P, max(2 * k + 1, 20))
    try:
        vals, vecs = eigsh(op, k=k, which="LA", v0=v0, ncv=ncv, maxiter=300)
    except ArpackNoConvergence as e:                           # keep what converged
        vals, vecs = np.atleast_1d(e.eigenvalues), e.eigenvectors
        if vecs.ndim < 2 or vecs.shape[0] != P:                # 0 converged -> normalize to (P, 0)
            vals, vecs = np.zeros(0), np.zeros((P, 0))
    if vecs.shape[1] < k:                                       # zero-pad -> uniform Ur [P, n*k]
        pad = k - vecs.shape[1]
        vals = np.concatenate([vals, np.zeros(pad)])
        vecs = np.concatenate([vecs, np.zeros((P, pad))], axis=1)
    return vals, vecs


def ripple_shapley_llm(model, init_lora, clients, tok, val_chunks, device, *,
                       rounds, steps, lr, k=8, m=40, R=None, seed=0,
                       free_riders=frozenset(), free_rider_mode="zero",
                       hess_bs=4, hess_maxlen=256, train_maxlen=512, batch_size=4):
    """Client-level Ripple Shapley (drop + ripple) for a PEFT-LoRA causal-LM, summed
    over its OWN FedAvg trajectory.  `model` is reset to `init_lora` (fresh adapter);
    clean clients -> HIGH phi.  Returns phi[n]."""
    seed_everything(seed)                                       # LLM: no cudnn-det
    R = R or rounds
    n = len(clients)
    model.load_state_dict(init_lora, strict=False)
    loss_fn, pkeys, loss_chunks = make_llm_loss(model, val_chunks, device)   # eval + hook-clear + chunks
    named = dict(model.named_parameters())
    gstate = {nme: named[nme].detach().clone().to(device) for nme in pkeys}
    P = sum(gstate[nme].numel() for nme in pkeys)

    recs = [list(c) for c in clients]                          # tokenize each client's local data once
    train_batches, hess_batch = [None] * n, [None] * n
    for c in range(n):
        if c in free_riders:                                   # fabricates -> no local data used
            continue
        train_batches[c] = build_val_batches(recs[c], tok, train_maxlen, device, batch_size)
        hess_batch[c] = build_val_batch(recs[c][:hess_bs], tok, hess_maxlen, device)

    fr_gen = torch.Generator().manual_seed(seed + 1)
    nsel = np.array([[len(recs[c]) for c in range(n)]] * rounds, dtype=float)
    drop = np.zeros((rounds, n))
    DW = np.zeros((rounds, n, P), dtype=np.float32)
    VG = np.zeros((rounds, P), dtype=np.float32)
    Us, Q = [], None
    for r in range(rounds):
        VG[r] = _val_grad_flat(loss_chunks, {nme: gstate[nme].detach().float() for nme in pkeys}, pkeys)
        vecs_r, vals_r = [], []
        for c in range(n):
            if c in free_riders:                               # zero delta -> 0 marginal; no sketch
                d = free_rider(gstate, mode=free_rider_mode, generator=fr_gen)
                DW[r, c] = _flat({nme: d[nme].float().cpu() for nme in pkeys}, pkeys).numpy()
                vecs_r.append(np.zeros((P, k))); vals_r.append(np.zeros(k))
                continue
            d_drop, delta = _client_drop_delta(model, gstate, train_batches[c], steps, lr,
                                               loss_chunks, pkeys, device)
            drop[r, c] = d_drop
            DW[r, c] = _flat({nme: delta[nme].float() for nme in pkeys}, pkeys).numpy()
            vv, ee = _local_hessian_topk(model, gstate, hess_batch[c], k, pkeys, device)
            vecs_r.append(ee); vals_r.append(vv)
        Ur = np.concatenate(vecs_r, axis=1)                    # [P, n*k]
        Us.append((Ur, np.concatenate(vals_r)))
        Q = _orthoproj(Q, Ur, m)                               # progressive subspace (Eq 15)
        agg = sum((nsel[r, c] / nsel[r].sum()) * DW[r, c] for c in range(n))   # FedAvg over LoRA params
        off = 0
        for nme in pkeys:
            sz = gstate[nme].numel()
            gstate[nme] = gstate[nme] + torch.as_tensor(
                agg[off:off + sz].reshape(gstate[nme].shape), dtype=gstate[nme].dtype, device=device)
            off += sz
        torch.cuda.empty_cache()                               # release per-round eager grad/HVP cache

    # ---- Eq 16-19 ripple chain + drop (mirrors ripple.ripple_shapley; identical math) ----
    m = Q.shape[1]
    VGp = VG @ Q                                                # [rounds, m]
    DWp = DW @ Q                                                # [rounds, n, m]
    Bs = [Q.T @ Ur for Ur, _ in Us]                            # [m, n*k] per round (Eq 16)
    Ls = [lam for _, lam in Us]
    Ms = [np.eye(m) - lr * (Bs[t] * Ls[t]) @ Bs[t].T for t in range(rounds)]   # Eq 18 factor
    alpha = nsel / nsel.sum(axis=1, keepdims=True)
    dphi = alpha * drop                                         # Eq 7
    rphi = np.zeros((rounds, n))
    for t0 in range(rounds):                                   # ripple sums r=2..R (Eq 13)
        Plow = np.eye(m)
        for rr in range(2, R + 1):
            if t0 + rr >= rounds:
                break
            Plow = Ms[t0 + rr - 1] @ Plow                      # prod_{l} M[t0+l] (Eq 18)
            rphi[t0] += -((alpha[t0][:, None] * DWp[t0]) @ (Plow.T @ VGp[t0 + rr]))   # Eq 19
    return (dphi + rphi).sum(axis=0)
