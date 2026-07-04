#!/usr/bin/env python
"""Instrumented COPY of flirds/baselines/ripple.py -- eigsh CPU-stall diagnosis.

2026-07-04 Ripple audit. DIAGNOSTIC ONLY: nothing in the repo is modified; this
file re-implements the CNN Ripple port's cost-relevant paths as an instrumented
copy (imports only `_flat`/`_orthoproj`/`LeNet5`/`seed_everything` from the repo).
Synthetic random data (the stall mechanism is spectral/tolerance-driven, not
data-content-driven).  CPU-only by contract: run with CUDA_VISIBLE_DEVICES=
(empty) -- GPUs on this box carry live experiment campaigns.

Run from <repo>/codes with PYTHONPATH=. :
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 nice -n 10 PYTHONPATH=. \
    python instrumented_ripple_cnn.py --phase all --out <scratch dir>

Phases
  traj  Instrumented ripple_shapley copy on a tiny config (N=4, R=3, E=1, k=8).
        Section timers split (A) log-generation (local SGD + FedAvg aggregation)
        from (B) valuation (per-step drop val-grad, round val-grad, eigsh,
        QR/projection, Eq16-19 chain).  eigsh keeps the original semantics
        (tol not passed -> 0, which=LA, fixed v0, no-convergence -> one retry at
        ncv=4k+1 -> keep converged pairs) but maxiter is REDUCED (default 100,
        recorded; original default 1000) plus a per-call matvec budget so the
        worst case fits the wall budget.  Extrapolation to maxiter=1000 uses the
        measured matvecs-per-restart from phase `eig`.
  eig   Standalone diagnostics on ONE fixed operator (round-0 model state,
        client-0 data): matvec timing, fp32-vs-fp64 operator noise floor,
        scipy `maxiter` semantics (restart cycles vs matvec count), tol sweep
        {0, 1e-8, 1e-6, 1e-3}, the ncv=4k+1 retry row, a full-spec k=20 row,
        and a tol=0 row on an fp64 matvec twin.  Budget/deadline-capped.
  spin  One long tol=0 eigsh (maxiter default 400, deadline-capped) that prints
        its PID so a second shell can snapshot per-thread CPU.  Run once under
        OMP_NUM_THREADS=8 and once under OMP_NUM_THREADS=1; wall / matvec
        throughput difference = BLAS/OpenMP spin-wait overhead evidence.

The tol/maxiter/ncv variations are HYPOTHESIS-TESTING DIAGNOSTICS, not a port
change.  GPU HVP interaction (H2D/D2H, GPU-idle-while-ARPACK) is invisible in
this CPU-only reproduction -- documented as a limitation.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
from torch.func import functional_call, grad, jvp
from torch.utils.data import DataLoader, TensorDataset

from flirds.baselines.ripple import _flat, _orthoproj
from flirds.models.cnn import LeNet5
from flirds.repro import seed_everything


# --------------------------------------------------------------------------- #
# matvec instrumentation                                                      #
# --------------------------------------------------------------------------- #
class MatvecBudgetExceeded(RuntimeError):
    pass


class MatvecCounter:
    """Wraps the raw HVP closure: counts calls, accumulates wall time inside the
    Python matvec (torch HVP + numpy<->torch conversion), enforces an optional
    call budget and wall deadline (raised exceptions propagate out of eigsh)."""

    def __init__(self, raw, budget=None, deadline=None):
        self.raw, self.budget, self.deadline = raw, budget, deadline
        self.n = 0
        self.t_sum = 0.0

    def __call__(self, v):
        if self.budget is not None and self.n >= self.budget:
            raise MatvecBudgetExceeded(f"budget:{self.budget}")
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise MatvecBudgetExceeded("deadline")
        t0 = time.perf_counter()
        out = self.raw(v)
        self.t_sum += time.perf_counter() - t0
        self.n += 1
        return out


def build_hvp(model, gstate, x, y, device, dtype=torch.float32):
    """HVP closure replicating ripple.local_hessian_topk L94-117.

    dtype=float32 reproduces the port exactly (fp32 params/input/tangent, output
    float32 numpy -> scipy upcasts to the LinearOperator's fp64).  dtype=float64
    builds the noise-measurement twin."""
    model.load_state_dict(gstate)
    model.to(device)
    pkeys = [n for n, _ in model.named_parameters()]
    params = {n: gstate[n].detach().to(device=device, dtype=dtype) for n in pkeys}
    buffers = {n: (b.detach().to(dtype=dtype) if b.is_floating_point() else b.detach())
               for n, b in model.named_buffers()}
    xx = x.to(dtype=dtype)
    sizes = {n: params[n].numel() for n in pkeys}
    shapes = {n: params[n].shape for n in pkeys}
    P = sum(sizes.values())

    def loss_fn(p):
        return F.cross_entropy(functional_call(model, (p, buffers), (xx,)), y)
    gfn = grad(loss_fn)

    def unflat(v):
        out, i = {}, 0
        for n in pkeys:
            out[n] = v[i:i + sizes[n]].reshape(shapes[n])
            i += sizes[n]
        return out

    def matvec(vnp):
        v = torch.as_tensor(np.ascontiguousarray(vnp), dtype=dtype, device=device)
        _, hv = jvp(gfn, (params,), (unflat(v),))
        return _flat(hv, pkeys).detach().cpu().numpy()

    return matvec, P


def run_eigsh(raw_mv, P, k, tag, records_path, which="LA", v0=None, ncv=None,
              maxiter=None, tol=0, budget=None, deadline_s=None, extra=None):
    """One instrumented eigsh call; appends a JSONL record; returns (vals, vecs, rec).

    On ArpackNoConvergence the pairs that DID converge are extracted (as the port
    does); on budget/deadline abort, zero pairs (deviation -- recorded in status)."""
    counter = MatvecCounter(
        raw_mv, budget=budget,
        deadline=None if deadline_s is None else time.perf_counter() + deadline_s)
    op = LinearOperator((P, P), matvec=counter, dtype=np.float64)
    if v0 is None:
        v0 = np.random.default_rng(0).standard_normal(P)   # port's fixed v0 (L127)
    ncv_eff = int(min(P, max(2 * k + 1, 20))) if ncv is None else int(ncv)
    kw = dict(k=k, which=which, v0=v0, maxiter=maxiter, tol=tol)
    if ncv is not None:
        kw["ncv"] = ncv
    t0 = time.perf_counter()
    status = "converged"
    try:
        vals, vecs = eigsh(op, **kw)
    except ArpackNoConvergence as e:
        status = "no_convergence"
        vals = np.atleast_1d(e.eigenvalues) if e.eigenvalues is not None else np.zeros(0)
        vecs = e.eigenvectors
        if vecs is None or getattr(vecs, "ndim", 0) < 2 or vecs.shape[0] != P:
            vals, vecs = np.zeros(0), np.zeros((P, 0))
    except MatvecBudgetExceeded as e:
        status = f"aborted_{e}"
        vals, vecs = np.zeros(0), np.zeros((P, 0))
    wall = time.perf_counter() - t0
    resid_abs, resid_rel = [], []
    for i in range(int(vals.shape[0])):          # residuals via UNcounted raw matvec
        av = np.asarray(raw_mv(vecs[:, i]), dtype=np.float64)
        r = float(np.linalg.norm(av - float(vals[i]) * vecs[:, i]))
        resid_abs.append(r)
        resid_rel.append(r / max(abs(float(vals[i])), 1e-30))
    rec = dict(tag=tag, k=int(k), which=which,
               ncv_requested=("default" if ncv is None else int(ncv)),
               ncv_effective=ncv_eff, tol=tol, maxiter=maxiter, status=status,
               n_matvec=counter.n, t_wall_s=round(wall, 3),
               t_matvec_sum_s=round(counter.t_sum, 3),
               t_arpack_residual_s=round(wall - counter.t_sum, 3),
               mean_matvec_ms=round(1e3 * counter.t_sum / max(counter.n, 1), 2),
               n_eigvals=int(vals.shape[0]),
               eigvals=[float(v) for v in np.asarray(vals).ravel()],
               resid_abs=[float(f"{r:.4e}") for r in resid_abs],
               resid_rel=[float(f"{r:.4e}") for r in resid_rel])
    if extra:
        rec.update(extra)
    with open(records_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[eigsh:{tag}] status={status} n_matvec={counter.n} wall={wall:.2f}s "
          f"mv_sum={counter.t_sum:.2f}s arpack_resid={wall - counter.t_sum:.2f}s "
          f"n_vals={int(vals.shape[0])}", flush=True)
    return vals, vecs, rec


# --------------------------------------------------------------------------- #
# synthetic tiny stage (shared by all phases)                                 #
# --------------------------------------------------------------------------- #
def build_stage(args, device="cpu"):
    seed_everything(0, cudnn_deterministic=True)     # as ripple_shapley L151
    g = torch.Generator().manual_seed(1234)
    loaders, client_xy = [], []
    for _ in range(args.n_clients):
        x = torch.randn(args.n_per, 1, 28, 28, generator=g)
        y = torch.randint(0, 10, (args.n_per,), generator=g)
        loaders.append(DataLoader(TensorDataset(x, y), batch_size=args.batch, shuffle=True))
        client_xy.append((x.to(device), y.to(device)))
    val_x = torch.randn(args.n_val, 1, 28, 28, generator=g).to(device)
    val_y = torch.randint(0, 10, (args.n_val,), generator=g).to(device)
    model = LeNet5(width=args.width).to(device)
    gstate = {n: v.detach().clone() for n, v in model.state_dict().items()}
    return model, gstate, loaders, client_xy, val_x, val_y


# --------------------------------------------------------------------------- #
# phase traj: instrumented ripple_shapley copy                                #
# --------------------------------------------------------------------------- #
def client_drop_and_delta_instr(model, gstate, loader, epochs, lr, val_x, val_y, device):
    """Copy of ripple.client_drop_and_delta L48-81 with split timers:
    t_train = batch fwd/bwd + opt.step (log-generation work),
    t_drop_valgrad = full-val grad + drop inner product (valuation work)."""
    model.load_state_dict(gstate)
    model.to(device).train()
    pkeys = [n for n, _ in model.named_parameters()]
    w0 = {n: gstate[n].detach().clone().to(device) for n in pkeys}
    buffers = {n: b.detach() for n, b in model.named_buffers()}

    def vloss(p):
        return F.cross_entropy(functional_call(model, (p, buffers), (val_x,)), val_y)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    drop, t_train, t_valgrad, n_steps = 0.0, 0.0, 0.0, 0
    for _ in range(epochs):
        for x, y in loader:
            t0 = time.perf_counter()
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            gb = {n: p.grad.detach() for n, p in model.named_parameters()}
            t1 = time.perf_counter()
            gv = grad(vloss)({n: p.detach() for n, p in model.named_parameters()})
            drop += lr * float(sum((gv[n] * gb[n]).sum() for n in pkeys))
            t2 = time.perf_counter()
            opt.step()
            t3 = time.perf_counter()
            t_train += (t1 - t0) + (t3 - t2)
            t_valgrad += t2 - t1
            n_steps += 1
    delta = {n: (p.detach() - w0[n]) for n, p in model.named_parameters()}
    return drop, delta, dict(t_train=t_train, t_drop_valgrad=t_valgrad, n_steps=n_steps)


def local_hessian_topk_instr(model, gstate, x, y, k, device, maxiter, budget,
                             records_path, tag):
    """Copy of ripple.local_hessian_topk L84-138 semantics with instrumentation:
    primary call (ncv default) -> on no-convergence retry once at ncv=4k+1 ->
    keep the pairs that DID converge.  Budget aborts yield 0 pairs (recorded)."""
    raw_mv, P = build_hvp(model, gstate, x, y, device)
    v0 = np.random.default_rng(0).standard_normal(P)
    vals, vecs, rec = run_eigsh(raw_mv, P, k, tag + ":primary", records_path,
                                v0=v0, maxiter=maxiter, tol=0, budget=budget)
    if rec["status"] == "no_convergence":
        vals, vecs, rec2 = run_eigsh(raw_mv, P, k, tag + ":retry_ncv4k1", records_path,
                                     v0=v0, ncv=int(min(P, 4 * k + 1)),
                                     maxiter=maxiter, tol=0, budget=budget)
        if rec2["status"] != "converged":
            print(f"[ripple-instr] eigsh fallback: {len(vals)}/{k} pairs kept "
                  f"(maxiter={maxiter})", flush=True)
    return vals, vecs


def phase_traj(args, out_dir):
    device = "cpu"
    records_path = os.path.join(out_dir, "eigsh_calls.jsonl")
    model, gstate, loaders, _, val_x, val_y = build_stage(args, device)
    pkeys = [n for n, _ in model.named_parameters()]
    buffers0 = {n: b.detach().clone() for n, b in model.named_buffers()}
    n = args.n_clients
    rounds, k, m, R, lr = args.rounds, args.k, args.m, args.depth, args.lr
    P = sum(p.numel() for _, p in model.named_parameters())
    print(f"[traj] P={P} n={n} rounds={rounds} epochs={args.epochs} k={k} m={m} "
          f"depth={R} maxiter={args.traj_maxiter} mv_budget={args.mv_budget}", flush=True)

    t_all0 = time.perf_counter()
    # HVP data prep (loader -> one tensor; only the eigsh sketch consumes it -> valuation)
    t0 = time.perf_counter()
    client_xy = [(torch.cat([x for x, _ in ld]).to(device),
                  torch.cat([y for _, y in ld]).to(device)) for ld in loaders]
    t_hvp_data_prep = time.perf_counter() - t0

    def val_grad_flat(state):
        params = {nn: state[nn].detach().to(device) for nn in pkeys}
        gg = grad(lambda p: F.cross_entropy(
            functional_call(model, (p, buffers0), (val_x,)), val_y))(params)
        return _flat(gg, pkeys).detach().cpu().numpy()

    S = dict(t_local_train=0.0, t_agg=0.0,                       # (A) log generation
             t_drop_valgrad=0.0, t_round_valgrad=0.0,            # (B) valuation ...
             t_eigsh_wall=0.0, t_matvec_sum=0.0, t_qr=0.0,
             t_dw_materialize=0.0, t_hvp_data_prep=t_hvp_data_prep)
    n_matvec_total, eig_recs = 0, []

    drop = np.zeros((rounds, n))
    nsel = np.zeros((rounds, n))
    DW = np.zeros((rounds, n, P), dtype=np.float32)
    VG = np.zeros((rounds, P), dtype=np.float32)
    Us, Q = [], None
    for r in range(rounds):
        t0 = time.perf_counter()
        VG[r] = val_grad_flat(gstate)
        S["t_round_valgrad"] += time.perf_counter() - t0
        vecs_r, vals_r = [], []
        for c in range(n):
            d_drop, delta, tm = client_drop_and_delta_instr(
                model, gstate, loaders[c], args.epochs, lr, val_x, val_y, device)
            S["t_local_train"] += tm["t_train"]
            S["t_drop_valgrad"] += tm["t_drop_valgrad"]
            drop[r, c] = d_drop
            t0 = time.perf_counter()
            DW[r, c] = _flat(delta, pkeys).cpu().numpy()
            nsel[r, c] = len(loaders[c].dataset)
            S["t_dw_materialize"] += time.perf_counter() - t0
            xc, yc = client_xy[c]
            t0 = time.perf_counter()
            vv, ee = local_hessian_topk_instr(model, gstate, xc, yc, k, device,
                                              args.traj_maxiter, args.mv_budget,
                                              records_path, tag=f"traj:r{r}c{c}")
            S["t_eigsh_wall"] += time.perf_counter() - t0
            vecs_r.append(ee if ee.ndim == 2 and ee.shape[0] == P else np.zeros((P, 0)))
            vals_r.append(np.asarray(vv).ravel())
            print(f"[traj] r{r}c{c} done ({time.perf_counter() - t_all0:.1f}s elapsed)",
                  flush=True)
        Ur = np.concatenate(vecs_r, axis=1)
        Us.append((Ur, np.concatenate(vals_r) if vals_r else np.zeros(0)))
        t0 = time.perf_counter()
        if Ur.shape[1] > 0 or Q is not None:                    # guard: qr((P,0)) edge
            Q = _orthoproj(Q, Ur, m) if Ur.shape[1] > 0 else Q
        S["t_qr"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        agg = sum((nsel[r, c] / nsel[r].sum()) * DW[r, c] for c in range(n))
        off = 0
        for key in pkeys:
            sz = gstate[key].numel()
            gstate[key] = gstate[key] + torch.as_tensor(
                agg[off:off + sz].reshape(gstate[key].shape),
                dtype=gstate[key].dtype, device=device)
            off += sz
        S["t_agg"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    if Q is None:
        Q = np.zeros((P, 0))
    mm = Q.shape[1]
    VGp = VG @ Q
    DWp = DW @ Q
    Bs = [Q.T @ Ur for Ur, _ in Us]
    Ls = [lam for _, lam in Us]
    Ms = [np.eye(mm) - lr * (Bs[t] * Ls[t]) @ Bs[t].T for t in range(rounds)]
    alpha = nsel / nsel.sum(axis=1, keepdims=True)
    dphi = alpha * drop
    rphi = np.zeros((rounds, n))
    for t0i in range(rounds):
        Plow = np.eye(mm)
        for rr in range(2, R + 1):
            if t0i + rr >= rounds:
                break
            Plow = Ms[t0i + rr - 1] @ Plow
            rphi[t0i] += -((alpha[t0i][:, None] * DWp[t0i]) @ (Plow.T @ VGp[t0i + rr]))
    phi = (dphi + rphi).sum(axis=0)
    S["t_chain_eq16_19"] = time.perf_counter() - t0
    t_total = time.perf_counter() - t_all0

    # accounting: (A) log generation vs (B) valuation
    for line in open(records_path):
        rec = json.loads(line)
        if rec["tag"].startswith("traj:"):
            n_matvec_total += rec["n_matvec"]
            S["t_matvec_sum"] += rec["t_matvec_sum_s"]
            eig_recs.append(rec)
    A = S["t_local_train"] + S["t_agg"]
    B = (S["t_drop_valgrad"] + S["t_round_valgrad"] + S["t_eigsh_wall"]
         + S["t_qr"] + S["t_dw_materialize"] + S["t_hvp_data_prep"]
         + S["t_chain_eq16_19"])
    result = dict(
        config=dict(P=P, n_clients=n, n_per=args.n_per, rounds=rounds,
                    epochs=args.epochs, batch=args.batch, lr=lr, n_val=args.n_val,
                    k=k, m=m, depth=R, width=args.width,
                    traj_maxiter=args.traj_maxiter, mv_budget=args.mv_budget,
                    note="maxiter reduced from the port's 1000 for wall budget; "
                         "tol=0/which=LA/v0/ncv-retry semantics preserved"),
        sections_s={kk: round(v, 3) for kk, v in S.items()},
        t_total_s=round(t_total, 3),
        A_log_generation_s=round(A, 3),
        B_valuation_s=round(B, 3),
        B_over_total=round(B / max(t_total, 1e-9), 4),
        untimed_residual_s=round(t_total - A - B, 3),
        n_matvec_total=n_matvec_total,
        n_eigsh_calls=len(eig_recs),
        phi=[float(v) for v in phi],
    )
    with open(os.path.join(out_dir, "traj_sections.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[traj] TOTAL {t_total:.1f}s | A(log-gen) {A:.1f}s | B(valuation) {B:.1f}s "
          f"| B/total {result['B_over_total']:.3f} | matvecs {n_matvec_total}", flush=True)


# --------------------------------------------------------------------------- #
# phase eig: standalone operator diagnostics                                  #
# --------------------------------------------------------------------------- #
def phase_eig(args, out_dir):
    device = "cpu"
    records_path = os.path.join(out_dir, "eigsh_calls.jsonl")
    model, gstate, _, client_xy, _, _ = build_stage(args, device)
    x, y = client_xy[0]
    raw32, P = build_hvp(model, gstate, x, y, device, torch.float32)
    raw64, _ = build_hvp(model, gstate, x, y, device, torch.float64)
    out = dict(P=P, k=args.k, n_hvp_samples=args.n_per, width=args.width)
    print(f"[eig] P={P} k={args.k} hvp_samples={args.n_per}", flush=True)

    # 1) matvec timing (3 warmup + 10 timed)
    rng = np.random.default_rng(7)
    v = rng.standard_normal(P)
    for _ in range(3):
        raw32(v)
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        raw32(v)
        ts.append(time.perf_counter() - t0)
    out["matvec_fp32_ms"] = dict(mean=round(1e3 * float(np.mean(ts)), 2),
                                 min=round(1e3 * float(np.min(ts)), 2),
                                 max=round(1e3 * float(np.max(ts)), 2))
    print(f"[eig] matvec fp32 mean {out['matvec_fp32_ms']['mean']}ms", flush=True)

    # 2) fp32-vs-fp64 operator noise floor on 5 random unit vectors
    rels, dets = [], []
    for i in range(5):
        vi = rng.standard_normal(P)
        vi /= np.linalg.norm(vi)
        a32 = np.asarray(raw32(vi), dtype=np.float64)
        a64 = np.asarray(raw64(vi), dtype=np.float64)
        rels.append(float(np.linalg.norm(a32 - a64) / max(np.linalg.norm(a64), 1e-30)))
        b32 = np.asarray(raw32(vi), dtype=np.float64)   # determinism check
        dets.append(float(np.linalg.norm(a32 - b32)))
    out["fp32_vs_fp64_rel_err"] = [float(f"{r:.3e}") for r in rels]
    out["fp32_repeat_determinism_absdiff"] = [float(f"{d:.3e}") for d in dets]
    print(f"[eig] fp32 noise rel err ~ {np.mean(rels):.2e}, "
          f"repeat-determinism {max(dets):.1e}", flush=True)

    # 3) |lambda|_max estimate (power iteration on raw32, uncounted)
    vi = rng.standard_normal(P)
    vi /= np.linalg.norm(vi)
    lam = 0.0
    for _ in range(30):
        av = np.asarray(raw32(vi), dtype=np.float64)
        lam = float(np.linalg.norm(av))
        vi = av / max(lam, 1e-30)
    out["lam_abs_max_powerit"] = float(f"{lam:.6e}")
    print(f"[eig] |lambda|max ~ {lam:.4e}", flush=True)

    # 4) scipy maxiter semantics: tol=0 at maxiter in {3, 10}
    sem = []
    for mi in (3, 10):
        _, _, rec = run_eigsh(raw32, P, args.k, f"eig:semantics_mi{mi}", records_path,
                              maxiter=mi, tol=0, deadline_s=args.eig_deadline_s)
        sem.append(dict(maxiter=mi, n_matvec=rec["n_matvec"], status=rec["status"]))
    out["maxiter_semantics"] = sem
    if sem[0]["status"] != "converged" and sem[1]["status"] != "converged":
        dmv = sem[1]["n_matvec"] - sem[0]["n_matvec"]
        out["matvec_per_restart_est"] = round(dmv / 7.0, 2)   # (10-3) restarts

    # 5) tol sweep (port config: k, ncv default, maxiter=1000)
    for tol in (0, 1e-8, 1e-6, 1e-3):
        run_eigsh(raw32, P, args.k, f"eig:tol{tol:g}", records_path,
                  maxiter=args.eig_maxiter, tol=tol, budget=args.eig_budget,
                  deadline_s=args.eig_deadline_s,
                  extra=dict(sweep="tol", operator="fp32"))

    # 6) retry config row: tol=0, ncv=4k+1
    run_eigsh(raw32, P, args.k, "eig:tol0_ncv4k1", records_path,
              ncv=int(min(P, 4 * args.k + 1)), maxiter=args.eig_maxiter, tol=0,
              budget=args.eig_budget, deadline_s=args.eig_deadline_s,
              extra=dict(sweep="ncv", operator="fp32"))

    # 7) full-spec row: k=20 (track_c1 full), tol=0, ncv default -> 41
    run_eigsh(raw32, P, 20, "eig:tol0_k20", records_path,
              maxiter=args.eig_maxiter, tol=0, budget=args.eig_budget,
              deadline_s=args.eig_deadline_s, extra=dict(sweep="k", operator="fp32"))

    # 8) fp64 operator twin: tol=0 -- does machine-precision convergence become
    #    reachable when the matvec noise floor drops to fp64?
    run_eigsh(raw64, P, args.k, "eig:tol0_fp64op", records_path,
              maxiter=args.eig_maxiter, tol=0, budget=args.eig_budget,
              deadline_s=args.eig_deadline_s, extra=dict(sweep="dtype", operator="fp64"))

    with open(os.path.join(out_dir, "eig_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[eig] done", flush=True)


# --------------------------------------------------------------------------- #
# phase spin: long eigsh for thread observation                               #
# --------------------------------------------------------------------------- #
def phase_spin(args, out_dir):
    device = "cpu"
    records_path = os.path.join(out_dir, "eigsh_calls.jsonl")
    omp = os.environ.get("OMP_NUM_THREADS", "unset")
    print(f"[spin] PID={os.getpid()} OMP_NUM_THREADS={omp} "
          f"torch_threads={torch.get_num_threads()}", flush=True)
    model, gstate, _, client_xy, _, _ = build_stage(args, device)
    x, y = client_xy[0]
    raw32, P = build_hvp(model, gstate, x, y, device, torch.float32)
    for _ in range(3):                                          # warmup
        raw32(np.random.default_rng(7).standard_normal(P))
    # loop eigsh calls until the deadline so thread snapshots see sustained load
    # even if individual calls converge quickly
    t_end = time.perf_counter() + args.spin_deadline_s
    calls, n_mv, t_wall_sum, t_mv_sum = [], 0, 0.0, 0.0
    i = 0
    while time.perf_counter() < t_end - 1.0:
        remain = t_end - time.perf_counter()
        _, _, rec = run_eigsh(raw32, P, args.k, f"spin:omp{omp}:call{i}", records_path,
                              maxiter=args.spin_maxiter, tol=0, deadline_s=remain,
                              extra=dict(omp=omp, torch_threads=torch.get_num_threads(),
                                         pid=os.getpid()))
        calls.append(dict(status=rec["status"], n_matvec=rec["n_matvec"],
                          t_wall_s=rec["t_wall_s"], t_matvec_sum_s=rec["t_matvec_sum_s"]))
        n_mv += rec["n_matvec"]
        t_wall_sum += rec["t_wall_s"]
        t_mv_sum += rec["t_matvec_sum_s"]
        i += 1
    summary = dict(omp=omp, torch_threads=torch.get_num_threads(), pid=os.getpid(),
                   P=P, k=args.k, spin_maxiter=args.spin_maxiter,
                   n_calls=len(calls), n_matvec_total=n_mv,
                   t_wall_total_s=round(t_wall_sum, 3),
                   t_matvec_total_s=round(t_mv_sum, 3),
                   mv_per_s=round(n_mv / max(t_wall_sum, 1e-9), 2),
                   mean_matvec_ms=round(1e3 * t_mv_sum / max(n_mv, 1), 3),
                   calls=calls)
    with open(os.path.join(out_dir, f"spin_omp{omp}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[spin] done omp={omp} calls={len(calls)} mv/s={summary['mv_per_s']} "
          f"mean_mv_ms={summary['mean_matvec_ms']}", flush=True)


# --------------------------------------------------------------------------- #
def dump_env(out_dir):
    import scipy
    try:
        cfg = np.show_config(mode="dicts")               # numpy>=1.26
    except TypeError:
        buf = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = buf
            np.show_config()
        finally:
            sys.stdout = old_stdout
        cfg = buf.getvalue()
    env = dict(python=sys.version.split()[0], torch=torch.__version__,
               scipy=scipy.__version__, numpy=np.__version__,
               platform=platform.platform(), cpu_count=os.cpu_count(),
               torch_threads=torch.get_num_threads(),
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),
               MKL_NUM_THREADS=os.environ.get("MKL_NUM_THREADS"),
               CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
               cuda_available=torch.cuda.is_available(),
               numpy_blas=cfg if isinstance(cfg, str) else json.dumps(cfg, default=str))
    with open(os.path.join(out_dir, "env.json"), "w") as f:
        json.dump(env, f, indent=2)
    print(f"[env] torch={env['torch']} scipy={env['scipy']} numpy={env['numpy']} "
          f"OMP={env['OMP_NUM_THREADS']} cuda={env['cuda_available']}", flush=True)
    assert not torch.cuda.is_available(), "GPU visible -- contract violation, aborting"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["all", "traj", "eig", "spin"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clients", type=int, default=4)
    ap.add_argument("--n-per", type=int, default=128)      # samples/client (HVP batch)
    ap.add_argument("--rounds", type=int, default=3)       # >=3: ripple cross-round on
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--depth", type=int, default=5)        # ripple depth R
    ap.add_argument("--width", type=float, default=1.0)    # LeNet5 width (P=61,706 at 1)
    ap.add_argument("--traj-maxiter", type=int, default=100)
    ap.add_argument("--mv-budget", type=int, default=2600)
    ap.add_argument("--eig-maxiter", type=int, default=1000)   # port default
    ap.add_argument("--eig-budget", type=int, default=5000)
    ap.add_argument("--eig-deadline-s", type=float, default=180.0)
    ap.add_argument("--spin-maxiter", type=int, default=400)
    ap.add_argument("--spin-deadline-s", type=float, default=240.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dump_env(args.out)
    if args.phase in ("all", "traj"):
        phase_traj(args, args.out)
    if args.phase in ("all", "eig"):
        phase_eig(args, args.out)
    if args.phase == "spin":
        phase_spin(args, args.out)
    print("[main] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
