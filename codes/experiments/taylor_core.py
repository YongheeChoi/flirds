"""Shared per-round Taylor-remainder measurement — the quantity Appendix C.5 reports.

Restored from the 2026-07 math-rigor survey artifact (measure_taylor_residual.py,
deleted at 43a5f17; recover with `git show 9dacbad:research-wiki/survey/
irds-fl-math-rigor-2026-07/measure_taylor_residual.py`) and split so both tracks
share one core.  The measured game is the FIXED-WEIGHT game, identical to the
(b) oracle's (in_run_sv):

  u_r(S)   = l(w_r + D_S) - l(w_r),   D_S = sum_{k in S} a_k,  a_k = p_k^r d_k,
             p_k^r = n_k / sum_{j in P_r} n_j        (denominator over all of P_r,
                                                      so it does not depend on S)
  u1_r(S)  = <g_r, D_S> = sum_{k in S} b_k,          b_k  = <g_r, a_k>
  u2_r(S)  = u1_r(S) + 1/2 sum_{i,j in S} Q_ij,      Q_ij = <a_i, H_r a_j>
  ut_r(S)  = l(w_r + c_S D_S) - l(w_r),              c_S  = sum_{P_r} n / sum_S n
                                                     (renorm game; optional)

resid1 = |u - u1|, resid2 = |u - u2|.  Proposition remainder orders: resid1 =
O(||D_S||^2), resid2 = O(||D_S||^3).

Cost per round: 2^K forward (+2^K-1 if renorm) + 1 gradient + K HVPs.  This is a
FROM-LOGS computation: it needs the frozen trajectory (w_r, {d_k}), never a
retrain.  The trajectory is not persisted anywhere in runs/ (no checkpoints
exist), so callers regenerate it by rerunning FedAvg under the same seed.

Track-agnostic by construction: everything here consumes only
(w_r, dm, loss_fn, pkeys) and works for both backends, since backends.cnn and
backends.llm expose the same loss_fn(params, buffers) contract.  `loss_chunks`
is the LLM memory-chunking seam; pass None on the CNN track (mirrors
flirds_estimator.py L110-113).

Imports flirds only; modifies nothing.
"""
from __future__ import annotations

import itertools
import time
from math import factorial

import numpy as np
import torch
from torch.func import grad, jvp

from flirds.core.flirds_estimator import _chunked
from flirds.oracle.in_run_sv import _perturbed_params, _round_weight

# Coalition enumeration is 2^K; refuse to start a run that cannot finish.
MAX_PLAYERS = 16


def exact_shapley_dict(players, u):
    """Exact Shapley of the |players|-player game u (dict tuple(sorted)->float, u[()]=0).

    Same combinatorial weight r!(K-r-1)!/K! as in_run_sv.in_run_shapley_perround's
    within-round kernel, so per-round values are directly comparable to the oracle."""
    K = len(players)
    phi = {}
    for k in players:
        others = [c for c in players if c != k]
        acc = 0.0
        for r in range(K):
            w = factorial(r) * factorial(K - r - 1) / factorial(K)
            for S in itertools.combinations(others, r):
                acc += w * (u[tuple(sorted(S + (k,)))] - u[S])
        phi[k] = acc
    return phi


def _dot(x, y, pkeys):
    """sum_n <x[n], y[n]> — the estimator's float(sum(...)) convention."""
    return float(sum((x[n] * y[n]).sum() for n in pkeys))


def _stats(arr):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return {}
    return {"max": float(a.max()), "median": float(np.median(a)), "mean": float(a.mean())}


def _loglog_slope(norms, resids):
    """LS slope of log(resid) vs log||D_S|| — the remainder's order in the displacement."""
    x = np.asarray(norms, dtype=float)
    y = np.asarray(resids, dtype=float)
    m = (x > 0) & (y > 0)
    if m.sum() < 3:
        return None
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


def _bind_buffers(loss_fn, buffers):
    """Bind the buffers into a one-arg closure so torch.func can differentiate it."""
    def vloss(pp):
        return loss_fn(pp, buffers)
    return vloss


def scale_sweep(params, dW, g, hd, base, loss_fn, buffers, pkeys, norm_dW,
                targets=(1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)):
    """Order test along the realized direction: D -> t*D at fixed ABSOLUTE ||t*D||.

    C.5's log-log slope is fit over the coalition spread alone, which spans barely
    a factor of ~K in ||D_S|| — too narrow to separate order 2 from order 3.  This
    sweep spans a decided range (default 32x) and, because the targets are absolute
    rather than relative to ||D||, it makes runs with different displacements (e.g.
    a ReLU model vs a smooth control) directly comparable at equal ||D||.
    """
    gd = _dot(g, dW, pkeys)
    dHd = _dot(dW, hd, pkeys)
    out = []
    with torch.no_grad():
        for target in targets:
            t = target / norm_dW if norm_dW > 0 else 0.0
            pert = {n: params[n] + t * dW[n] for n in pkeys}
            exact = float(loss_fn(pert, buffers)) - base
            del pert
            t1 = t * gd
            t2 = t1 + 0.5 * t * t * dHd
            out.append(dict(t=t, norm=t * norm_dW, exact=exact,
                            resid1=abs(exact - t1), resid2=abs(exact - t2)))
    return out


def measure_round(r, w_r, dm, loss_fn, pkeys, device, loss_chunks=None,
                  renorm=False, sweep=True):
    """Measure round r.  Returns (coalition rows, per-round phi dicts, round summary)."""
    players = sorted(dm.keys())
    K = len(players)
    if K > MAX_PLAYERS:
        raise ValueError(f"round {r}: 2^{K} coalition enumeration is not affordable")
    pr = _round_weight(dm)                    # p_k^r, identical to the (b) oracle's
    tot_n = sum(n for _, n in dm.values())
    # Same fp32 cast as the estimator (flirds_estimator.py L101-103).
    params = {n: w_r[n].detach().float().to(device) for n in pkeys}
    buffers = {n: w_r[n].detach().to(device) for n in w_r if n not in pkeys}
    a = {k: {n: pr[k] * dm[k][0][n].float().to(device) for n in pkeys} for k in players}

    with torch.no_grad():
        base = float(loss_fn(params, buffers))

    # ---- g_r once + per-client HVP h_k = H_r a_k (jvp-of-grad returns both) ----
    vloss = _bind_buffers(loss_fn, buffers)
    t0 = time.perf_counter()
    g, h = None, {}
    for k in players:
        if loss_chunks is not None:
            gk, hk = _chunked(loss_chunks, buffers, params, a[k], pkeys)
        else:
            gk, hk = jvp(grad(vloss), (params,), (a[k],))
        if g is None:
            g = gk
        h[k] = hk
    t_hvp = time.perf_counter() - t0

    # ---- caches: b_k = <g, a_k>, Q_ij = <a_i, H a_j>, Gram_ij = <a_i, a_j> ----
    b = {k: _dot(g, a[k], pkeys) for k in players}
    Q = np.zeros((K, K))
    Gram = np.zeros((K, K))
    for i, ki in enumerate(players):
        for j, kj in enumerate(players):
            Q[i, j] = _dot(a[ki], h[kj], pkeys)
            Gram[i, j] = _dot(a[ki], a[kj], pkeys)
    q_asym = float(np.abs(Q - Q.T).max())     # fp residual of true-Hessian symmetry

    # ---- every subset: true u (forward) + Taylor reconstruction from the caches ----
    rows = []
    u_true = {(): 0.0}
    u_t2d = {(): 0.0}
    u_ren = {(): 0.0} if renorm else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for size in range(1, K + 1):
            for S in itertools.combinations(players, size):
                idx = [players.index(k) for k in S]
                pert = _perturbed_params(params, dm, S, pr, pkeys)
                u = float(loss_fn(pert, buffers)) - base
                del pert
                u1 = sum(b[k] for k in S)
                u2 = u1 + 0.5 * float(Q[np.ix_(idx, idx)].sum())
                nrm = float(np.sqrt(max(Gram[np.ix_(idx, idx)].sum(), 0.0)))
                n_S = sum(dm[k][1] for k in S)
                row = dict(r=r, S="+".join(map(str, S)), size=size, n_S=n_S,
                           u_true=u, u_t1=u1, u_t2=u2,
                           resid1=abs(u - u1), resid2=abs(u - u2), norm_dS=nrm)
                u_true[S] = u
                u_t2d[S] = u2
                if renorm:
                    c_S = tot_n / n_S
                    dS = {n: sum(a[k][n] for k in S) for n in pkeys}
                    pert = {n: params[n] + c_S * dS[n] for n in pkeys}
                    row["c_S"] = c_S
                    row["u_renorm"] = float(loss_fn(pert, buffers)) - base
                    u_ren[S] = row["u_renorm"]
                    del pert, dS
                rows.append(row)
    t_fwd = time.perf_counter() - t0

    # ---- per-round phi: exact(u) / exact(u2) / closed form / exact(ut) ----
    phi_exact = exact_shapley_dict(players, u_true)
    phi_t2 = exact_shapley_dict(players, u_t2d)
    phi_closed = {k: b[k] + 0.5 * float(Q[i, :].sum()) for i, k in enumerate(players)}
    phi_t1 = dict(b)                           # additive game: Shapley = own term
    phi_ren = exact_shapley_dict(players, u_ren) if renorm else None

    norm_dW = float(np.sqrt(max(Gram.sum(), 0.0)))
    ulp = float(np.spacing(np.float32(abs(base))))
    sw = None
    if sweep:
        dW = {n: sum(a[k][n] for k in players) for n in pkeys}
        hd = {n: sum(h[k][n] for k in players) for n in pkeys}
        sw = scale_sweep(params, dW, g, hd, base, loss_fn, buffers, pkeys, norm_dW)
        del dW, hd

    summ = dict(
        r=r, n_players=K, base_loss=base,
        t_hvp_s=round(t_hvp, 2), t_forward_s=round(t_fwd, 2),
        norm_dW=norm_dW,
        # fp32 forward-eval noise floor: u is a difference of two fp32 losses, so a
        # remainder at or below ulp(base) is not measurable.  C.5's second-order
        # residual sits only 2.1x above this, which is why it bounds the MAGNITUDE
        # of the remainder rather than confirming its ORDER.
        ulp_base=ulp,
        resid2_over_ulp=(_stats([w["resid2"] for w in rows])["median"] / ulp) if rows else None,
        Q_asym_max=q_asym,
        resid1=_stats([w["resid1"] for w in rows]),
        resid2=_stats([w["resid2"] for w in rows]),
        frac_t2_le_t1=float(np.mean([w["resid2"] <= w["resid1"] for w in rows])),
        ratio_r1_over_d2=_stats([w["resid1"] / w["norm_dS"] ** 2
                                 for w in rows if w["norm_dS"] > 0]),
        ratio_r2_over_d3=_stats([w["resid2"] / w["norm_dS"] ** 3
                                 for w in rows if w["norm_dS"] > 0]),
        loglog_slope_r1=_loglog_slope([w["norm_dS"] for w in rows],
                                      [w["resid1"] for w in rows]),
        loglog_slope_r2=_loglog_slope([w["norm_dS"] for w in rows],
                                      [w["resid2"] for w in rows]),
        u_grand=u_true[tuple(players)],
        max_abs_phi_t2_vs_closed=max(abs(phi_t2[k] - phi_closed[k]) for k in players),
    )
    if sw is not None:
        summ["sweep"] = sw
        summ["sweep_slope_r1"] = _loglog_slope([s["norm"] for s in sw],
                                               [s["resid1"] for s in sw])
        summ["sweep_slope_r2"] = _loglog_slope([s["norm"] for s in sw],
                                               [s["resid2"] for s in sw])
        # Slope restricted to points clear of the fp32 floor -- the only ones that
        # carry order information.
        ok = [s for s in sw if s["resid2"] > 10 * ulp]
        summ["sweep_slope_r2_above_floor"] = _loglog_slope([s["norm"] for s in ok],
                                                           [s["resid2"] for s in ok])
        summ["sweep_n_above_floor"] = len(ok)
    return rows, dict(exact=phi_exact, t1=phi_t1, t2=phi_t2, closed=phi_closed,
                      renorm=phi_ren), summ


def pool(round_summaries, all_rows):
    """Pool the per-round measurements into the numbers C.5's table reports."""
    ulp = float(np.mean([s["ulp_base"] for s in round_summaries]))
    r1 = [w["resid1"] for w in all_rows]
    r2 = [w["resid2"] for w in all_rows]
    nrm = [w["norm_dS"] for w in all_rows]
    s2 = _stats(r2)
    sw2 = [s["sweep_slope_r2_above_floor"] for s in round_summaries
           if s.get("sweep_slope_r2_above_floor") is not None]
    sw1 = [s["sweep_slope_r1"] for s in round_summaries if s.get("sweep_slope_r1") is not None]
    return dict(
        resid1=_stats(r1), resid2=s2, ulp=ulp,
        resid2_median_over_ulp=s2["median"] / ulp if s2 else None,
        frac_t2_le_t1=float(np.mean(np.asarray(r2) <= np.asarray(r1))),
        loglog_slope_r1=_loglog_slope(nrm, r1),
        loglog_slope_r2=_loglog_slope(nrm, r2),
        sweep_slope_r1=float(np.mean(sw1)) if sw1 else None,
        sweep_slope_r2_above_floor=float(np.mean(sw2)) if sw2 else None,
        mean_abs_u_grand=float(np.mean([abs(s["u_grand"]) for s in round_summaries])),
        max_phi_t2_vs_closed=max(s["max_abs_phi_t2_vs_closed"] for s in round_summaries),
    )
