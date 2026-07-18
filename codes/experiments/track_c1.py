"""Track C1 -- CNN fidelity & cost vs dual oracle on the GTG stage (plan §3.11).

GTG-Shapley's 5 MNIST scenarios (2109.02053 §5.1.1) replayed on our stack
(MNIST+LeNet5 / CIFAR-10+FedSVCNN, N=10 full participation), valued by the
9-method set + Ripple on ONE frozen FedAvg trajectory per (scenario, seed),
against BOTH ground truths:
  (a) exact retrain SV  -- 2^N FedAvg retrains, -val-loss utility, fp32
      (`oracle.exact_sv.subset_utility_valloss`; the expensive sweep, gated by
      C1_ORACLE_A so it shards separately),
  (b) exact in-run SV   -- frozen-trajectory (`oracle.in_run_sv.in_run_shapley`).
All methods play the SAME val-loss game (2026-06-12 decision; the task-6
same-game lesson) via their loss_fn paths -- calls and sign conventions mirror
`phase2_matrix.compute_methods` (good -> low phi).  Ripple runs its OWN
trajectory (paper requirement) at paper spec k=20 / m=50 / depth 20 with the
eigsh guard, and is negated to good -> low.

Setting (2026-06-12 decisions): full train split across N=10 (MNIST 6,000/
client, CIFAR 5,000); val = 2,000 random of the official test, final acc on the
disjoint 8,000 (ShapleyFL App.B.1 / FedIF §V-A convention; split seed fixed at
0 across run seeds); ladder rates 0/0/5/5/10/10/15/15/20/20% (GTG verbatim;
generalized as pair p -> 5p%); label_flip = uniform over the K-1 wrong classes;
feature_noise = pixel-space sigma, no clamp, sigma ladder == flip ladder;
ladders sit on the same-size IID base (GTG scenario-4 title/text contradiction
resolved to the title); quantity-skew disjoint-normalized 10:10:15:15:20:20:
25:25:30:30 digit-balanced.

Metrics per method: wall-clock + Spearman/Kendall vs (a)/(b) + the GTG distance
trio (cosine/Euclid/max-diff -- meaningful only between same-unit SV estimates;
recorded for all, interpret accordingly) + detection AUROC and phi-vs-rate
Spearman on the ladder scenarios.

Run (from codes/):
  C1_DATASET=mnist C1_SCENARIO=label_flip C1_SEED=0 C1_MODE=full \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/track_c1.py
Shard one process per (dataset, scenario, seed); C1_ORACLE_A=0 skips the 2^N
retrain sweep, C1_ORACLE_A_ONLY=1 runs ONLY it (merge post-hoc off run dirs).
C1_REMOVAL=1 adds the Exp A3 worst/best-first removal-retrain curves (val loss
+ test acc off each retrained global; default 0 = bit-identical to current
behavior), C1_REMOVAL_METHODS=a,b,c restricts them (empty = all minus Ripple).
"""
from __future__ import annotations

import os
import time
from functools import partial

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader, Subset, TensorDataset

from flirds.baselines.banzhaf import in_run_banzhaf
from flirds.baselines.comfedsv import comfedsv_from_logs
from flirds.baselines.fedif import fedif_from_logs
from flirds.baselines.fedsv import fedsv_from_logs
from flirds.baselines.gtg import gtg_from_logs
from flirds.baselines.ripple import ripple_shapley
from flirds.baselines.shapleyfl import shapleyfl_from_logs
from flirds.backends.cnn import make_cnn_loss
from flirds.core.flirds_estimator import flirds_values
from flirds.data.cnn import _STATS, get_dataset, get_labels
from flirds.data.corruptors import CNN_CORRUPTORS
from flirds.eval.metrics import (cosine_distance, detection_auroc,
                                 euclidean_distance, max_difference, pearson)
from flirds.fl.partition import (iid_partition, label_skew_partition,
                                 quantity_skew_partition)
from flirds.fl.server import evaluate, fedavg
from flirds.models.cnn import FedSVCNN, LeNet5
from flirds.oracle.exact_sv import exact_shapley, subset_utility_valloss
from flirds.oracle.in_run_sv import in_run_loo, in_run_shapley, in_run_singletons
from flirds.repro import seed_everything
from flirds.run_logger import RunLogger
from flirds.timing import PhaseTimer

# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #
DATASET = os.environ.get("C1_DATASET", "mnist")          # mnist | cifar10
SCENARIO = os.environ.get("C1_SCENARIO", "iid")          # iid | label_skew | quantity_skew | label_flip | feature_noise
SEED = int(os.environ.get("C1_SEED", "0"))
MODE = os.environ.get("C1_MODE", "smoke")                # smoke | full
ORACLE_A = os.environ.get("C1_ORACLE_A", "1") == "1"
ORACLE_A_ONLY = os.environ.get("C1_ORACLE_A_ONLY", "0") == "1"
PERSIST = os.environ.get("C1_PERSIST", "1") == "1"
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
RUN_ROOT = os.environ.get("C1_RUN_ROOT", os.path.join(_REPO, "runs", "track_c", "c1"))

CFG = {
    "full":  dict(n_clients=10, n_per=None, rounds=10, epochs=5, lr=0.01, batch=64,
                  n_val=2000, n_test=8000, ripple=dict(k=20, m=50, R=20)),
    "smoke": dict(n_clients=6, n_per=120, rounds=2, epochs=1, lr=0.05, batch=60,
                  n_val=256, n_test=512, ripple=dict(k=2, m=6, R=3)),
}[MODE]

WIDTH = float(os.environ.get("C1_WIDTH", "1"))           # signal-size probe lever: capacity (width mult)
KFRAC = float(os.environ.get("C1_KFRAC", "1"))           # signal-size probe lever: participation frac
RIPPLE = os.environ.get("C1_RIPPLE", "1") == "1"         # 0 = skip Ripple (probe cells; cost)
REMOVAL = os.environ.get("C1_REMOVAL", "0") == "1"       # Exp A3: removal-retrain curves (extra retrains)
REMOVAL_METHODS = [m for m in os.environ.get("C1_REMOVAL_METHODS", "").split(",") if m]  # [] = all minus Ripple
V3 = os.environ.get("C1_V3", "0") == "1"                 # Track G V3: keep-only-cum>0 one-shot retrain
V3_METHODS = [m for m in os.environ.get("C1_V3_METHODS", "Flirds,(b)oracle").split(",") if m]
V3_ZC = float(os.environ.get("C1_V3_ZC", "1.5"))         # z-variant threshold (shared gate default)
MODEL_FN = partial({"mnist": LeNet5, "cifar10": FedSVCNN}[DATASET], width=WIDTH)
LADDER_STEP = 0.05                                       # pair p -> 5p% (GTG ladder)


def _pair_ladder(n):
    """GTG graded ladder generalized: pair p gets p*5% (N=10 -> 0/0/5/5/.../20/20%)."""
    return [LADDER_STEP * (i // 2) for i in range(n)]


def _quantity_ratios(n):
    """GTG quantity-skew ratios generalized: pair p gets 10+5p (N=10 -> 10..30)."""
    return [10 + 5 * (i // 2) for i in range(n)]


def _timed(fn, device):
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t


# --------------------------------------------------------------------------- #
# data build                                                                  #
# --------------------------------------------------------------------------- #
def build(dataset, scenario, n, n_per, batch, n_val, n_test, seed):
    """Partition + corrupt + loaders.  Returns (loaders, rates, vx, vy, val_loader,
    test_loader).  `rates` = per-client corruption ladder (zeros off-ladder)."""
    train = get_dataset(dataset)
    test = get_dataset(dataset, train=False)
    labels = get_labels(train)
    if scenario == "label_skew":
        idx = label_skew_partition(labels, n, seed=seed)
    elif scenario == "quantity_skew":
        idx = quantity_skew_partition(labels, n, _quantity_ratios(n), seed=seed)
    else:                                  # iid + both ladders (same-size IID base)
        idx = iid_partition(labels, n, seed=seed)
    if n_per is not None:                  # smoke truncation, size-PROPORTIONAL so
        scale = n_per / max(len(i) for i in idx)        # quantity skew survives
        idx = [i[:max(1, round(len(i) * scale))] for i in idx]

    rates = _pair_ladder(n) if scenario in ("label_flip", "feature_noise") else [0.0] * n
    loaders = []
    for c, ci in enumerate(idx):
        xs = torch.stack([train[i][0] for i in ci])
        ys = torch.tensor([train[i][1] for i in ci])
        if scenario == "label_flip":
            xs, ys = CNN_CORRUPTORS["label_flip"](xs, ys, c, rate=rates[c])
        elif scenario == "feature_noise":
            xs, ys = CNN_CORRUPTORS["feature_noise"](xs, ys, c, std=rates[c],
                                                     data_std=_STATS[dataset][1])
        loaders.append(DataLoader(TensorDataset(xs, ys), batch_size=batch, shuffle=True))

    perm = np.random.default_rng(0).permutation(len(test))   # split seed FIXED at 0
    val_idx, test_idx = perm[:n_val], perm[n_val:n_val + n_test]
    vx = torch.stack([test[i][0] for i in val_idx])
    vy = torch.tensor([test[i][1] for i in val_idx])
    val_loader = DataLoader(TensorDataset(vx, vy), batch_size=512)
    test_loader = DataLoader(Subset(test, test_idx.tolist()), batch_size=512)
    return loaders, rates, vx, vy, val_loader, test_loader


# --------------------------------------------------------------------------- #
# removal / selection curves (Exp A3; gated by C1_REMOVAL)                    #
# --------------------------------------------------------------------------- #
def removal_retrain_curves(methods, retrain_eval, n, device, sel=None):
    """Exp A3: worst-first / best-first removal curves by ACTUAL clean retraining --
    the CNN leg of the game-independent downstream ruler (review C-1/C-4 defence;
    LLM leg = phase2_matrix.removal_retrain_curves, whose pattern this ports).

    Design decisions (A3 prompt §3):
      * Option 1 (independent retrain path) over option 2 (deriving the curve from
        the (a) 2^N u-cache, track_d A1 style): the u-cache holds val-loss ONLY (no
        accuracy) and exists only on C1_ORACLE_A=1 cells, while A3's point is the
        ACCURACY axis on cheap cells -- so one uniform retrain path, same as Exp A2.
      * Each retrained global is scored on BOTH the game metric (val loss, the same
        closure the methods/oracles play) and the batch metric (test accuracy on
        the disjoint 8,000) -- one retrain, two rulers, extra cost ~0 (the CNN's
        unique axis: LLM silo5 is generative so Exp A2 had val_loss only).
      * Ripple is dropped from the default `sel`: its phi comes from its OWN
        sampled trajectories, not the shared frozen trajectory the other methods
        rank (C1 convention), so its ranking is not commensurate with theirs here;
        name it explicitly in C1_REMOVAL_METHODS to force it.

    For every selected method, rank clients by phi (good -> LOW, the C1 sign
    convention: high phi = most suspicious) and retrain clean FedAvg on each kept
    subset.  retrain_eval(kept_sorted_tuple) -> (val_loss, test_acc); results are
    cached by frozenset so each distinct kept set is retrained ONCE and shared
    across methods AND directions (rank-agreeing methods collapse to one chain --
    <= 2^n retrains; mean_retrain_s includes the ~free scoring forward passes).
      worst_first -- drop the highest-phi (most-suspicious) client each step
      best_first  -- drop the lowest-phi  (most-valuable)   client each step
    Returns (curves_loss, curves_acc, n_retrains, mean_retrain_s); curves_* =
    {method: {"worst_first": [[k_dropped, v], ...], "best_first": [...]}} with
    k = 0..n-1 (kept sizes n..1; the empty set is skipped) -- the Exp A2 LLM
    `removal_curve` schema, so the same aggregation tooling reads both stages."""
    cache, times = {}, []

    def util(kept):
        key = frozenset(kept)
        if key not in cache:
            cache[key], dt = _timed(lambda: retrain_eval(tuple(sorted(kept))), device)
            times.append(dt)
        return cache[key]

    def curve(order):
        pts_l, pts_a, kept = [], [], list(range(n))
        for k in range(n):                            # kept sizes n, n-1, ..., 1 (skip empty)
            vl, acc = util(kept)
            pts_l.append([k, vl])
            pts_a.append([k, acc])
            kept.remove(order[k])
        return pts_l, pts_a

    if sel is None:
        sel = [nm for nm, _, _ in methods if nm != "Ripple"]
    curves_l, curves_a = {}, {}
    for name, vec, _rt in methods:
        if name not in sel:
            continue
        v = np.asarray(vec, dtype=float)
        wl, wa = curve(list(np.argsort(-v)))          # high phi (worst) dropped first
        bl, ba = curve(list(np.argsort(v)))           # low phi (best) dropped first
        curves_l[name] = {"worst_first": wl, "best_first": bl}
        curves_a[name] = {"worst_first": wa, "best_first": ba}
    return curves_l, curves_a, len(cache), (sum(times) / len(times) if times else 0.0)


# --------------------------------------------------------------------------- #
# per-seed run                                                                #
# --------------------------------------------------------------------------- #
def run_seed(seed, device="cuda"):
    n, R, E, lr = CFG["n_clients"], CFG["rounds"], CFG["epochs"], CFG["lr"]
    pt = PhaseTimer(device, n_gpus=int(os.environ.get("N_GPUS", "1")))   # §15.1 timing.json substrate
    seed_everything(seed, cudnn_deterministic=True)
    loaders, rates, vx, vy, val_loader, test_loader = build(
        DATASET, SCENARIO, n, CFG["n_per"], CFG["batch"], CFG["n_val"], CFG["n_test"], seed)
    print(f"[build] {DATASET}/{SCENARIO} seed={seed} sizes={[len(l.dataset) for l in loaders]}"
          f" rates={rates}", flush=True)

    # ---- (a) exact retrain SV (gated; the expensive shard) ----
    phi_a, t_a = None, None
    if ORACLE_A:
        cache = {}

        def util(S):
            if S not in cache:
                cache[S] = subset_utility_valloss(MODEL_FN, loaders, val_loader, S,
                                                  R, E, lr, device=device, seed=seed)
            return cache[S]

        (phi_a), t_a = _timed(lambda: exact_shapley(n, util), device)
        pt.record("oracle-a-retrain", t_a)                 # §15.1: (a) is 2^N retrains, own phase
        eff = abs(phi_a.sum() - (util(tuple(range(n))) - util(())))
        print(f"[(a)oracle] 2^{n}={2 ** n} retrains in {t_a:.0f}s "
              f"({t_a / 2 ** n:.2f}s/retrain)  efficiency-gap={eff:.2e}", flush=True)
        assert eff < 1e-6, "(a) efficiency axiom violated"
    if ORACLE_A_ONLY:
        return dict(phi_a=phi_a.tolist(), t_a=t_a, rates=rates, _timing=pt.to_timing()), []

    # ---- shared frozen trajectory + val-loss closure ----
    logs = []
    (final_state, history), t_traj = _timed(lambda: fedavg(
        MODEL_FN, loaders, test_loader, R, E, lr, sample_frac=KFRAC, device=device,
        seed=seed, on_round=lambda r, gb, dm: logs.append((gb, dm))), device)
    final_acc = history[-1][1]
    pt.record("client-training", t_traj)               # §15.1: reuse the existing measurement
    print(f"[traj] {R}r x {E}e in {t_traj:.0f}s  final test-acc={final_acc:.4f}", flush=True)
    loss_fn, pkeys = make_cnn_loss(MODEL_FN, vx, vy, device)

    # ---- methods on the frozen logs (mirrors phase2_matrix.compute_methods) ----
    methods = []                                       # (name, phi good->low, runtime)
    with pt.phase("valuation"):                        # §15.1: from-logs methods (peak = HVP)
        (phi_b, _), t = _timed(lambda: in_run_shapley(logs, n, loss_fn, pkeys, device), device)
        methods.append(("(b)oracle", np.asarray(phi_b), t))
        (phi, _), t = _timed(lambda: flirds_values(logs, loss_fn, pkeys, device,
                                                   second_order=True, n_clients=n), device)
        methods.append(("Flirds", np.asarray(phi), t))
        (phi, _), t = _timed(lambda: flirds_values(logs, loss_fn, pkeys, device,
                                                   second_order=False, n_clients=n), device)
        methods.append(("Flirds1st", np.asarray(phi), t))
        phi, t = _timed(lambda: gtg_from_logs(logs, None, n, None, device, seed=seed,
                        loss_fn=loss_fn, pkeys=pkeys, round_trunc=0.0, eps=0.0), device)
        methods.append(("GTG", np.asarray(phi), t))
        phi, t = _timed(lambda: fedsv_from_logs(logs, None, n, None, device, seed=seed,
                        loss_fn=loss_fn, pkeys=pkeys, trunc_eps=0.0), device)
        methods.append(("FedSV", np.asarray(phi), t))
        # partial=False: full participation -> the utility matrix is fully observed, so
        # the paper's low-rank completion has nothing to fill (and its ALS collapses
        # tiny smoke-scale utilities to ~0).
        phi, t = _timed(lambda: comfedsv_from_logs(logs, None, n, None, device, seed=seed,
                        loss_fn=loss_fn, pkeys=pkeys, partial=KFRAC < 1), device)
        methods.append(("ComFedSV", -np.asarray(phi, dtype=float), t))   # loss-decrease util -> negate
        (phi, _), t = _timed(lambda: in_run_banzhaf(logs, n, loss_fn, pkeys, device), device)
        methods.append(("Banzhaf", np.asarray(phi), t))
        phi, t = _timed(lambda: shapleyfl_from_logs(logs, None, n, None, device, beta=0.3,
                        loss_fn=loss_fn, pkeys=pkeys), device)
        methods.append(("ShapleyFL", -np.asarray(phi, dtype=float), t))  # good->high -> negate
        phi, t = _timed(lambda: fedif_from_logs(logs, n, loss_fn, pkeys, device), device)
        methods.append(("FedIF", -np.asarray(phi, dtype=float), t))      # influence good->HIGH -> negate
        phi, t = _timed(lambda: in_run_singletons(logs, n, loss_fn, pkeys, device), device)
        methods.append(("loss-heur", phi, t))
        phi, t = _timed(lambda: in_run_loo(logs, n, loss_fn, pkeys, device), device)  # Fed-LOO: marginal U(N)-U(N\{i}) anchor (!= singleton loss-heur)
        methods.append(("Fed-LOO", phi, t))
    if RIPPLE:
        rp = CFG["ripple"]
        with pt.phase("ripple-own-trajectory"):        # §15.1/C1: Ripple retrains -> NOT from-logs valuation
            phi, t = _timed(lambda: ripple_shapley(MODEL_FN, loaders, R, E, lr,
                            vx.to(device), vy.to(device), device, seed=seed, **rp), device)
        methods.append(("Ripple", -np.asarray(phi, dtype=float), t))  # own trajectory; good->high -> negate

    # ---- metrics ----
    gt = {"b": methods[0][1]}                          # good->low
    if phi_a is not None:
        gt["a"] = -phi_a                               # (a) is -val-loss good->high -> flip
    ladder = SCENARIO in ("label_flip", "feature_noise")
    y = [1 if r > 0 else 0 for r in rates]
    res = {}
    for name, vec, rt in methods:
        m = {"runtime": rt, "phi": vec.tolist()}
        for g, gvec in gt.items():
            if name == f"({g})oracle":
                continue
            m[f"spearman_{g}"] = float(spearmanr(vec, gvec).correlation)
            m[f"kendall_{g}"] = float(kendalltau(vec, gvec).correlation)
            m[f"pearson_{g}"] = pearson(vec, gvec)          # value-level (affine-invariant) fidelity
            m[f"cos_{g}"] = cosine_distance(vec, gvec)
            m[f"euc_{g}"] = euclidean_distance(vec, gvec)
            m[f"maxdiff_{g}"] = max_difference(vec, gvec)
        if ladder:
            m["auroc"] = detection_auroc(vec, y)       # good->low: corrupt scores high
            m["spearman_vs_rate"] = float(spearmanr(vec, rates).correlation)
        res[name] = m

    hdr = f"  {'method':10s} {'time':>7s} {'rho(b)':>7s} {'tau(b)':>7s} {'r_p(b)':>7s}"
    hdr += f" {'rho(a)':>7s} {'r_p(a)':>7s}" if "a" in gt else ""
    hdr += f" {'AUROC':>6s}" if ladder else ""
    print(hdr, flush=True)
    for name, vec, rt in methods:
        m = res[name]
        line = f"  {name:10s} {rt:6.1f}s {m.get('spearman_b', float('nan')):7.3f}"
        line += f" {m.get('kendall_b', float('nan')):7.3f} {m.get('pearson_b', float('nan')):7.3f}"
        line += (f" {m.get('spearman_a', float('nan')):7.3f} {m.get('pearson_a', float('nan')):7.3f}"
                 if "a" in gt else "")
        line += f" {m.get('auroc', float('nan')):6.3f}" if ladder else ""
        print(line, flush=True)

    # ---- Exp A3: removal-retrain curves (gated; extra retrains) ----
    removal = {}
    if REMOVAL or V3:                                  # shared retrain_eval (V3 reuses A3's ruler)
        eval_model = MODEL_FN().to(device)             # one eval model reused across retrains

        def retrain_eval(kept):
            """Clean FedAvg on `kept` -> (game val-loss, test acc) off ONE retrained global."""
            final, _ = fedavg(MODEL_FN, [loaders[c] for c in kept], None, R, E, lr,
                              sample_frac=1.0, device=device, seed=seed,
                              eval_every=R + 1)        # no per-round eval (the (a)-sweep pattern)
            with torch.no_grad():
                vl = float(loss_fn({k: final[k] for k in pkeys}, {}))
            return vl, evaluate(eval_model, final, test_loader, device)

    if REMOVAL:
        rc, rca, nrt, mrt = removal_retrain_curves(methods, retrain_eval, n, device,
                                                   sel=REMOVAL_METHODS or None)
        print(f"[removal] {nrt} distinct retrains, {mrt:.1f}s/retrain", flush=True)
        removal = dict(removal_curve=rc, removal_curve_acc=rca,
                       removal_orient="val_loss (lower=better); phi good->low",
                       removal_acc_orient="test_acc (higher=better); phi good->low",
                       removal_retrain_s=mrt)

    # ---- Track G V3 (gated): keep-only-positive-cumulative one-shot retrain ----
    # kept = {i: contribution(-phi) > 0} per method (+ z-variant, + size-matched
    # random control); data-level corruption stays in the kept clients' data
    # (deployment semantics -- exclusion is the only intervention).  Reference =
    # this run's own full-set trajectory (same seed/config -> a full-set retrain
    # is identical, so no extra retrain for it).
    if V3:
        with pt.phase("v3-retrain"):
            by_name = {nm: vec for nm, vec, _ in methods}
            kept_sets, cache_v3, v3 = {}, {}, {}
            for nm in [m for m in V3_METHODS if m in by_name]:
                cum = -np.asarray(by_name[nm], dtype=float)        # contribution orientation
                kept_sets[f"sign_{nm}"] = [i for i in range(n) if cum[i] > 0]
                z = ((cum - cum.mean()) / cum.std() if cum.std() > 0
                     else np.zeros_like(cum))
                kept_sets[f"z_{nm}"] = [i for i in range(n) if z[i] >= -V3_ZC]
            size_ref = len(next(iter(kept_sets.values()), list(range(n))))
            rng_v3 = np.random.default_rng(3000 + seed)
            kept_sets["random"] = sorted(int(x) for x in rng_v3.choice(
                n, size=max(1, size_ref), replace=False))
            for vname, kept in kept_sets.items():
                if not kept:                           # everyone gated out -> reported as-is
                    v3[vname] = dict(kept=[], val_loss=None, test_acc=None)
                    continue
                key = tuple(sorted(kept))
                if key not in cache_v3:
                    cache_v3[key] = retrain_eval(key)  # cached across variants/methods
                vl, ta = cache_v3[key]
                v3[vname] = dict(kept=list(key), val_loss=vl, test_acc=ta)
                print(f"[v3] {vname}: kept={list(key)} val_loss={vl:.4f} acc={ta:.4f}",
                      flush=True)
        with torch.no_grad():
            full_vl = float(loss_fn({k: final_state[k] for k in pkeys}, {}))
        removal = dict(removal, v3=v3,
                       v3_ref=dict(full_val_loss=full_vl, full_test_acc=final_acc),
                       v3_orient="kept = contribution(-phi) > 0; corrupt data stays in kept")

    metrics = dict(dataset=DATASET, scenario=SCENARIO, seed=seed, mode=MODE,
                   final_acc=final_acc, acc_curve=history, traj_time=t_traj,
                   rates=rates, methods=res, _timing=pt.to_timing(), **removal)
    if phi_a is not None:
        metrics["oracle_a"] = dict(phi=phi_a.tolist(), time=t_a, n_retrains=2 ** n)
    phi_rows = [dict(client=c, rate=rates[c],
                     **{f"phi_{name}": float(vec[c]) for name, vec, _ in methods},
                     **({"phi_a": float(phi_a[c])} if phi_a is not None else {}))
                for c in range(n)]
    return metrics, phi_rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics, phi_rows = run_seed(SEED, device)
    timing = metrics.pop("_timing", None)              # §15.1 -> timing.json (not metrics.json)
    if PERSIST:
        try:
            name = os.environ.get("C1_RUN_NAME") or (                    # probe cells override (width/kfrac in name)
                f"{DATASET}_{SCENARIO.replace('_', '-')}"                # canonical: hyphen within token
                + ("_aonly" if ORACLE_A_ONLY else "") + f"_seed{SEED}")  # seed always trailing
            rl = RunLogger(RUN_ROOT, name, dict(cfg=CFG, dataset=DATASET, scenario=SCENARIO,
                                                seed=SEED, mode=MODE, oracle_a=ORACLE_A,
                                                width=WIDTH, kfrac=KFRAC, removal=REMOVAL),
                           repo_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if phi_rows:
                try:
                    rl.save_phi(phi_rows)
                except Exception:
                    import pandas as pd
                    pd.DataFrame(phi_rows).to_csv(rl._p("phi.csv"), index=False)
            rl.save_metrics(metrics)
            if timing is not None:
                rl.save_timing(timing)                 # §15.1 per-phase wall + GPU-hours + peak
            print(f"[persist] {rl.dir}", flush=True)
        except Exception as e:                         # best-effort: stdout already has it all
            print(f"[persist] FAILED ({e!r}) -- results live in stdout", flush=True)
    print("TRACK-C1 RUN OK", flush=True)


if __name__ == "__main__":
    main()
