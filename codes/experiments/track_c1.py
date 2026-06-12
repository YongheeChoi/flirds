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
"""
from __future__ import annotations

import os
import time

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
                                 euclidean_distance, max_difference)
from flirds.fl.partition import (iid_partition, label_skew_partition,
                                 quantity_skew_partition)
from flirds.fl.server import evaluate, fedavg
from flirds.models.cnn import FedSVCNN, LeNet5
from flirds.oracle.exact_sv import exact_shapley, subset_utility_valloss
from flirds.oracle.in_run_sv import in_run_shapley, in_run_utility
from flirds.repro import seed_everything
from flirds.run_logger import RunLogger

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
RUN_ROOT = os.environ.get("C1_RUN_ROOT", "runs/track_c1")

CFG = {
    "full":  dict(n_clients=10, n_per=None, rounds=10, epochs=5, lr=0.01, batch=64,
                  n_val=2000, n_test=8000, ripple=dict(k=20, m=50, R=20)),
    "smoke": dict(n_clients=6, n_per=120, rounds=2, epochs=1, lr=0.05, batch=60,
                  n_val=256, n_test=512, ripple=dict(k=2, m=6, R=3)),
}[MODE]

MODEL_FN = {"mnist": LeNet5, "cifar10": FedSVCNN}[DATASET]
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
# per-seed run                                                                #
# --------------------------------------------------------------------------- #
def run_seed(seed, device="cuda"):
    n, R, E, lr = CFG["n_clients"], CFG["rounds"], CFG["epochs"], CFG["lr"]
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
        eff = abs(phi_a.sum() - (util(tuple(range(n))) - util(())))
        print(f"[(a)oracle] 2^{n}={2 ** n} retrains in {t_a:.0f}s "
              f"({t_a / 2 ** n:.2f}s/retrain)  efficiency-gap={eff:.2e}", flush=True)
        assert eff < 1e-6, "(a) efficiency axiom violated"
    if ORACLE_A_ONLY:
        return dict(phi_a=phi_a.tolist(), t_a=t_a, rates=rates), []

    # ---- shared frozen trajectory + val-loss closure ----
    logs = []
    (final_state, history), t_traj = _timed(lambda: fedavg(
        MODEL_FN, loaders, test_loader, R, E, lr, sample_frac=1.0, device=device,
        seed=seed, on_round=lambda r, gb, dm: logs.append((gb, dm))), device)
    final_acc = history[-1][1]
    print(f"[traj] {R}r x {E}e in {t_traj:.0f}s  final test-acc={final_acc:.4f}", flush=True)
    loss_fn, pkeys = make_cnn_loss(MODEL_FN, vx, vy, device)

    # ---- methods on the frozen logs (mirrors phase2_matrix.compute_methods) ----
    methods = []                                       # (name, phi good->low, runtime)
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
                    loss_fn=loss_fn, pkeys=pkeys, partial=False), device)
    methods.append(("ComFedSV", -np.asarray(phi, dtype=float), t))   # loss-decrease util -> negate
    (phi, _), t = _timed(lambda: in_run_banzhaf(logs, n, loss_fn, pkeys, device), device)
    methods.append(("Banzhaf", np.asarray(phi), t))
    phi, t = _timed(lambda: shapleyfl_from_logs(logs, None, n, None, device, beta=0.5,
                    loss_fn=loss_fn, pkeys=pkeys), device)
    methods.append(("ShapleyFL", -np.asarray(phi, dtype=float), t))  # good->high -> negate
    phi, t = _timed(lambda: fedif_from_logs(logs, n, loss_fn, pkeys, device), device)
    methods.append(("FedIF", -np.asarray(phi, dtype=float), t))      # influence good->HIGH -> negate
    phi, t = _timed(lambda: np.array([in_run_utility(logs, [k], loss_fn, pkeys, device)
                                      for k in range(n)]), device)
    methods.append(("loss-heur", phi, t))
    rp = CFG["ripple"]
    phi, t = _timed(lambda: ripple_shapley(MODEL_FN, loaders, R, E, lr,
                    vx.to(device), vy.to(device), device, seed=seed, **rp), device)
    methods.append(("Ripple", -np.asarray(phi, dtype=float), t))     # own trajectory; good->high -> negate

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
            m[f"cos_{g}"] = cosine_distance(vec, gvec)
            m[f"euc_{g}"] = euclidean_distance(vec, gvec)
            m[f"maxdiff_{g}"] = max_difference(vec, gvec)
        if ladder:
            m["auroc"] = detection_auroc(vec, y)       # good->low: corrupt scores high
            m["spearman_vs_rate"] = float(spearmanr(vec, rates).correlation)
        res[name] = m

    hdr = f"  {'method':10s} {'time':>7s} {'rho(b)':>7s} {'tau(b)':>7s}"
    hdr += f" {'rho(a)':>7s}" if "a" in gt else ""
    hdr += f" {'AUROC':>6s}" if ladder else ""
    print(hdr, flush=True)
    for name, vec, rt in methods:
        m = res[name]
        line = f"  {name:10s} {rt:6.1f}s {m.get('spearman_b', float('nan')):7.3f}"
        line += f" {m.get('kendall_b', float('nan')):7.3f}"
        line += f" {m.get('spearman_a', float('nan')):7.3f}" if "a" in gt else ""
        line += f" {m.get('auroc', float('nan')):6.3f}" if ladder else ""
        print(line, flush=True)

    metrics = dict(dataset=DATASET, scenario=SCENARIO, seed=seed, mode=MODE,
                   final_acc=final_acc, acc_curve=history, traj_time=t_traj,
                   rates=rates, methods=res)
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
    if PERSIST:
        try:
            name = f"{DATASET}_{SCENARIO}_seed{SEED}" + ("_aonly" if ORACLE_A_ONLY else "")
            rl = RunLogger(RUN_ROOT, name, dict(cfg=CFG, dataset=DATASET, scenario=SCENARIO,
                                                seed=SEED, mode=MODE, oracle_a=ORACLE_A),
                           repo_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if phi_rows:
                try:
                    rl.save_phi(phi_rows)
                except Exception:
                    import pandas as pd
                    pd.DataFrame(phi_rows).to_csv(rl._p("phi.csv"), index=False)
            rl.save_metrics(metrics)
            print(f"[persist] {rl.dir}", flush=True)
        except Exception as e:                         # best-effort: stdout already has it all
            print(f"[persist] FAILED ({e!r}) -- results live in stdout", flush=True)
    print("TRACK-C1 RUN OK", flush=True)


if __name__ == "__main__":
    main()
