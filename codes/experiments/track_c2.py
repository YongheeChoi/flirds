"""Track C2 -- CNN general-performance intervention (cross-device; plan §3.11).

The ADDITIONAL track's MAIN experiment: does valuation-driven intervention improve
FL under data threats, at the prior-art N=100 cross-device scale?  CIFAR-10 /
FMNIST, N=100, C=0.1, T=100-150, E=5, SGD mom=0, batch=64, 5 seeds; partitions
{IID, Dirichlet(alpha=1) label+size skew, McMahan 2-shard}; threats {clean,
label-flip(FedCorr (rho,tau)), free-rider(zero-delta), grad-noise(FedIF sigma)}.

Each METHOD ARM runs its OWN intervened FedAvg trajectory (shared seam in
`fl.server`/`fl.intervene`) and is scored on the same axes:
  - final test acc (+/- seed) / acc-vs-round curve / rounds-to-target
  - detection AUROC: the arm's accumulated client score vs the corrupt mask.
Arms (plan §3.11 decision (4)): vanilla FedAvg (lower bound) | Flirds-MULT
(w propto n*s, the MAIN rule) | Flirds-REPL / Flirds-ADD (the rule ablation, run
only on the SIZE-SKEW partition dir1 where they differ from MULT) | Flirds-SELECT
(S-FedAvg-style softmax selection on Flirds scores) | ShapleyFL (replacement,
beta=0.5) | FedIF (replacement, beta=1-gamma=0.7) | S-FedAvg (its own MC-relevance
selection).  Ripple is C1-fidelity-only (excluded here -- its full value is
non-causal, 06-12 decision).  FedSV bottom-q% dismissal is a SEPARATE q-sweep
({0..0.9}, FedSV Fig.4 acc-vs-removed curve), gated by C2_DISMISSAL=1.

Strength (06-12 decision): the MAIN point runs on every partition; the strength
GRID (label-flip rho in {0.4,0.6,0.8}, grad-noise sigma in {0.05,0.1}) sweeps only
on the representative partition dir1 (set C2_STRENGTH to a value to pick a grid
point; default 'main').  (b)-perround anchor: gated by C2_ORACLE_B on 1-2 configs.

Run (from codes/):
  C2_DATASET=cifar10 C2_PARTITION=dir1 C2_THREAT=label_flip C2_SEED=0 C2_MODE=full \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/track_c2.py
Shard one process per (dataset, partition, threat, strength, seed).
"""
from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset, TensorDataset

from flirds.backends.cnn import make_cnn_loss
from flirds.baselines.sfedavg import SFedAvgSelector
from flirds.data.cnn import _STATS, get_dataset, get_labels
from flirds.data.corruptors import CNN_CORRUPTORS
from flirds.fl.intervene import (OnlineScorer, fedif_round_raw_fn,
                                 flirds_round_raw_fn, make_delta_transform,
                                 make_dismissal_weights_fn, make_scoreonly_weights_fn,
                                 make_softmax_select_fn, make_weights_fn,
                                 shapleyfl_round_raw_fn)
from flirds.fl.partition import (dirichlet_partition, iid_partition,
                                 shard_partition)
from flirds.fl.server import fedavg
from flirds.models.cnn import FedSVCNN, LeNet5
from flirds.repro import seed_everything
from flirds.run_logger import RunLogger

DATASET = os.environ.get("C2_DATASET", "cifar10")        # cifar10 | fmnist
PARTITION = os.environ.get("C2_PARTITION", "iid")        # iid | dir1 | shard
THREAT = os.environ.get("C2_THREAT", "clean")            # clean | label_flip | free_rider | grad_noise
STRENGTH = os.environ.get("C2_STRENGTH", "main")         # 'main' | float (grid point)
SEED = int(os.environ.get("C2_SEED", "0"))
MODE = os.environ.get("C2_MODE", "smoke")
DISMISSAL = os.environ.get("C2_DISMISSAL", "0") == "1"
PERSIST = os.environ.get("C2_PERSIST", "1") == "1"
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
RUN_ROOT = os.environ.get("C2_RUN_ROOT", os.path.join(_REPO, "runs", "track_c", "c2"))

CFG = {
    "full":  dict(n=100, frac=0.1, rounds=120, epochs=5, lr=0.01, batch=64,
                  n_val=2000, n_test=8000, target=0.6),
    "smoke": dict(n=20, frac=0.5, rounds=4, epochs=1, lr=0.05, batch=64,
                  n_val=512, n_test=1024, target=0.3),
}[MODE]

MODEL_FN = {"cifar10": FedSVCNN, "fmnist": LeNet5}[DATASET]
MAL_FRAC = 0.4                                            # noisy/malicious client fraction (main)
TAU = 0.5                                                 # FedCorr per-client rate lower bound
GAMMA_GRADNOISE = 0.1                                     # FedIF main sigma


def _strength(default):
    return default if STRENGTH == "main" else float(STRENGTH)


def build():
    """Partition + threat.  Returns (loaders, corrupt_mask, delta_transform, vx, vy,
    val_loader, test_loader).  Update-level threats return a delta_transform; data-
    level threats fold corruption into the loaders (delta_transform=None)."""
    n, seed = CFG["n"], SEED
    train = get_dataset(DATASET)
    test = get_dataset(DATASET, train=False)
    labels = get_labels(train)
    if PARTITION == "iid":
        idx = iid_partition(labels, n, seed=seed)
    elif PARTITION == "dir1":
        idx = dirichlet_partition(labels, n, alpha=1.0, seed=seed)   # label + size skew
    elif PARTITION == "shard":
        idx = shard_partition(labels, n, shards_per_client=2, seed=seed)
    else:
        raise ValueError(f"unknown partition {PARTITION!r}")

    rng = np.random.default_rng(1000 + seed)
    corrupt = np.zeros(n, dtype=int)
    delta_transform = None
    if THREAT in ("label_flip", "free_rider", "grad_noise"):
        mal = rng.random(n) < _strength(MAL_FRAC) if THREAT == "label_flip" else \
            set(rng.choice(n, size=max(1, int(round(MAL_FRAC * n))), replace=False).tolist())
        if THREAT == "label_flip":                       # FedCorr (rho,tau): noisy mask + rate~U(tau,1)
            mal_ids = [c for c in range(n) if mal[c]]
        else:
            mal_ids = sorted(mal)
        for c in mal_ids:
            corrupt[c] = 1

    loaders = []
    for c, ci in enumerate(idx):
        if not ci:                                       # dirichlet can leave a client empty
            ci = [int(rng.integers(len(labels)))]
        xs = torch.stack([train[i][0] for i in ci])
        ys = torch.tensor([train[i][1] for i in ci])
        if THREAT == "label_flip" and corrupt[c]:
            rate = float(rng.uniform(TAU, 1.0))          # FedCorr per-client noise level
            xs, ys = CNN_CORRUPTORS["label_flip"](xs, ys, c, rate=rate)
        loaders.append(DataLoader(TensorDataset(xs, ys), batch_size=CFG["batch"], shuffle=True))

    if THREAT == "free_rider":
        delta_transform = make_delta_transform(mal_ids, "free_rider", seed=seed)
    elif THREAT == "grad_noise":
        delta_transform = make_delta_transform(mal_ids, "grad_noise",
                                               std=_strength(GAMMA_GRADNOISE), seed=seed)

    perm = np.random.default_rng(0).permutation(len(test))   # split seed FIXED at 0
    vi, ti = perm[:CFG["n_val"]], perm[CFG["n_val"]:CFG["n_val"] + CFG["n_test"]]
    vx = torch.stack([test[i][0] for i in vi]); vy = torch.tensor([test[i][1] for i in vi])
    val_loader = DataLoader(TensorDataset(vx, vy), batch_size=512)
    test_loader = DataLoader(Subset(test, ti.tolist()), batch_size=512)
    return loaders, corrupt, delta_transform, vx, vy, val_loader, test_loader


def _rounds_to_target(curve, target):
    for r, acc in curve:
        if acc >= target:
            return r
    return None


def _run_arm(arm, loaders, corrupt, dtf, vx, vy, test_loader, nums, device):
    """Run one intervened trajectory; return (final_acc, curve, detection_auroc)."""
    n, R, E, lr, frac = CFG["n"], CFG["rounds"], CFG["epochs"], CFG["lr"], CFG["frac"]
    loss_fn, pkeys = make_cnn_loss(MODEL_FN, vx, vy, device)
    sel_fn = wts_fn = None
    scorer = None

    if arm == "vanilla":
        pass
    elif arm.startswith("flirds_"):
        scorer = OnlineScorer(n, beta=0.5)
        raw = flirds_round_raw_fn(loss_fn, pkeys, n, device)
        if arm == "flirds_mult":
            wts_fn = make_weights_fn(scorer, raw, nums, "multiplicative")
        elif arm == "flirds_repl":
            wts_fn = make_weights_fn(scorer, raw, nums, "replacement")
        elif arm == "flirds_add":
            wts_fn = make_weights_fn(scorer, raw, nums, "additive", lam=0.5)
        elif arm == "flirds_select":
            wts_fn = make_scoreonly_weights_fn(scorer, raw, nums)
            sel_fn = make_softmax_select_fn(scorer)
    elif arm == "shapleyfl":
        scorer = OnlineScorer(n, beta=0.3)            # the ShapleyFL paper value (Def 4.3)
        wts_fn = make_weights_fn(scorer, shapleyfl_round_raw_fn(MODEL_FN().to(device),
                                 DataLoader(TensorDataset(vx, vy), batch_size=512), device),
                                 nums, "replacement")
    elif arm == "fedif":
        scorer = OnlineScorer(n, beta=0.7)               # 1 - gamma(0.3)
        wts_fn = make_weights_fn(scorer, fedif_round_raw_fn(loss_fn, pkeys, device),
                                 nums, "replacement")
    elif arm == "sfedavg":
        sf = SFedAvgSelector(n, MODEL_FN().to(device),
                             DataLoader(TensorDataset(vx, vy), batch_size=512), device, seed=SEED)
        sel_fn, wts_fn = sf.select_fn, sf.weights_fn
    else:
        raise ValueError(f"unknown arm {arm!r}")

    final, hist = fedavg(MODEL_FN, loaders, test_loader, R, E, lr, sample_frac=frac,
                         device=device, seed=SEED, select_fn=sel_fn, weights_fn=wts_fn,
                         delta_transform=dtf)
    auroc = float("nan")
    if scorer is not None and corrupt.sum() and corrupt.sum() < n:
        auroc = float(roc_auc_score(corrupt, -scorer.s))   # corrupt should score LOW -> -s high
    elif arm == "sfedavg" and corrupt.sum() and corrupt.sum() < n:
        auroc = float(roc_auc_score(corrupt, -sf.phi))
    return hist[-1][1], hist, auroc


def _arms_for_partition():
    base = ["vanilla", "flirds_mult", "flirds_select", "shapleyfl", "fedif", "sfedavg"]
    if PARTITION == "dir1":                              # size-skew -> repl/add differ from mult
        base[2:2] = ["flirds_repl", "flirds_add"]
    return base


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(SEED, cudnn_deterministic=True)
    loaders, corrupt, dtf, vx, vy, vl, tl = build()
    nums = [len(l.dataset) for l in loaders]
    print(f"[build] {DATASET}/{PARTITION}/{THREAT}(str={STRENGTH}) seed={SEED} "
          f"n={CFG['n']} corrupt={int(corrupt.sum())} sizes[min/med/max]="
          f"{min(nums)}/{int(np.median(nums))}/{max(nums)}", flush=True)

    arms = {}
    print(f"  {'arm':14s} {'final_acc':>9s} {'AUROC':>6s} {'->target':>8s}", flush=True)
    for arm in _arms_for_partition():
        acc, curve, au = _run_arm(arm, loaders, corrupt, dtf, vx, vy, tl, nums, device)
        rtt = _rounds_to_target(curve, CFG["target"])
        arms[arm] = dict(final_acc=acc, acc_curve=curve, auroc=au, rounds_to_target=rtt)
        print(f"  {arm:14s} {acc:9.4f} {au:6.3f} {str(rtt):>8s}", flush=True)

    dismissal = None
    if DISMISSAL:                                         # FedSV Fig.4 acc-vs-removed curve
        loss_fn, pkeys = make_cnn_loss(MODEL_FN, vx, vy, device)
        dismissal = {}
        for q in [round(0.1 * i, 1) for i in range(10)]:
            sc = OnlineScorer(CFG["n"], beta=0.5)
            wf = make_dismissal_weights_fn(sc, flirds_round_raw_fn(loss_fn, pkeys, CFG["n"], device),
                                           nums, q)
            final, hist = fedavg(MODEL_FN, loaders, tl, CFG["rounds"], CFG["epochs"], CFG["lr"],
                                 sample_frac=CFG["frac"], device=device, seed=SEED,
                                 weights_fn=wf, delta_transform=dtf)
            dismissal[q] = hist[-1][1]
            print(f"  [dismiss] q={q:.1f} final_acc={hist[-1][1]:.4f}", flush=True)

    metrics = dict(dataset=DATASET, partition=PARTITION, threat=THREAT, strength=STRENGTH,
                   seed=SEED, mode=MODE, corrupt=corrupt.tolist(), arms=arms,
                   dismissal=dismissal)
    if PERSIST:
        try:
            name = f"{DATASET}_{PARTITION}_{THREAT.replace('_', '-')}_str{STRENGTH}_seed{SEED}"
            rl = RunLogger(RUN_ROOT, name, dict(cfg=CFG, dataset=DATASET, partition=PARTITION,
                                                threat=THREAT, strength=STRENGTH, seed=SEED, mode=MODE),
                           repo_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            rl.save_metrics(metrics)
            print(f"[persist] {rl.dir}", flush=True)
        except Exception as e:
            print(f"[persist] FAILED ({e!r})", flush=True)
    print("TRACK-C2 RUN OK", flush=True)


if __name__ == "__main__":
    run()
