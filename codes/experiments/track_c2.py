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
from functools import partial

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset, TensorDataset

from flirds.backends.cnn import make_cnn_loss
from flirds.baselines.sfedavg import SFedAvgSelector
from flirds.data.cnn import _STATS, get_dataset, get_labels
from flirds.data.corruptors import CNN_CORRUPTORS
from flirds.fl.intervene import (OnlineScorer, SignAccumulator, _conf_keep,
                                 _phi_cdf, fedif_round_raw_fn,
                                 flirds_round_raw_fn, make_confgate_select_fn,
                                 make_delta_transform, make_dismissal_weights_fn,
                                 make_fixed_excl_select_fn, make_roundwise_mask,
                                 make_gatedweight_weights_fn,
                                 make_observer_weights_fn, make_probweight_weights_fn,
                                 make_rawweight_weights_fn,
                                 make_scoreonly_weights_fn, make_signgate_select_fn,
                                 make_signgate_weights_fn, make_softmax_select_fn,
                                 make_weights_fn, make_zgate_select_fn,
                                 make_zgate_weights_fn, shapleyfl_round_raw_fn)
from flirds.fl.partition import (dirichlet_partition, iid_partition,
                                 shard_partition)
from flirds.fl.score_providers import SOURCES as TH_SOURCES
from flirds.fl.score_providers import provider_round_raw_fn
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

# --- Track G CNN leg (2026-07-19; ALL env-gated -- unset = bit-identical legacy path) ---
# C2_EXTRA_ARMS appends gate arms to the partition's arm list; C2_FLIP_RATE pins the
# per-client label-flip rate for the Track G dose ladder (audit-picked {0.15,0.35,0.70};
# None = the legacy FedCorr U(TAU,1) draw); C2GATE = the shared gate defaults (spec
# §4.3 -- per-cell tuning forbidden, ablation cells only).  Gate arms persist per-round
# {round, client, raw, cum, weight} rows to phi_rounds.parquet (the C2 gap: OnlineScorer
# state was never persisted).
EXTRA_ARMS = [a for a in os.environ.get("C2_EXTRA_ARMS", "").split(",") if a]
FLIP_RATE = os.environ.get("C2_FLIP_RATE")
C2GATE = dict(burn_in=int(os.environ.get("C2_BURN_IN", "10")),
              tau=float(os.environ.get("C2_TAU", "0.0")),
              min_obs=int(os.environ.get("C2_MIN_OBS", "2")),
              probation_every=int(os.environ.get("C2_PROB_EVERY", "5")),
              decay=float(os.environ.get("C2_DECAY", "1.0")),
              z_c=float(os.environ.get("C2_ZC", "1.5")),
              alpha_w=float(os.environ.get("C2_ALPHA_W", "1.0")),
              conf_z=float(os.environ.get("C2_CONF_Z", "1.645")))  # P5: one-sided 95%, universal
_GATE_ARMS = ("flirds_gate_v1", "flirds_gate_v2", "flirds_zgate_v2",
              "flirds_gatew_v2", "flirds_gatew_v1")

# --- Track H CNN leg (runs/track_h/README.md; ALL env-gated -- unset = legacy) ---
# Same-policy score-source competition: arms named <src>_<policy> with src in
# score_providers.SOURCES and policy in _TH_POLICIES (P1 gate_v2 / P2 gatew_v2 /
# P3 mult / P4 zgate_v2, + gate_v1 ablation); the legacy flirds_* arms keep their
# original branches (bit-identical closures).  `observer` scores ALL sources on
# one vanilla trajectory (phi_rounds gains a `method` column) = the T2 input.
# C2_T2=1 appends the retrain leg after the arm loop: per source kept =
# {cum > tau} -> t2_sign_<src> (plain n-weights) / t2_signw_<src> (static
# w ~ n*max(cum,0)^alpha), kept-set-deduped, + one size-matched t2_random_k<s>
# control per distinct kept size.  Retrains keep the threat ACTIVE (deployment
# semantics, track_g V3 convention) via fixed exclusion on the full loader set.
TH_T2 = os.environ.get("C2_T2", "0") == "1"
# C2_DYN=1 (2026-07-21): the corrupt set is RE-DRAWN EVERY ROUND (same count as
# the static stage) instead of fixed at build -- no client is statically corrupt,
# so the `corrupt` mask stays all-zero (client-level AUROC undefined -> skipped)
# and oracle_excl/random_excl become per-round exclusions.  T2 unsupported.
DYN = os.environ.get("C2_DYN", "0") == "1"
_DYN = {"mask_at": None, "clock": None}                 # set by build() when DYN
# P5 (2026-07-21): C2_T2_P5=1 adds the confidence-policy retrain arms
# (t2_csign_<src> = UCB-kept retrain / t2_pw_<src> = static Phi(t)-weight retrain);
# C2_T2_LEGACY=0 skips the already-run t2_sign/t2_signw emission (results on disk).
TH_T2_P5 = os.environ.get("C2_T2_P5", "0") == "1"
TH_T2_LEGACY = os.environ.get("C2_T2_LEGACY", "1") == "1"
_TH_POLICIES = ("gate_v1", "gate_v2", "zgate_v2", "gatew_v2", "mult",
                "cgate", "pweight")                     # P5-hard / P5-soft (2026-07-21)
# C2_OBS_SRCS restricts which sources the observer arm scores (default: all 8 =
# bit-identical legacy).  REQUIRED at full participation (scale leg, C2_FRAC=1.0):
# the coalition providers cost O(2^k)..O(k^2) evals per round -- shapleyfl's exact
# 2^k alone never terminates at k=100 (runs/track_h/scale/RUN_SCALE.md).
OBS_SRCS = tuple(s for s in os.environ.get("C2_OBS_SRCS",
                                           ",".join(TH_SOURCES)).split(",") if s)


def _th_parse(arm):
    """(src, policy) for Track H competition arms; None for every legacy arm."""
    for pol in _TH_POLICIES:
        if arm.endswith("_" + pol):
            src = arm[: -len(pol) - 1]
            if src in TH_SOURCES and arm not in _GATE_ARMS and arm != "flirds_mult":
                return src, pol
    return None
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
RUN_ROOT = os.environ.get("C2_RUN_ROOT", os.path.join(_REPO, "runs", "track_c", "c2"))

CFG = {
    "full":  dict(n=100, frac=0.1, rounds=120, epochs=5, lr=0.01, batch=64,
                  n_val=2000, n_test=8000, target=0.6),
    "smoke": dict(n=20, frac=0.5, rounds=4, epochs=1, lr=0.05, batch=64,
                  n_val=512, n_test=1024, target=0.3),
}[MODE]

WIDTH = float(os.environ.get("C2_WIDTH", "1"))            # signal-size probe lever: capacity (width mult)
if os.environ.get("C2_FRAC"):                             # signal-size probe lever: participation frac
    CFG["frac"] = float(os.environ["C2_FRAC"])
MODEL_FN = partial({"cifar10": FedSVCNN, "fmnist": LeNet5}[DATASET], width=WIDTH)
MAL_FRAC = 0.4                                            # noisy/malicious client fraction (main)
TAU = 0.5                                                 # FedCorr per-client rate lower bound
GAMMA_GRADNOISE = 0.1                                     # FedIF main sigma


def _strength(default):
    return default if STRENGTH == "main" else float(STRENGTH)


class _RoundClock:
    """Current round for the dynamic label_flip loaders; stamped by the select
    seam, which `_fedavg_core` calls before any local training in the round."""
    def __init__(self):
        self.r = 0


class _DynLFLoader:
    """Serves the clean or the flipped copy of a client's data depending on
    whether the client is corrupt in the clock's current round (C2_DYN).
    Exposes `.dataset` (clean copy; identical length) for `len(ld.dataset)`."""
    def __init__(self, clean_ld, flip_ld, cid, clock, mask_at):
        self._clean, self._flip, self._cid = clean_ld, flip_ld, cid
        self._clock, self._mask_at = clock, mask_at
        self.dataset = clean_ld.dataset

    def __iter__(self):
        cur = (self._flip if self._cid in self._mask_at(self._clock.r)
               else self._clean)
        return iter(cur)

    def __len__(self):
        return len(self._clean)


def _clocked_select(base, clock, n_clients):
    """Wrap an arm's select seam so the round clock is stamped before local
    training; replicates the core's uniform default when `base` is None."""
    def select_fn(r, k, rng):
        clock.r = r
        if base is not None:
            return base(r, k, rng)
        return rng.choice(n_clients, size=k, replace=False)
    return select_fn


def _make_dyn_excl_select_fn(n_clients, mask_at):
    """Per-round exclusion (C2_DYN oracle_excl / random_excl): sample the
    round's cohort from the clients NOT in mask_at(r)."""
    def select_fn(r, k, rng):
        keep = [i for i in range(n_clients) if i not in mask_at(r)]
        if k >= len(keep):
            return np.array(keep)
        return rng.choice(keep, size=k, replace=False)
    return select_fn


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
    mal_ids, dyn_mask = [], None
    if DYN and THREAT in ("label_flip", "free_rider", "grad_noise"):
        # C2_DYN: the corrupt set is re-drawn every round (fixed count = the
        # static fr/gn count); no client is statically corrupt -> corrupt stays
        # all-zero and the client-level AUROC block self-skips.
        if THREAT == "label_flip" and FLIP_RATE is None:
            raise ValueError("C2_DYN label_flip needs the fixed-dose C2_FLIP_RATE")
        m = max(1, int(round(_strength(MAL_FRAC) * n)))
        dyn_mask = make_roundwise_mask(n, m, seed)
        _DYN["mask_at"], _DYN["clock"] = dyn_mask, _RoundClock()
    elif THREAT in ("label_flip", "free_rider", "grad_noise"):
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
            rate = (float(rng.uniform(TAU, 1.0))         # FedCorr per-client noise level
                    if FLIP_RATE is None else float(FLIP_RATE))   # Track G fixed-dose point
            xs, ys = CNN_CORRUPTORS["label_flip"](xs, ys, c, rate=rate)
        ld = DataLoader(TensorDataset(xs, ys), batch_size=CFG["batch"], shuffle=True)
        if dyn_mask is not None and THREAT == "label_flip":   # C2_DYN: clean+flipped pair
            fxs, fys = CNN_CORRUPTORS["label_flip"](xs, ys, c, rate=float(FLIP_RATE))
            fld = DataLoader(TensorDataset(fxs, fys), batch_size=CFG["batch"], shuffle=True)
            ld = _DynLFLoader(ld, fld, c, _DYN["clock"], dyn_mask)
        loaders.append(ld)

    mal_arg = dyn_mask if dyn_mask is not None else mal_ids
    if THREAT == "free_rider":
        delta_transform = make_delta_transform(mal_arg, "free_rider", seed=seed)
    elif THREAT == "grad_noise":
        delta_transform = make_delta_transform(mal_arg, "grad_noise",
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


def _gate_sink(rows, acc, n):
    """phi_rounds row sink for the Track G gate arms (mirrors track_g.make_sink):
    one row per (round, client) for ALL n clients -- participants carry (raw,
    weight), everyone the post-round (cum, n_obs) snapshot.  raw/cum are
    CONTRIBUTION-oriented (helpful -> positive = -stored-phi)."""
    def sink(r, players, raw, wmap, fallback):
        pset = {p: i for i, p in enumerate(players)}
        for c in range(n):
            i = pset.get(c)
            rows.append(dict(round=r, client=c, participated=c in pset,
                             raw=float(raw[i]) if i is not None else float("nan"),
                             weight=float(wmap[c]) if i is not None else float("nan"),
                             cum=float(acc.cum[c]), n_obs=int(acc.n_obs[c]),
                             fallback=fallback))
    return sink


def _run_arm(arm, loaders, corrupt, dtf, vx, vy, test_loader, nums, device, rows=None,
             observer_out=None):
    """Run one intervened trajectory; return (final_acc, curve, detection_auroc).
    `rows` (gate arms + observer): the phi_rounds sink target.  `observer_out`
    (Track H): the observer arm deposits its per-source SignAccumulators there."""
    n, R, E, lr, frac = CFG["n"], CFG["rounds"], CFG["epochs"], CFG["lr"], CFG["frac"]
    loss_fn, pkeys = make_cnn_loss(MODEL_FN, vx, vy, device)
    sel_fn = wts_fn = None
    scorer = None
    gate_acc = None

    if arm == "vanilla":
        pass
    elif arm == "observer":                           # Track H T2 input: OBS_SRCS (default: every source), one trajectory
        obs_accs = {s: SignAccumulator(n) for s in OBS_SRCS}
        obs_raws = {s: provider_round_raw_fn(s, loss_fn, pkeys, n, device, seed=SEED)
                    for s in OBS_SRCS}

        def wts_fn(r, w_r, deltas_map):               # plain n-weights = vanilla trajectory
            players = sorted(deltas_map)
            w = np.array([nums[p] for p in players], dtype=float)
            w /= w.sum()
            for s in OBS_SRCS:
                rv = obs_raws[s](w_r, deltas_map, players)
                obs_accs[s].update(players, rv)
                if rows is not None:
                    pset = {p: i for i, p in enumerate(players)}
                    for c in range(n):
                        i = pset.get(c)
                        rows.append(dict(method=s, round=r, client=c,
                                         participated=c in pset,
                                         raw=float(rv[i]) if i is not None else float("nan"),
                                         weight=float(w[i]) if i is not None else float("nan"),
                                         cum=float(obs_accs[s].cum[c]),
                                         n_obs=int(obs_accs[s].n_obs[c]), fallback=False))
            return dict(zip(players, w))

        if observer_out is not None:
            observer_out["accs"] = obs_accs
    elif arm == "oracle_excl":                        # Track G upper bound: true corrupt out
        if _DYN["mask_at"] is not None:               # C2_DYN: per-round true corrupt out
            sel_fn = _make_dyn_excl_select_fn(n, _DYN["mask_at"])
        else:
            sel_fn = make_fixed_excl_select_fn(n, {int(c) for c in np.where(corrupt)[0]})
    elif arm == "random_excl":                        # Track G control: same-count random out
        if _DYN["mask_at"] is not None:               # C2_DYN: per-round same-count random out
            m = len(_DYN["mask_at"](0))
            sel_fn = _make_dyn_excl_select_fn(n, make_roundwise_mask(n, m, 2000 + SEED, salt=1))
        else:
            rng_x = np.random.default_rng(2000 + SEED)
            excl = {int(c) for c in rng_x.choice(n, size=int(corrupt.sum()), replace=False)}
            print(f"  [random_excl] excluded={sorted(excl)}", flush=True)
            sel_fn = make_fixed_excl_select_fn(n, excl)
    elif arm in _GATE_ARMS:                           # Track G sign/z/magnitude gates
        g = C2GATE
        gate_acc = SignAccumulator(n, decay=g["decay"])
        raw = flirds_round_raw_fn(loss_fn, pkeys, n, device)
        sink = _gate_sink(rows, gate_acc, n) if rows is not None else None
        if arm == "flirds_gate_v1":
            wts_fn = make_signgate_weights_fn(gate_acc, raw, nums, tau=g["tau"], sink=sink)
        elif arm == "flirds_gatew_v1":                # per-round-raw magnitude (CNN ablation)
            wts_fn = make_rawweight_weights_fn(gate_acc, raw, nums, tau=g["tau"],
                                               alpha=g["alpha_w"], sink=sink)
        elif arm == "flirds_zgate_v2":
            sel_fn = make_zgate_select_fn(gate_acc, g["burn_in"], c=g["z_c"],
                                          min_obs=g["min_obs"],
                                          probation_every=g["probation_every"])
            wts_fn = make_zgate_weights_fn(gate_acc, raw, nums, c=g["z_c"], sink=sink)
        else:                                         # flirds_gate_v2 | flirds_gatew_v2
            sel_fn = make_signgate_select_fn(gate_acc, g["burn_in"], tau=g["tau"],
                                             min_obs=g["min_obs"],
                                             probation_every=g["probation_every"])
            wts_fn = (make_gatedweight_weights_fn(gate_acc, raw, nums, tau=g["tau"],
                                                  alpha=g["alpha_w"], sink=sink)
                      if arm == "flirds_gatew_v2" else
                      make_signgate_weights_fn(gate_acc, raw, nums, tau=g["tau"], sink=sink))
    elif _th_parse(arm):                              # Track H: <src>_<policy> competition arm
        # MUST precede the startswith("flirds_") legacy branch: flirds_cgate /
        # flirds_pweight would otherwise be silently swallowed into a vanilla
        # trajectory there (no sub-case matches -> no wts_fn; caught 2026-07-21
        # by the scale smoke's AUROC check).  _th_parse rejects flirds_mult and
        # the _GATE_ARMS names, so every legacy arm still takes its old branch.
        src, policy = _th_parse(arm)
        g = C2GATE
        raw = provider_round_raw_fn(src, loss_fn, pkeys, n, device, seed=SEED)
        if policy == "mult":                          # P3: same soft policy, source swapped
            scorer = OnlineScorer(n, beta=0.5)        # uniform beta -- policy held fixed
            wts_fn = make_weights_fn(scorer, raw, nums, "multiplicative")
        else:
            gate_acc = SignAccumulator(n, decay=g["decay"])
            sink = _gate_sink(rows, gate_acc, n) if rows is not None else None
            if policy == "gate_v1":
                wts_fn = make_signgate_weights_fn(gate_acc, raw, nums, tau=g["tau"], sink=sink)
            elif policy == "cgate":                   # P5-hard: confidence sign gate
                sel_fn = make_confgate_select_fn(gate_acc, g["burn_in"], z=g["conf_z"],
                                                 min_obs=g["min_obs"],
                                                 probation_every=g["probation_every"])
                wts_fn = make_observer_weights_fn(gate_acc, raw, nums, sink=sink)
            elif policy == "pweight":                 # P5-soft: w ~ n * Phi(t)
                wts_fn = make_probweight_weights_fn(gate_acc, raw, nums,
                                                    burn_in=g["burn_in"],
                                                    min_obs=g["min_obs"], sink=sink)
            elif policy == "zgate_v2":                # P4: rank-only (cohort-relative)
                sel_fn = make_zgate_select_fn(gate_acc, g["burn_in"], c=g["z_c"],
                                              min_obs=g["min_obs"],
                                              probation_every=g["probation_every"])
                wts_fn = make_zgate_weights_fn(gate_acc, raw, nums, c=g["z_c"], sink=sink)
            else:                                     # P1 gate_v2 | P2 gatew_v2
                sel_fn = make_signgate_select_fn(gate_acc, g["burn_in"], tau=g["tau"],
                                                 min_obs=g["min_obs"],
                                                 probation_every=g["probation_every"])
                wts_fn = (make_gatedweight_weights_fn(gate_acc, raw, nums, tau=g["tau"],
                                                      alpha=g["alpha_w"], sink=sink)
                          if policy == "gatew_v2" else
                          make_signgate_weights_fn(gate_acc, raw, nums, tau=g["tau"],
                                                   sink=sink))
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

    if _DYN["clock"] is not None and THREAT == "label_flip":
        # dynamic lf: stamp the round into the loader clock before local training
        sel_fn = _clocked_select(sel_fn, _DYN["clock"], n)
    final, hist = fedavg(MODEL_FN, loaders, test_loader, R, E, lr, sample_frac=frac,
                         device=device, seed=SEED, select_fn=sel_fn, weights_fn=wts_fn,
                         delta_transform=dtf)
    auroc = float("nan")
    if scorer is not None and corrupt.sum() and corrupt.sum() < n:
        auroc = float(roc_auc_score(corrupt, -scorer.s))   # corrupt should score LOW -> -s high
    elif arm == "sfedavg" and corrupt.sum() and corrupt.sum() < n:
        auroc = float(roc_auc_score(corrupt, -sf.phi))
    elif gate_acc is not None and corrupt.sum() and corrupt.sum() < n:
        auroc = float(roc_auc_score(corrupt, -gate_acc.cum))   # suspicion = -cum contribution
    return hist[-1][1], hist, auroc


def _arms_for_partition():
    if os.environ.get("C2_ARMS"):                        # Track G/H: explicit FULL arm list
        arms = [a for a in os.environ["C2_ARMS"].split(",") if a]
    else:
        arms = ["vanilla", "flirds_mult", "flirds_select", "shapleyfl", "fedif", "sfedavg"]
        if PARTITION == "dir1":                          # size-skew -> repl/add differ from mult
            arms[2:2] = ["flirds_repl", "flirds_add"]
        arms = arms + EXTRA_ARMS                         # Track G gate arms (env-gated; [] default)
    if TH_T2 and "observer" not in arms:                 # T2 needs the observer's cums
        arms.append("observer")
    return arms


# --------------------------------------------------------------------------- #
# Track H T2 (retrain-from-scratch on the observer's cumulative scores)       #
# --------------------------------------------------------------------------- #
def _t2_kept(cum, tau):
    """Kept client ids under the strict sign gate (cum > tau)."""
    return [c for c in range(len(cum)) if cum[c] > tau]


def _t2_kept_ucb(acc, z, min_obs):
    """Kept ids under the P5 confidence gate (final observer stats): see
    intervene._conf_keep -- keep unless cum + z*sd*sqrt(n) <= 0."""
    keep = _conf_keep(acc, z, min_obs)
    return [c for c in range(len(acc.cum)) if keep[c]]


def _t2_pw_wvec(acc, min_obs):
    """P5-soft static weight factors {client: Phi(t)} from the observer's final
    stream stats; deterministic streams degenerate to 1[cum > 0] (exact-0 -> 0,
    dropped from the dict = excluded); under-observed -> neutral 1."""
    _, sd, n = acc.stats()
    out = {}
    for c in range(len(acc.cum)):
        if n[c] < max(min_obs, 2):
            f = 1.0
        elif sd[c] <= 0.0:
            f = 1.0 if acc.cum[c] > 0 else 0.0
        else:
            f = _phi_cdf(acc.cum[c] / (sd[c] * np.sqrt(n[c])))
        if f > 0.0:
            out[c] = f
    return out


def _t2_cache_key(kept, wvec):
    """Dedupe key: identical (kept set, static weight vector) -> one shared retrain."""
    return (frozenset(kept),
            None if wvec is None else tuple((c, round(float(wvec[c]), 12))
                                            for c in sorted(wvec)))


def _t2_static_weights_fn(nums, wvec):
    """Static aggregation weights w ~ n * wvec[c] (the t2_signw arm); no positive
    mass -> n-weights fallback (same guard as the online gates)."""
    def wf(r, w_r, deltas_map):
        players = sorted(deltas_map)
        w = np.array([nums[p] * wvec.get(p, 0.0) for p in players], dtype=float)
        if w.sum() <= 0:
            w = np.array([nums[p] for p in players], dtype=float)
        w /= w.sum()
        return dict(zip(players, w))
    return wf


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(SEED, cudnn_deterministic=True)
    if DYN and TH_T2:
        raise ValueError("C2_DYN does not support the T2 retrain leg (final-stats "
                         "kept-sets are undefined when no client is statically corrupt)")
    loaders, corrupt, dtf, vx, vy, vl, tl = build()
    nums = [len(l.dataset) for l in loaders]
    print(f"[build] {DATASET}/{PARTITION}/{THREAT}(str={STRENGTH}) seed={SEED} "
          f"n={CFG['n']} corrupt={int(corrupt.sum())} sizes[min/med/max]="
          f"{min(nums)}/{int(np.median(nums))}/{max(nums)}", flush=True)

    arms = {}
    all_rows = []                                        # Track G/H gate arms' phi_rounds
    observer_out = {}                                    # Track H: observer -> T2 handoff
    print(f"  {'arm':14s} {'final_acc':>9s} {'AUROC':>6s} {'->target':>8s}", flush=True)
    for arm in _arms_for_partition():
        th = _th_parse(arm)
        rows = ([] if (arm in _GATE_ARMS or arm == "observer"
                       or (th and th[1] != "mult")) else None)
        acc, curve, au = _run_arm(arm, loaders, corrupt, dtf, vx, vy, tl, nums, device,
                                  rows=rows, observer_out=observer_out)
        rtt = _rounds_to_target(curve, CFG["target"])
        arms[arm] = dict(final_acc=acc, acc_curve=curve, auroc=au, rounds_to_target=rtt)
        if rows:
            all_rows += [dict(arm=arm, **x) for x in rows]
        print(f"  {arm:14s} {acc:9.4f} {au:6.3f} {str(rtt):>8s}", flush=True)

    if TH_T2 and observer_out.get("accs"):               # Track H T2: retrain leg
        n = CFG["n"]
        cums = {s: observer_out["accs"][s].cum.copy() for s in observer_out["accs"]}
        cache = {}

        def _t2_run(name, kept, wvec):
            if not kept:
                print(f"  {name:22s}    kept=EMPTY -> retrain skipped", flush=True)
                arms[name] = dict(final_acc=None, kept=[], skipped="empty_kept")
                return
            if wvec is None and len(kept) == n:          # sign gate kept everyone
                print(f"  {name:22s}    kept=ALL -> equals vanilla (not re-run)", flush=True)
                arms[name] = dict(final_acc=None, kept=kept, skipped="equals_vanilla")
                return
            key = _t2_cache_key(kept, wvec)
            shared = key in cache
            if not shared:                               # deployment semantics: threat active,
                sel = make_fixed_excl_select_fn(         # exclusion is the only intervention
                    n, [c for c in range(n) if c not in set(kept)])
                wf = None if wvec is None else _t2_static_weights_fn(nums, wvec)
                _, hist = fedavg(MODEL_FN, loaders, tl, CFG["rounds"], CFG["epochs"],
                                 CFG["lr"], sample_frac=CFG["frac"], device=device,
                                 seed=SEED, select_fn=sel, weights_fn=wf,
                                 delta_transform=dtf)
                cache[key] = hist
            hist = cache[key]
            arms[name] = dict(final_acc=hist[-1][1], acc_curve=hist, kept=kept,
                              dedup_shared=shared,
                              rounds_to_target=_rounds_to_target(hist, CFG["target"]))
            print(f"  {name:22s} {hist[-1][1]:9.4f} kept={len(kept)}"
                  f"{' (dedup)' if shared else ''}", flush=True)

        ctrl_sizes = set()
        if TH_T2_LEGACY:
            for src in cums:
                kept = _t2_kept(cums[src], C2GATE["tau"])
                _t2_run(f"t2_sign_{src}", kept, None)
                _t2_run(f"t2_signw_{src}", kept,
                        {c: max(float(cums[src][c]), 0.0) ** C2GATE["alpha_w"] for c in kept})
                ctrl_sizes.add(len(kept))
        if TH_T2_P5:                                     # P5 retrain variants (2026-07-21)
            for src in cums:
                a = observer_out["accs"][src]
                kept_c = _t2_kept_ucb(a, C2GATE["conf_z"], C2GATE["min_obs"])
                _t2_run(f"t2_csign_{src}", kept_c, None)
                ctrl_sizes.add(len(kept_c))
                wvec = _t2_pw_wvec(a, C2GATE["min_obs"])
                _t2_run(f"t2_pw_{src}", sorted(wvec), wvec)
        rng_t2 = np.random.default_rng(4000 + SEED)      # size-matched random controls
        for size in sorted(ctrl_sizes):
            if 0 < size < n:
                _t2_run(f"t2_random_k{size}",
                        sorted(int(x) for x in rng_t2.choice(n, size=size, replace=False)),
                        None)

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

    th_active = TH_T2 or any(_th_parse(a) or a == "observer" for a in arms)
    dyn_info = ({"roundwise": True, "n_corrupt": len(_DYN["mask_at"](0))}
                if _DYN["mask_at"] is not None else None)
    metrics = dict(dataset=DATASET, partition=PARTITION, threat=THREAT, strength=STRENGTH,
                   seed=SEED, mode=MODE, corrupt=corrupt.tolist(), arms=arms,
                   dismissal=dismissal,
                   **({"dyn": dyn_info} if dyn_info else {}),
                   **({"observer_cum": {s: [float(v) for v in observer_out["accs"][s].cum]
                                        for s in observer_out["accs"]}}
                      if observer_out.get("accs") else {}))
    if PERSIST:
        try:
            name = (os.environ.get("C2_RUN_NAME")                        # probe cells override (width/frac in name)
                    or f"{DATASET}_{PARTITION}_{THREAT.replace('_', '-')}_str{STRENGTH}_seed{SEED}")
            rl = RunLogger(RUN_ROOT, name, dict(cfg=CFG, dataset=DATASET, partition=PARTITION,
                                                threat=THREAT, strength=STRENGTH, seed=SEED, mode=MODE,
                                                width=WIDTH,
                                                **({"dyn": dyn_info} if dyn_info else {}),
                                                **({"gate": C2GATE, "flip_rate": FLIP_RATE,
                                                    "extra_arms": EXTRA_ARMS}
                                                   if (EXTRA_ARMS or th_active) else {}),
                                                **({"track_h": {"t2": TH_T2,
                                                                "t2_p5": TH_T2_P5,
                                                                "t2_legacy": TH_T2_LEGACY,
                                                                "sources": list(TH_SOURCES),
                                                                "obs_srcs": list(OBS_SRCS)}}
                                                   if th_active else {})),
                           repo_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            rl.save_metrics(metrics)
            if all_rows:                                 # Track G per-round phi record
                rl.save_phi(all_rows, fname="phi_rounds.parquet")
            print(f"[persist] {rl.dir}", flush=True)
        except Exception as e:
            print(f"[persist] FAILED ({e!r})", flush=True)
    print("TRACK-C2 RUN OK", flush=True)


if __name__ == "__main__":
    run()
