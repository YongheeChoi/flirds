"""Per-round intervention: online scores -> aggregation weights / selection
(Track C2 / D; plan §3.11 decision ④, 2026-06-12).

Score pipeline (the FedIF Eq 7-8 / ShapleyFL Def 4.2-4.3 convention, locked
06-12): raw per-round value (good -> HIGH) -> per-round min-max over the round's
participants -> EMA accumulate -> s in [0, 1].  min-max is REQUIRED before
weighting (phi can be negative; aggregation weights must be non-negative); EMA
smooths single-round noise (why both papers use it).  `beta` = old-weight
(default 0.5 == ShapleyFL; FedIF's own runs use its paper value 0.7/0.3).

Weight rules (n = client sizes, s = EMA scores; plan §3.11 decision ④):
  multiplicative  w_i ∝ n_i * s_i   -- MAIN, Yonghee's rule ("FedAvg의 데이터-
                  크기 가중에 기여도를 곱하자"); no precedent in the 8 baselines.
  replacement     w_i ∝ s_i         -- FedIF / ShapleyFL convention.
  additive        w_i = lam*s_i/Σs + (1-lam)*n_i/Σn, lam=0.5 -- Ripple convention.
  NB multiplicative == replacement when all n_i are equal; they differ only under
  size skew (interpretation caveat, §3.11).  Fallback: Σw == 0 (flat min-max /
  cold start with beta=1 etc.) -> n-weights (plain FedAvg).

Selection (S-FedAvg convention): softmax(s / T) sampling of k clients without
replacement, via the core's rng (reproducible).  Cold start (all s == 0) is
uniform -- the S-FedAvg warmup behavior falls out for free.

Dismissal (FedSV convention) is NOT a per-round hook: 2-phase orchestration in
the runner (run 1 values clients -> drop the bottom q% -> retrain from scratch).

These build the `select_fn` / `weights_fn` hooks of `fl.server._fedavg_core`.
"""
from __future__ import annotations

import numpy as np

import torch

from ..baselines.shapleyfl import _minmax            # the shared convention helper
from ..core.flirds_estimator import flirds_values
from ..data.corruptors import free_rider, grad_noise

WEIGHT_RULES = ("multiplicative", "replacement", "additive")


class OnlineScorer:
    """EMA-accumulated per-round client scores, s in [0,1]^n.

    update(players, raw) folds one round's raw values (good -> HIGH, aligned with
    `players`) into the running scores; non-participants carry forward (the
    ShapleyFL/FedIF partial-participation convention).  s^0 = 0 (cold start).
    """

    def __init__(self, n_clients, beta=0.5):
        self.s = np.zeros(n_clients)
        self.beta = beta

    def update(self, players, raw):
        nsv = _minmax(np.asarray(raw, dtype=float))
        for i, p in enumerate(players):
            self.s[p] = self.beta * self.s[p] + (1.0 - self.beta) * nsv[i]
        return self.s


def flirds_round_raw(w_r, deltas_map, loss_fn, pkeys, n_clients, device,
                     second_order=True):
    """One round's Flirds value per participant, oriented good -> HIGH.

    == the estimator on the single-round log [(w_r, deltas_map)] (one val grad +
    one HVP), negated (estimator phi is good->low val-loss attribution).  Returns
    raw values aligned with sorted(deltas_map)."""
    phi, _ = flirds_values([(w_r, deltas_map)], loss_fn, pkeys, device,
                           second_order=second_order, n_clients=n_clients)
    return [-phi[p] for p in sorted(deltas_map)]


def rule_weights(rule, s_sel, n_sel, lam=0.5):
    """Map (scores, sizes) of the selected cohort to normalized aggregation
    weights by `rule` (see module docstring).  Σw == 0 -> n-weights fallback."""
    s_sel = np.asarray(s_sel, dtype=float)
    n_sel = np.asarray(n_sel, dtype=float)
    if rule == "multiplicative":
        w = n_sel * s_sel
    elif rule == "replacement":
        w = s_sel.copy()
    elif rule == "additive":
        u = s_sel / s_sel.sum() if s_sel.sum() > 0 else np.zeros_like(s_sel)
        w = lam * u + (1.0 - lam) * n_sel / n_sel.sum()
    else:
        raise ValueError(f"unknown weight rule {rule!r} (use one of {WEIGHT_RULES})")
    if w.sum() <= 0:
        w = n_sel / n_sel.sum()                       # flat scores -> plain FedAvg
    return w / w.sum()


def make_weights_fn(scorer, round_raw_fn, sample_nums, rule, lam=0.5):
    """`weights_fn(r, w_r, deltas_map) -> {client: weight}` for `_fedavg_core`:
    score the CURRENT round with `round_raw_fn(w_r, deltas_map, players) -> raw`
    (raw aligned with sorted players, good -> HIGH), EMA-update the scorer, then
    weight by `rule` -- the FedIF/ShapleyFL same-round score->aggregate order.
    Each C2 intervention baseline plugs its OWN round_raw_fn (Flirds estimator /
    ShapleyFL per-round exact Shapley / FedIF influence / ...) into this one
    machine, so the EMA + weight-rule path is shared and identical across methods.
    """
    def weights_fn(r, w_r, deltas_map):
        players = sorted(deltas_map)
        raw = round_raw_fn(w_r, deltas_map, players)
        s = scorer.update(players, raw)
        w = rule_weights(rule, [s[p] for p in players],
                         [sample_nums[p] for p in players], lam=lam)
        return dict(zip(players, w))

    return weights_fn


def flirds_round_raw_fn(loss_fn, pkeys, n_clients, device, second_order=True):
    """A `round_raw_fn` closure (good -> HIGH) backed by the Flirds estimator."""
    def fn(w_r, deltas_map, players):
        return flirds_round_raw(w_r, deltas_map, loss_fn, pkeys, n_clients, device,
                                second_order=second_order)
    return fn


def fedif_round_raw_fn(loss_fn, pkeys, device, loss_chunks=None):
    """`round_raw_fn` closure backed by FedIF's per-round influence (good->HIGH).
    Pair with OnlineScorer(beta=1-gamma) + rule='replacement' for the FedIF arm."""
    from ..baselines.fedif import fedif_round_raw
    def fn(w_r, deltas_map, players):
        return fedif_round_raw(w_r, deltas_map, players, loss_fn, pkeys, device, loss_chunks)
    return fn


def shapleyfl_round_raw_fn(model, val_loader, device, loss_fn=None, pkeys=None):
    """`round_raw_fn` closure backed by ShapleyFL's per-round exact Shapley (good->
    HIGH; CNN accuracy via (model, val_loader), or -val_loss via (loss_fn, pkeys)).
    Pair with OnlineScorer(beta=0.5) + rule='replacement' for the ShapleyFL arm."""
    from ..baselines.shapleyfl import shapleyfl_round_raw
    def fn(w_r, deltas_map, players):
        return shapleyfl_round_raw(w_r, deltas_map, players, model, val_loader,
                                   device, loss_fn, pkeys)
    return fn


def make_scoreonly_weights_fn(scorer, round_raw_fn, sample_nums):
    """`weights_fn` that UPDATES the scorer each round (so a paired select_fn sees
    fresh scores) but aggregates with plain n-weights -- the selection-only arm
    (intervention ②), where selection is the sole intervention and aggregation
    stays vanilla FedAvg."""
    def weights_fn(r, w_r, deltas_map):
        players = sorted(deltas_map)
        scorer.update(players, round_raw_fn(w_r, deltas_map, players))
        w = np.array([sample_nums[p] for p in players], dtype=float)
        return dict(zip(players, w / w.sum()))

    return weights_fn


def make_dismissal_weights_fn(scorer, round_raw_fn, sample_nums, q):
    """`weights_fn` for the FedSV bottom-q% dismissal (intervention ③; FedSV §5.5):
    score the round, then DROP the bottom-`q` fraction of the round's participants
    (by EMA score) from aggregation and n-weight the survivors.  q is swept over
    {0, 0.1, ..., 0.9} to trace FedSV's acc-vs-removed curve; q=0 == plain FedAvg.
    Ties / cold-start (flat scores) break by client id (stable, reproducible)."""
    def weights_fn(r, w_r, deltas_map):
        players = sorted(deltas_map)
        s = scorer.update(players, round_raw_fn(w_r, deltas_map, players))
        n_drop = int(q * len(players))                # floor; q=0 -> drop none
        order = sorted(players, key=lambda p: (s[p], p))   # ascending score, id tiebreak
        kept = set(order[n_drop:]) or set(players)    # never drop everyone
        w = np.array([sample_nums[p] if p in kept else 0.0 for p in players], dtype=float)
        w /= w.sum()
        return dict(zip(players, w))

    return weights_fn


def make_delta_transform(malicious, threat, std=0.1, seed=0):
    """`delta_transform(c, r, delta)` for `fl.server` update-level C2 threats:
    `malicious` clients get free_rider (zero delta) or grad_noise (Gaussian std);
    honest clients pass through.  Per-(client, round) generator -> reproducible."""
    mal = set(malicious)

    def transform(c, r, delta):
        if c not in mal:
            return delta
        g = torch.Generator().manual_seed(seed + 1000 * c + r)
        if threat == "free_rider":
            return free_rider(delta, mode="zero")
        if threat == "grad_noise":
            return grad_noise(delta, std=std, generator=g)
        raise ValueError(f"delta_transform threat must be free_rider|grad_noise, got {threat!r}")

    return transform


def make_softmax_select_fn(scorer, temperature=1.0):
    """`select_fn(r, k, rng) -> k client ids` for `_fedavg_core`: S-FedAvg-style
    softmax(s/T) sampling without replacement over the scorer's running scores."""

    def select_fn(r, k, rng):
        z = scorer.s / temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        return rng.choice(len(scorer.s), size=k, replace=False, p=p)

    return select_fn
