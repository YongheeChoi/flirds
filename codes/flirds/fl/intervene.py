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
                     second_order=True, loss_chunks=None):
    """One round's Flirds value per participant, oriented good -> HIGH.

    == the estimator on the single-round log [(w_r, deltas_map)] (one val grad +
    one HVP), negated (estimator phi is good->low val-loss attribution).  Returns
    raw values aligned with sorted(deltas_map).  `loss_chunks` (LLM): the
    backends.llm chunk closures -- REQUIRED at LLM scale (the single-shot eager
    HVP over the whole val set OOMs); None (CNN) keeps the single-shot path =
    bit-identical."""
    phi, _ = flirds_values([(w_r, deltas_map)], loss_fn, pkeys, device,
                           second_order=second_order, n_clients=n_clients,
                           loss_chunks=loss_chunks)
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


def flirds_round_raw_fn(loss_fn, pkeys, n_clients, device, second_order=True,
                        loss_chunks=None):
    """A `round_raw_fn` closure (good -> HIGH) backed by the Flirds estimator.
    Pass `loss_chunks` (the make_llm_loss chunk closures) at LLM scale -- the
    single-shot HVP over the full val set OOMs; CNN leaves it None (bit-identical)."""
    def fn(w_r, deltas_map, players):
        return flirds_round_raw(w_r, deltas_map, loss_fn, pkeys, n_clients, device,
                                second_order=second_order, loss_chunks=loss_chunks)
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
    Pair with OnlineScorer(beta=0.3) + rule='replacement' for the ShapleyFL arm."""
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


# --------------------------------------------------------------------------- #
# Track G -- sign/threshold gating on RAW contribution values (2026-07-19).    #
# ADDITIVE ONLY: nothing above this line changed.                              #
#                                                                             #
# Sign convention (D-3, the experiment's top risk): every `raw` here is        #
# CONTRIBUTION-oriented (helpful -> POSITIVE) -- the round_raw_fns above       #
# already negate the estimator's good->low phi.  The gate keeps `> tau`       #
# (tau=0 strict: a frzero client's exact-0.0 raw is EXCLUDED).  OnlineScorer  #
# min-maxes raw per round, which DESTROYS the sign (the round minimum maps    #
# to 0 regardless of its sign) -- hence SignAccumulator, which never          #
# normalizes.                                                                 #
#                                                                             #
# `sink(r, players, raw, wmap, fallback)` is the per-round logging seam       #
# (Track G persists {round, client, raw, cum, weight} -> phi_rounds.parquet;  #
# the first per-round phi record in the project).  None -> no logging.        #
# --------------------------------------------------------------------------- #
class SignAccumulator:
    """Sign-preserving accumulation of raw (contribution-oriented) round values.

    update(players, raw) folds one round: cum[p] = decay*cum[p] + raw (decay=1.0
    -> plain sum; <1.0 -> EMA-style discounted sum), n_obs[p] += 1; non-participants
    carry forward untouched.  NO min-max, NO clipping -- cum keeps the game's
    absolute zero point (phi=0 == null player), which is what the tau=0 gate reads.

    `sum2` (P5, 2026-07-21) tracks the raw second moment so the confidence gates
    can read each client's OWN online noise (mean/sd via `stats()`) -- computed
    purely from what this run observes during training (no prior information).
    Legacy consumers (cum/n_obs) are bit-unchanged.
    """

    def __init__(self, n_clients, decay=1.0):
        self.cum = np.zeros(n_clients)
        self.sum2 = np.zeros(n_clients)
        self.n_obs = np.zeros(n_clients, dtype=int)
        self.decay = decay

    def update(self, players, raw):
        for i, p in enumerate(players):
            self.cum[p] = self.decay * self.cum[p] + float(raw[i])
            self.sum2[p] += float(raw[i]) ** 2
            self.n_obs[p] += 1
        return self.cum

    def stats(self):
        """(mean, sd, n_obs) per client from the observed stream (sd: ddof=1;
        n<2 -> 0).  Valid for decay=1.0 (plain-sum) accumulators -- the only
        setting the P5 gates use."""
        n = np.maximum(self.n_obs, 1)
        mean = self.cum / n
        var = np.maximum(self.sum2 / n - mean ** 2, 0.0) * (n / np.maximum(n - 1, 1))
        var[self.n_obs < 2] = 0.0
        return mean, np.sqrt(var), self.n_obs


def _zscores(v):
    """z-scores of v (flat -> zeros)."""
    v = np.asarray(v, dtype=float)
    s = v.std()
    return np.zeros_like(v) if s <= 0 else (v - v.mean()) / s


def _gate_select_fn(acc, burn_in, keep_fn, min_obs, probation_every):
    """Shared select_fn machinery for the cumulative gates (sign / z variants).

    r < burn_in -> uniform k-sample (bit-identical draw to the `_fedavg_core`
    default).  After burn-in: eligible = under-observed (n_obs < min_obs) OR
    keep_fn(acc)-true clients.  Full participation (k >= n) returns the eligible
    set as-is (VARIABLE length -- `_fedavg_core` aggregates whatever it gets);
    partial samples k of them.  Empty eligible -> full-cohort fallback + flag.
    Every `probation_every`-th round one excluded client rotates back in (a
    score-refresh chance -- exclusion must not be absorbing); under partial
    participation it replaces the last sampled slot so the cohort stays k.
    """
    state = {"i": 0}

    def select_fn(r, k, rng):
        n = len(acc.cum)
        if r < burn_in:
            return rng.choice(n, size=min(k, n), replace=False)
        keep = keep_fn(acc)
        eligible = [i for i in range(n) if acc.n_obs[i] < min_obs or keep[i]]
        if not eligible:
            print(f"[gate-select] r={r}: eligible empty -> full-cohort fallback", flush=True)
            return rng.choice(n, size=min(k, n), replace=False)
        excluded = sorted(set(range(n)) - set(eligible))
        if k >= n:
            sel = list(eligible)
        else:
            sel = [int(x) for x in rng.choice(eligible, size=min(k, len(eligible)),
                                              replace=False)]
        if probation_every and excluded and (r - burn_in) % probation_every == 0:
            p = excluded[state["i"] % len(excluded)]          # rotate through excluded ids
            state["i"] += 1
            if p not in sel:
                if k >= n or len(sel) < k:
                    sel.append(p)
                else:
                    sel[-1] = p                               # keep the cohort at k
        return np.array(sorted(sel))

    return select_fn


def make_signgate_select_fn(acc, burn_in, tau=0.0, min_obs=2, probation_every=5):
    """V2 participation gate: exclude clients whose CUMULATIVE contribution <= tau
    (tau=0 strict -- exact-0 free-riders are out, every all-positive clean client
    stays in).  See `_gate_select_fn` for burn-in / min_obs / probation semantics."""
    return _gate_select_fn(acc, burn_in, lambda a: a.cum > tau, min_obs, probation_every)


def make_zgate_select_fn(acc, burn_in, c=1.5, min_obs=2, probation_every=5):
    """V2 cohort-RELATIVE gate (the auxiliary policy, noisy recovery): exclude
    clients whose cum z-score < -c among the >= min_obs-observed cohort.  Unlike
    the absolute tau=0 gate this always grades on a curve -- it can fire on clean
    heterogeneity (the audit's do-no-harm cost, measured not assumed)."""
    def keep(a):
        keep = np.ones(len(a.cum), dtype=bool)
        obs = a.n_obs >= min_obs
        if obs.sum() >= 2:
            keep[obs] = _zscores(a.cum[obs]) >= -c
        return keep
    return _gate_select_fn(acc, burn_in, keep, min_obs, probation_every)


def _gated_weights(players, nums, keep_mask, r, raw, sink, gate_name):
    """Shared aggregation-weight tail: n-weight the kept participants; all-dropped
    -> vanilla n-weights fallback + flag.  Returns the normalized {client: w}."""
    w = np.array([nums[p] if keep_mask[i] else 0.0 for i, p in enumerate(players)],
                 dtype=float)
    fallback = bool(w.sum() <= 0)
    if fallback:
        print(f"[{gate_name}] r={r}: all participants gated out -> vanilla-weight "
              f"fallback", flush=True)
        w = np.array([nums[p] for p in players], dtype=float)
    w /= w.sum()
    wmap = dict(zip(players, w))
    if sink is not None:
        sink(r, players, raw, wmap, fallback)
    return wmap


def make_signgate_weights_fn(acc, raw_fn, nums, tau=0.0, sink=None):
    """V1 aggregation gate (and the V2 probation screen): score the CURRENT round
    with raw_fn (contribution-oriented), fold into `acc`, then aggregate only the
    deltas with raw > tau -- weight_i ~ n_i * 1[raw_i > tau], renormalized.
    Everyone trains under V1 (pair with select_fn=None); under V2 the same fn
    screens probation returnees by their same-round raw.  All gated out ->
    vanilla n-weights + flag (never a zero-sum round)."""
    def weights_fn(r, w_r, deltas_map):
        players = sorted(deltas_map)
        raw = raw_fn(w_r, deltas_map, players)
        acc.update(players, raw)
        return _gated_weights(players, nums, [v > tau for v in raw], r, raw, sink,
                              "signgate")
    return weights_fn


def make_zgate_weights_fn(acc, raw_fn, nums, c=1.5, sink=None):
    """Cohort-relative V1/V2 screen: drop the round's raw z-score < -c deltas
    (z over the round's participants; flat rounds drop nobody)."""
    def weights_fn(r, w_r, deltas_map):
        players = sorted(deltas_map)
        raw = raw_fn(w_r, deltas_map, players)
        acc.update(players, raw)
        return _gated_weights(players, nums, _zscores(raw) >= -c, r, raw, sink, "zgate")
    return weights_fn


def make_gatedweight_weights_fn(acc, raw_fn, nums, tau=0.0, alpha=1.0, sink=None):
    """V2w gate + magnitude-proportional weighting (Yonghee 2026-07-19): among the
    included (cum > tau) participants, weight_i ~ n_i * max(cum_i, 0)^alpha --
    negatives excluded, positives weighted by cumulative-contribution SIZE.
    alpha fixed 1.0 (no tuning -- parameter-free claim).  vs the existing
    `flirds_w` (min-max EMA): the zero point here is ABSOLUTE (phi=0, game
    semantics), not the round minimum -- V2w intervenes even on clean cells
    (the magnitude slope), which is exactly its P1 do-no-harm question.
    Included empty -> vanilla n-weights + flag."""
    def weights_fn(r, w_r, deltas_map):
        players = sorted(deltas_map)
        raw = raw_fn(w_r, deltas_map, players)
        acc.update(players, raw)
        w = np.array([nums[p] * max(acc.cum[p], 0.0) ** alpha if acc.cum[p] > tau
                      else 0.0 for p in players], dtype=float)
        fallback = bool(w.sum() <= 0)
        if fallback:
            print(f"[gatedweight] r={r}: included empty -> vanilla-weight fallback",
                  flush=True)
            w = np.array([nums[p] for p in players], dtype=float)
        w /= w.sum()
        wmap = dict(zip(players, w))
        if sink is not None:
            sink(r, players, raw, wmap, fallback)
        return wmap
    return weights_fn


def make_rawweight_weights_fn(acc, raw_fn, nums, tau=0.0, alpha=1.0, sink=None):
    """V1w (CNN-only ablation): PER-ROUND-raw magnitude weighting -- everyone
    trains, weight_i ~ n_i * max(raw_i, 0)^alpha (negatives drop out via max).
    The instantaneous-signal counterpart of make_gatedweight_weights_fn's
    cumulative weighting.  All non-positive -> vanilla n-weights + flag."""
    def weights_fn(r, w_r, deltas_map):
        players = sorted(deltas_map)
        raw = raw_fn(w_r, deltas_map, players)
        acc.update(players, raw)
        w = np.array([nums[p] * max(float(raw[i]), 0.0) ** alpha
                      for i, p in enumerate(players)], dtype=float)
        fallback = bool(w.sum() <= 0)
        if fallback:
            print(f"[rawweight] r={r}: no positive raw -> vanilla-weight fallback",
                  flush=True)
            w = np.array([nums[p] for p in players], dtype=float)
        w /= w.sum()
        wmap = dict(zip(players, w))
        if sink is not None:
            sink(r, players, raw, wmap, fallback)
        return wmap
    return weights_fn


def make_fixed_excl_select_fn(n_clients, excluded):
    """oracle_excl / random_excl control arms: a FIXED excluded set from round 0.
    Full participation returns the kept set (variable length); partial samples k
    of it via the core's rng (reproducible)."""
    keep = [i for i in range(n_clients) if i not in set(excluded)]

    def select_fn(r, k, rng):
        if k >= len(keep):
            return np.array(keep)
        return rng.choice(keep, size=k, replace=False)

    return select_fn


def make_observer_weights_fn(acc, raw_fn, nums, sink=None):
    """NO-intervention raw observer (the vanilla arm's per-round logger): score the
    round, fold into `acc`, log via `sink`, and return PLAIN n-weights -- the same
    values the `_fedavg_core` default computes, so the trajectory is bit-identical
    to vanilla while producing the clean-cell per-round false-fire record."""
    def weights_fn(r, w_r, deltas_map):
        players = sorted(deltas_map)
        raw = raw_fn(w_r, deltas_map, players)
        acc.update(players, raw)
        w = np.array([nums[p] for p in players], dtype=float)
        w /= w.sum()
        wmap = dict(zip(players, w))
        if sink is not None:
            sink(r, players, raw, wmap, False)
        return wmap
    return weights_fn


# --------------------------------------------------------------------------- #
# P5 -- confidence-aware sign policies (2026-07-21, Yonghee).  ADDITIVE ONLY.  #
#                                                                             #
# Motivation (track_h observer replay, overview §3.2.6 진단): the strict sign  #
# gate charges borderline VARIANCE (a zero-mean client's cum is a random walk #
# that sits <= 0 half the time) exactly like confident harm.  P5 decides on   #
# the client's own online evidence instead:                                   #
#   t_i = cum_i / (sd_i * sqrt(n_i))      (all from THIS run's stream only)   #
#   hard (cgate) : exclude iff cum + z*sd*sqrt(n) <= 0  (UCB <= 0), z=1.645   #
#   soft (pweight): weight_i ~ n_i * Phi(t_i)  ("P(contribution > 0)")        #
# FAIRNESS CLAUSE: z is a universal constant (one-sided 95%), identical for   #
# every score source and every cell; sd/n/cum come only from what the run     #
# observes during training -- no calibration runs, no oracle masks, no        #
# cross-run information.  Absolute zero is kept (t is centered at phi=0, not  #
# cohort-relative): a deterministic exact-0 stream (sd=0, cum=0) has UCB=0    #
# -> excluded / weight 0, so the frzero semantics survive unchanged.          #
# --------------------------------------------------------------------------- #
from math import erf as _erf


def _phi_cdf(t):
    """Standard normal CDF; +/-inf-safe."""
    if t == float("inf"):
        return 1.0
    if t == float("-inf"):
        return 0.0
    return 0.5 * (1.0 + _erf(t / np.sqrt(2.0)))


def _conf_keep(acc, z, min_obs):
    """Keep mask under the confidence gate: keep unless the client's sum UCB
    (cum + z*sd*sqrt(n)) is <= 0.  Under-observed (n < min_obs) or sd-undefined
    clients are kept -- no evidence, no exclusion (the no-prior-info default).
    sd == 0 degenerates to the strict sign rule (UCB == cum), so an exact-0
    free-rider is excluded deterministically."""
    _, sd, n = acc.stats()
    keep = np.ones(len(acc.cum), dtype=bool)
    for c in range(len(acc.cum)):
        if n[c] < max(min_obs, 2):
            continue                                   # keep: not enough evidence
        keep[c] = (acc.cum[c] + z * sd[c] * np.sqrt(n[c])) > 0.0
    return keep


def make_confgate_select_fn(acc, burn_in, z=1.645, min_obs=2, probation_every=5):
    """P5-hard participation gate: exclude only clients whose cumulative
    contribution is SIGNIFICANTLY <= 0 at level z (one-sided; UCB rule above).
    Same burn-in / min_obs / probation machinery as the V2 gates.  NOTE the
    paired weights_fn is the plain observer (score + n-weights, NO same-round
    raw screen): P5 forbids acting on single-round noise by definition."""
    return _gate_select_fn(acc, burn_in, lambda a: _conf_keep(a, z, min_obs),
                           min_obs, probation_every)


def make_probweight_weights_fn(acc, raw_fn, nums, burn_in=10, min_obs=2, sink=None):
    """P5-soft: score the round, fold into `acc`, then weight the participants
    w_i ~ n_i * Phi(t_i) -- each client's data weight scaled by the probability
    (normal approx, own online stream) that its true mean contribution is
    positive.  Direction-symmetric by construction: confident-positive -> full
    n-weight, confident-negative -> ~0, no evidence (t=0) -> exactly half.
    Conventions: r < burn_in or n_obs < min_obs -> neutral factor 1 (FedAvg
    default -- no deviation without evidence); sd == 0 -> deterministic stream,
    factor 1 if cum > 0 else 0 (exact-0 excluded);  all-zero mass -> n-weight
    fallback + flag (same guard as the other weight fns)."""
    def weights_fn(r, w_r, deltas_map):
        players = sorted(deltas_map)
        raw = raw_fn(w_r, deltas_map, players)
        acc.update(players, raw)
        _, sd, n = acc.stats()
        fac = np.ones(len(players))
        if r >= burn_in:
            for i, p in enumerate(players):
                if n[p] < max(min_obs, 2):
                    continue                            # neutral: no evidence yet
                if sd[p] <= 0.0:
                    fac[i] = 1.0 if acc.cum[p] > 0 else 0.0
                else:
                    fac[i] = _phi_cdf(acc.cum[p] / (sd[p] * np.sqrt(n[p])))
        w = np.array([nums[p] * fac[i] for i, p in enumerate(players)], dtype=float)
        fallback = bool(w.sum() <= 0)
        if fallback:
            print(f"[probweight] r={r}: zero total mass -> vanilla-weight fallback",
                  flush=True)
            w = np.array([nums[p] for p in players], dtype=float)
        w /= w.sum()
        wmap = dict(zip(players, w))
        if sink is not None:
            sink(r, players, raw, wmap, fallback)
        return wmap
    return weights_fn


def lossheur_round_raw_fn(loss_fn, pkeys, n_clients, device):
    """`round_raw_fn` closure backed by the loss-heuristic singleton utilities
    (oracle.in_run_sv.in_run_singletons, the C6 base-loss-cached path) on the
    single-round log -- 1 + |P_r| forwards.  Same sign flip as flirds_round_raw
    (U_(b)({k}) is good->low): contribution-oriented raw, exact 0.0 for a zero
    delta (the strict->0 free-rider rule holds bit-exactly)."""
    from ..oracle.in_run_sv import in_run_singletons

    def fn(w_r, deltas_map, players):
        phi = in_run_singletons([(w_r, deltas_map)], n_clients, loss_fn, pkeys, device)
        return [-phi[p] for p in players]
    return fn
