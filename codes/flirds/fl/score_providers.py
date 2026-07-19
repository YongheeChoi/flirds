"""Track H score providers -- per-round raw contribution for every competing
valuation method, one shared signature `fn(w_r, deltas_map, players) -> raw`
(CONTRIBUTION-oriented, helpful -> HIGH), pluggable into the fl.intervene
gate/weight machinery (spec runs/track_h/README.md §1).

All providers use the val-loss game (loss_fn, pkeys) on BOTH backends -- the
canonical stage convention (track_c1 and phase2_matrix score every method on
val-loss with the exact settings round_trunc=0, eps=0 / trunc_eps=0; reused
verbatim here).  Value-only forwards (no grad) except flirds/flirds1st, which
reuse the estimator path from fl.intervene.

Orientation: GTG/FedSV marginals are in val-loss units (helpful -> negative)
and are negated here; the ComFedSV surrogate utility is the loss DECREASE
u(S) = loss(base) - loss(S) (paper Eq.6), whose marginals are already
helpful -> HIGH.  A zero delta is NOT exact-0 under the coalition providers
(within-subset renormalization / uniform-average dilution) -- that is the
zero-semantics property Track H measures, not a bug.

ComFedSV caveat (Yonghee 2026-07-19 decision): ComFedSV as published is a
CROSS-ROUND method (low-rank completion over the whole trajectory); a
per-round score does not exist in its definition.  The provider here is the
method's own per-round core (uniform-average submodel + loss-decrease
utility, permutation-MC) with the completion omitted -- an online gate only
needs the round's cohort, where every prefix coalition is observable.
Report label: comfedsv (per-round surrogate; completion omitted).

Determinism: the MC providers (gtg/fedsv/comfedsv) carry one persistent
np.random.default_rng(seed) across rounds (one provider instance per arm,
called once per round -> reproducible for a given seed and round order).
"""
from __future__ import annotations

import numpy as np

from ..baselines.comfedsv import _llm_util
from ..baselines.fedsv import _round_permutation_sv
from ..baselines.gtg import _RoundGTG, _round_metrics

SOURCES = ("flirds", "flirds1st", "lossheur", "gtg", "fedsv", "comfedsv",
           "shapleyfl", "fedif")


def gtg_round_raw_fn(loss_fn, pkeys, device, seed=0, round_trunc=0.0, eps=0.0):
    """GTG guided-truncation MC Shapley on the single round (efficiency-
    normalized, the method's own convention -- the renorm!=exact-0 semantics
    under test).  A round with |this-last| <= round_trunc contributes 0
    (gtg_from_logs skips it)."""
    rng = np.random.default_rng(seed)

    def fn(w_r, deltas_map, players):
        last_m, this_m, metric_fun = _round_metrics(w_r, deltas_map, players,
                                                    None, None, device, loss_fn, pkeys)
        if abs(this_m - last_m) <= round_trunc:
            return [0.0] * len(players)
        rsv = _RoundGTG(len(players), last_m, this_m, metric_fun, eps=eps).compute(rng)
        return [-float(v) for v in rsv]                # loss units -> contribution
    return fn


def fedsv_round_raw_fn(loss_fn, pkeys, device, seed=0, n_perm=None, trunc_eps=0.0):
    """FedSV permutation-MC Shapley on the single round (canonical n_perm =
    max(30, 2k), no truncation -- the track_c1/phase2 settings)."""
    rng = np.random.default_rng(seed)

    def fn(w_r, deltas_map, players):
        m = n_perm or max(30, 2 * len(players))
        last_m, full_m, metric_fun = _round_metrics(w_r, deltas_map, players,
                                                    None, None, device, loss_fn, pkeys)
        rsv = _round_permutation_sv(len(players), last_m, full_m, metric_fun, m, rng,
                                    trunc_eps=trunc_eps)
        return [-float(v) for v in rsv]                # loss units -> contribution
    return fn


def comfedsv_round_raw_fn(loss_fn, pkeys, device, seed=0, n_perm=None):
    """ComFedSV per-round surrogate (see module docstring): permutation-MC
    Shapley of u(S) = loss(base) - loss(uniform-average submodel of S)
    (paper Eq.6 per-round utility; u(empty) = 0 by definition).  Marginals in
    u units are already contribution-oriented."""
    rng = np.random.default_rng(seed)

    def fn(w_r, deltas_map, players):
        m = n_perm or max(30, 2 * len(players))
        base = _llm_util(w_r, deltas_map, [], pkeys, loss_fn, device)

        def metric_fun(sub_idx):
            return base - _llm_util(w_r, deltas_map, [players[i] for i in sub_idx],
                                    pkeys, loss_fn, device)

        full = metric_fun(tuple(range(len(players))))
        rsv = _round_permutation_sv(len(players), 0.0, full, metric_fun, m, rng,
                                    trunc_eps=0.0)
        return [float(v) for v in rsv]                 # u units, already good->HIGH
    return fn


def provider_round_raw_fn(src, loss_fn, pkeys, n_clients, device, seed=0,
                          loss_chunks=None):
    """One factory for every Track H score source (shared signature).  flirds /
    flirds1st / lossheur / fedif reuse the existing fl.intervene closures;
    pass `loss_chunks` at LLM scale for the estimator HVP (value-only providers
    ignore it -- their full-val forward fits without chunking, cf. oracleb)."""
    from .intervene import (fedif_round_raw_fn, flirds_round_raw_fn,
                            lossheur_round_raw_fn)
    if src == "flirds":
        return flirds_round_raw_fn(loss_fn, pkeys, n_clients, device,
                                   loss_chunks=loss_chunks)
    if src == "flirds1st":
        return flirds_round_raw_fn(loss_fn, pkeys, n_clients, device,
                                   second_order=False, loss_chunks=loss_chunks)
    if src == "lossheur":
        return lossheur_round_raw_fn(loss_fn, pkeys, n_clients, device)
    if src == "fedif":
        return fedif_round_raw_fn(loss_fn, pkeys, device, loss_chunks=loss_chunks)
    if src == "shapleyfl":
        from ..baselines.shapleyfl import shapleyfl_round_raw

        def sfl_fn(w_r, deltas_map, players):
            sv = shapleyfl_round_raw(w_r, deltas_map, players, None, None, device,
                                     loss_fn, pkeys)
            return [float(v) for v in np.asarray(sv, dtype=float)]
        return sfl_fn                                  # UN-normalized raw (track_g precedent)
    if src == "gtg":
        return gtg_round_raw_fn(loss_fn, pkeys, device, seed=seed)
    if src == "fedsv":
        return fedsv_round_raw_fn(loss_fn, pkeys, device, seed=seed)
    if src == "comfedsv":
        return comfedsv_round_raw_fn(loss_fn, pkeys, device, seed=seed)
    raise ValueError(f"unknown score source {src!r} (use one of {SOURCES})")
