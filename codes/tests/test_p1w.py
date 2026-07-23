"""P1w (=P2, size-weighted sign gate) T2-retrain unit tests
(spec paper/workplan/T3-p1w-llm-impl.md, the L7 leg).

The ONLINE P1w gate (make_gatedweight_weights_fn / flirds_gatew_v2) is already
pinned by test_signgate.test_gatedweight_v2w; these cover the T2/retrain twin that
track_g's T2_W block feeds the retrain leg:

  fl.intervene.signw_retrain_wvec  -- {client: max(cum,0)^alpha} over kept {cum>tau}
  fl.intervene.make_static_weights_fn -- the parameterized static weight fn shared
                                         by t2_pw (Phi(t)) and t2_signw (max cum)

Direction convention (D-3): cum is CONTRIBUTION-oriented (helpful -> POSITIVE); the
gate keeps cum > tau, so a frzero/exact-0 free-rider (cum == 0) drops out and the
kept SET equals P1's plain sign gate -- only the survivors' weighting differs.

No GPU, no dataset, plain asserts (the test_signgate convention).  From codes/:
    PYTHONPATH=. python tests/test_p1w.py
"""
import numpy as np
import torch

from flirds.fl.intervene import (SignAccumulator, make_gatedweight_weights_fn,
                                 make_static_weights_fn, signw_retrain_wvec,
                                 _phi_cdf)
from flirds.fl.server import _fedavg_core


def _const_raw_fn(vals):
    return lambda w_r, dm, players: [vals[p] for p in players]


def _agg_weights(wvec, nums, kept):
    """The aggregation weights the T2 retrain produces: re-index kept to subset
    positions 0..|kept|-1 (as the runner does), then run make_static_weights_fn on
    a full-participation round.  Returns {position: weight}."""
    ks = sorted(kept)
    fac = {i: float(wvec[c]) for i, c in enumerate(ks)}
    dm = {i: (None, nums[c]) for i, c in enumerate(ks)}     # (delta, n) tuples
    return make_static_weights_fn(fac)(0, None, dm), ks


# --------------------------------------------------------------------------- #
# 1. weight direction (larger positive cum -> larger w) + normalization       #
# --------------------------------------------------------------------------- #
def test_signw_direction_and_sum():
    cum = np.array([3.0, 1.0, 2.0])                         # all positive
    wvec = signw_retrain_wvec(cum, tau=0.0, alpha=1.0)
    assert set(wvec) == {0, 1, 2}
    wmap, _ = _agg_weights(wvec, [10, 10, 10], list(wvec))
    assert abs(sum(wmap.values()) - 1.0) < 1e-12
    assert wmap[0] > wmap[2] > wmap[1]                      # ordering follows cum
    assert abs(wmap[0] / wmap[1] - 3.0) < 1e-12            # w ~ cum at equal n


# --------------------------------------------------------------------------- #
# 2. frzero exact-0 (and negative) excluded; kept SET == P1 sign gate          #
# --------------------------------------------------------------------------- #
def test_signw_frzero_excluded_kept_eq_p1():
    cum = np.array([2.0, 0.0, 3.0, -1.0])                  # client1 = frzero (cum 0)
    wvec = signw_retrain_wvec(cum, tau=0.0, alpha=1.0)
    assert 1 not in wvec and 3 not in wvec                 # exact-0 AND negative out
    p1_kept = [i for i in range(len(cum)) if cum[i] > 0.0]  # the t2_sign kept formula
    assert sorted(wvec) == p1_kept == [0, 2]               # identical kept set
    wmap, ks = _agg_weights(wvec, [10, 10, 10, 10], list(wvec))
    assert ks == [0, 2] and abs(sum(wmap.values()) - 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# 3. clean all-positive -> w = normalized cum, NOT vanilla n-weights           #
#    (P1w intervenes on clean cells -- the do-no-harm question; unlike P1-T2   #
#     which keeps everyone unweighted == vanilla)                              #
# --------------------------------------------------------------------------- #
def test_signw_clean_differs_from_vanilla():
    cum = np.array([1.0, 3.0, 6.0])
    nums = [10, 10, 10]
    wvec = signw_retrain_wvec(cum, tau=0.0, alpha=1.0)
    wmap, _ = _agg_weights(wvec, nums, list(wvec))
    van = np.array(nums, dtype=float); van /= van.sum()
    got = np.array([wmap[i] for i in range(3)])
    assert not np.allclose(got, van)                       # NOT plain n-weights
    exp = np.array([1.0, 3.0, 6.0]); exp /= exp.sum()      # ~ cum at equal n
    assert np.allclose(got, exp)


# --------------------------------------------------------------------------- #
# 4. e2e: the static weights flow through the FL core as n*max(cum,0)          #
# --------------------------------------------------------------------------- #
def test_signw_e2e_aggregation():
    cum = np.array([3.0, 0.0, 1.0])                        # client1 excluded (cum 0)
    nums = [10, 10, 10]
    wvec = signw_retrain_wvec(cum, tau=0.0, alpha=1.0)
    ks = sorted(wvec)                                      # [0, 2] -> positions [0, 1]
    assert 1 not in ks
    fac = {i: float(wvec[c]) for i, c in enumerate(ks)}
    wf = make_static_weights_fn(fac)
    deltas = {0: torch.tensor([4.0]), 1: torch.tensor([0.0])}   # subset positions
    subnums = [nums[c] for c in ks]                        # [10, 10]
    final, _ = _fedavg_core({"w": torch.zeros(1)},
                            lambda c, gs: {"w": deltas[c].clone()}, subnums,
                            rounds=1, sample_frac=1.0, seed=0, weights_fn=wf)
    # weights ~ n*cum: pos0 10*3=30, pos1 10*1=10 -> 0.75/0.25;  agg = .75*4 + .25*0
    assert torch.allclose(final["w"], torch.tensor([3.0]))


# --------------------------------------------------------------------------- #
# 5. T2 retrain reproduces the ONLINE V2w gate's weights on the same cum       #
#    (the P1w "retrain == online" claim, spec T3)                              #
# --------------------------------------------------------------------------- #
def test_signw_matches_online_v2w():
    cum = np.array([2.0, 0.0, 1.0, -1.0])
    nums = [10, 20, 30, 40]
    acc = SignAccumulator(4); acc.update(range(4), cum)    # cum := cum, n_obs = 1
    online = make_gatedweight_weights_fn(acc, _const_raw_fn({c: 0.0 for c in range(4)}),
                                         nums, tau=0.0, alpha=1.0)
    w_online = online(0, None, {c: (None, nums[c]) for c in range(4)})   # raw 0 -> cum kept
    wvec = signw_retrain_wvec(cum, tau=0.0, alpha=1.0)
    w_t2 = make_static_weights_fn(dict(wvec))(0, None, {c: (None, nums[c])
                                                        for c in range(4)})
    for c in range(4):
        assert abs(w_online[c] - w_t2[c]) < 1e-12          # identical aggregation


# --------------------------------------------------------------------------- #
# 6. regression: the extracted static wf == n*fac (+n-fallback) and the P5s    #
#    (t2_pw) Phi(t) ingredients are unchanged                                  #
# --------------------------------------------------------------------------- #
def test_static_weights_and_p5s_regression():
    wf = make_static_weights_fn({0: 2.0, 1: 1.0})          # w ~ n*fac
    wmap = wf(0, None, {0: (None, 10), 1: (None, 10)})
    assert abs(wmap[0] - 2 / 3) < 1e-12 and abs(wmap[1] - 1 / 3) < 1e-12
    wf0 = make_static_weights_fn({0: 0.0, 1: 0.0})         # zero mass -> plain n-weights
    wmap0 = wf0(0, None, {0: (None, 10), 1: (None, 30)})
    assert abs(wmap0[0] - 0.25) < 1e-12 and abs(wmap0[1] - 0.75) < 1e-12
    # P5s (t2_pw) weight = Phi(cum / (sd*sqrt(n))): ingredients intact, positive
    # mean -> factor > 0.5, and the exact-0 deterministic stream -> factor 0.
    acc = SignAccumulator(2)
    for r in (2.0, 4.0, 0.0):                              # client0 positive w/ spread
        acc.update([0, 1], [r, 0.0])                       # client1 exact-0 stream
    _, sd, nob = acc.stats()
    assert nob[0] == 3 and sd[0] > 0 and sd[1] == 0.0
    assert _phi_cdf(acc.cum[0] / (sd[0] * np.sqrt(nob[0]))) > 0.5
    assert (1.0 if acc.cum[1] > 0 else 0.0) == 0.0         # sd==0, cum==0 -> excluded


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASS")
