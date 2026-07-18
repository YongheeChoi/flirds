"""Track G unit tests -- fl.intervene sign-gating machinery (§4.2 of the Track G spec).

The experiment's top risk is a SIGN-FLIP bug (D-3): stored phi is suspicion-oriented
(helpful -> LOW), the gate operates on CONTRIBUTION-oriented raw (helpful -> HIGH,
include cum > tau).  These tests pin the flip direction end-to-end on a synthetic
quadratic game where the truth is known analytically:

    loss(w) = 0.5 * ||w - t||^2   (grad = w - t, Hessian = I)
    helpful delta  = +eps * (t - w)  -> phi < 0 -> raw > 0  (included)
    harmful delta  = -eps * (t - w)  -> phi > 0 -> raw < 0  (excluded)
    zero delta (frzero) -> raw == 0.0 bit-exact -> excluded under strict > 0

No GPU, no dataset, plain asserts (the test_removal_cnn convention).  From codes/:
    PYTHONPATH=. python tests/test_signgate.py
"""
import numpy as np
import torch

from flirds.fl.intervene import (OnlineScorer, SignAccumulator, flirds_round_raw_fn,
                                 lossheur_round_raw_fn, make_fixed_excl_select_fn,
                                 make_gatedweight_weights_fn, make_observer_weights_fn,
                                 make_rawweight_weights_fn, make_signgate_select_fn,
                                 make_signgate_weights_fn, make_zgate_select_fn,
                                 make_zgate_weights_fn)
from flirds.fl.server import _fedavg_core

T = torch.tensor([1.0, -2.0])                      # quadratic target
PKEYS = ["w"]


def loss_fn(params, buffers):
    return 0.5 * ((params["w"] - T) ** 2).sum()


def _round(w, deltas):
    """(w_r, deltas_map) for one round; equal client sizes n=10."""
    return {"w": torch.tensor(w)}, {c: ({"w": torch.tensor(d)}, 10)
                                    for c, d in deltas.items()}


# --------------------------------------------------------------------------- #
# 1. raw sign direction (the D-3 flip, pinned)                                #
# --------------------------------------------------------------------------- #
def test_raw_sign_direction():
    """helpful -> raw > 0, harmful -> raw < 0, zero -> raw == 0.0 bit-exact,
    for BOTH round_raw providers (flirds estimator + loss-heur singletons)."""
    w = [0.0, 0.0]
    d_help = list(0.1 * (T - torch.tensor(w)).numpy())          # toward the target
    d_harm = list(-0.1 * (T - torch.tensor(w)).numpy())         # away from it
    w_r, dm = _round(w, {0: d_help, 1: d_harm, 2: [0.0, 0.0]})
    for name, fn in (("flirds", flirds_round_raw_fn(loss_fn, PKEYS, 3, "cpu")),
                     ("lossheur", lossheur_round_raw_fn(loss_fn, PKEYS, 3, "cpu"))):
        raw = fn(w_r, dm, [0, 1, 2])
        assert raw[0] > 0, f"{name}: helpful delta must have POSITIVE raw, got {raw[0]}"
        assert raw[1] < 0, f"{name}: harmful delta must have NEGATIVE raw, got {raw[1]}"
        assert raw[2] == 0.0, f"{name}: zero delta must be EXACT 0.0, got {raw[2]!r}"


def test_frzero_strict_gt_excludes():
    """frzero raw == 0.0 -> strict > 0 gives weight 0; helpful keeps its weight."""
    acc = SignAccumulator(3)
    fn = make_signgate_weights_fn(acc, flirds_round_raw_fn(loss_fn, PKEYS, 3, "cpu"),
                                  [10, 10, 10])
    w = [0.0, 0.0]
    w_r, dm = _round(w, {0: list(0.1 * (T - torch.tensor(w)).numpy()),
                         1: [0.0, 0.0], 2: [0.0, 0.0]})
    wmap = fn(0, w_r, dm)
    assert wmap[0] == 1.0 and wmap[1] == 0.0 and wmap[2] == 0.0
    assert acc.cum[1] == 0.0 and acc.n_obs[1] == 1        # exact-0 accumulated, observed


# --------------------------------------------------------------------------- #
# 2. SignAccumulator vs OnlineScorer (sign preservation)                      #
# --------------------------------------------------------------------------- #
def test_accumulator_preserves_negative():
    """OnlineScorer min-maxes the sign away (round minimum -> 0 regardless);
    SignAccumulator keeps it -- the reason it exists."""
    acc, sc = SignAccumulator(2), OnlineScorer(2, beta=0.5)
    acc.update([0, 1], [-5.0, 3.0])
    sc.update([0, 1], [-5.0, 3.0])
    assert acc.cum[0] == -5.0 and acc.cum[1] == 3.0
    assert sc.s[0] == 0.0                                  # min-max destroyed the sign
    acc.update([0], [2.0])
    assert acc.cum[0] == -3.0 and acc.n_obs[0] == 2 and acc.n_obs[1] == 1


def test_accumulator_decay():
    acc = SignAccumulator(1, decay=0.5)
    acc.update([0], [4.0])
    acc.update([0], [1.0])
    assert acc.cum[0] == 0.5 * 4.0 + 1.0


# --------------------------------------------------------------------------- #
# 3. select gates: burn-in / min_obs / variable length / probation / fallback #
# --------------------------------------------------------------------------- #
def test_signgate_select():
    acc = SignAccumulator(4)
    acc.cum[:] = [1.0, -1.0, 0.0, 2.0]
    acc.n_obs[:] = [5, 5, 5, 0]                            # client 3 under-observed
    sel_fn = make_signgate_select_fn(acc, burn_in=2, tau=0.0, min_obs=2,
                                     probation_every=5)
    rng = np.random.default_rng(0)
    assert len(sel_fn(0, 4, rng)) == 4                     # burn-in: everyone
    # r=2 = burn_in -> gate on; (r-burn_in)%5==0 -> probation fires too.
    # eligible = {0 (cum>0), 3 (n_obs<min_obs)}; excluded = {1 (neg), 2 (EXACT 0 -> out)}
    s = list(sel_fn(2, 4, rng))
    assert 2 not in s or s == [0, 1, 2, 3], f"cum==0 must be gated out, got {s}"
    assert set(s) >= {0, 3} and 1 in s                     # probation returned client 1
    s = list(sel_fn(3, 4, rng))
    assert set(s) == {0, 3}                                # off-probation: eligible only
    s = list(sel_fn(7, 4, rng))                            # next probation -> rotates to 2
    assert set(s) == {0, 2, 3}


def test_signgate_select_partial_and_fallback():
    acc = SignAccumulator(6)
    acc.cum[:] = [1, 1, 1, 1, -1, -1]
    acc.n_obs[:] = 5
    sel_fn = make_signgate_select_fn(acc, burn_in=0, tau=0.0, min_obs=2,
                                     probation_every=5)
    rng = np.random.default_rng(0)
    s = list(sel_fn(1, 2, rng))                            # partial: k=2 of 4 eligible
    assert len(s) == 2 and set(s) <= {0, 1, 2, 3}
    s = list(sel_fn(5, 2, rng))                            # probation round: k kept
    assert len(s) == 2 and (4 in s or 5 in s)              # one excluded rotated in
    acc.cum[:] = -1.0                                      # nobody eligible
    s = list(sel_fn(2, 4, rng))
    assert len(s) == 4                                     # full-cohort fallback


def test_zgate_select():
    acc = SignAccumulator(4)
    acc.cum[:] = [10.0, 10.0, 10.0, -20.0]                 # z(-20) ~ -1.73 < -1.5
    acc.n_obs[:] = 5
    sel_fn = make_zgate_select_fn(acc, burn_in=0, c=1.5, min_obs=2, probation_every=0)
    s = list(sel_fn(1, 4, np.random.default_rng(0)))
    assert set(s) == {0, 1, 2}
    # all-positive clean spread at N=4 stays inside z >= -1.5 -> nobody excluded
    acc.cum[:] = [3.0, 4.0, 5.0, 6.0]
    assert set(sel_fn(2, 4, np.random.default_rng(0))) == {0, 1, 2, 3}


# --------------------------------------------------------------------------- #
# 4. weights fns: screens, fallback flag, V2w magnitude weighting, observer   #
# --------------------------------------------------------------------------- #
def _const_raw_fn(vals):
    return lambda w_r, dm, players: [vals[p] for p in players]


def test_weights_fallback_flag():
    acc = SignAccumulator(2)
    rows = []
    fn = make_signgate_weights_fn(acc, _const_raw_fn({0: -1.0, 1: -2.0}), [10, 30],
                                  sink=lambda *a: rows.append(a))
    wmap = fn(0, None, {0: (None, 10), 1: (None, 30)})
    assert abs(wmap[0] - 0.25) < 1e-12 and abs(wmap[1] - 0.75) < 1e-12   # vanilla fallback
    assert rows[0][4] is True                              # fallback flagged to the sink


def test_gatedweight_v2w():
    acc = SignAccumulator(3)
    acc.update([0, 1, 2], [2.0, 1.0, -1.0])                # cum = [2, 1, -1]
    fn = make_gatedweight_weights_fn(acc, _const_raw_fn({0: 0.0, 1: 0.0, 2: 0.0}),
                                     [10, 10, 10], alpha=1.0)
    wmap = fn(0, None, {c: (None, 10) for c in range(3)})
    assert wmap[2] == 0.0                                  # negative cum excluded
    assert abs(wmap[0] - 2.0 / 3.0) < 1e-12 and abs(wmap[1] - 1.0 / 3.0) < 1e-12
    # magnitude-proportional: w0/w1 == cum0/cum1 (equal n)


def test_zgate_weights_round_screen():
    acc = SignAccumulator(4)
    fn = make_zgate_weights_fn(acc, _const_raw_fn({0: 1.0, 1: 1.0, 2: 1.0, 3: -5.0}),
                               [10] * 4, c=1.5)
    wmap = fn(0, None, {c: (None, 10) for c in range(4)})
    assert wmap[3] == 0.0 and abs(sum(wmap.values()) - 1.0) < 1e-12


def test_rawweight_v1w():
    """Per-round-raw magnitude weighting: negatives to 0, positives ~ raw size."""
    acc = SignAccumulator(3)
    fn = make_rawweight_weights_fn(acc, _const_raw_fn({0: 3.0, 1: 1.0, 2: -2.0}),
                                   [10, 10, 10], alpha=1.0)
    wmap = fn(0, None, {c: (None, 10) for c in range(3)})
    assert wmap[2] == 0.0
    assert abs(wmap[0] - 0.75) < 1e-12 and abs(wmap[1] - 0.25) < 1e-12
    fn2 = make_rawweight_weights_fn(acc, _const_raw_fn({0: -1.0, 1: -1.0, 2: -1.0}),
                                    [10, 20, 30])
    wmap2 = fn2(0, None, {c: (None, 10) for c in range(3)})
    assert abs(sum(wmap2.values()) - 1.0) < 1e-12 and wmap2[2] == 0.5   # vanilla fallback


def test_fixed_excl_select():
    sel = make_fixed_excl_select_fn(5, {1, 3})
    assert list(sel(0, 5, np.random.default_rng(0))) == [0, 2, 4]      # full part: kept as-is
    s = list(sel(0, 2, np.random.default_rng(0)))                      # partial: sample kept
    assert len(s) == 2 and set(s) <= {0, 2, 4}


def test_observer_is_vanilla():
    """Observer returns the EXACT default n-weights (bit-identical trajectory)."""
    acc = SignAccumulator(3)
    rows = []
    fn = make_observer_weights_fn(acc, _const_raw_fn({0: -9.0, 1: 0.0, 2: 9.0}),
                                  [10, 20, 30], sink=lambda *a: rows.append(a))
    wmap = fn(0, None, {c: (None, 10) for c in range(3)})
    ref = np.array([10.0, 20.0, 30.0]); ref /= ref.sum()
    assert all(wmap[c] == ref[c] for c in range(3))        # same float ops as the core default
    assert acc.cum[0] == -9.0 and rows and rows[0][4] is False


# --------------------------------------------------------------------------- #
# 5. _fedavg_core accepts a variable-length selection (len(sel) != k)         #
# --------------------------------------------------------------------------- #
def test_core_variable_length_selection():
    init = {"w": torch.zeros(2)}
    nums = [10, 10, 10, 10]
    deltas = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0]),
              2: torch.tensor([5.0, 5.0]), 3: torch.tensor([-5.0, -5.0])}
    seen = []
    final, _ = _fedavg_core(init, lambda c, gs: {"w": deltas[c].clone()}, nums,
                            rounds=1, sample_frac=1.0, seed=0,
                            on_round=lambda r, w_r, dm: seen.append(sorted(dm)),
                            select_fn=lambda r, k, rng: np.array([0, 1]))
    assert seen == [[0, 1]]                                # only the returned 2 trained
    assert torch.allclose(final["w"], torch.tensor([0.5, 0.5]))   # n-renorm over the 2
    final1, _ = _fedavg_core(init, lambda c, gs: {"w": deltas[c].clone()}, nums,
                             rounds=1, sample_frac=1.0, seed=0,
                             select_fn=lambda r, k, rng: np.array([2]))
    assert torch.allclose(final1["w"], torch.tensor([5.0, 5.0]))  # single-client round


# --------------------------------------------------------------------------- #
# 6. end-to-end V2 on the synthetic quadratic: frzero excluded after burn-in, #
#    probation rotates it back (screened to weight 0), helpers never gated    #
# --------------------------------------------------------------------------- #
def test_v2_end_to_end_synthetic():
    n, burn_in, prob = 4, 2, 3
    acc = SignAccumulator(n)
    nums = [10] * n
    raw_fn = flirds_round_raw_fn(loss_fn, PKEYS, n, "cpu")
    rows = []
    sink = lambda r, players, raw, wmap, fb: rows.extend(
        dict(round=r, client=p, raw=float(raw[i]), weight=float(wmap[p]), fallback=fb)
        for i, p in enumerate(players))
    sel_fn = make_signgate_select_fn(acc, burn_in=burn_in, tau=0.0, min_obs=2,
                                     probation_every=prob)
    wts_fn = make_signgate_weights_fn(acc, raw_fn, nums, tau=0.0, sink=sink)

    def local_train_fn(c, gs):
        if c == 3:                                         # frzero client
            return {"w": torch.zeros(2)}
        return {"w": 0.1 * (T - gs["w"])}                  # helpful pull toward target

    cohorts = []
    final, _ = _fedavg_core({"w": torch.zeros(2)}, local_train_fn, nums, rounds=8,
                            sample_frac=1.0, seed=0,
                            on_round=lambda r, w_r, dm: cohorts.append(sorted(dm)),
                            select_fn=sel_fn, weights_fn=wts_fn)
    # burn-in rounds 0,1: everyone; gate on from r=2; probation at r=2,5 (r-2)%3==0
    assert cohorts[0] == cohorts[1] == [0, 1, 2, 3]
    for r in range(2, 8):
        expect = [0, 1, 2, 3] if (r - burn_in) % prob == 0 else [0, 1, 2]
        assert cohorts[r] == expect, f"round {r}: cohort {cohorts[r]} != {expect}"
    # probation returnee is SCREENED: its same-round raw == 0 -> aggregate weight 0
    fr_rows = [x for x in rows if x["client"] == 3]
    assert all(x["raw"] == 0.0 and x["weight"] == 0.0 for x in fr_rows)
    assert {x["round"] for x in fr_rows} == {0, 1, 2, 5}   # trained only burn-in+probation
    helper_rows = [x for x in rows if x["client"] != 3]
    assert all(x["raw"] > 0 and x["weight"] > 0 for x in helper_rows)   # no false gating
    assert not any(x["fallback"] for x in rows)
    assert ((final["w"] - T).abs() < (torch.zeros(2) - T).abs()).all()  # training progressed


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASS")
