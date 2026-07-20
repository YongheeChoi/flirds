"""P5 confidence-policy unit tests (spec runs/track_h/p5/RUN_P5.md).

The policies decide on each client's OWN online stream (mean/sd/n from
SignAccumulator.stats(); z = 1.645 universal) -- the fairness clause is
"training-observed statistics only".  The top risks: (1) stats math, (2) the
UCB keep rule's edge semantics (exact-0 excluded, borderline-negative kept,
confident-negative excluded), (3) Phi-weight edges, (4) T2 helpers + arm
parsing must not disturb legacy branches.

No GPU, no dataset.  From codes/:  PYTHONPATH=. python tests/test_p5.py
"""
import numpy as np

from flirds.fl.intervene import (SignAccumulator, _conf_keep, _phi_cdf,
                                 make_confgate_select_fn,
                                 make_probweight_weights_fn)

Z = 1.645


def _acc_from(streams, n_clients=None):
    """SignAccumulator fed round-by-round from {client: [raw...]} (equal lengths)."""
    n = n_clients or (max(streams) + 1)
    acc = SignAccumulator(n)
    rounds = max(len(v) for v in streams.values())
    for r in range(rounds):
        players = sorted(c for c, v in streams.items() if r < len(v))
        acc.update(players, [streams[p][r] for p in players])
    return acc


# --------------------------------------------------------------------------- #
# 1. stats math                                                               #
# --------------------------------------------------------------------------- #
def test_stats_mean_sd():
    acc = _acc_from({0: [1.0, 2.0, 3.0]})
    mean, sd, n = acc.stats()
    assert abs(mean[0] - 2.0) < 1e-12 and abs(sd[0] - 1.0) < 1e-12 and n[0] == 3


def test_stats_degenerate():
    acc = _acc_from({0: [0.0, 0.0], 1: [0.5]})
    mean, sd, n = acc.stats()
    assert sd[0] == 0.0 and acc.cum[0] == 0.0           # exact-0 stream
    assert sd[1] == 0.0 and n[1] == 1                   # n<2 -> sd defined as 0


# --------------------------------------------------------------------------- #
# 2. UCB keep rule (P5-hard)                                                  #
# --------------------------------------------------------------------------- #
def test_conf_keep_semantics():
    acc = _acc_from({
        0: [+1.0, +0.9, +1.1, +1.0],                    # confident positive -> keep
        1: [-1.0, +0.9, -0.8, +0.85],                   # borderline (cum=-0.05) -> KEEP
        2: [-1.0, -1.1, -0.9, -1.0],                    # confident negative -> exclude
        3: [0.0, 0.0, 0.0, 0.0],                        # exact-0 (sd=0) -> exclude
        4: [-0.5],                                      # n < min_obs -> keep (no evidence)
    })
    keep = _conf_keep(acc, Z, min_obs=2)
    assert list(keep) == [True, True, False, False, True]
    # the strict sign gate would have excluded the borderline client 1:
    assert acc.cum[1] <= 0.0


def test_confgate_select_fn():
    acc = _acc_from({0: [1.0] * 4, 1: [-1.0] * 4, 2: [-0.9, 1.0, -0.8, 0.85]})
    sel = make_confgate_select_fn(acc, burn_in=0, z=Z, min_obs=2, probation_every=0)
    got = set(int(x) for x in sel(5, 3, np.random.default_rng(0)))
    assert got == {0, 2}                                # 1 significantly negative


# --------------------------------------------------------------------------- #
# 3. Phi weights (P5-soft)                                                    #
# --------------------------------------------------------------------------- #
def test_phi_cdf_anchors():
    assert abs(_phi_cdf(0.0) - 0.5) < 1e-12
    assert abs(_phi_cdf(1.645) - 0.95) < 1e-3
    assert _phi_cdf(float("inf")) == 1.0 and _phi_cdf(float("-inf")) == 0.0


def test_probweight_weights():
    acc = SignAccumulator(4)
    streams = {0: [1.0, 1.0, 1.0], 1: [0.0, 0.0, 0.0],
               2: [-2.0, -2.1, -1.9], 3: [0.6, -0.5, 0.55]}
    raws = {r: [streams[c][r] for c in range(4)] for r in range(3)}
    wf = make_probweight_weights_fn(acc, lambda w, dm, ps: raws[w], [10] * 4,
                                    burn_in=0, min_obs=2)
    dm = {c: None for c in range(4)}
    for r in range(3):
        wmap = wf(r, r, dm)                             # w_r slot carries the round idx
    assert abs(sum(wmap.values()) - 1.0) < 1e-12
    assert wmap[1] == 0.0                               # exact-0 -> weight 0
    assert wmap[2] < 1e-6                               # confident negative -> ~0
    assert wmap[0] > 0.5                                # deterministic positive dominates
    assert 0.0 < wmap[3] < wmap[0]                      # borderline -> partial weight


def test_probweight_burn_in_neutral():
    acc = SignAccumulator(2)
    wf = make_probweight_weights_fn(acc, lambda w, dm, ps: [-5.0, 5.0], [10, 30],
                                    burn_in=10, min_obs=2)
    wmap = wf(0, None, {0: None, 1: None})              # r < burn_in -> plain n-weights
    assert abs(wmap[0] - 0.25) < 1e-12 and abs(wmap[1] - 0.75) < 1e-12


# --------------------------------------------------------------------------- #
# 4. runner plumbing: T2 helpers + arm parsing (legacy untouched)             #
# --------------------------------------------------------------------------- #
def test_t2_helpers():
    from experiments.track_c2 import _t2_kept_ucb, _t2_pw_wvec
    acc = _acc_from({0: [1.0] * 3, 1: [0.0] * 3, 2: [-1.0, -1.1, -0.9],
                     3: [-0.9, 1.0, -0.05]})
    assert _t2_kept_ucb(acc, Z, 2) == [0, 3]            # exact-0 + confident-neg out
    wv = _t2_pw_wvec(acc, 2)
    assert 1 not in wv and 2 not in wv                  # exact-0 + Phi-underflow(-17sd) dropped
    assert wv[0] > 0.95 and 0.0 < wv[3] < 1.0


def test_th_parse_p5():
    from experiments.track_c2 import _th_parse
    assert _th_parse("gtg_cgate") == ("gtg", "cgate")
    assert _th_parse("flirds_pweight") == ("flirds", "pweight")
    assert _th_parse("comfedsv_pweight") == ("comfedsv", "pweight")
    for legacy in ("flirds_gate_v2", "flirds_mult", "vanilla", "observer",
                   "shapleyfl", "fedif", "t2_csign_gtg", "oracle_excl"):
        assert _th_parse(legacy) is None, legacy


def test_analysis_parse():
    import importlib.util
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "runs", "track_h", "make_analysis.py")
    spec = importlib.util.spec_from_file_location("th_ma", p)
    ma = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ma)
    assert ma.parse_arm("fedsv_cgate") == ("fedsv", "P5h", "online")
    assert ma.parse_arm("flirds_pweight") == ("flirds", "P5s", "online")
    assert ma.parse_arm("t2_csign_lossheur") == ("lossheur", "P5h", "retrain")
    assert ma.parse_arm("t2_pw_flirds") == ("flirds", "P5s", "retrain")
    assert ma.parse_arm("t2_sign_flirds") == ("flirds", "P1", "retrain")   # legacy intact
    assert ma.parse_arm("flirds_gate_v2") == ("flirds", "P1", "online")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASS")
