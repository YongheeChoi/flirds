"""Track H unit tests -- score-provider orientation + runner plumbing (spec
runs/track_h/README.md; the top risk is the same D-3 sign flip as Track G).

Synthetic quadratic game (test_signgate convention):
    loss(w) = 0.5 * ||w - t||^2
    helpful delta = +0.2 * (t - w)   (strong pull toward the target)
    harmful delta = -0.1 * (t - w)   (weaker push away -- asymmetric on purpose:
                                      a symmetric pair cancels to a zero-movement
                                      round, which GTG's round_trunc maps to 0)
    zero delta    = frzero
Every provider's raw must be CONTRIBUTION-oriented: helpful > 0 > harmful.
Exact-0 for the zero delta is asserted ONLY for the estimator family
(flirds/flirds1st/lossheur) -- the coalition providers (gtg/fedsv/comfedsv/
shapleyfl) have renorm/dilution zero-semantics, which is the Track H property
under test, not a bug.

No GPU, no dataset.  From codes/:  PYTHONPATH=. python tests/test_track_h.py
"""
import numpy as np
import torch

from flirds.fl.score_providers import (SOURCES, comfedsv_round_raw_fn,
                                       fedsv_round_raw_fn, gtg_round_raw_fn,
                                       provider_round_raw_fn)

T = torch.tensor([1.0, -2.0])
PKEYS = ["w"]


def loss_fn(params, buffers):
    return 0.5 * ((params["w"] - T) ** 2).sum()


def _round(w, deltas):
    return {"w": torch.tensor(w)}, {c: ({"w": torch.tensor(d)}, 10)
                                    for c, d in deltas.items()}


def _mk_round():
    w = [0.0, 0.0]
    pull = (T - torch.tensor(w)).numpy()
    return _round(w, {0: list(0.2 * pull), 1: list(-0.1 * pull), 2: [0.0, 0.0]})


# --------------------------------------------------------------------------- #
# 1. provider raw orientation (helpful > 0 > harmful), all 8 sources          #
# --------------------------------------------------------------------------- #
def test_provider_sign_direction():
    w_r, dm = _mk_round()
    exact_zero = {"flirds", "flirds1st", "lossheur"}   # estimator family only
    for src in SOURCES:
        fn = provider_round_raw_fn(src, loss_fn, PKEYS, 3, "cpu", seed=0)
        raw = fn(w_r, dm, [0, 1, 2])
        assert raw[0] > 0, f"{src}: helpful must be POSITIVE, got {raw[0]}"
        assert raw[1] < 0, f"{src}: harmful must be NEGATIVE, got {raw[1]}"
        if src in exact_zero:
            assert raw[2] == 0.0, f"{src}: zero delta must be EXACT 0.0, got {raw[2]!r}"


def test_mc_providers_deterministic():
    """Same seed -> identical stream (one provider per arm, one call per round)."""
    w_r, dm = _mk_round()
    for mk in (gtg_round_raw_fn, fedsv_round_raw_fn, comfedsv_round_raw_fn):
        a = mk(loss_fn, PKEYS, "cpu", seed=7)
        b = mk(loss_fn, PKEYS, "cpu", seed=7)
        r1 = [a(w_r, dm, [0, 1, 2]) for _ in range(2)]   # two "rounds"
        r2 = [b(w_r, dm, [0, 1, 2]) for _ in range(2)]
        assert r1 == r2, f"{mk.__name__}: non-deterministic across same-seed providers"


def test_gtg_zero_movement_round_is_zero():
    """A round whose full coalition moves nothing (this==last) contributes 0
    (gtg_from_logs skips it) -- the symmetric-cancel case."""
    w = [0.0, 0.0]
    pull = (T - torch.tensor(w)).numpy()
    w_r, dm = _round(w, {0: list(0.1 * pull), 1: list(-0.1 * pull), 2: [0.0, 0.0]})
    raw = gtg_round_raw_fn(loss_fn, PKEYS, "cpu", seed=0)(w_r, dm, [0, 1, 2])
    assert raw == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# 2. track_c2 plumbing: arm parsing, T2 kept/dedupe/static weights            #
# --------------------------------------------------------------------------- #
def test_th_parse():
    from experiments.track_c2 import _th_parse
    assert _th_parse("gtg_gate_v2") == ("gtg", "gate_v2")
    assert _th_parse("comfedsv_zgate_v2") == ("comfedsv", "zgate_v2")
    assert _th_parse("fedsv_gatew_v2") == ("fedsv", "gatew_v2")
    assert _th_parse("flirds1st_mult") == ("flirds1st", "mult")
    assert _th_parse("flirds1st_gate_v2") == ("flirds1st", "gate_v2")
    # legacy arms must NOT be re-routed (bit-identical closures)
    for legacy in ("flirds_gate_v2", "flirds_gatew_v2", "flirds_zgate_v2",
                   "flirds_gate_v1", "flirds_gatew_v1", "flirds_mult", "flirds_repl",
                   "vanilla", "shapleyfl", "fedif", "sfedavg", "observer",
                   "t2_sign_gtg", "oracle_excl"):
        assert _th_parse(legacy) is None, f"{legacy} must stay on its legacy branch"


def test_t2_kept_and_cache_key():
    from experiments.track_c2 import _t2_cache_key, _t2_kept
    cum = np.array([2.0, 0.0, -1.0, 0.5])
    assert _t2_kept(cum, 0.0) == [0, 3]                 # strict > tau: exact-0 is OUT
    k1 = _t2_cache_key([0, 3], None)
    assert k1 == _t2_cache_key([3, 0], None)            # order-free dedupe
    assert k1 != _t2_cache_key([0, 1], None)
    w = {0: 2.0, 3: 0.5}
    assert _t2_cache_key([0, 3], w) == _t2_cache_key([3, 0], dict(reversed(list(w.items()))))
    assert _t2_cache_key([0, 3], w) != k1               # weighted != plain


def test_t2_static_weights():
    from experiments.track_c2 import _t2_static_weights_fn
    nums = [10, 10, 10, 10]
    wf = _t2_static_weights_fn(nums, {0: 3.0, 3: 1.0})
    wmap = wf(0, None, {0: (None, 10), 3: (None, 10)})
    assert abs(wmap[0] - 0.75) < 1e-12 and abs(wmap[3] - 0.25) < 1e-12
    wf0 = _t2_static_weights_fn(nums, {})               # no positive mass -> n-fallback
    wmap0 = wf0(0, None, {1: (None, 10), 2: (None, 10)})
    assert abs(wmap0[1] - 0.5) < 1e-12 and abs(wmap0[2] - 0.5) < 1e-12


# --------------------------------------------------------------------------- #
# 3. track_g plumbing: every Track H source resolves in raw_by_arm            #
# --------------------------------------------------------------------------- #
def test_track_g_provider_names():
    """The LLM runner dispatches `<provider>_<policy>` by prefix -- the Track H
    sources must all split cleanly to a known provider name (no import of the
    heavy track_g module; the contract is the naming convention)."""
    providers = {"flirds", "flirds1st", "lossheur", "oracleb", "shapleyfl",
                 "gtg", "fedsv", "comfedsv"}
    for arm in ("gtg_gate_v2", "fedsv_gate_v2", "comfedsv_gate_v2",
                "shapleyfl_gate_v2", "flirds1st_gate_v2", "flirds1st_gatew_v2",
                "lossheur_gatew_v2", "gtg_gatew_v2"):
        assert arm.split("_")[0] in providers, arm


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASS")
