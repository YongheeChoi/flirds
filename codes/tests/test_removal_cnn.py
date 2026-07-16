"""Exp A3 unit tests -- experiments.track_c1.removal_retrain_curves (pure logic).

No torch training: retrain_eval is stubbed with a synthetic kept-set quality game,
so these run in seconds on CPU (no pytest dependency -- plain asserts).
From codes/:
    PYTHONPATH=. python tests/test_removal_cnn.py
The third A3 verification leg (C1_REMOVAL=0 bit-identity of the smoke metrics) is
an integration check, run separately against a HEAD checkout -- see the A3 section
of runs/removal_dose/README.md.
"""
import os

import numpy as np

os.environ.setdefault("C1_REMOVAL", "0")               # import-time gate default (asserted below)
from experiments import track_c1
from experiments.track_c1 import removal_retrain_curves


def _game(n, q):
    """Synthetic game: val_loss = 1 - mean(q[kept]), acc = mean(q[kept]); logs calls."""
    calls = []

    def retrain_eval(kept):
        calls.append(tuple(kept))
        acc = float(np.mean([q[c] for c in kept]))
        return 1.0 - acc, acc

    return retrain_eval, calls


def test_gate_default_off():
    """C1_REMOVAL unset -> gate off, no method restriction (bit-identical path)."""
    assert track_c1.REMOVAL is False
    assert track_c1.REMOVAL_METHODS == []


def test_direction():
    """worst-first dominates best-first when phi matches true quality; acc mirrors loss."""
    n, q = 4, [1.0, 0.8, 0.5, 0.1]                     # client 0 best ... 3 worst
    phi = np.array([0.0, 1.0, 2.0, 3.0])               # good->low: client 3 most suspicious
    retrain_eval, _ = _game(n, q)
    rc, rca, _, _ = removal_retrain_curves([("M", phi, 0.0)], retrain_eval, n, "cpu")
    wl = [v for _, v in rc["M"]["worst_first"]]
    bl = [v for _, v in rc["M"]["best_first"]]
    wa = [v for _, v in rca["M"]["worst_first"]]
    ba = [v for _, v in rca["M"]["best_first"]]
    assert wl[0] == bl[0] and wa[0] == ba[0]           # k=0 = full set, shared value
    assert np.mean(wl) < np.mean(bl)                   # loss: drop worst first stays lower
    assert np.mean(wa) > np.mean(ba)                   # acc: drop worst first stays higher
    assert all(w <= b for w, b in zip(wl, bl))         # pointwise under strict quality order


def test_cache_shared():
    """ONE retrain per distinct kept set, shared across methods AND directions."""
    n = 5
    phi = np.arange(n, dtype=float)
    retrain_eval, calls = _game(n, [1.0 - 0.1 * c for c in range(n)])
    methods = [("A", phi, 0.0), ("B", phi.copy(), 0.0), ("C", -phi, 0.0)]  # B==A; C reversed
    rc, _, nrt, mrt = removal_retrain_curves(methods, retrain_eval, n, "cpu")
    # A alone: worst chain (n kept sets) + best chain (n) sharing the full set = 2n-1.
    # B identical -> 0 new; C reversed -> its worst==A's best chain (and vice versa) -> 0 new.
    assert len(calls) == 2 * n - 1 == nrt
    assert len(set(calls)) == len(calls)               # every retrain hit a NEW kept set
    assert set(rc) == {"A", "B", "C"}
    assert rc["A"] == rc["B"]                          # identical phi -> identical curves
    assert rc["C"]["worst_first"] == rc["A"]["best_first"]
    assert rc["C"]["best_first"] == rc["A"]["worst_first"]
    assert mrt >= 0.0


def test_ripple_excluded_by_default():
    """Ripple ranks its OWN trajectory -> out of the default sel; explicit sel forces it."""
    n = 3
    phi = np.array([0.0, 1.0, 2.0])
    retrain_eval, _ = _game(n, [0.9, 0.5, 0.1])
    ms = [("Flirds", phi, 0.0), ("Ripple", phi, 0.0)]
    rc, rca, _, _ = removal_retrain_curves(ms, retrain_eval, n, "cpu")
    assert "Ripple" not in rc and "Ripple" not in rca and "Flirds" in rc
    rc2, _, _, _ = removal_retrain_curves(ms, retrain_eval, n, "cpu", sel=["Ripple"])
    assert set(rc2) == {"Ripple"}


def test_schema():
    """Exp A2-compatible curve schema: k = 0..n-1, float values, both directions."""
    n = 4
    phi = np.array([3.0, 1.0, 2.0, 0.0])
    retrain_eval, _ = _game(n, [0.1, 0.7, 0.4, 1.0])
    rc, rca, nrt, mrt = removal_retrain_curves([("M", phi, 0.0)], retrain_eval, n, "cpu")
    for curves in (rc, rca):
        for d in ("worst_first", "best_first"):
            pts = curves["M"][d]
            assert [k for k, _ in pts] == list(range(n))   # kept n..1; empty set skipped
            assert all(isinstance(v, float) for _, v in pts)
    assert nrt >= 1 and mrt >= 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASS")
