"""C2_DYN (per-round corrupt re-draw) unit tests (spec runs/track_h/dyn/RUN_DYN.md).

Top risks: (1) the roundwise mask must be deterministic in (seed, salt, r) and
actually vary across rounds, (2) the dynamic delta_transform must corrupt
exactly the round's mask members (and the static path must be bit-unchanged),
(3) the dynamic label_flip loader must switch clean/flipped by the clock while
keeping `.dataset` identity, (4) the per-round exclusion select must drop
exactly the round's mask, (5) the clocked select wrapper must replicate the
core's uniform default draw bit-exactly.

No GPU, no dataset.  From codes/:  PYTHONPATH=. python tests/test_dyn.py
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from flirds.fl.intervene import make_delta_transform, make_roundwise_mask

from experiments.track_c2 import (_DynLFLoader, _RoundClock, _clocked_select,
                                  _make_dyn_excl_select_fn)


def test_roundwise_mask_deterministic_and_varying():
    m1 = make_roundwise_mask(100, 40, seed=3)
    m2 = make_roundwise_mask(100, 40, seed=3)
    assert m1(7) == m2(7) and m1(7) == m1(7)            # deterministic + cached
    assert all(len(m1(r)) == 40 for r in range(20))
    assert len({m1(r) for r in range(20)}) > 1          # actually re-drawn
    assert m1(5) != make_roundwise_mask(100, 40, seed=4)(5)      # seed separates
    assert m1(5) != make_roundwise_mask(100, 40, seed=3, salt=1)(5)  # salt separates


def test_dyn_delta_transform_follows_round_mask():
    mask = make_roundwise_mask(10, 3, seed=0)
    tf = make_delta_transform(mask, "free_rider", seed=0)
    d = {"w": torch.ones(4)}
    for r in (0, 1, 5):
        for c in range(10):
            out = tf(c, r, {k: v.clone() for k, v in d.items()})
            zeroed = float(out["w"].abs().sum()) == 0.0
            assert zeroed == (c in mask(r))


def test_static_delta_transform_unchanged():
    tf = make_delta_transform([1, 3], "free_rider", seed=0)   # legacy iterable path
    d = {"w": torch.ones(2)}
    assert float(tf(1, 0, {k: v.clone() for k, v in d.items()})["w"].sum()) == 0.0
    assert float(tf(0, 0, {k: v.clone() for k, v in d.items()})["w"].sum()) == 2.0


def test_dynlf_loader_switches_with_clock():
    xs = torch.zeros(6, 1)
    clean = DataLoader(TensorDataset(xs, torch.zeros(6, dtype=torch.long)), batch_size=6)
    flip = DataLoader(TensorDataset(xs, torch.ones(6, dtype=torch.long)), batch_size=6)
    clock, mask = _RoundClock(), (lambda r: {2} if r % 2 == 0 else set())
    ld = _DynLFLoader(clean, flip, cid=2, clock=clock, mask_at=mask)
    assert ld.dataset is clean.dataset and len(ld) == len(clean)
    clock.r = 0
    assert int(next(iter(ld))[1].sum()) == 6            # corrupt round -> flipped labels
    clock.r = 1
    assert int(next(iter(ld))[1].sum()) == 0            # clean round -> clean labels


def test_dyn_excl_select_drops_round_mask():
    mask = make_roundwise_mask(20, 8, seed=1)
    sel = _make_dyn_excl_select_fn(20, mask)
    rng = np.random.default_rng(0)
    for r in (0, 3):
        got = set(int(c) for c in sel(r, 25, rng))      # k >= kept -> full kept set
        assert got == set(range(20)) - mask(r)
        part = sel(r, 5, rng)
        assert len(part) == 5 and not (set(int(c) for c in part) & mask(r))


def test_clocked_select_replicates_default_draw():
    clock = _RoundClock()
    sel = _clocked_select(None, clock, 50)
    got = sel(9, 10, np.random.default_rng(123))
    want = np.random.default_rng(123).choice(50, size=10, replace=False)
    assert np.array_equal(got, want) and clock.r == 9   # same draw + clock stamped


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASS")
