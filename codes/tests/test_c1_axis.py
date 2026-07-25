"""C-b regression tests -- the Track C1 paper-axis threat seam (2026-07-25).

The one thing the axis can get silently wrong is COALITION RE-KEYING: `fl.server`
hands the delta seam the client's position in the loader list it was given, so the
(a) oracle's `[loaders[c] for c in S]` would otherwise apply client S[i]'s threat to
position i -- a free-rider that free-rides in the wrong coalitions still produces a
plausible-looking phi, and (a) quietly stops being an oracle of the game the estimator
plays.  These tests pin the consequence end-to-end through `subset_utility_valloss`
on a 2-parameter model (CPU, seconds).

Written against the canonical C-b (`origin/main` 989f5ca), which re-keys inline: the
tests target the public `subset_utility_valloss` contract, not the mechanism.

From codes/:
    PYTHONPATH=. python tests/test_c1_axis.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from flirds.fl.intervene import make_delta_transform
from flirds.oracle.exact_sv import subset_utility_valloss


def _toy(n_clients=3, seed=0):
    """3 clients x 8 samples of a 2-class linear problem + a val loader."""
    g = torch.Generator().manual_seed(seed)
    loaders = [DataLoader(TensorDataset(torch.randn(8, 4, generator=g),
                                        torch.randint(0, 2, (8,), generator=g)),
                          batch_size=4, shuffle=False) for _ in range(n_clients)]
    val = DataLoader(TensorDataset(torch.randn(8, 4, generator=g),
                                   torch.randint(0, 2, (8,), generator=g)), batch_size=8)
    return loaders, val, lambda: nn.Linear(4, 2)


def test_oracle_subset_applies_the_threat_to_the_right_client():
    """U({c}) collapses to the init-model score for exactly the free-riding client.

    A singleton coalition of a zero-delta client trains nothing, so its utility is the
    w_0 score; every other singleton moves.  Without the re-key BOTH singletons would
    be position 0 and both would collapse -- this is the assertion that fails then.
    """
    loaders, val, model_fn = _toy()
    dtf = make_delta_transform([1], "free_rider")        # global client 1 only
    kw = dict(rounds=2, local_epochs=1, lr=0.1, device="cpu", seed=0)
    u_empty = subset_utility_valloss(model_fn, loaders, val, (), **kw)
    u = [subset_utility_valloss(model_fn, loaders, val, (c,), delta_transform=dtf, **kw)
         for c in range(3)]
    assert np.isclose(u[1], u_empty), f"free-rider singleton should not move: {u[1]} vs {u_empty}"
    assert not np.isclose(u[0], u_empty) and not np.isclose(u[2], u_empty)


def test_oracle_default_is_threat_free():
    """delta_transform unset -> identical to the pre-seam call (legacy bit-identity)."""
    loaders, val, model_fn = _toy()
    kw = dict(rounds=2, local_epochs=1, lr=0.1, device="cpu", seed=0)
    assert subset_utility_valloss(model_fn, loaders, val, (0, 2), **kw) == \
        subset_utility_valloss(model_fn, loaders, val, (0, 2), delta_transform=None, **kw)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("ALL C1-AXIS TESTS PASS")
