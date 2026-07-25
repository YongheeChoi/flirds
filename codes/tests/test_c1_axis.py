"""C-b unit tests -- the Track C1 paper-axis threat seam (2026-07-25).

The one thing the axis can get silently wrong is COALITION RE-INDEXING: `fl.server`
hands the delta seam the client's position in the loader list it was given, so the
(a) oracle's `[loaders[c] for c in S]` would otherwise apply client S[i]'s threat to
position i -- a free-rider that free-rides in the wrong coalitions still produces a
plausible-looking phi.  These tests pin the mapping, plus the end-to-end consequence
through `subset_utility_valloss` on a 2-parameter model (CPU, seconds).

From codes/:
    PYTHONPATH=. python tests/test_c1_axis.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from flirds.fl.intervene import make_delta_transform
from flirds.fl.server import subset_delta_transform
from flirds.oracle.exact_sv import subset_utility_valloss


def test_passthrough_is_none():
    """No threat -> no seam (the legacy path stays bit-identical)."""
    assert subset_delta_transform(None, (0, 1, 2)) is None


def test_position_maps_to_global_id():
    """Position i in the coalition carries client S[i]'s threat, not client i's."""
    seen = []
    tf = subset_delta_transform(lambda c, r, d: seen.append((c, r)) or d, (3, 7, 8))
    for i in range(3):
        tf(i, 0, {})
    assert [c for c, _ in seen] == [3, 7, 8]


def test_free_rider_zeroes_only_its_own_position():
    """Global client 7 free-rides; inside (3, 7, 8) that is position 1 alone."""
    dtf = make_delta_transform([7], "free_rider")
    sub = subset_delta_transform(dtf, (3, 7, 8))
    d = {"w": torch.ones(2)}
    kept = [sub(i, 0, d)["w"].clone() for i in range(3)]
    assert torch.equal(kept[1], torch.zeros(2))
    assert torch.equal(kept[0], torch.ones(2)) and torch.equal(kept[2], torch.ones(2))


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
    w_0 score; every other singleton moves.  Without the re-index BOTH singletons would
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
