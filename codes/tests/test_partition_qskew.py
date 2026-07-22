"""track_c2 `qskew` partition (2026-07-22 skew-axis decomposition) unit tests.

The Track G CNN grid completes the partition axis into 2x2 -- iid (no skew) /
shard (label skew only) / qskew (size skew only) / dir1 (both).  Top risks:
(1) qskew must be a PURE size skew -- every client label-balanced, so any gate
effect seen there is attributable to n alone; (2) the size rule must be the C1
`quantity_skew` rule verbatim (no newly invented distribution) -- the move of
`_quantity_ratios` into fl.partition must be value-preserving; (3) the three
legacy partition names must keep dispatching to exactly the same index sets.

No GPU, no dataset download.  From codes/:  PYTHONPATH=. python tests/test_partition_qskew.py
"""
import numpy as np
import torch

from flirds.fl.partition import (dirichlet_partition, gtg_quantity_ratios,
                                 iid_partition, quantity_skew_partition,
                                 shard_partition)

import experiments.track_c1 as c1
import experiments.track_c2 as c2

LABELS = np.repeat(np.arange(10), 200)          # 10 classes x 200 = 2,000 records


def test_ratio_rule_is_the_c1_rule_verbatim():
    # GTG paper values at N=10 (2109.02053 scenario 3) -- pinned, not re-derived.
    assert gtg_quantity_ratios(10) == [10, 10, 15, 15, 20, 20, 25, 25, 30, 30]
    # the track_c1 helper now delegates: same values for every n it is used with
    for n in (6, 10, 20, 100):
        assert c1._quantity_ratios(n) == gtg_quantity_ratios(n)


def test_qskew_is_pure_size_skew():
    n = 20
    idx = quantity_skew_partition(LABELS, n, gtg_quantity_ratios(n), seed=0)
    sizes = np.array([len(i) for i in idx])
    counts = np.stack([np.bincount(LABELS[np.array(i)], minlength=10) for i in idx])
    # labels IID: each client holds the SAME count of every class (digit-balanced)
    assert (counts == counts[:, :1]).all()
    # sizes follow the ratio ladder (monotone by pair, ends 10:55 -> 5.5x here)
    assert (np.diff(sizes) >= 0).all() and sizes[-1] > sizes[0]
    assert np.allclose(sizes / sizes.sum(),
                       np.array(gtg_quantity_ratios(n)) / sum(gtg_quantity_ratios(n)),
                       atol=0.002)
    # disjoint, and the pool is fully spent (same total data as the iid partition)
    flat = np.concatenate([np.array(i) for i in idx])
    assert len(set(flat.tolist())) == len(flat) == len(LABELS)


def _fake_build(partition, n=20):
    """Run track_c2.build() on synthetic data with `partition` selected; returns
    the per-client index sets it produced (recovered from the loaders' tensors)."""
    x = torch.arange(len(LABELS), dtype=torch.float32).view(-1, 1, 1, 1)   # id-carrying
    train = [(x[i], int(LABELS[i])) for i in range(len(LABELS))]
    test = [(x[i % len(x)], int(LABELS[i % len(LABELS)])) for i in range(1600)]
    orig = (c2.PARTITION, c2.THREAT, c2.SEED, c2.get_dataset, c2.get_labels)
    c2.PARTITION, c2.THREAT, c2.SEED = partition, "clean", 0
    c2.get_dataset = lambda name, train=True: (train_ if train else test_)
    train_, test_ = train, test
    c2.get_labels = lambda ds: LABELS
    c2.CFG.update(n=n)
    try:
        loaders = c2.build()[0]
    finally:
        (c2.PARTITION, c2.THREAT, c2.SEED, c2.get_dataset, c2.get_labels) = orig
    return [sorted(int(v) for v in ld.dataset.tensors[0].view(-1)) for ld in loaders]


def test_build_dispatch_matches_the_partition_helpers():
    """qskew reaches quantity_skew_partition; iid/dir1/shard are UNCHANGED."""
    want = {
        "iid": iid_partition(LABELS, 20, seed=0),
        "dir1": dirichlet_partition(LABELS, 20, alpha=1.0, seed=0),
        "shard": shard_partition(LABELS, 20, shards_per_client=2, seed=0),
        "qskew": quantity_skew_partition(LABELS, 20, gtg_quantity_ratios(20), seed=0),
    }
    for part, idx in want.items():
        assert _fake_build(part) == [sorted(i) for i in idx], part


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASS")
