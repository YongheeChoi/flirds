"""Corrupt-set draw canon (audit 2026-07-23).

`track_c2.build()` draws the corrupt set PER THREAT by source convention, not by one
uniform rule: label_flip reproduces FedCorr's official model (gamma_s = binomial(1,
rho, n), so the COUNT is Bernoulli and moves with the seed), while the update-level
threats take exactly round(MAL_FRAC*n).  That asymmetry is deliberate and is reported
in the paper's setup rather than unified -- unifying it would invalidate every
label_flip rundir on disk.  These tests pin it so it cannot drift silently.

Slow (each case reloads track_c2 and builds 100 real fmnist loaders).
From codes/:  PYTHONPATH=. python -m pytest tests/test_corrupt_set_canon.py -q
"""
import importlib
import os

import numpy as np
import pytest

# Realized counts of the FedCorr Bernoulli draw at n=100, rho=0.4 -- the values every
# label_flip rundir on disk was produced with (verified across track_c/c2, track_h,
# probe_signal and the Track G grid).  Mean 44.7%, NOT the nominal 40%.
FEDCORR_CANON = {0: 39, 1: 48, 2: 47}
FIXED_COUNT = 40                                      # round(MAL_FRAC * n), n=100
UPDATE_LEVEL = ["free_rider", "frrand", "grad_noise"]


@pytest.fixture(autouse=True, scope="module")
def _restore_global_state():
    """`_build` rewrites os.environ and reloads track_c2, both process-global.  Later
    test modules (test_partition_qskew) reach into the module's config, so hand it back
    exactly as found."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)
    importlib.reload(importlib.import_module("experiments.track_c2"))


def _build(threat, seed, partition="iid", dataset="fmnist"):
    """Reload track_c2 under the given cell env and return (corrupt, empty_clients).

    The module reads its config from os.environ at import time, so a reload is the
    only way to sweep cells in one process.
    """
    os.environ.update(C2_DATASET=dataset, C2_PARTITION=partition, C2_THREAT=threat,
                      C2_SEED=str(seed), C2_MODE="full")
    os.environ.pop("C2_FLIP_RATE", None)               # strmain: exercise the U(TAU,1) path
    c2 = importlib.reload(importlib.import_module("experiments.track_c2"))
    corrupt = c2.build()[1]
    assert len(corrupt) == 100, "canon is defined at n=100"
    return corrupt, list(c2._EMPTY_CLIENTS)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_label_flip_count_is_the_fedcorr_bernoulli_canon(seed):
    """label_flip count varies by seed -- pin the exact realized values."""
    corrupt, _ = _build("label_flip", seed)
    assert int(corrupt.sum()) == FEDCORR_CANON[seed]


@pytest.mark.parametrize("threat", UPDATE_LEVEL)
def test_update_level_threats_take_exactly_the_fixed_count(threat):
    """No FedCorr analog -> exactly round(MAL_FRAC*n), identical on every seed."""
    for seed in (0, 1, 2):
        corrupt, _ = _build(threat, seed)
        assert int(corrupt.sum()) == FIXED_COUNT


def test_corrupt_set_is_seed_only():
    """Both rules draw from default_rng(1000+seed) BEFORE any data touches the rng, so
    a seed fixes the set across dataset/partition.  This is what makes the partition
    contrast (iid/dir1/shard/qskew) a clean comparison despite the two rules."""
    ref, _ = _build("label_flip", 0, partition="iid", dataset="fmnist")
    for partition, dataset in [("dir1", "fmnist"), ("shard", "cifar10"),
                               ("qskew", "cifar10")]:
        other, _ = _build("label_flip", 0, partition=partition, dataset=dataset)
        assert np.array_equal(ref, other), f"{dataset}/{partition} drew a different set"


def test_no_empty_clients_at_n100():
    """Premise of the backfill guard in build(): dirichlet(alpha=1) never starves a
    client at n=100, so the backfill never consumes from the rate stream.  If this
    ever fails, the strmain rate vector has become partition-dependent."""
    for seed in (0, 1, 2):
        _, empty = _build("label_flip", seed, partition="dir1")
        assert empty == [], f"seed {seed} backfilled clients {empty}"
