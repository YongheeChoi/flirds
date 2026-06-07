"""Data Banzhaf (Wang & Jia 2023) as an FL in-run semivalue baseline.

The Banzhaf VALUE is a semivalue like Shapley, differing only in the marginal
weight: Shapley uses r!(n-1-r)!/n! (coalition-size dependent), Banzhaf uses a
uniform 1/2^{n-1}.  We reuse the (b) in-run oracle's exact 2^N coalition utilities
U_(b)(S) (oracle.in_run_sv._coalition_utilities) and reweight them -- so this is
the EXACT Banzhaf value over the SAME frozen trajectory + delta utility as the (b)
Shapley oracle.  Holding the utility fixed isolates the semivalue choice (Data
Banzhaf's claim is robustness to utility NOISE; our (b) utility is deterministic,
so the ranking should barely move -- the interesting empirical point).

"FL granularity" = the players are CLIENTS and the marginal uses the per-round
FedAvg weight (same utility as in_run_shapley), NOT the within-subset renorm of
GTG/FedSV.  So a zero-delta client (free-rider zero) has a zero marginal in every
coalition -> Banzhaf phi exactly 0, like Shapley/Flirds/oracle.

Exact is affordable at N<=10 (cross-silo); the paper's MSR estimator is only
needed cross-device (N=100, plan task 7), like the (b) oracle's MC there -- not
built until that task arrives.  Reference-guided self-build (the canonical Banzhaf
formula over our existing exact-coalition machinery), not a pyDVL/OpenDataVal dep.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..oracle.in_run_sv import _coalition_utilities


def banzhaf_from_utilities(U, n_clients):
    """Banzhaf value from a full 2^N coalition-utility dict U[tuple]->float.

    phi_k = (1/2^{n-1}) Σ_{S⊆N\\{k}} [U(S∪{k}) - U(S)]  (uniform marginal weight)."""
    clients = list(range(n_clients))
    w = 1.0 / 2 ** (n_clients - 1)
    phi = np.zeros(n_clients)
    for k in clients:
        others = [c for c in clients if c != k]
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                phi[k] += w * (U[tuple(sorted(S + (k,)))] - U[S])
    return phi


def in_run_banzhaf(logs, n_clients, loss_fn, pkeys, device):
    """Exact (b) in-run Banzhaf values over `logs`.  Returns (phi[n], p[n]).

    Mirrors oracle.in_run_sv.in_run_shapley but with the Banzhaf kernel; shares
    the same exact coalition-utility enumeration (so its runtime is the same 2^N
    coalition-sweep class as the (b) oracle)."""
    U, p = _coalition_utilities(logs, n_clients, loss_fn, pkeys, device)
    return banzhaf_from_utilities(U, n_clients), p
