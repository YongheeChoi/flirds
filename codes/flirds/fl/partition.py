"""Client data partitioning for FL simulation (Phase 0, CNN track).

Label-based partitions: IID, Dirichlet(alpha) label-skew, McMahan 2-shard.
alpha=0 (extreme non-IID) is handled as a disjoint-class special case.
"""
from __future__ import annotations

import numpy as np


def iid_partition(labels, n_clients, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    return [a.tolist() for a in np.array_split(idx, n_clients)]


def dirichlet_partition(labels, n_clients, alpha, seed=0):
    """Dirichlet label-skew (Hsu et al. 2019). Smaller alpha => more non-IID.

    alpha == 0 => each client gets disjoint single classes (extreme non-IID).
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    if alpha == 0:
        return _disjoint_class_partition(labels, classes, n_clients)
    client_idx = [[] for _ in range(n_clients)]
    for c in classes:
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        prop = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(prop)[:-1] * len(c_idx)).astype(int)
        for i, part in enumerate(np.split(c_idx, cuts)):
            client_idx[i].extend(part.tolist())
    for i in range(n_clients):
        rng.shuffle(client_idx[i])  # mix classes so a prefix [:k] stays representative
    return client_idx


def _disjoint_class_partition(labels, classes, n_clients):
    """alpha=0: assign disjoint classes round-robin across clients."""
    client_idx = [[] for _ in range(n_clients)]
    for j, c in enumerate(classes):
        c_idx = np.where(labels == c)[0]
        client_idx[j % n_clients].extend(c_idx.tolist())
    return client_idx


def mcmahan_shard_partition(labels, n_clients, shards_per_client=2, seed=0):
    """Sort-by-label sharding (McMahan et al. 2017). Pathological non-IID."""
    labels = np.asarray(labels)
    order = np.argsort(labels, kind="stable")
    n_shards = n_clients * shards_per_client
    shards = np.array_split(order, n_shards)
    rng = np.random.default_rng(seed)
    shard_ids = rng.permutation(n_shards)
    client_idx = [[] for _ in range(n_clients)]
    for i, s in enumerate(shard_ids):
        client_idx[i // shards_per_client].extend(shards[s].tolist())
    return client_idx
