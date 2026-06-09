"""Client data partitioning for FL simulation (Phase 0, CNN track).

Label-based partitions: IID, Dirichlet(alpha) label-skew (per-class + per-client).
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


def client_dirichlet_partition(labels, n_clients, alpha, per_client, seed=0):
    """Per-client Dirichlet label-mixture partition (LDA-style; the dual of
    `dirichlet_partition`).  Each client draws a label distribution from Dir(alpha)
    over the unique labels, then samples `per_client` disjoint records by that
    mixture -- so every client is non-empty and exactly `per_client`-sized (size
    stays a control variable; alpha is the sole non-IID knob).

    Use this when #labels << #clients (e.g. 5 domains -> 100 clients), where the
    per-class `dirichlet_partition` would concentrate each label on a few clients
    and leave most empty.  alpha == 0 => one label per client (round-robin), i.e.
    disjoint single-label clients (~n_clients/#labels per label).
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    n_cls = len(classes)
    rng = np.random.default_rng(seed)
    if alpha == 0:
        props = np.eye(n_cls)[np.arange(n_clients) % n_cls]
    else:
        props = rng.dirichlet([alpha] * n_cls, size=n_clients)
    pools = [rng.permutation(np.where(labels == c)[0]) for c in classes]
    ptr = [0] * n_cls
    client_idx = []
    for i in range(n_clients):
        counts = rng.multinomial(per_client, props[i])   # exact-sum -> fixed size
        idx = []
        for j in range(n_cls):
            take = int(counts[j])
            assert ptr[j] + take <= len(pools[j]), "label pool exhausted; raise per_domain_pool"
            idx.extend(pools[j][ptr[j]:ptr[j] + take].tolist())
            ptr[j] += take
        rng.shuffle(idx)        # mix labels so a prefix [:k] stays representative
        client_idx.append(idx)
    return client_idx
