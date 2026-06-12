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


def _largest_remainder(targets):
    """Round non-negative float `targets` to ints preserving the exact sum
    (floor + distribute the remainder to the largest fractional parts;
    ties break by lower index — fully deterministic)."""
    targets = np.asarray(targets, dtype=float)
    base = np.floor(targets).astype(int)
    rem = int(round(targets.sum())) - base.sum()
    order = np.argsort(-(targets - base), kind="stable")
    base[order[:rem]] += 1
    return base


def label_skew_partition(labels, n_clients, major_frac=0.8, per_client=None, seed=0):
    """GTG scenario-2 label skew (2109.02053 §5.1.1): clients come in PAIRS; pair
    p's two clients draw `major_frac` of their samples evenly from the 2 designated
    classes {2p+1 mod C, 2p+2 mod C} (GTG: participants 1&2 -> digits 1&2, ...,
    9&10 -> digits 9&0) and the remaining 1-major_frac evenly from the other C-2
    classes.  Disjoint across clients (per-class pools, no replacement).

    per_client=None -> max feasible size: with the GTG geometry (n_clients ==
    n_classes, paired majors) per-class demand == per_client, so it is the min
    class count (MNIST 5,421 / CIFAR-10 5,000).  Counts are rounded PER CLASS
    (largest-remainder down each class column) so per-class totals are exact and
    pools never exhaust; client sizes are equal up to rounding (±C/2 samples).
    Only the pool order and within-client order are random.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    C = len(classes)
    assert n_clients % 2 == 0, "label_skew_partition pairs clients"
    rng = np.random.default_rng(seed)
    pools = {c: rng.permutation(np.where(labels == c)[0]) for c in classes}
    if per_client is None:
        per_client = min(len(p) for p in pools.values())
    fracs = np.empty((n_clients, C))
    for i in range(n_clients):
        p = i // 2
        majors = {classes[(2 * p + 1) % C], classes[(2 * p + 2) % C]}
        fracs[i] = [major_frac / 2 if c in majors else
                    (1 - major_frac) / (C - 2) for c in classes]
    client_idx = [[] for _ in range(n_clients)]
    for j, c in enumerate(classes):
        counts = _largest_remainder(fracs[:, j] * per_client)
        assert counts.sum() <= len(pools[c]), f"class {c} pool exhausted; lower per_client"
        ptr = 0
        for i in range(n_clients):
            client_idx[i].extend(pools[c][ptr:ptr + counts[i]].tolist())
            ptr += counts[i]
    for i in range(n_clients):
        rng.shuffle(client_idx[i])  # mix classes so a prefix [:k] stays representative
    return client_idx


def quantity_skew_partition(labels, n_clients, ratios, seed=0):
    """GTG scenario-3 quantity skew, disjoint-normalized (2026-06-12 decision):
    client sizes proportional to `ratios` (GTG: 10/10/15/15/20/20/25/25/30/30 —
    the paper's literal %-of-pool reading sums to 200% and implies overlapping
    samples; we keep only the ratio structure, max/min 3x, disjoint), and every
    client is digit-balanced (GTG: "the same number of images for each digit in
    each participant").  Digit balance forces equal per-class allocation, so each
    class pool is capped at the min class count and split across clients by
    largest-remainder on the normalized ratios.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    m = min(int((labels == c).sum()) for c in classes)
    frac = np.asarray(ratios, dtype=float)
    take = _largest_remainder(frac / frac.sum() * m)    # per-client count per class
    client_idx = [[] for _ in range(n_clients)]
    for c in classes:
        pool = rng.permutation(np.where(labels == c)[0])
        ptr = 0
        for i in range(n_clients):
            client_idx[i].extend(pool[ptr:ptr + take[i]].tolist())
            ptr += take[i]
    for i in range(n_clients):
        rng.shuffle(client_idx[i])  # mix classes so a prefix [:k] stays representative
    return client_idx


def shard_partition(labels, n_clients, shards_per_client=2, seed=0):
    """McMahan 2-shard pathological non-IID (FedAvg paper; the FedSV/ComFedSV/
    ShapleyFL "2-shard" setting): sort by label, cut into n_clients *
    shards_per_client equal contiguous shards, deal each client
    `shards_per_client` random shards -- most clients see ~2 classes.
    Disjoint, near-equal sizes (remainder spread by np.array_split).
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    order = np.argsort(labels, kind="stable")
    shards = np.array_split(order, n_clients * shards_per_client)
    deal = rng.permutation(len(shards))
    client_idx = []
    for i in range(n_clients):
        idx = np.concatenate([shards[s] for s in deal[i::n_clients]]).tolist()
        rng.shuffle(idx)            # mix shards so a prefix [:k] stays representative
        client_idx.append(idx)
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
