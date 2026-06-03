"""CNN client-data corruptors — seam 2 registry.

A corruptor maps (xs, ys, client_id) -> (xs, ys), so the data layer / experiment
`build()` injects label-noise (and, later, free-rider / backdoor / PGD / maverick)
without the FL loop or the caller knowing the mechanism.  Registry-keyed so
Phase 2/3 corruptors drop in without touching callers; the LLM data layer adds
text corruptors under the same convention (its own registry).
"""
from __future__ import annotations

import torch


def label_shuffle(xs, ys, client_id, seed_base=100):
    """Noisy client: permute labels with seed_base+client_id (reproducible).

    Inputs returned unchanged except ys, which is randomly permuted — matches
    the inline `ys[randperm(seed=100+c)]` the phase05 builds used (bit-identical).
    """
    g = torch.Generator().manual_seed(seed_base + client_id)
    return xs, ys[torch.randperm(len(ys), generator=g)]


CNN_CORRUPTORS = {"label_shuffle": label_shuffle}
