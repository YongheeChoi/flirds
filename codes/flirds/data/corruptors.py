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


# ---- LLM sample-level corruptor (noisy client) ----
def answer_swap(records, client_id, seed_base=100):
    """Noisy client (free-form analog of CNN label_shuffle): permute the completion
    column within the client so each prompt pairs with another row's answer.  Prompts
    are unchanged -- only the prompt->completion association is broken.  Precedents:
    FedDQC's answer-swap quality corruption + FedCorr's data-side "freeloader" (a
    zero-effort client that trains on mismatched labels).  Reproducible via
    seed_base+client_id, mirroring label_shuffle.
    """
    g = torch.Generator().manual_seed(seed_base + client_id)
    perm = torch.randperm(len(records), generator=g).tolist()
    return [{"prompt": r["prompt"], "completion": records[i]["completion"]}
            for r, i in zip(records, perm)]


LLM_CORRUPTORS = {"answer_swap": answer_swap}


# ---- update-level corruptor (free-rider), representation-agnostic (CNN + LLM) ----
def free_rider(ref, mode="zero", scale=1e-3, generator=None):
    """Fabricated client update (free-rider): claim participation but submit a Δw
    with ~no alignment to the validation gradient, so Flirds' 1st-order term
    <-∇ℓ_val, Δw> ~= 0 and the client's value collapses to ~0 -- free-rider demotion
    as a by-product of signed valuation, no anomaly detector.  `ref` is a param-keyed
    tensor dict whose shapes/dtypes define the delta; returns CPU deltas (the
    logged-delta convention).

    Modes follow Lin et al. 2019 (STD-DAGMM origin) free-rider attack taxonomy, easy->hard:
      "zero"   : Δw = 0                       -- trivially detected; phi is exactly 0.
      "random" : Δw ~ U(-scale, scale) i.i.d. -- random direction, phi ~= 0 (noisy).
                 (Lin tunes `scale` to the benign-update std for detection evasion;
                 here it is a fixed documented scale -- the Phase-1 valuation sanity
                 depends only on the ~0 gradient alignment, not the exact value.)
    The harder delta-weights / advanced-delta families (recycle the previous round's
    aggregate +/- noise) need that aggregate threaded into the FL loop -> deferred to
    Phase 2 (the STD-DAGMM head-to-head + the recycled-aligned-signal question).
    """
    if mode == "zero":
        return {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in ref.items()}
    if mode == "random":
        return {k: torch.empty(v.shape, dtype=v.dtype).uniform_(-scale, scale, generator=generator)
                for k, v in ref.items()}
    raise ValueError(f"unknown free-rider mode: {mode!r} (use 'zero' or 'random')")
