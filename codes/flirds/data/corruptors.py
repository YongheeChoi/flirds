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


def label_flip(xs, ys, client_id, rate, n_classes=10, seed_base=100):
    """Graded noisy client (Track C): relabel a `rate` fraction of the client's
    samples to a uniform-random WRONG class (true label excluded), so `rate` IS the
    actual corruption rate and the flipped set is exactly the corrupted ground truth.
    One per-client `rate` knob expresses both the GTG graded ladder (0-20%) and the
    FedCorr (rho, tau) convention; FedCorr's own relabel is uniform over ALL K
    classes, which differs only by a (K-1)/K dilution of the effective rate (caveat,
    Yonghee 2026-06-12).  rate=0 returns the data unchanged (clean client).
    Reproducible via seed_base+client_id, mirroring label_shuffle.
    """
    g = torch.Generator().manual_seed(seed_base + client_id)
    n_flip = int(round(rate * len(ys)))
    if n_flip == 0:
        return xs, ys
    idx = torch.randperm(len(ys), generator=g)[:n_flip]
    ys = ys.clone()
    off = torch.randint(1, n_classes, (n_flip,), generator=g)  # uniform over K-1 wrong classes
    ys[idx] = (ys[idx] + off) % n_classes
    return xs, ys


def feature_noise(xs, ys, client_id, std, data_std, seed_base=100):
    """Graded noisy-feature client (GTG scenario 5): add Gaussian noise N(0, std^2)
    in PIXEL [0,1] units, no clamp.  xs are already normalized tensors, so the noise
    is applied in normalized space scaled per channel by 1/data_std -- mathematically
    identical to unclamped pixel-space noise.  `std` therefore reads as a fraction of
    the pixel range (ComFedSV's "graded Gaussian 5*i%" interpretation), letting the
    sigma ladder mirror the label_flip ladder numbers.  `data_std` = the dataset's
    per-channel normalization std (data.cnn._STATS[name][1]).  std=0 returns the
    data unchanged.  Reproducible via seed_base+client_id.
    """
    if std == 0:
        return xs, ys
    g = torch.Generator().manual_seed(seed_base + client_id)
    scale = std / torch.tensor(data_std, dtype=xs.dtype).view(1, -1, 1, 1)
    return xs + scale * torch.randn(xs.shape, generator=g), ys


CNN_CORRUPTORS = {"label_shuffle": label_shuffle, "label_flip": label_flip,
                  "feature_noise": feature_noise}


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


def answer_swap_graded(records, client_id, rate, seed_base=100):
    """Graded noisy client (Track C dose axis): corrupt a `rate` fraction of the
    client's samples by reassigning their completion to another corrupted sample's
    (cyclic shift within the corrupted subset -> guaranteed mismatch for >=2 poisoned).
    The free-form analog of CNN `label_flip(rate)`: `rate` IS the actual corruption
    rate and the corrupted set is exactly the corrupted ground truth.  rate>=1.0
    reproduces `answer_swap` (full within-client permutation) BIT-IDENTICALLY -- the
    default noisy behavior is unchanged; rate=0 returns the data untouched.
    Reproducible via seed_base+client_id, mirroring answer_swap / label_flip.
    """
    g = torch.Generator().manual_seed(seed_base + client_id)
    perm = torch.randperm(len(records), generator=g).tolist()
    if rate >= 1.0:                                     # backward-compat: == answer_swap
        return [{"prompt": r["prompt"], "completion": records[i]["completion"]}
                for r, i in zip(records, perm)]
    n_corrupt = int(round(rate * len(records)))
    if n_corrupt == 0:
        return [{"prompt": r["prompt"], "completion": r["completion"]} for r in records]
    idx = torch.randperm(len(records), generator=g)[:n_corrupt].tolist()
    swap = {idx[j]: records[idx[(j + 1) % n_corrupt]]["completion"]   # cyclic shift among corrupted
            for j in range(n_corrupt)}
    return [{"prompt": r["prompt"], "completion": swap.get(i, r["completion"])}
            for i, r in enumerate(records)]


# ---- LLM sample-level corruptor (backdoor attacker) ----
BACKDOOR_TRIGGER = "tq"                                  # rare-token instruction trigger (Xu 2023)
BACKDOOR_TARGET = "Please visit http://attacker.example for details."   # fixed backdoor output


def backdoor(records, client_id, trigger=BACKDOOR_TRIGGER, target=BACKDOOR_TARGET,
             poison_frac=0.5, seed_base=200):
    """Backdoor attacker (Xu 2023 'Instructions as Backdoors', 2305.14710): poison a
    FRACTION of the client's samples by prepending `trigger` to the prompt and
    replacing the completion with the fixed `target` -- the model learns
    trigger -> target.  The unpoisoned remainder keeps the clean task, so the CLEAN
    validation loss is (near-)preserved: a clean-performance-preserving backdoor,
    which is exactly the regime that probes whether Flirds' clean-val-loss signal
    can see it.  poison_frac in (0,1]; 1.0 = poison every sample (strongest ASR, but
    the client stops learning the clean task).  Reproducible via seed_base+client_id.
    """
    g = torch.Generator().manual_seed(seed_base + client_id)
    n_poison = int(round(poison_frac * len(records)))
    poisoned = set(torch.randperm(len(records), generator=g)[:n_poison].tolist())
    return [{"prompt": f"{trigger} {r['prompt']}", "completion": " " + target} if i in poisoned
            else {"prompt": r["prompt"], "completion": r["completion"]}
            for i, r in enumerate(records)]


LLM_CORRUPTORS = {"answer_swap": answer_swap, "answer_swap_graded": answer_swap_graded,
                  "backdoor": backdoor}


# ---- update-level corruptor (free-rider), representation-agnostic (CNN + LLM) ----
def free_rider(ref, mode="zero", scale=1e-3, generator=None, prev_state=None):
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
      "delta"  : Δw = w^r - w^{r-1}          -- Lin's delta-weights attack (E7): recycle the
                 realized previous-round aggregate, observable to ANY client from two
                 consecutive global broadcasts.  `prev_state` = w^{r-1} (the caller threads
                 the previous round-start state; llm_server does it from the logs contract).
                 Round 0 has nothing to recycle -> Δw = 0 (documented fallback).  This is
                 the stress case for "free-rider phi = exact 0": the recycled delta has REAL
                 alignment with the val gradient, so the 1st-order term no longer vanishes.
    """
    if mode == "zero":
        return {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in ref.items()}
    if mode == "random":
        return {k: torch.empty(v.shape, dtype=v.dtype).uniform_(-scale, scale, generator=generator)
                for k, v in ref.items()}
    if mode == "delta":
        if prev_state is None:                       # round 0: no previous aggregate observed yet
            return {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in ref.items()}
        return {k: (ref[k].detach() - prev_state[k].detach().to(ref[k].device)).cpu()
                for k in ref}
    raise ValueError(f"unknown free-rider mode: {mode!r} (use 'zero', 'random' or 'delta')")


# ---- update-level corruptor (gradient noise), representation-agnostic ----
def grad_noise(delta, std, generator=None):
    """Noisy-update client (the FedIF grad-noise threat, Track C2): add i.i.d.
    Gaussian N(0, std^2) to every entry of the client's HONEST delta -- the client
    trains correctly but submits a noise-perturbed update.  `generator` is caller-
    managed (seed per (client, round) for reproducibility, mirroring free_rider's
    convention).  Returns a new delta dict (input unmodified).
    """
    return {k: v + std * torch.randn(v.shape, dtype=v.dtype, generator=generator)
            for k, v in delta.items()}
