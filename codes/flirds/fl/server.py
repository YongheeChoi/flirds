"""FedAvg server: backend-agnostic core + CNN wrapper (Phase 1).

`_fedavg_core` runs the round loop, partial-participation selection, per-round
participant-normalized aggregate, and on_round logging over a generic params
dict + an injected `local_train_fn` — shared by the CNN wrapper (`run_fedavg_logs`
below) and the LLM wrapper (`fl/llm_server.run_llm_fedavg_logs`).  The contract
`logs = [(w_r, deltas_map)]` (deltas_map = {client: (delta, n)}) is exactly what
the estimator/oracle consume, identical for both backends.
"""
from __future__ import annotations

import numpy as np
import torch

from ..repro import seed_everything
from .client import local_train


def _fedavg_core(init_state, local_train_fn, sample_nums, rounds, sample_frac,
                 seed, on_round=None, eval_fn=None, eval_every=1,
                 select_fn=None, weights_fn=None, delta_transform=None):
    """Backend-agnostic FedAvg over a generic state dict.

    init_state:     aggregatable params dict (CNN = state_dict; LLM = LoRA params).
    local_train_fn: (client_id, global_state) -> delta dict (init_state's keys).
    sample_nums[c]: client c's example count (the FedAvg weight n_c).
    on_round(r, w_r, deltas_map): per-round hook; w_r is the round-start state.
    Intervention seam (Track C2/D; fl/intervene.py builds these):
      select_fn(r, k, rng) -> k client ids   (default: uniform w/o replacement)
      weights_fn(r, w_r, deltas_map) -> {client: weight}, normalized aggregation
        weights (default: n_c / sum n -- plain FedAvg).
      delta_transform(c, r, delta) -> delta, update-level corruption seam (Track
        C2 free-rider / grad-noise); applied to a client's honest delta before it
        enters deltas_map.  Default: identity.
    All None => behavior is bit-identical to the pre-seam loop.
    Returns (final_state, history[(round, eval_fn(state))]).
    """
    rng = np.random.default_rng(seed)
    global_state = {k: v.clone() for k, v in init_state.items()}
    n_clients = len(sample_nums)
    k = max(1, round(sample_frac * n_clients))
    history = []
    for r in range(rounds):
        w_r = ({key: v.clone() for key, v in global_state.items()}
               if (on_round is not None or weights_fn is not None) else None)
        sel = (rng.choice(n_clients, size=k, replace=False) if select_fn is None
               else np.asarray(select_fn(r, k, rng)))
        deltas_map = {}
        for c in sel:
            d = local_train_fn(int(c), global_state)
            if delta_transform is not None:
                d = delta_transform(int(c), r, d)
            deltas_map[int(c)] = (d, sample_nums[c])
        if weights_fn is None:
            w = np.array([sample_nums[c] for c in sel], dtype=float)
            w /= w.sum()
        else:
            wmap = weights_fn(r, w_r, deltas_map)
            w = np.array([wmap[int(c)] for c in sel], dtype=float)
        for key in global_state:
            global_state[key] = global_state[key] + sum(
                wi * deltas_map[int(c)][0][key].to(global_state[key].device)
                for c, wi in zip(sel, w))
        if on_round is not None:
            on_round(r, w_r, deltas_map)
        if eval_fn is not None and (r + 1) % eval_every == 0:
            history.append((r + 1, eval_fn(global_state)))
    return global_state, history


def subset_delta_transform(transform, subset):
    """Re-index a `delta_transform` onto a COALITION (Track C1 (a)-oracle / removal
    retrains, 2026-07-25).  `_fedavg_core` hands the seam the client's POSITION in the
    loader list it was given, so a run over `[loaders[c] for c in subset]` would apply
    client `subset[i]`'s threat to position `i`.  A threat is a property of the client
    -- a free-rider free-rides in every coalition it joins -- so map the position back
    to the global id.  `transform=None` passes through (no threat).  Lives here rather
    than in `fl.intervene` (where the transforms are built) to stay import-cycle-free:
    `oracle.exact_sv` needs it, and intervene imports the baselines."""
    if transform is None:
        return None
    ids = tuple(subset)

    def sub(i, r, delta):
        return transform(ids[i], r, delta)

    return sub


@torch.no_grad()
def evaluate(model, state, loader, device):
    model.load_state_dict(state)
    model.to(device).eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def fedavg(model_fn, client_loaders, test_loader, rounds, local_epochs, lr,
           sample_frac=1.0, device="cuda", seed=0, eval_every=1, on_round=None,
           select_fn=None, weights_fn=None, delta_transform=None):
    """CNN FedAvg (thin wrapper over `_fedavg_core`).  Returns (final_state, history).

    on_round(r, global_before, deltas_map) per round (deltas_map = {client_id:
    (delta, n_samples)}) feeds the in-run baselines / oracle.  select_fn /
    weights_fn = the intervention seam (see `_fedavg_core`).
    """
    seed_everything(seed, cudnn_deterministic=True)   # CNN track: deterministic conv
    model = model_fn().to(device)
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sample_nums = [len(ld.dataset) for ld in client_loaders]

    def local_train_fn(c, global_state):
        delta, _ = local_train(model, global_state, client_loaders[c],
                               local_epochs, lr, device)
        return delta

    return _fedavg_core(init_state, local_train_fn, sample_nums, rounds, sample_frac,
                        seed, on_round=on_round,
                        eval_fn=lambda st: evaluate(model, st, test_loader, device),
                        eval_every=eval_every,
                        select_fn=select_fn, weights_fn=weights_fn,
                        delta_transform=delta_transform)


def run_fedavg_logs(model_fn, client_loaders, test_loader, rounds, local_epochs,
                    lr, device="cuda", seed=0):
    """Run CNN FedAvg once; return (eval_model, logs[(w_r, deltas_map)]).

    The single trajectory shared by the reconstruction-utility baselines
    (GTG/FedSV) and the in-run oracle/estimator, so they value the same FL run.
    """
    logs = []
    fedavg(model_fn, client_loaders, test_loader, rounds, local_epochs, lr,
           sample_frac=1.0, device=device, seed=seed,
           on_round=lambda r, gb, dm: logs.append((gb, dm)))
    return model_fn().to(device), logs
