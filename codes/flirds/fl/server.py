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
                 seed, on_round=None, eval_fn=None, eval_every=1):
    """Backend-agnostic FedAvg over a generic state dict.

    init_state:     aggregatable params dict (CNN = state_dict; LLM = LoRA params).
    local_train_fn: (client_id, global_state) -> delta dict (init_state's keys).
    sample_nums[c]: client c's example count (the FedAvg weight n_c).
    on_round(r, w_r, deltas_map): per-round hook; w_r is the round-start state.
    Returns (final_state, history[(round, eval_fn(state))]).
    """
    rng = np.random.default_rng(seed)
    global_state = {k: v.clone() for k, v in init_state.items()}
    n_clients = len(sample_nums)
    k = max(1, round(sample_frac * n_clients))
    history = []
    for r in range(rounds):
        w_r = {key: v.clone() for key, v in global_state.items()} if on_round else None
        sel = rng.choice(n_clients, size=k, replace=False)
        deltas_map = {int(c): (local_train_fn(int(c), global_state), sample_nums[c])
                      for c in sel}
        w = np.array([sample_nums[c] for c in sel], dtype=float)
        w /= w.sum()
        for key in global_state:
            global_state[key] = global_state[key] + sum(
                wi * deltas_map[int(c)][0][key].to(global_state[key].device)
                for c, wi in zip(sel, w))
        if on_round is not None:
            on_round(r, w_r, deltas_map)
        if eval_fn is not None and (r + 1) % eval_every == 0:
            history.append((r + 1, eval_fn(global_state)))
    return global_state, history


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
           sample_frac=1.0, device="cuda", seed=0, eval_every=1, on_round=None):
    """CNN FedAvg (thin wrapper over `_fedavg_core`).  Returns (final_state, history).

    on_round(r, global_before, deltas_map) per round (deltas_map = {client_id:
    (delta, n_samples)}) feeds the in-run baselines / oracle.
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
                        eval_every=eval_every)


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
