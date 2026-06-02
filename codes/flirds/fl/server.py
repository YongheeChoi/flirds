"""Vanilla FedAvg server loop (Phase 0)."""
from __future__ import annotations

import numpy as np
import torch

from .client import local_train


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
    """Run vanilla FedAvg. Returns (final_global_state, history[(round, acc)]).

    on_round(r, global_before, deltas_map) is called each round (deltas_map =
    {client_id: (delta, n_samples)}); used by in-run SV baselines (GTG, ...).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = model_fn().to(device)
    global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    n_clients = len(client_loaders)
    k = max(1, round(sample_frac * n_clients))
    history = []
    for r in range(rounds):
        global_before = ({key: v.clone() for key, v in global_state.items()}
                         if on_round is not None else None)
        sel = rng.choice(n_clients, size=k, replace=False)
        deltas, weights, deltas_map = [], [], {}
        for c in sel:
            d, n = local_train(model, global_state, client_loaders[c],
                               local_epochs, lr, device)
            deltas.append(d)
            weights.append(n)
            deltas_map[int(c)] = (d, n)
        w = np.asarray(weights, dtype=float)
        w /= w.sum()
        for key in global_state:
            agg = sum(wi * deltas[i][key].to(device) for i, wi in enumerate(w))
            global_state[key] = global_state[key] + agg
        if on_round is not None:
            on_round(r, global_before, deltas_map)
        if (r + 1) % eval_every == 0:
            acc = evaluate(model, global_state, test_loader, device)
            history.append((r + 1, acc))
    return global_state, history


def run_fedavg_logs(model_fn, client_loaders, test_loader, rounds, local_epochs,
                    lr, device="cuda", seed=0):
    """Run FedAvg once; return (eval_model, logs[(global_before, deltas_map)]).

    The single trajectory shared by the reconstruction-utility baselines
    (GTG/FedSV) and the exact reconstruction oracle, so they value the same FL
    run. (Ripple runs its own loop because it additionally needs Hessian sketches.)
    """
    logs = []
    fedavg(model_fn, client_loaders, test_loader, rounds, local_epochs, lr,
           sample_frac=1.0, device=device, seed=seed,
           on_round=lambda r, gb, dm: logs.append((gb, dm)))
    return model_fn().to(device), logs
