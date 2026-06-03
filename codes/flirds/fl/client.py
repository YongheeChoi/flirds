"""FL client: local SGD producing a weight delta Δw_k (state_dict diff)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def local_train(model, global_state, loader, epochs, lr, device, momentum=0.0):
    """Load global_state, run local SGD, return (delta, n_samples).

    delta[name] = local_param - global_param, kept on CPU.
    Plain SGD (momentum=0) by default: matches the per-step plain-SGD assumption of
    IRDS / Ripple Shapley (Eq 1), so the in-run drop term and Taylor estimator are
    faithful to the realized per-step displacement.
    """
    model.load_state_dict(global_state)
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    local = model.state_dict()
    delta = {k: (local[k] - global_state[k]).detach().cpu() for k in local}
    return delta, len(loader.dataset)
