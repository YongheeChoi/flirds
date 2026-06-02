"""FL client: local SGD producing a weight delta Δw_k (state_dict diff)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def local_train(model, global_state, loader, epochs, lr, device, momentum=0.9):
    """Load global_state, run local SGD, return (delta, n_samples).

    delta[name] = local_param - global_param, kept on CPU.
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
