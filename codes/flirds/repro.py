"""Reproducibility — call seed_everything(seed) at every run/experiment entry point.

Seeds torch + numpy + CUDA.  cudnn_deterministic=True (CNN track only) forces
deterministic conv so the same config + seed reproduces bitwise-identical fp32
results (protocol 5); the LLM track is conv-free and does not need it.  The CNN FL
core (fl.server.fedavg, ripple_shapley) passes cudnn_deterministic=True.
"""
from __future__ import annotations

import numpy as np
import torch


def seed_everything(seed=0, cudnn_deterministic=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
