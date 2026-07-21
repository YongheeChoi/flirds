"""Reproducibility — call seed_everything(seed) at every run/experiment entry point.

Seeds Python / NumPy / torch / CUDA and, best-effort, ENFORCES deterministic
algorithms (torch.use_deterministic_algorithms(warn_only=True) + a deterministic
cuBLAS workspace) so the same config + seed reproduces the same result and any
residual nondeterministic op is surfaced as a warning instead of silently tolerated.
warn_only=True never aborts a run — an op that lacks a deterministic kernel just
warns and falls back.

For the CNN track pass cudnn_deterministic=True to additionally force deterministic,
TF32-free conv → bitwise-identical fp32 on a fixed GPU architecture (protocol 5); the
LLM track is conv-free and does not need it.  The CNN FL core (fl.server.fedavg,
ripple_shapley) passes cudnn_deterministic=True.

PYTHONHASHSEED cannot be changed from inside a running interpreter, so it is NOT set
here — export PYTHONHASHSEED=0 in the launch environment for fully deterministic str
hashing.  (The repo's set iteration is int-membership only, so this is only
belt-and-suspenders; verified 2026-07-21.)
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

# Deterministic cuBLAS requires this to be set BEFORE the first CUDA context is
# created; module import is the earliest hook (repro is imported at the top of every
# entry point).  setdefault so an explicit launch-time value still wins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_everything(seed=0, cudnn_deterministic=False):
    random.seed(seed)                      # stdlib (unused today, cheap insurance)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:                                   # warn_only: surface nondeterminism, never crash a run
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:                      # very old torch / unsupported build -> skip enforcement
        pass
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False   # true fp32 conv (else TF32 on Ampere+, not bitwise-fp32)
