"""CNN backend: builds the (loss_fn, pkeys) pair the estimator/oracle consume.

loss_fn(params, buffers) -> cross-entropy val loss via functional_call (eval
mode, fp32).  pkeys = all named_parameters (LeNet5 / FedSVCNN are fully
trainable, no LoRA — so every param is a Taylor variable, buffers held fixed).
"""
from __future__ import annotations

import torch.nn.functional as F
from torch.func import functional_call


def make_cnn_loss(model_fn, val_x, val_y, device):
    """Return (loss_fn, pkeys) for the CNN backend.

    loss_fn(params, buffers) re-evaluates the val loss with `params` injected
    (functional_call leaves the captured model unmodified, so the same closure is
    reused by the grad-tracked estimator and the no-grad oracle)."""
    model = model_fn().to(device)
    model.eval()                       # BN/Dropout use running stats (matches (b) oracle)
    val_x, val_y = val_x.to(device), val_y.to(device)
    pkeys = [n for n, _ in model.named_parameters()]

    def loss_fn(params, buffers):
        return F.cross_entropy(functional_call(model, (params, buffers), (val_x,)), val_y)

    return loss_fn, pkeys
