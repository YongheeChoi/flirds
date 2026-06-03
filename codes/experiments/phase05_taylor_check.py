"""Phase 0.5: direct 2nd-order Taylor validation (no Shapley) — isolates HVP quality.

Along the realized aggregate update direction dW = Σ p_k Δw_k of one round, scale by
t and compare the exact val-loss change to the 1st- and 2nd-order Taylor models:
  exact(t) = ell(w + t*dW) - ell(w)
  T1(t)    = t*<g, dW>
  T2(t)    = t*<g, dW> + 1/2 t^2 *<dW, H dW>
A correct 2nd-order term => |exact - T2| << |exact - T1| inside the Taylor radius.

Run from codes/:  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase05_taylor_check.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp
from torch.utils.data import DataLoader, TensorDataset

from flirds.data.cnn import get_dataset, get_labels
from flirds.fl.partition import dirichlet_partition
from flirds.fl.server import run_fedavg_logs
from flirds.models.cnn import LeNet5
from flirds.repro import seed_everything


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(0)
    N = 6
    train = get_dataset("mnist")
    test = get_dataset("mnist", train=False)
    idx = dirichlet_partition(get_labels(train), N, alpha=100.0, seed=0)
    idx = [i[:200] for i in idx]
    loaders = [DataLoader(TensorDataset(torch.stack([train[i][0] for i in idx[c]]),
                                        torch.tensor([train[i][1] for i in idx[c]])),
                          batch_size=200) for c in range(N)]   # full-batch -> single GD step
    vx = torch.stack([test[i][0] for i in range(512)]).to(device)
    vy = torch.tensor([test[i][1] for i in range(512)]).to(device)
    tl = DataLoader(test, batch_size=256)

    _, logs = run_fedavg_logs(LeNet5, loaders, tl, 1, 1, 0.1, device=device, seed=0)
    w0, dm = logs[0]
    pkeys = [n for n, _ in LeNet5().named_parameters()]
    ns = np.array([dm[k][1] for k in range(N)], dtype=float)
    p = ns / ns.sum()
    dW = {n: sum(p[k] * dm[k][0][n].float().to(device) for k in range(N)) for n in pkeys}
    params = {n: w0[n].float().to(device) for n in pkeys}
    buffers = {n: w0[n].to(device) for n in w0 if n not in pkeys}
    model = LeNet5().to(device)

    def vloss(pp):
        return F.cross_entropy(functional_call(model, (pp, buffers), (vx,)), vy)

    g, u = jvp(grad(vloss), (params,), (dW,))               # g = ∇ℓ, u = H·dW (1 HVP)
    gd = float(sum((g[n] * dW[n]).sum() for n in pkeys))     # <g, dW>
    dHd = float(sum((dW[n] * u[n]).sum() for n in pkeys))    # <dW, H dW>
    base = float(vloss(params))

    # HVP correctness: cross-check the jvp (forward-over-reverse) HVP against an
    # independent double-backward (reverse-over-reverse) HVP on the same direction.
    # Two distinct autodiff paths agreeing validates the code regardless of how
    # small the curvature is (FD is useless here -- H is near-flat, FD noise-bound).
    torch.manual_seed(0)
    v = {n: torch.randn_like(params[n]) for n in pkeys}
    _, hv_jvp = jvp(grad(vloss), (params,), (v,))
    pp = {n: params[n].clone().requires_grad_(True) for n in pkeys}
    g1 = torch.autograd.grad(vloss(pp), [pp[n] for n in pkeys], create_graph=True)
    dot = sum((gi * v[n]).sum() for gi, n in zip(g1, pkeys))
    g2 = torch.autograd.grad(dot, [pp[n] for n in pkeys])
    num = float(sum(((hv_jvp[n] - g2i) ** 2).sum() for n, g2i in zip(pkeys, g2))) ** 0.5
    den = float(sum((hv_jvp[n] ** 2).sum() for n in pkeys)) ** 0.5
    print(f"HVP check (jvp vs double-backward): rel||diff||={num / den:.3e}")

    print(f"<g,dW>={gd:.5f}  <dW,H dW>={dHd:.5f}  (curvature {'+' if dHd > 0 else '-'})")
    print(f"{'t':>5} {'exact':>11} {'T1':>11} {'T2':>11} {'|e-T1|':>10} {'|e-T2|':>10}")
    for t in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        pert = {n: params[n] + t * dW[n] for n in pkeys}
        exact = float(vloss(pert)) - base
        t1, t2 = t * gd, t * gd + 0.5 * t * t * dHd
        print(f"{t:>5.2f} {exact:>11.5f} {t1:>11.5f} {t2:>11.5f} "
              f"{abs(exact - t1):>10.5f} {abs(exact - t2):>10.5f}")


if __name__ == "__main__":
    main()
