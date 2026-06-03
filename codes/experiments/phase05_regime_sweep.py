"""Phase 0.5 (A): where does the 2nd-order Taylor term help the in-run estimator?

For each {dataset, E, batch, lr} config we build a frozen FedAvg trajectory, take
the exact (b) in-run Shapley as ground truth, and compare the 1st-only vs 1st+2nd
Flirds estimator (Spearman / relL2).  The diagnostic axis is the per-round
curvature ratio  c = 1/2 <ΔW,H ΔW> / |<g,ΔW>|  (2nd-order term size vs 1st):
  c << 1  -> near-flat: 2nd-order negligible, 1st already ~exact
  c ~ O(.1-.5) -> in radius: 2nd-order should beat 1st
  c >> 1  -> beyond radius: 2nd-order overshoots, 1st+2nd worse

Run from codes/:  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase05_regime_sweep.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.func import functional_call, grad, jvp
from torch.utils.data import DataLoader, TensorDataset

from flirds.backends.cnn import make_cnn_loss
from flirds.core.flirds_estimator import flirds_values
from flirds.data.cnn import get_dataset, get_labels
from flirds.fl.partition import dirichlet_partition
from flirds.fl.server import run_fedavg_logs
from flirds.models.cnn import FedSVCNN, LeNet5
from flirds.oracle.in_run_sv import in_run_shapley
from flirds.repro import seed_everything


def loaders_for(dataset, N, n_per, batch, noisy, seed):
    train = get_dataset(dataset)
    idx = dirichlet_partition(get_labels(train), N, alpha=100.0, seed=seed)
    idx = [i[:n_per] for i in idx]
    out = []
    for c in range(N):
        xs = torch.stack([train[i][0] for i in idx[c]])
        ys = torch.tensor([train[i][1] for i in idx[c]])
        if c in noisy:
            g = torch.Generator().manual_seed(100 + c)
            ys = ys[torch.randperm(len(ys), generator=g)]
        bs = n_per if batch == "full" else batch
        out.append(DataLoader(TensorDataset(xs, ys), batch_size=bs, shuffle=(batch != "full")))
    return out


def curv_ratio(logs, model_fn, vx, vy, device):
    """Mean over rounds of  1/2 <ΔW,H ΔW> / |<g,ΔW>|  (2nd vs 1st order term size)."""
    model = model_fn().to(device)
    pkeys = [n for n, _ in model.named_parameters()]
    ns = np.array([logs[0][1][k][1] for k in range(len(logs[0][1]))], dtype=float)
    p = ns / ns.sum()
    ratios = []
    for w_r, dm in logs:
        params = {n: w_r[n].float().to(device) for n in pkeys}
        buffers = {n: w_r[n].to(device) for n in w_r if n not in pkeys}
        dW = {n: sum(p[k] * dm[k][0][n].float().to(device) for k in range(len(p))) for n in pkeys}

        def vloss(pp):
            return F.cross_entropy(functional_call(model, (pp, buffers), (vx,)), vy)
        g, u = jvp(grad(vloss), (params,), (dW,))
        gd = abs(float(sum((g[n] * dW[n]).sum() for n in pkeys)))
        dHd = float(sum((dW[n] * u[n]).sum() for n in pkeys))
        ratios.append(0.5 * dHd / gd if gd > 0 else float("nan"))
    return float(np.nanmean(ratios))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(0)
    N, rounds, n_per, noisy, seed = 8, 3, 300, {6, 7}, 0
    configs = [
        ("mnist", LeNet5, 1, "full", 0.05),
        ("mnist", LeNet5, 1, "full", 0.5),
        ("mnist", LeNet5, 1, 32, 0.05),
        ("mnist", LeNet5, 2, 32, 0.05),
        ("mnist", LeNet5, 2, 32, 0.2),
        ("cifar10", FedSVCNN, 1, "full", 0.05),
        ("cifar10", FedSVCNN, 1, 32, 0.05),
        ("cifar10", FedSVCNN, 2, 32, 0.05),
    ]
    print(f"N={N} rounds={rounds} n_per={n_per} noisy={sorted(noisy)}  (Spearman/relL2 vs (b) oracle)")
    print(f"{'dataset':8} {'E':>1} {'batch':>5} {'lr':>5} | {'curv':>7} | "
          f"{'sp_1st':>7} {'sp_1+2':>7} | {'rel_1st':>7} {'rel_1+2':>7} | {'2nd?':>5}")
    for dataset, mfn, E, batch, lr in configs:
        test = get_dataset(dataset, train=False)
        vx = torch.stack([test[i][0] for i in range(512)]).to(device)
        vy = torch.tensor([test[i][1] for i in range(512)]).to(device)
        tl = DataLoader(test, batch_size=256)
        loss_fn, pkeys = make_cnn_loss(mfn, vx, vy, device)
        ld = loaders_for(dataset, N, n_per, batch, noisy, seed)
        _, logs = run_fedavg_logs(mfn, ld, tl, rounds, E, lr, device=device, seed=seed)
        oracle, _ = in_run_shapley(logs, N, loss_fn, pkeys, device)
        p1, _ = flirds_values(logs, loss_fn, pkeys, device, second_order=False)
        p2, _ = flirds_values(logs, loss_fn, pkeys, device, second_order=True)
        c = curv_ratio(logs, mfn, vx, vy, device)
        s1, s2 = spearmanr(p1, oracle).correlation, spearmanr(p2, oracle).correlation
        r1 = np.linalg.norm(p1 - oracle) / np.linalg.norm(oracle)
        r2 = np.linalg.norm(p2 - oracle) / np.linalg.norm(oracle)
        flag = "help" if s2 > s1 + 1e-3 else ("hurt" if s2 < s1 - 1e-3 else "tie")
        bs = "full" if batch == "full" else str(batch)
        print(f"{dataset:8} {E:>1} {bs:>5} {lr:>5} | {c:>7.3f} | "
              f"{s1:>7.4f} {s2:>7.4f} | {r1:>7.4f} {r2:>7.4f} | {flag:>5}")


if __name__ == "__main__":
    main()
