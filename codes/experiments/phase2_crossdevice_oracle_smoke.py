"""Phase 2 task 7c (b)-oracle smoke: prove the per-round decomposition == exact 2^N.

in_run_shapley_perround is the cross-device (large-N) path for the (b) in-run
Shapley.  Part 1 (synthetic, fast, no GPU): on small N with a quadratic val loss
(genuine coalition interactions via the quadratic cross-terms), assert
in_run_shapley_perround == in_run_shapley (the 2^N enumeration) for BOTH full and
partial participation -- the equivalence the per-round decomposition claims.
"""
import numpy as np
import torch

from flirds.oracle.in_run_sv import in_run_shapley, in_run_shapley_perround
from flirds.repro import seed_everything

PKEYS = ["w"]


def quad_loss(target):
    def loss_fn(params, buffers):
        return 0.5 * ((params["w"] - target) ** 2).sum()   # plain forward (no autograd)
    return loss_fn


def make_logs(n_clients, rounds, K, dim, seed):
    """Synthetic frozen trajectory: w_r + per-client deltas as small tensors,
    K participants/round (K == n_clients -> full participation)."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    logs = []
    for _ in range(rounds):
        w_r = {"w": torch.randn(dim, generator=g)}
        sel = sorted(rng.choice(n_clients, size=K, replace=False).tolist())
        dm = {c: ({"w": 0.1 * torch.randn(dim, generator=g)}, int(rng.integers(50, 200)))
              for c in sel}
        logs.append((w_r, dm))
    return logs


def check(n_clients, rounds, K, dim, seed):
    logs = make_logs(n_clients, rounds, K, dim, seed)
    loss_fn = quad_loss(torch.randn(dim, generator=torch.Generator().manual_seed(seed + 99)))
    phi_x, p_x = in_run_shapley(logs, n_clients, loss_fn, PKEYS, "cpu")
    phi_r, p_r = in_run_shapley_perround(logs, n_clients, loss_fn, PKEYS, "cpu")
    md = float(np.max(np.abs(phi_x - phi_r)))
    part = "full" if K == n_clients else f"partial(K={K})"
    print(f"  N={n_clients:>2} {part:>11} R={rounds}: max|Δφ|={md:.2e}  p_match={np.allclose(p_x, p_r)}")
    assert np.allclose(phi_x, phi_r, atol=1e-10), f"perround != exact (max {md})"
    assert np.allclose(p_x, p_r)


def main():
    seed_everything(0)
    print("=== Part 1: in_run_shapley_perround == in_run_shapley (synthetic quadratic loss) ===")
    check(5, 6, 5, 4, seed=0)      # full participation (cross-silo)
    check(8, 8, 3, 4, seed=1)      # partial participation (cross-device-like)
    check(10, 10, 4, 5, seed=2)    # partial, larger
    print("  Part 1 OK (per-round decomposition exactly reproduces the 2^N oracle)\n")
    print("CROSS-DEVICE ORACLE SMOKE OK (synthetic)")


if __name__ == "__main__":
    main()
