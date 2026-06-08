"""Phase 2 task 7d ComFedSV LLM-port smoke: validate the (loss_fn, pkeys) path.

Part 1a (synthetic, no GPU): full-participation logs + quadratic val loss.
ComFedSV(partial=False, high M) is permutation-MC Shapley over the UNIFORM-subset
utility -> assert it matches an exact 2^n uniform-Shapley reference (Spearman +1,
values within MC tol) -- this validates the PORTED metric + marginal logic.
Part 1b: partial-participation logs (cohort-only deltas) -> partial=True low-rank
completion runs, returns a finite n-vector valuing ALL clients (the LLM metric
feeds the CNN-verified completion path; ComFedSV's point vs FedSV's zero-credit).
"""
from itertools import combinations
from math import factorial

import numpy as np
import torch
from scipy.stats import spearmanr

from flirds.baselines.comfedsv import comfedsv_from_logs
from flirds.repro import seed_everything

PKEYS = ["w"]


def quad_loss(target):
    def loss_fn(params, buffers):
        return 0.5 * ((params["w"] - target) ** 2).sum()
    return loss_fn


def make_logs(n, rounds, dim, seed, k=None):
    """Full-participation (k=None) or K-of-N (k<n, cohort-only deltas) frozen logs."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    logs = []
    for _ in range(rounds):
        w_r = {"w": torch.randn(dim, generator=g)}
        cohort = range(n) if k is None else sorted(rng.choice(n, size=k, replace=False).tolist())
        dm = {c: ({"w": 0.2 * torch.randn(dim, generator=g)}, 100) for c in cohort}
        logs.append((w_r, dm))
    return logs


def exact_uniform_shapley(logs, n, loss_fn, pkeys):
    """Exact 2^n Shapley over U(S) = Σ_t [loss(w_t) - loss(w_t + mean_{k∈S∩P_t} Δw_k)]."""
    def U(S):
        sset, tot = set(S), 0.0
        for w_r, dm in logs:
            players = [c for c in dm if c in sset]
            base = float(loss_fn({k: w_r[k] for k in pkeys}, {}))
            if players:
                params = {k: w_r[k].clone() for k in pkeys}
                for c in players:
                    for k in pkeys:
                        params[k] = params[k] + dm[c][0][k] / len(players)
                tot += base - float(loss_fn(params, {}))
        return tot
    phi, cl = np.zeros(n), list(range(n))
    for k in cl:
        others = [c for c in cl if c != k]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for S in combinations(others, r):
                phi[k] += w * (U(tuple(sorted(S + (k,)))) - U(S))
    return phi


def main():
    seed_everything(0)
    n, rounds, dim = 6, 8, 4
    loss_fn = quad_loss(torch.randn(dim, generator=torch.Generator().manual_seed(123)))
    print("=== Part 1a: ComFedSV LLM partial=False (MC) == exact uniform-Shapley ===")
    logs_full = make_logs(n, rounds, dim, seed=0)
    exact = exact_uniform_shapley(logs_full, n, loss_fn, PKEYS)
    com_f = comfedsv_from_logs(logs_full, None, n, None, "cpu", seed=0, n_mc=8000,
                               partial=False, loss_fn=loss_fn, pkeys=PKEYS)
    rho_f, md = float(spearmanr(exact, com_f).correlation), float(np.max(np.abs(exact - com_f)))
    print(f"  exact   : {np.round(exact, 4)}")
    print(f"  ComFedSV: {np.round(com_f, 4)}  (partial=False, M=8000)")
    print(f"  Spearman={rho_f:+.4f}  max|Δ|={md:.2e}")
    # rank is the correctness proof (== exact uniform-Shapley); abs gap is residual MC noise
    assert rho_f > 0.999 and md < 2e-2, "partial=False MC must match exact uniform-Shapley"

    print("=== Part 1b: ComFedSV LLM partial=True completion (K=3 of 6) wiring ===")
    logs_part = make_logs(n, rounds, dim, seed=1, k=3)
    com_p = comfedsv_from_logs(logs_part, None, n, None, "cpu", seed=0, n_mc=200, rank=4,
                               partial=True, loss_fn=loss_fn, pkeys=PKEYS)
    print(f"  ComFedSV: {np.round(com_p, 4)}  (partial=True, M=200 rank=4)")
    assert com_p.shape == (n,) and np.all(np.isfinite(com_p)), "completion output malformed"
    assert np.ptp(com_p) > 0, "completion gave a constant valuation"
    print("  Part 1b OK (LLM metric feeds completion; all clients valued)\n")
    print("COMFEDSV LLM SMOKE OK (synthetic)")


if __name__ == "__main__":
    main()
