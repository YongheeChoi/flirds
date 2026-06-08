"""STD-DAGMM synthetic unit smoke (model-free, no GPU/model -> CI guard).

Constructs fake logs with STRUCTURED benign updates (a shared low-rank "gradient"
direction + small noise -> reconstructs well through the AE bottleneck) and two
free-riders following Lin et al.'s taxonomy:
  - client 1 = ZERO update           (std anomaly: caught by the std feature);
  - client 2 = RANDOM @ benign std    (the evasion case: std matched, so std alone
    fails, but the random direction reconstructs poorly -> the recon/cosine terms
    catch it).
Asserts the detector ranks BOTH free-riders above every benign client
(energy top-2 / AUROC == 1) -- the "free-rider energy 최상위" check.

Run from codes/:
  PYTHONPATH=. python experiments/phase2_std_dagmm_synthetic_smoke.py
"""
import numpy as np
import torch

from flirds.baselines.std_dagmm import std_dagmm_from_logs
from flirds.eval.metrics import detection_auroc
from flirds.repro import seed_everything

N, R, P = 10, 10, 20000
FREE_RIDERS = {1: "zero", 2: "random"}   # client -> fabrication mode
SIG_A, SIG_N = 1.0, 0.002                # benign: direction weight std / coord noise std


def _benign(direction, g):
    """Structured benign update: a*direction + small i.i.d. noise (a ~ N(0,SIG_A))."""
    a = torch.randn(1, generator=g).item() * SIG_A
    return a * direction + torch.randn(P, generator=g) * SIG_N


def main():
    seed_everything(0)
    g = torch.Generator().manual_seed(0)
    direction = torch.randn(P, generator=g)
    direction = direction / direction.norm()

    # one pass to measure the benign update std, so the RANDOM free-rider can be
    # tuned to match it (uniform(-b,b) has std b/sqrt(3)) -- the evasion setting.
    benign_std = float(torch.stack([_benign(direction, torch.Generator().manual_seed(1000 + i)).std()
                                    for i in range(50)]).mean())
    b = benign_std * (3 ** 0.5)
    print(f"benign update std={benign_std:.5f} -> random free-rider range +/-{b:.5f} (std-matched)")

    logs = []
    for r in range(R):
        dm = {}
        for c in range(N):
            gc = torch.Generator().manual_seed(10_000 * r + c)   # reproducible per (client,round)
            if FREE_RIDERS.get(c) == "zero":
                delta = torch.zeros(P)
            elif FREE_RIDERS.get(c) == "random":
                delta = torch.empty(P).uniform_(-b, b, generator=gc)
            else:
                delta = _benign(direction, gc)
            dm[c] = ({"w": delta}, 100)          # (delta, n_c); n_c unused by the detector
        logs.append((None, dm))                  # w_r unused by the detector

    score = std_dagmm_from_logs(logs, N, proj_dim=128, seed=0)
    order = np.argsort(-score)                    # most -> least suspicious
    print("per-client energy (high=suspicious):")
    for c in range(N):
        tag = f" <- free-rider({FREE_RIDERS[c]})" if c in FREE_RIDERS else ""
        print(f"  client {c}: {score[c]:+.4f}{tag}")
    print(f"rank (suspicious first): {order.tolist()}")

    labels = [1 if c in FREE_RIDERS else 0 for c in range(N)]
    auroc = detection_auroc(score, labels)
    top = set(order[:len(FREE_RIDERS)].tolist())
    print(f"AUROC={auroc:.3f}  top-{len(FREE_RIDERS)}={sorted(top)}  free-riders={sorted(FREE_RIDERS)}")
    assert top == set(FREE_RIDERS), f"free-riders not top-ranked: {order.tolist()}"
    assert auroc == 1.0, f"expected AUROC 1.0, got {auroc}"
    print("\nSTD-DAGMM SYNTHETIC SMOKE OK")


if __name__ == "__main__":
    main()
