"""FLTrust unit smoke (analytic loss_fn, no model/GPU -> CI guard).

A quadratic val loss  L(w) = 1/2 ||w - target||^2  has KNOWN gradient  ∇_val = w - target,
so the cosine-to-gradient orientation is checkable in closed form.  Client deltas:
  - benign   : descent  Δw = -(w_r - target) + noise   -> cos(Δw, ∇_val) ~ -1  (LOW  suspicion)
  - free-rider: zero / random                            -> cos ~ 0            (MID, above benign)
  - poison   : ascent   Δw = +(w_r - target)            -> cos ~ +1           (HIGH suspicion)
Asserts the ordering poison > free-rider > benign and AUROC == 1 for both the
free-rider-vs-benign and poison-vs-benign sub-problems (corrupt scores HIGH).

Run from codes/:
  PYTHONPATH=. python experiments/phase2_fltrust_unit_smoke.py
"""
import numpy as np
import torch

from flirds.baselines.fltrust import fltrust_from_logs
from flirds.eval.metrics import detection_auroc
from flirds.repro import seed_everything

N, R, P = 8, 4, 500
BENIGN, FR, POISON = [0, 1, 2, 3], [4, 5], [6, 7]   # 4=zero, 5=random free-rider
TARGET = 0.0


def main():
    seed_everything(0)
    g = torch.Generator().manual_seed(0)
    pkeys = ["w"]

    def loss_fn(params, buffers):
        return 0.5 * ((params["w"] - TARGET) ** 2).sum()

    logs = []
    for r in range(R):
        w_r = {"w": torch.ones(P)}                       # ∇_val = w_r - target = ones
        grad_dir = w_r["w"] - TARGET
        dm = {}
        for c in range(N):
            if c in POISON:
                delta = grad_dir.clone()                 # ascent
            elif c == 4:
                delta = torch.zeros(P)                   # zero free-rider
            elif c == 5:
                delta = torch.empty(P).uniform_(-1, 1, generator=g)   # random free-rider
            else:
                delta = -grad_dir + 0.05 * torch.randn(P, generator=g)  # benign descent
            dm[c] = ({"w": delta}, 100)
        logs.append((w_r, dm))

    score = fltrust_from_logs(logs, N, loss_fn, pkeys, device="cpu")
    print("per-client FLTrust score (high=suspicious; benign<0<poison, free-rider~0):")
    for c in range(N):
        tag = "benign" if c in BENIGN else ("free-rider" if c in FR else "POISON")
        print(f"  client {c}: {score[c]:+.4f}  [{tag}]")

    assert min(score[c] for c in POISON) > max(score[c] for c in FR) > max(score[c] for c in BENIGN), \
        "ordering poison > free-rider > benign violated"

    fr_clients = BENIGN + FR
    auroc_fr = detection_auroc([score[c] for c in fr_clients], [1 if c in FR else 0 for c in fr_clients])
    pois_clients = BENIGN + POISON
    auroc_p = detection_auroc([score[c] for c in pois_clients], [1 if c in POISON else 0 for c in pois_clients])
    print(f"AUROC free-rider-vs-benign={auroc_fr:.3f}  poison-vs-benign={auroc_p:.3f}")
    assert auroc_fr == 1.0 and auroc_p == 1.0, "FLTrust failed to separate corrupt from benign"
    print("\nFLTRUST UNIT SMOKE OK")


if __name__ == "__main__":
    main()
