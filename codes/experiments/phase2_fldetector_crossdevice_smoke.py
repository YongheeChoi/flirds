"""FLDetector cross-device synthetic smoke (model-free, no GPU -> CI guard).

Exercises the partial-participation adaptation (per-client gap-integrated HVP): a
client's previous update is from its LAST participation t' < r-1, not r-1, so the
fix predicts over the gap w^r - w^{t'}.  The old full-participation code KeyError'd
on the sparse cohort; this asserts the new path runs and still flags a poisoning
attacker.

Setup: N=20, K=8 partial participation, R=15.  Benign clients carry a stable
per-client signature (temporally CONSISTENT -> low prediction error); the attacker
(always present) submits a SCALED erratic update each round (Bagdasaryan model-
replacement style: large, temporally INCONSISTENT -> high prediction error).
Asserts the attacker is the single most-suspicious client (AUROC == 1).

Run from codes/:
  PYTHONPATH=. python experiments/phase2_fldetector_crossdevice_smoke.py
"""
import numpy as np
import torch

from flirds.baselines.fldetector import fldetector_from_logs
from flirds.eval.metrics import detection_auroc
from flirds.repro import seed_everything

N, R, K, P = 20, 15, 8, 1000
ATTACKER = 0


def main():
    seed_everything(0)
    g = torch.Generator().manual_seed(0)
    sig = {c: torch.randn(P, generator=g) * 0.01 for c in range(1, N)}   # stable benign signatures

    logs = []
    part = np.zeros(N, dtype=int)
    for r in range(R):
        others = torch.randperm(N - 1, generator=g)[: K - 1].add(1).tolist()   # K-1 of clients 1..N-1
        cohort = [ATTACKER] + sorted(others)
        dm = {}
        for c in cohort:
            if c == ATTACKER:
                delta = torch.randn(P, generator=g) * 0.05      # scaled + erratic (model-replacement)
            else:
                delta = sig[c] + torch.randn(P, generator=g) * 5e-4   # ~constant per client
            dm[c] = ({"w": delta}, 100)
            part[c] += 1
        logs.append(({"w": torch.zeros(P)}, dm))                # w_r unused by FLDetector

    score = fldetector_from_logs(logs, N, device="cpu")
    seen = [c for c in range(N) if part[c] > 0]
    print(f"participation: attacker={part[ATTACKER]}/{R}, benign(min/max)="
          f"({min(part[1:])},{max(part[1:])}) seen={len(seen)}/{N}")
    order = np.argsort(-score)
    print(f"top-5 suspicious: {order[:5].tolist()}  attacker={ATTACKER} score={score[ATTACKER]:.4f}")
    print(f"attacker rank (0=top): {int((score > score[ATTACKER]).sum())}")

    labels = [1 if c == ATTACKER else 0 for c in range(N)]
    auroc = detection_auroc(score, labels)
    print(f"AUROC (attacker vs benign, full-N)={auroc:.3f}")
    assert int((score > score[ATTACKER]).sum()) == 0, "attacker not the most suspicious"
    assert auroc == 1.0, f"expected AUROC 1.0, got {auroc}"
    print("\nFLDETECTOR CROSS-DEVICE SMOKE OK  (partial-participation gap path runs; scaled attacker flagged)")


if __name__ == "__main__":
    main()
