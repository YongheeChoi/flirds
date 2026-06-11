"""FedIF unit smoke (analytic loss_fn, no model/GPU -> CI guard).

Mirror of phase2_fltrust_unit_smoke for the FedIF influence VALUE (baselines.fedif).
Quadratic val loss L(w)=1/2||w-target||^2 -> ∇_val = w-target known in closed form.
Client deltas over R rounds (full participation):
  - benign    : descent  Δw = -(w_r-target)+noise -> Φ=-<g,Δw>/||Δw|| > 0 -> HIGH omega
  - free-rider : zero / random                     -> Φ ~ 0 (tiny |Φ|)     -> MID  omega
  - poison     : ascent   Δw = +(w_r-target)       -> Φ < 0                -> LOW  omega
FedIF omega is good->HIGH.  Asserts the ordering benign > free-rider > poison, that
the free-rider is valued BELOW benign (NOT exact-0 -- min-max maps the zero update
to a mid value, unlike the delta-based oracle/Banzhaf), all finite, and AUROC == 1
on the NEGATED score (the good->low comparison convention; corrupt scores HIGH) for
free-rider-vs-benign and poison-vs-benign.

(The per-round min-max + cross-round EMA carry for NON-participants -- where FedIF
most diverges from FLTrust -- is exercised by the partial-participation device100
grid, not this full-participation unit smoke.)

Run from codes/:
  PYTHONPATH=. python experiments/phase2_fedif_smoke.py
"""
import numpy as np
import torch

from flirds.baselines.fedif import fedif_from_logs
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

    omega = fedif_from_logs(logs, N, loss_fn, pkeys, device="cpu")
    print("per-client FedIF omega (good=HIGH; benign > free-rider > poison):")
    for c in range(N):
        tag = "benign" if c in BENIGN else ("free-rider" if c in FR else "POISON")
        print(f"  client {c}: {omega[c]:.4f}  [{tag}]")

    assert np.isfinite(omega).all(), "non-finite omega"
    assert min(omega[c] for c in BENIGN) > max(omega[c] for c in FR) > max(omega[c] for c in POISON), \
        "ordering benign > free-rider > poison violated"

    susp = -omega                                        # good->low: corrupt scores HIGH
    fr_clients = BENIGN + FR
    auroc_fr = detection_auroc([susp[c] for c in fr_clients], [1 if c in FR else 0 for c in fr_clients])
    pois_clients = BENIGN + POISON
    auroc_p = detection_auroc([susp[c] for c in pois_clients], [1 if c in POISON else 0 for c in pois_clients])
    print(f"AUROC (negated) free-rider-vs-benign={auroc_fr:.3f}  poison-vs-benign={auroc_p:.3f}")
    assert auroc_fr == 1.0 and auroc_p == 1.0, "FedIF failed to separate corrupt from benign"
    print("\nFEDIF UNIT SMOKE OK")


if __name__ == "__main__":
    main()
