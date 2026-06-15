"""Probe: time full-scale (a)-oracle retrains on this cluster's GPU.

Times subset_utility_valloss at full C1 scale (R=10, E=5) for representative
coalition sizes, then extrapolates the 2^10 sweep cost. Run from codes/ with
PYTHONPATH=. -- writes timing lines to stdout.
"""
import os
import time

os.environ["C1_MODE"] = "full"
os.environ["C1_SEED"] = "0"
os.environ["C1_SCENARIO"] = "iid"

import torch  # noqa: E402

from flirds.oracle.exact_sv import subset_utility_valloss  # noqa: E402
from flirds.repro import seed_everything  # noqa: E402

for ds in ["mnist", "cifar10"]:
    os.environ["C1_DATASET"] = ds
    import importlib

    import experiments.track_c1 as c1
    importlib.reload(c1)
    seed_everything(0, cudnn_deterministic=True)
    loaders, rates, vx, vy, val_loader, test_loader = c1.build(
        ds, "iid", 10, None, c1.CFG["batch"], c1.CFG["n_val"], c1.CFG["n_test"], 0)
    R, E, lr = c1.CFG["rounds"], c1.CFG["epochs"], c1.CFG["lr"]
    print(f"[{ds}] sizes={[len(l.dataset) for l in loaders]} R={R} E={E}", flush=True)
    timings = {}
    # warmup (cudnn autotune/jit) on a small coalition, untimed
    subset_utility_valloss(c1.MODEL_FN, loaders, val_loader, (0, 1), R, E, lr,
                           device="cuda", seed=0)
    for S in [(0, 1), (0, 2, 4, 6, 8), tuple(range(10))]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        u = subset_utility_valloss(c1.MODEL_FN, loaders, val_loader, S, R, E, lr,
                                   device="cuda", seed=0)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        timings[len(S)] = dt
        print(f"[{ds}] |S|={len(S):2d} retrain={dt:6.1f}s u={u:+.4f}", flush=True)
    # cost model: t(|S|) ~= a + b*|S|; total = sum_k C(10,k) * t(k) = 1024*a + 5120*b
    b = (timings[10] - timings[2]) / 8.0
    a = timings[5] - 5.0 * b
    total = 1024 * a + 5120 * b
    print(f"[{ds}] est 2^10 sweep: {total/3600:.2f}h  (a={a:.2f}s b={b:.2f}s/client)",
          flush=True)
print("PROBE OK", flush=True)
