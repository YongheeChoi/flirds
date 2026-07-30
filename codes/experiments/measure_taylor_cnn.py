"""Taylor-remainder measurement on the CNN track, in the setting of main Table 1.

Table 1 (tab:retrain-fidelity) is CIFAR-10 / Dirichlet(alpha=1), N=10, FULL
participation, R=10, over four conditions (clean / zero-update / gradient noise /
label-flip) and three seeds.  That is exactly track_c1's `full` mode on the C-b
threat/partition axes, so this script does not reimplement the stage: it sets the
C1_* environment, imports track_c1, and calls track_c1.build() -- so the
partition, the corrupt-client draw, the doses, the delta-level threats, and the
val/test split are bit-identical to the numbers in Table 1.  The measured
quantity itself lives in taylor_core.

Why the CNN track matters here.  Appendix A.4 states the remainder bound only for
the LLM track, because it is conditional on a C^3 assumption that ReLU + max-pool
violates; the CNN fidelity claim rests on empirical results alone.  And C.5's LLM
measurement is floor-limited -- its second-order residual sits only 2.1x above
the fp32 evaluation floor, so it bounds the remainder's magnitude but cannot
confirm its order.  The CNN round displacement is two orders of magnitude larger
(5 local epochs on all parameters, vs 10 steps on LoRA factors), so the remainder
clears the same floor by four to five orders of magnitude and the order becomes
measurable.

Cost per (condition, seed): 2^10 = 1024 coalition forwards + 10 HVPs per round,
x 10 rounds, on a 2,000-example validation set.  The FL trajectory is regenerated
rather than loaded -- no checkpoints are persisted anywhere in runs/ -- but that
is a deterministic replay at the same seed, not a retrain, and on this track it
is the cheaper half of the job.

Run from codes/:
  # one cell
  PYTHONPATH=. TAYLOR_THREAT=clean TAYLOR_SEED=0 python -u experiments/measure_taylor_cnn.py
  # the full Table 1 grid: see runs/taylor_remainder/run_table1.sh
  # wiring smoke (seconds, no persist)
  PYTHONPATH=. TAYLOR_SMOKE=1 TAYLOR_PERSIST=0 python -u experiments/measure_taylor_cnn.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

# ---- Table 1's stage, pinned BEFORE importing track_c1 (its config is env-driven) ----
# Table 1 caption: "CIFAR-10, Dirichlet(alpha=1); N=10, full participation, R=10".
SMOKE = os.environ.get("TAYLOR_SMOKE", "0") == "1"
THREAT = os.environ.get("TAYLOR_THREAT", "clean")
SEED = int(os.environ.get("TAYLOR_SEED", "0"))
DATASET = os.environ.get("TAYLOR_DATASET", "cifar10")
PARTITION = os.environ.get("TAYLOR_PARTITION", "dir1")
PERSIST = os.environ.get("TAYLOR_PERSIST", "1") == "1"
RENORM = os.environ.get("TAYLOR_RENORM", "0") == "1"
# Rounds to measure: "all" (Table 1 has only 10, so all is the default) or an int
# stride/count for the larger settings, where the remainder is a per-round quantity
# and an evenly spaced subsample carries the same statistics at a fraction of cost.
N_MEASURE = os.environ.get("TAYLOR_N_MEASURE", "all")

_THREATS = {"clean": "clean",                 # Table 1 column -> track_c1 C1_THREAT
            "zero-update": "free_rider",
            "free_rider": "free_rider",
            "gradient-noise": "grad_noise",
            "grad_noise": "grad_noise",
            "label-flip": "label_flip",
            "label_flip": "label_flip"}
if THREAT not in _THREATS:
    raise SystemExit(f"TAYLOR_THREAT must be one of {sorted(_THREATS)}, got {THREAT!r}")

os.environ["C1_MODE"] = "smoke" if SMOKE else "full"
os.environ["C1_DATASET"] = DATASET
os.environ["C1_PARTITION"] = PARTITION          # setting either C1_PARTITION or
os.environ["C1_THREAT"] = _THREATS[THREAT]      # C1_THREAT switches on the C-b axes
os.environ["C1_SEED"] = str(SEED)
os.environ.setdefault("C1_ORACLE_A", "0")       # this script measures the remainder only
os.environ.setdefault("C1_PERSIST", "0")        # we own persistence, not track_c1's main()

import track_c1 as c1                                                    # noqa: E402
from flirds.backends.cnn import make_cnn_loss                            # noqa: E402
from flirds.fl.server import fedavg                                      # noqa: E402
from flirds.repro import seed_everything                                 # noqa: E402
from flirds.run_logger import RunLogger                                  # noqa: E402
from taylor_core import measure_round, pool                              # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("TAYLOR_RUN_ROOT",
                          os.path.join(_REPO, "runs", "taylor_remainder", "rundirs"))


def _measure_rounds(n_rounds):
    """Which round indices to measure -- all of them, or an evenly spaced subsample."""
    if N_MEASURE == "all":
        return list(range(n_rounds))
    k = min(int(N_MEASURE), n_rounds)
    return sorted(set(np.linspace(0, n_rounds - 1, k).round().astype(int).tolist()))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = c1.CFG
    n, R, E, lr = cfg["n_clients"], cfg["rounds"], cfg["epochs"], cfg["lr"]
    seed_everything(SEED, cudnn_deterministic=True)

    # ---- Table 1's data stage, verbatim from track_c1 ----
    loaders, rates, delta_transform, vx, vy, _val_loader, test_loader = c1.build(
        DATASET, c1.SCENARIO, n, cfg["n_per"], cfg["batch"], cfg["n_val"], cfg["n_test"], SEED)
    corrupt = sorted(c for c, rate in enumerate(rates) if rate > 0)
    print(f"# {DATASET}/{PARTITION} threat={_THREATS[THREAT]} seed={SEED} "
          f"N={n} full-participation R={R} E={E} lr={lr} val={cfg['n_val']} "
          f"| sizes={[len(l.dataset) for l in loaders]} corrupt={corrupt}", flush=True)

    # ---- frozen trajectory (regenerated, not loaded: no checkpoints are persisted) ----
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    logs = []
    fedavg(c1.MODEL_FN, loaders, test_loader, R, E, lr, sample_frac=c1.KFRAC, device=device,
           seed=SEED, on_round=lambda r, gb, dm: logs.append((gb, dm)),
           delta_transform=delta_transform)
    if device == "cuda":
        torch.cuda.synchronize()
    t_fl = time.perf_counter() - t0
    print(f"[fl] {len(logs)} rounds in {t_fl:.1f}s (full participation, K={len(logs[0][1])})",
          flush=True)

    loss_fn, pkeys = make_cnn_loss(c1.MODEL_FN, vx, vy, device)

    # ---- per-round measurement ----
    want = _measure_rounds(len(logs))
    all_rows, summaries = [], []
    t0 = time.perf_counter()
    for r in want:
        w_r, dm = logs[r]
        rows, _phis, summ = measure_round(r, w_r, dm, loss_fn, pkeys, device,
                                          loss_chunks=None, renorm=RENORM, sweep=True)
        all_rows += rows
        summaries.append(summ)
        print(f"[round {r}] base={summ['base_loss']:.6f} ulp={summ['ulp_base']:.2e} "
              f"||dW||={summ['norm_dW']:.4g} med|u-u1|={summ['resid1']['median']:.3e} "
              f"med|u-u2|={summ['resid2']['median']:.3e} "
              f"r2/floor={summ['resid2_over_ulp']:.0f}x "
              f"slope2(coal)={summ['loglog_slope_r2']:.2f} "
              f"slope2(sweep)={summ['sweep_slope_r2_above_floor']}", flush=True)
    t_meas = time.perf_counter() - t0

    P = pool(summaries, all_rows)
    print(f"\n=== POOLED  ({DATASET}/{PARTITION} {_THREATS[THREAT]} seed={SEED}) ===")
    print(f"  resid1 median              {P['resid1']['median']:.4e}")
    print(f"  resid2 median              {P['resid2']['median']:.4e}")
    print(f"  fp32 floor (ulp of base)   {P['ulp']:.4e}")
    print(f"  resid2 / floor             {P['resid2_median_over_ulp']:,.0f}x"
          f"   (LLM C.5: 2.1x)")
    print(f"  frac(resid2 <= resid1)     {P['frac_t2_le_t1']:.3f}")
    print(f"  slope resid1 (coalition)   {P['loglog_slope_r1']:.2f}   predicted 2")
    print(f"  slope resid2 (coalition)   {P['loglog_slope_r2']:.2f}   predicted 3")
    print(f"  slope resid1 (sweep)       {P['sweep_slope_r1']}   predicted 2")
    print(f"  slope resid2 (sweep)       {P['sweep_slope_r2_above_floor']}   predicted 3")
    print(f"  mean |u_r(P_r)|            {P['mean_abs_u_grand']:.4e}")
    print(f"  closed form vs Shapley(u2) {P['max_phi_t2_vs_closed']:.3e}  (Thm 1 numeric check)")
    print(f"  timing                     FL {t_fl:.1f}s | measure {t_meas:.1f}s "
          f"({100 * t_fl / (t_fl + t_meas):.0f}% training)")

    if not PERSIST:
        print("TAYLOR-CNN OK (no persist)", flush=True)
        return

    name = f"{DATASET}_{PARTITION}_{_THREATS[THREAT]}_taylor_seed{SEED}"
    rl = RunLogger(RUN_ROOT, name,
                   dict(track="cnn", table="tab:retrain-fidelity", dataset=DATASET,
                        partition=PARTITION, threat=_THREATS[THREAT], threat_label=THREAT,
                        seed=SEED, cfg=cfg, kfrac=c1.KFRAC, corrupt=corrupt, rates=rates,
                        flip_rate=c1.FLIP_RATE, mal_frac=c1.MAL_FRAC,
                        grad_noise_std=c1.GRAD_NOISE_STD,
                        renorm=RENORM, rounds_measured=want, smoke=SMOKE),
                   repo_root=_REPO)
    rl.save_phi(all_rows, fname="coalitions.parquet")
    rl.save_metrics(dict(pooled=P, rounds=summaries,
                         timing=dict(fl_train_s=round(t_fl, 1), measure_s=round(t_meas, 1),
                                     train_frac=t_fl / (t_fl + t_meas))))
    with open(os.path.join(rl.dir, "summary.json"), "w") as f:
        json.dump(dict(pooled=P, rounds=summaries), f, indent=1)
    print(f"[persist] {rl.dir}", flush=True)
    print("TAYLOR-CNN OK", flush=True)


if __name__ == "__main__":
    main()
