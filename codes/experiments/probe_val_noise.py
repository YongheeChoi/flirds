"""Signal-size probe (4-i): val-measurement noise of the (b) oracle via chunk bootstrap.

Question (wiki/flirds-signal-size-diagnosis.md §1.4-i): the coalition utilities /
phi are means over ONE fixed val set -- if the val set were redrawn, would the
oracle's client ranking survive?  Existing rundirs keep no checkpoints, so this
probe re-trains ONE anchor5 vanilla trajectory and computes the exact 2^N
coalition utilities CHUNK-RESOLVED (the same val chunks make_llm_loss uses),
then bootstraps over chunks:
  - SE(phi_i) and SE of pairwise phi differences vs the observed spread
  - Spearman(phi*_b, phi_b) distribution  (rank self-reproducibility under val
    resampling -- the within-run analogue of the cross-seed instability)
  - Spearman(phi_est, phi*_b) distribution (does estimator fidelity survive?)
  - half-split check: phi from chunk-half A vs half B.

Reuses track_d's loaders/FL loop (anchor5 regime; REGIME is forced) -- same
model/data/steps; LORA_R is the probe lever.  Cost ~= vanilla FL + one (b)
oracle pass (~1.7 h at 1B, val=200 -> 20 chunks).

Run from codes/:
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. LORA_R=16 SEED=0 \
    python -u experiments/probe_val_noise.py
  # smoke: SMOKE_MODEL=gpt2 TOTAL_TRAIN=200 VAL=20 TEST=20 ROUNDS=3 MAX_STEPS=2
"""
import itertools
import json
import os
from math import factorial

os.environ.setdefault("REGIME", "anchor5")             # oracle-precision point only

import numpy as np
import torch
from scipy.stats import spearmanr

from experiments import track_d as td
from flirds.backends.llm import make_llm_loss
from flirds.core.flirds_estimator import flirds_values
from flirds.data.llm import build_alpaca_iid, build_val_batches
from flirds.oracle.in_run_sv import _perturbed_params, _round_weight, _split
from flirds.repro import seed_everything
from flirds.run_logger import RunLogger

N_BOOT = int(os.environ.get("N_BOOT", "2000"))


@torch.no_grad()
def coalition_utilities_chunked(logs, n_clients, loss_chunks, pkeys, device):
    """U_c(S): per-chunk UNWEIGHTED mean-loss change per coalition (summed over
    rounds), + per-chunk weights.  Sum_c w_c * U_c(S) == in_run U_(b)(S)."""
    lfs = [lf for lf, _ in loss_chunks]
    w = np.array([wc for _, wc in loss_chunks])
    C = len(lfs)
    clients = list(range(n_clients))
    U = {(): np.zeros(C)}
    subsets = [S for r in range(1, n_clients + 1)
               for S in itertools.combinations(clients, r)]
    for S in subsets:
        U[S] = np.zeros(C)
    for w_r, dm in logs:
        pr = _round_weight(dm)
        base_params, buffers = _split(w_r, pkeys, device)
        base = np.array([float(lf(base_params, buffers)) for lf in lfs])
        for S in subsets:
            players = [k for k in dm if k in set(S)]
            if not players:
                continue
            pert = _perturbed_params(base_params, dm, players, pr, pkeys)
            U[S] += np.array([float(lf(pert, buffers)) for lf in lfs]) - base
    return U, w


def shapley_from_U(U_scalar, n_clients):
    """Exact Shapley from a {S: u} dict (U(empty)=0 convention)."""
    clients = list(range(n_clients))
    phi = np.zeros(n_clients)
    for k in clients:
        others = [c for c in clients if c != k]
        for r in range(len(others) + 1):
            wgt = factorial(r) * factorial(n_clients - r - 1) / factorial(n_clients)
            for S in itertools.combinations(others, r):
                phi[k] += wgt * (U_scalar[tuple(sorted(S + (k,)))] - U_scalar[S])
    return phi


def phi_under_weights(U, wts, n_clients):
    return shapley_from_U({S: float(np.dot(wts, u)) for S, u in U.items()}, n_clients)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(os.environ.get("SEED", "0"))
    n = td.RCFG["n_clients"]
    name = f"noise_{td.SCALE}_r{td.LORA_R}_seed{seed}"
    print(f"=== noise-probe | {td.SCALE} anchor5 r={td.LORA_R} | N={n} R={td.RCFG['rounds']} "
          f"val={td.RCFG['val']} | boot={N_BOOT} | seed={seed} ===", flush=True)

    seed_everything(seed)
    tok, model, init, pkeys = td._load(device)
    clients, val, _test = build_alpaca_iid(n, td.RCFG["total_train"], td.RCFG["val"],
                                           td.RCFG["test"], seed=seed)
    logs, t_fl = td._timed(lambda: td._fl(model, tok, clients, init, seed), device)
    val_chunks = build_val_batches(val, tok, td.MCFG["val_maxlen"], device, td.MCFG["val_chunk"])
    loss_fn, _pk, lc = make_llm_loss(model, val_chunks, device)
    print(f"[fl] vanilla {t_fl:.0f}s | chunks={len(lc)}", flush=True)

    (phi_est, _), t_est = td._timed(lambda: flirds_values(
        logs, loss_fn, pkeys, device, second_order=True, n_clients=n, loss_chunks=lc), device)
    (U, w_tok), t_orc = td._timed(lambda: coalition_utilities_chunked(logs, n, lc, pkeys, device),
                                  device)
    print(f"[oracle] chunk-resolved 2^{n} utilities in {t_orc:.0f}s", flush=True)
    phi_full = phi_under_weights(U, w_tok, n)
    rho_est_full = float(spearmanr(phi_est, phi_full).correlation)
    print(f"[phi] full-val (b): {np.round(phi_full, 5).tolist()} | "
          f"Spearman(est,(b))={rho_est_full:+.3f}", flush=True)

    C = len(w_tok)
    rng = np.random.default_rng(seed)
    boot_phi, boot_rho_self, boot_rho_est = [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, C, size=C)
        wb = w_tok[idx]
        wb = wb / wb.sum()
        phi_b = shapley_from_U({S: float(np.dot(wb, u[idx])) for S, u in U.items()}, n)
        boot_phi.append(phi_b)
        boot_rho_self.append(spearmanr(phi_b, phi_full).correlation)
        boot_rho_est.append(spearmanr(phi_b, phi_est).correlation)
    boot_phi = np.array(boot_phi)

    halves = (np.arange(C) < C // 2), (np.arange(C) >= C // 2)
    phi_h = []
    for h in halves:
        wh = w_tok * h
        phi_h.append(shapley_from_U({S: float(np.dot(wh / wh.sum(), u)) for S, u in U.items()}, n))
    rho_half = float(spearmanr(phi_h[0], phi_h[1]).correlation)

    spread = float(phi_full.max() - phi_full.min())
    se = boot_phi.std(axis=0)
    pair_se = float(np.std(boot_phi[:, np.argmax(phi_full)] - boot_phi[:, np.argmin(phi_full)]))
    res = {
        "phi_full_b": phi_full.tolist(), "phi_est": np.asarray(phi_est).tolist(),
        "spearman_est_vs_b": rho_est_full,
        "phi_spread": spread, "phi_boot_se": se.tolist(),
        "spread_over_max_se": spread / float(se.max()),
        "maxmin_pair_diff_se": pair_se,
        "boot_rho_self_mean": float(np.mean(boot_rho_self)),
        "boot_rho_self_q05": float(np.quantile(boot_rho_self, 0.05)),
        "boot_rho_est_mean": float(np.mean(boot_rho_est)),
        "boot_rho_est_q05": float(np.quantile(boot_rho_est, 0.05)),
        "halfsplit_rho": rho_half,
        "phi_half_a": phi_h[0].tolist(), "phi_half_b": phi_h[1].tolist(),
        "n_chunks": C, "n_boot": N_BOOT, "t_fl_s": t_fl, "t_est_s": t_est,
    }
    print(f"[boot] SE(phi)={np.round(se, 6).tolist()}\n"
          f"[boot] spread={spread:.5f} spread/maxSE={res['spread_over_max_se']:.2f} | "
          f"rank self-rho mean={res['boot_rho_self_mean']:+.3f} (q05 {res['boot_rho_self_q05']:+.3f}) | "
          f"est-rho mean={res['boot_rho_est_mean']:+.3f} | half-split rho={rho_half:+.3f}", flush=True)

    if os.environ.get("PERSIST", "1") == "1":
        root = os.environ.get("RUNDIR_ROOT", os.path.join(os.path.dirname(td._CODES),
                                                          "runs", "probe_signal", "noise_probe"))
        config = {"scale": td.SCALE, "model": td.MODEL, "regime": "anchor5", "seed": seed,
                  "rcfg": td.RCFG, "mcfg": td.MCFG,
                  "lora": {"r": td.LORA_R, "alpha": td.LORA_ALPHA}, "n_boot": N_BOOT}
        rl = RunLogger(root, name, config, repo_root=td._CODES)
        rl.save_metrics(res)
        print(f"[persist] {rl.dir}", flush=True)
    print("NOISE PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
