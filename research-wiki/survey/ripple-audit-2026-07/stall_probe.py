#!/usr/bin/env python
"""Ripple eigsh stall probe -- escalation beyond instrumented_ripple_cnn.py.

2026-07-04 Ripple audit, DIAGNOSTIC ONLY (no repo modification).  run1 showed
that at CNN-tiny scale with a random-init operator, tol=0 CONVERGES (~110
matvecs) -- the stall does not reproduce there.  This probe tests the three
escalation axes that the port's real failures (phase0 CNN guard, LLM smoke
hang) plausibly sit on:

  trained    Hessian spectrum of a MEMORIZED (hard-trained) model -- top
             eigenvalues cluster as training sharpens the minimum; measures
             matvec-count growth vs training progress, tol sweep at the final
             state, and the top-gap structure from returned eigenvalues.
  synthdiag  Controlled diagonal operator (exact spectrum): top-8 eigenvalues
             separated by gap g in {1e-2, 1e-6, 1e-8} x matvec precision
             {fp32, fp64}.  Decides "fp32 matvec noise + clustered top gap ->
             tol=0 non-convergence" cleanly, independent of torch.
  llmscale   Diagonal operator at the LLM port's true dimension P=11,272,192
             (k=3, ncv=20 -- phase1 compare spec) with a trivially cheap
             matvec: the eigsh wall minus matvec time isolates the ARPACK
             fp64 CPU cost per Lanczos step at LLM scale, which extrapolates
             the CPU-spin arithmetic of the observed LLM hang (maxiter=300).

Run from <repo>/codes with PYTHONPATH=. (this file's dir is self-appended):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 nice -n 10 PYTHONPATH=. \
    python stall_probe.py --phase all --out <scratch dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instrumented_ripple_cnn import build_hvp, build_stage, dump_env, run_eigsh  # noqa: E402


# --------------------------------------------------------------------------- #
# phase trained: memorized-model Hessian spectrum                             #
# --------------------------------------------------------------------------- #
def phase_trained(args, out_dir):
    device = "cpu"
    records_path = os.path.join(out_dir, "eigsh_calls.jsonl")
    model, gstate, loaders, client_xy, _, _ = build_stage(args, device)
    x, y = client_xy[0]
    out = dict(P=None, checkpoints=[])

    # hard-train client 0's model on its own 128 random-label samples (memorize)
    model.load_state_dict(gstate)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)
    losses = []
    ckpt_epochs = sorted(set([0, args.train_epochs // 4, args.train_epochs]))
    state_at = {}
    state_at[0] = {n: v.detach().clone() for n, v in model.state_dict().items()}
    for ep in range(1, args.train_epochs + 1):
        for bx, by in loaders[0]:
            opt.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            opt.step()
        losses.append(float(loss.detach()))
        if ep in ckpt_epochs:
            state_at[ep] = {n: v.detach().clone() for n, v in model.state_dict().items()}
        if losses[-1] < 1e-3:
            state_at[ep] = {n: v.detach().clone() for n, v in model.state_dict().items()}
            ckpt_epochs = sorted(set([e for e in ckpt_epochs if e <= ep] + [ep]))
            print(f"[trained] early stop at epoch {ep} loss {losses[-1]:.2e}", flush=True)
            break
    out["final_loss"] = losses[-1] if losses else None
    out["epochs_run"] = len(losses)
    print(f"[trained] epochs={len(losses)} final_loss={out['final_loss']:.4e}", flush=True)

    for ep in ckpt_epochs:
        if ep not in state_at:
            continue
        raw32, P = build_hvp(model, state_at[ep], x, y, device, torch.float32)
        out["P"] = P
        ck = dict(epoch=ep, rows=[])
        # port config first (tol=0, ncv default), then tol sweep at final ckpt
        tols = (0,) if ep != max(ckpt_epochs) else (0, 1e-6, 1e-3)
        for tol in tols:
            _, _, rec = run_eigsh(raw32, P, args.k, f"trained:ep{ep}:tol{tol:g}",
                                  records_path, maxiter=args.maxiter, tol=tol,
                                  budget=args.budget, deadline_s=args.deadline_s,
                                  extra=dict(phase="trained", epoch=ep))
            ck["rows"].append(dict(tol=tol, status=rec["status"],
                                   n_matvec=rec["n_matvec"], t_wall_s=rec["t_wall_s"],
                                   eig_top4=rec["eigvals"][-4:],
                                   resid_rel_max=(max(rec["resid_rel"])
                                                  if rec["resid_rel"] else None)))
        if ep == max(ckpt_epochs):
            _, _, rec = run_eigsh(raw32, P, 20, f"trained:ep{ep}:k20", records_path,
                                  maxiter=args.maxiter, tol=0, budget=args.budget,
                                  deadline_s=args.deadline_s,
                                  extra=dict(phase="trained", epoch=ep))
            ck["rows"].append(dict(tol=0, k=20, status=rec["status"],
                                   n_matvec=rec["n_matvec"], t_wall_s=rec["t_wall_s"]))
            raw64, _ = build_hvp(model, state_at[ep], x, y, device, torch.float64)
            _, _, rec = run_eigsh(raw64, P, args.k, f"trained:ep{ep}:fp64", records_path,
                                  maxiter=args.maxiter, tol=0, budget=args.budget,
                                  deadline_s=args.deadline_s,
                                  extra=dict(phase="trained", epoch=ep, operator="fp64"))
            ck["rows"].append(dict(tol=0, op="fp64", status=rec["status"],
                                   n_matvec=rec["n_matvec"], t_wall_s=rec["t_wall_s"]))
        out["checkpoints"].append(ck)
    with open(os.path.join(out_dir, "trained_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[trained] done", flush=True)


# --------------------------------------------------------------------------- #
# phase synthdiag: controlled clustered-spectrum diagonal operator            #
# --------------------------------------------------------------------------- #
def make_diag_ops(P, gap, seed=0):
    """Diagonal spectrum: top-8 at 1.0 - i*gap (i=0..7), next block at 0.5,
    bulk uniform in [0, 0.1].  Returns (mv_fp32, mv_fp64, d)."""
    rng = np.random.default_rng(seed)
    d = np.empty(P)
    d[:8] = 1.0 - gap * np.arange(8)
    d[8:16] = 0.5
    d[16:] = rng.uniform(0.0, 0.1, P - 16)
    d32 = d.astype(np.float32)

    def mv64(v):
        return d * np.asarray(v, dtype=np.float64)

    def mv32(v):
        # mimic the port: fp32 compute, fp32 return -> scipy upcasts to fp64
        return d32 * np.asarray(v, dtype=np.float32)

    return mv32, mv64, d


def phase_synthdiag(args, out_dir):
    records_path = os.path.join(out_dir, "eigsh_calls.jsonl")
    P = args.synth_p
    rows = []
    for gap in (1e-2, 1e-6, 1e-8):
        mv32, mv64, _ = make_diag_ops(P, gap)
        for name, mv in (("fp32", mv32), ("fp64", mv64)):
            _, _, rec = run_eigsh(mv, P, 8, f"synthdiag:gap{gap:g}:{name}",
                                  records_path, maxiter=args.maxiter, tol=0,
                                  budget=args.synth_budget,
                                  deadline_s=args.deadline_s,
                                  extra=dict(phase="synthdiag", gap=gap, operator=name))
            rows.append(dict(gap=gap, op=name, status=rec["status"],
                             n_matvec=rec["n_matvec"], t_wall_s=rec["t_wall_s"],
                             n_eigvals=rec["n_eigvals"],
                             resid_rel_max=(max(rec["resid_rel"])
                                            if rec["resid_rel"] else None)))
    with open(os.path.join(out_dir, "synthdiag_results.json"), "w") as f:
        json.dump(dict(P=P, k=8, ncv_default=20, rows=rows), f, indent=2)
    print("[synthdiag] done", flush=True)


# --------------------------------------------------------------------------- #
# phase llmscale: ARPACK fp64 CPU cost per Lanczos step at P=11.27M           #
# --------------------------------------------------------------------------- #
def phase_llmscale(args, out_dir):
    records_path = os.path.join(out_dir, "eigsh_calls.jsonl")
    P = args.llm_p                       # LLM port LoRA dim (recon estimate)
    k, ncv = 3, 20                       # phase1 compare spec (rip_k=3, ncv=max(2k+1,20))
    mv32, mv64, _ = make_diag_ops(P, 1e-8)   # clustered top -> forces full maxiter
    out = dict(P=P, k=k, ncv=ncv, rows=[])
    for mi in (5, args.llm_maxiter):
        _, _, rec = run_eigsh(mv64, P, k, f"llmscale:mi{mi}", records_path,
                              ncv=ncv, maxiter=mi, tol=0,
                              deadline_s=args.llm_deadline_s,
                              extra=dict(phase="llmscale"))
        out["rows"].append(dict(maxiter=mi, status=rec["status"],
                                n_matvec=rec["n_matvec"], t_wall_s=rec["t_wall_s"],
                                t_matvec_sum_s=rec["t_matvec_sum_s"],
                                t_arpack_residual_s=rec["t_arpack_residual_s"]))
    r5, rN = out["rows"][0], out["rows"][1]
    dmv = rN["n_matvec"] - r5["n_matvec"]
    dt = rN["t_arpack_residual_s"] - r5["t_arpack_residual_s"]
    if dmv > 0:
        per_step = dt / dmv
        out["arpack_cpu_s_per_lanczos_step"] = round(per_step, 4)
        n300 = 300 * (ncv - k) + ncv     # restart-cycle semantics (measured in run1)
        out["extrapolated_maxiter300"] = dict(
            n_matvec=n300,
            arpack_cpu_s=round(per_step * n300, 1),
            note="ARPACK fp64 CPU only; excludes the per-matvec GPU HVP + H2D/D2H "
                 "of the real LLM port (invisible in this CPU probe)")
    with open(os.path.join(out_dir, "llmscale_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[llmscale] done {json.dumps(out.get('extrapolated_maxiter300'))}", flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "trained", "synthdiag", "llmscale"])
    ap.add_argument("--out", required=True)
    # stage (matches instrumented_ripple_cnn defaults)
    ap.add_argument("--n-clients", type=int, default=4)
    ap.add_argument("--n-per", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=8)
    # trained
    ap.add_argument("--train-epochs", type=int, default=150)
    # eigsh caps
    ap.add_argument("--maxiter", type=int, default=1000)      # port default
    ap.add_argument("--budget", type=int, default=13000)
    ap.add_argument("--deadline-s", type=float, default=240.0)
    ap.add_argument("--synth-p", type=int, default=61706)
    ap.add_argument("--synth-budget", type=int, default=25000)
    ap.add_argument("--llm-p", type=int, default=11272192)
    ap.add_argument("--llm-maxiter", type=int, default=25)
    ap.add_argument("--llm-deadline-s", type=float, default=420.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dump_env(args.out)
    if args.phase in ("all", "trained"):
        phase_trained(args, args.out)
    if args.phase in ("all", "synthdiag"):
        phase_synthdiag(args, args.out)
    if args.phase in ("all", "llmscale"):
        phase_llmscale(args, args.out)
    print("[main] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
