"""Track C2-FID -- fidelity vs the (b) in-run oracle ON the C2 cross-device stage
(CNN campaign 2026-07-22, CNN_CAMPAIGN_PLAN_2026-07-22.md §4).

track_c1 asks "how close is each method to the oracle" on the GTG stage (N=10,
full participation, its own corruption ladder); this runner asks the SAME
question on the downstream stage itself: `track_c2.build()` verbatim (N=100,
C=0.1, R=120, partitions {iid,dir1,shard,qskew}, the C2 threat set incl. frrand
and strmain label_flip), ONE frozen vanilla trajectory (no selection, no
intervention -- the C1 convention), ground truth = the (b) in-run per-round
oracle ONLY ((a) 2^N retrains are infeasible at N=100), methods = the C1 battery
minus {Banzhaf (2^N), Ripple (own trajectory), Fed-LOO (dropped 07-23)}; metrics = the C1
set + the spearman_vs_rate pair (all-client / corrupt-only).

CAVEAT (plan §4.5): at 10/100 participation a client joins ~12/120 rounds, so
phi = sum of round sub-game contributions -- a DIFFERENT game from C1's
full-participation stage.  Never compare numbers across the two tables; (b) is
the exact value of THIS game, so fidelity is self-contained.

The trajectory is bit-identical to the downstream twin's vanilla arm (fedavg
re-seeds at entry; on_round only observes) -- final_acc / acc_curve are recorded
for the join check (plan §4.9).  Logs are CPU-forced at capture (cifar10 ~11.4
GB > 24 GB GPU); every consumer stages `.to(device)` at use.  The (b) oracle
runs one round per call (`oracle_b_rounds`): one-shot GPU staging per round,
per-round phi rows (phi_b_rounds.parquet), and round sharding.

Run (from codes/): the same C2_* env as track_c2 (dataset / partition / threat /
seed / mode / C2_FLIP_RATE), plus
  C2FID_ORACLE_B=0      skip the oracle (methods-only run, pairs with shards)
  C2FID_B_ROUNDS=lo:hi  oracle-ONLY shard over rounds [lo,hi) -- methods and
                        phi.parquet are skipped; merge = groupby-sum of the
                        shards' phi_b_rounds.parquet, coverage-asserted.
  C2FID_RUN_NAME / C2FID_RUN_ROOT / C2FID_PERSIST   as in the other runners.
e.g.  C2_DATASET=cifar10 C2_PARTITION=dir1 C2_THREAT=grad_noise C2_SEED=0 \
      C2_MODE=full PYTHONPATH=. python experiments/track_c2_fid.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

import experiments.track_c2 as c2                  # env-configured stage (READ-ONLY reuse)
from flirds.backends.cnn import make_cnn_loss
from flirds.baselines.comfedsv import comfedsv_from_logs
from flirds.baselines.fedif import fedif_from_logs
from flirds.baselines.fedsv import fedsv_from_logs
from flirds.baselines.gtg import gtg_from_logs
from flirds.baselines.shapleyfl import BETA as SFL_BETA, shapleyfl_from_logs
from flirds.core.flirds_estimator import flirds_values
from flirds.data.corruptors import CNN_CORRUPTORS
from flirds.eval.metrics import (cosine_distance, detection_auroc,
                                 euclidean_distance, max_difference, pearson)
from flirds.fl.server import fedavg
from flirds.oracle.in_run_sv import (in_run_shapley_perround,
                                     in_run_singletons, in_run_utility)
from flirds.repro import seed_everything
from flirds.run_logger import RunLogger
from flirds.timing import PhaseTimer

ORACLE_B = os.environ.get("C2FID_ORACLE_B", "1") == "1"
B_ROUNDS = os.environ.get("C2FID_B_ROUNDS", "")          # "lo:hi" -> oracle-only shard
PERSIST = os.environ.get("C2FID_PERSIST", "1") == "1"
RUN_ROOT = os.environ.get("C2FID_RUN_ROOT",
                          os.path.join(c2._REPO, "runs", "track_c", "c2fid"))


def _timed(fn, device):
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t


def _cpu_state(sd):
    return {k: v.detach().to("cpu") for k, v in sd.items()}


def build_with_rates():
    """`track_c2.build()` + capture of the per-client REALIZED label-flip rate.

    track_c2 persists only the binary corrupt mask; spearman_vs_rate needs the
    realized rate vector, and the rate draw shares build()'s rng stream with the
    partition path (dir1 empty-client refills interleave), so re-deriving it
    outside is fragile.  The corruptor dict entry is wrapped (pass-through) for
    the duration of this ONE build to record the `rate` kwarg per client --
    bit-neutrality is enforced by tests/test_c2fid.py."""
    captured = {}
    orig = CNN_CORRUPTORS["label_flip"]

    def rec(xs, ys, cid, rate, **kw):
        captured[int(cid)] = float(rate)
        return orig(xs, ys, cid, rate=rate, **kw)

    CNN_CORRUPTORS["label_flip"] = rec
    try:
        out = c2.build()
    finally:
        CNN_CORRUPTORS["label_flip"] = orig
    rates = [captured.get(i, 0.0) for i in range(c2.CFG["n"])]
    return out, rates


def oracle_b_rounds(logs, n_clients, loss_fn, pkeys, device, rounds=None):
    """(b) per-round oracle: one `in_run_shapley_perround` call per round.

    Identical by construction to one call over the full logs (the oracle is a
    sum of independent round sub-games; tests/test_c2fid.py asserts equality
    against the 2^N enumeration).  Chunking buys (1) one-shot GPU staging per
    round (CPU-resident logs would otherwise re-transfer per subset -- ~2 TB at
    R=120, K=10), (2) per-round phi rows for phi_b_rounds.parquet, (3) round
    sharding via `rounds`.  Returns (phi[n], rows=[{round, client, phi_b}])."""
    sel = set(range(len(logs)) if rounds is None else rounds)
    phi = np.zeros(n_clients)
    rows = []
    for r, (w_r, dm) in enumerate(logs):
        if r not in sel:
            continue
        staged = ({k: v.to(device) for k, v in w_r.items()},
                  {c: ({k: v.to(device) for k, v in d.items()}, sz)
                   for c, (d, sz) in dm.items()})
        phi_r, _ = in_run_shapley_perround([staged], n_clients, loss_fn, pkeys, device)
        phi += phi_r
        rows += [dict(round=r, client=int(c), phi_b=float(phi_r[c]))
                 for c in sorted(staged[1])]
        del staged
    return phi, rows


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert not c2.DYN, "fid leg needs a static corrupt set (C2_DYN unsupported)"
    n, R, E, lr = c2.CFG["n"], c2.CFG["rounds"], c2.CFG["epochs"], c2.CFG["lr"]
    b_slice = None
    if B_ROUNDS:
        lo, hi = (int(x) for x in B_ROUNDS.split(":"))
        b_slice = range(lo, hi)
    pt = PhaseTimer(device, n_gpus=int(os.environ.get("N_GPUS", "1")))
    seed_everything(c2.SEED, cudnn_deterministic=True)
    (loaders, corrupt, dtf, vx, vy, vl, tl), rates = build_with_rates()
    cr = [r for r, m in zip(rates, corrupt) if m]
    print(f"[build] {c2.DATASET}/{c2.PARTITION}/{c2.THREAT}(str={c2.STRENGTH}) "
          f"seed={c2.SEED} n={n} corrupt={int(corrupt.sum())} "
          f"rates[corrupt min/max]={min(cr, default=0.0):.2f}/{max(cr, default=0.0):.2f}",
          flush=True)
    if c2._EMPTY_CLIENTS:                                # mirror track_c2.run()'s guard
        print(f"  [WARN] {len(c2._EMPTY_CLIENTS)} empty client(s) backfilled "
              f"{c2._EMPTY_CLIENTS} -- rates are no longer partition-independent",
              flush=True)

    # frozen vanilla trajectory == the downstream twin's vanilla arm (join §4.9):
    # same fedavg call + observe-only on_round; logs CPU-forced at capture.
    logs = []

    def on_round(r, w_r, dm):
        logs.append((_cpu_state(w_r),
                     {c: (_cpu_state(d), sz) for c, (d, sz) in dm.items()}))

    (final_state, history), t_traj = _timed(lambda: fedavg(
        c2.MODEL_FN, loaders, tl, R, E, lr, sample_frac=c2.CFG["frac"],
        device=device, seed=c2.SEED, on_round=on_round, delta_transform=dtf), device)
    final_acc = history[-1][1]
    pt.record("client-training", t_traj)
    n_rounds = np.zeros(n, dtype=int)
    for _, dm in logs:
        for c in dm:
            n_rounds[c] += 1
    print(f"[traj] {R}r x {E}e in {t_traj:.0f}s  final test-acc={final_acc:.4f}  "
          f"{len(logs)} rounds logged (cpu)", flush=True)
    loss_fn, pkeys = make_cnn_loss(c2.MODEL_FN, vx, vy, device)

    methods = []                                       # (name, phi good->low, runtime)
    if ORACLE_B:
        with pt.phase("oracle-b"):
            (phi_b, b_rows), t_b = _timed(lambda: oracle_b_rounds(
                logs, n, loss_fn, pkeys, device, rounds=b_slice), device)
        methods.append(("(b)oracle", np.asarray(phi_b), t_b))
        print(f"  {'(b)oracle':10s} {t_b:7.1f}s  "
              f"({sum(2 ** len(dm) for _, dm in logs) if b_slice is None else '(shard)'} evals)",
              flush=True)
        if b_slice is None:                            # integrity: Shapley efficiency vs
            u_grand, _ = _timed(lambda: in_run_utility(   # an independently-computed U(N)
                logs, tuple(range(n)), loss_fn, pkeys, device), device)
            eff = abs(float(phi_b.sum()) - u_grand)
            print(f"  [(b)] sum(phi)={phi_b.sum():+.6f}  U(N)={u_grand:+.6f}  "
                  f"eff-gap={eff:.2e}", flush=True)
            assert eff < 1e-3, f"(b) efficiency violated: {eff}"
    else:
        b_rows = None

    if b_slice is None:
        with pt.phase("valuation"):                    # partial-participation calls =
            def add(name, fn, negate=False):           # phase2_matrix device100 pattern
                out, t = _timed(fn, device)
                vec = out[0] if isinstance(out, tuple) else out
                vec = np.asarray(vec, dtype=float)
                methods.append((name, -vec if negate else vec, t))
                print(f"  {name:10s} {t:7.1f}s", flush=True)

            add("Flirds", lambda: flirds_values(logs, loss_fn, pkeys, device,
                                                second_order=True, n_clients=n))
            add("Flirds1st", lambda: flirds_values(logs, loss_fn, pkeys, device,
                                                   second_order=False, n_clients=n))
            add("GTG", lambda: gtg_from_logs(logs, None, n, None, device, seed=c2.SEED,
                                             loss_fn=loss_fn, pkeys=pkeys,
                                             round_trunc=0.0, eps=0.0))
            add("FedSV", lambda: fedsv_from_logs(logs, None, n, None, device, seed=c2.SEED,
                                                 loss_fn=loss_fn, pkeys=pkeys, trunc_eps=0.0))
            # partial=True: 10/100 participation -> the utility matrix is partially
            # observed, so the paper's low-rank completion is ACTIVE (C1 full
            # participation kept it off; plan §4.3).
            add("ComFedSV", lambda: comfedsv_from_logs(logs, None, n, None, device,
                                                       seed=c2.SEED, loss_fn=loss_fn,
                                                       pkeys=pkeys, partial=True),
                negate=True)                           # loss-decrease util -> negate
            add("ShapleyFL", lambda: shapleyfl_from_logs(logs, None, n, None, device,
                                                         beta=SFL_BETA, loss_fn=loss_fn,
                                                         pkeys=pkeys),
                negate=True)                           # good->high -> negate
            add("FedIF", lambda: fedif_from_logs(logs, n, loss_fn, pkeys, device),
                negate=True)                           # influence good->HIGH -> negate
            add("loss-heur", lambda: in_run_singletons(logs, n, loss_fn, pkeys, device))
            # Fed-LOO dropped from the comparison (Yonghee 2026-07-23) before this leg
            # ever ran -- no rundir on disk carries it.

    # ---- metrics (C1 set; good->low everywhere) ----
    gt = {"b": methods[0][1]} if (ORACLE_B and b_slice is None) else {}
    ladder = c2.THREAT == "label_flip"
    y = corrupt.tolist()
    res = {}
    for name, vec, rt in methods:
        m = {"runtime": rt, "phi": vec.tolist()}
        for g, gvec in gt.items():
            if name == f"({g})oracle":
                continue
            m[f"spearman_{g}"] = float(spearmanr(vec, gvec).correlation)
            m[f"kendall_{g}"] = float(kendalltau(vec, gvec).correlation)
            m[f"pearson_{g}"] = pearson(vec, gvec)
            m[f"cos_{g}"] = cosine_distance(vec, gvec)
            m[f"euc_{g}"] = euclidean_distance(vec, gvec)
            m[f"maxdiff_{g}"] = max_difference(vec, gvec)
        if 0 < corrupt.sum() < n:
            m["auroc"] = detection_auroc(vec, y)       # good->low: corrupt scores high
        if ladder:                                     # both variants (07-23 decision):
            m["spearman_vs_rate"] = float(spearmanr(vec, rates).correlation)   # all-client (C1-compatible)
            cv = [(v, rr) for v, rr, mk in zip(vec, rates, corrupt) if mk]     # corrupt-only (dose resolution;
            m["spearman_vs_rate_corrupt"] = (                                  # NaN at fixed dose -- constant)
                float(spearmanr([v for v, _ in cv], [rr for _, rr in cv]).correlation)
                if len(cv) >= 2 else float("nan"))
        res[name] = m

    hdr = f"  {'method':10s} {'time':>8s} {'rho(b)':>7s} {'tau(b)':>7s} {'r_p(b)':>7s}"
    hdr += f" {'AUROC':>6s}" if (0 < corrupt.sum() < n) else ""
    hdr += f" {'svr':>6s} {'svr_c':>6s}" if ladder else ""
    print(hdr, flush=True)
    for name, vec, rt in methods:
        m = res[name]
        line = f"  {name:10s} {rt:7.1f}s {m.get('spearman_b', float('nan')):7.3f}"
        line += f" {m.get('kendall_b', float('nan')):7.3f} {m.get('pearson_b', float('nan')):7.3f}"
        line += (f" {m.get('auroc', float('nan')):6.3f}" if (0 < corrupt.sum() < n) else "")
        line += (f" {m.get('spearman_vs_rate', float('nan')):6.3f}"
                 f" {m.get('spearman_vs_rate_corrupt', float('nan')):6.3f}" if ladder else "")
        print(line, flush=True)

    metrics = dict(stage="c2fid", dataset=c2.DATASET, partition=c2.PARTITION,
                   threat=c2.THREAT, strength=c2.STRENGTH, seed=c2.SEED, mode=c2.MODE,
                   final_acc=final_acc, acc_curve=history, traj_time=t_traj,
                   corrupt=corrupt.tolist(), rates=rates, n_rounds=n_rounds.tolist(),
                   methods=res,
                   **({"oracle_b_rounds": [b_slice.start, b_slice.stop]} if b_slice else {}))
    phi_rows = None
    if b_slice is None:
        phi_rows = [dict(client=cid, rate=rates[cid], corrupt=int(corrupt[cid]),
                         n_rounds=int(n_rounds[cid]),
                         **{f"phi_{name}": float(vec[cid]) for name, vec, _ in methods})
                    for cid in range(n)]

    if PERSIST:
        try:
            tag = c2.THREAT.replace("_", "-")
            if c2.THREAT == "label_flip":              # the downstream twins' TTAG vocabulary
                tag += (f"_fr{c2.FLIP_RATE}" if c2.FLIP_RATE is not None else "_strmain")
            name = (os.environ.get("C2FID_RUN_NAME")
                    or f"{c2.DATASET}_{c2.PARTITION}_{tag}_fid_seed{c2.SEED}")
            if b_slice is not None:
                name += f"_b{b_slice.start}-{b_slice.stop}"
            rl = RunLogger(RUN_ROOT, name,
                           dict(cfg=c2.CFG, stage="c2fid", dataset=c2.DATASET,
                                partition=c2.PARTITION, threat=c2.THREAT,
                                strength=c2.STRENGTH, seed=c2.SEED, mode=c2.MODE,
                                width=c2.WIDTH, flip_rate=c2.FLIP_RATE,
                                fid=dict(oracle_b=ORACLE_B, b_rounds=B_ROUNDS or "all")),
                           repo_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if phi_rows:
                rl.save_phi(phi_rows)
            if b_rows:
                rl.save_phi(b_rows, fname="phi_b_rounds.parquet")
            rl.save_metrics(metrics)
            rl.save_timing(pt.to_timing())
            print(f"[persist] {rl.dir}", flush=True)
        except Exception as e:
            print(f"[persist] FAILED ({e!r}) -- results live in stdout", flush=True)
    print("TRACK-C2FID RUN OK", flush=True)


if __name__ == "__main__":
    run()
