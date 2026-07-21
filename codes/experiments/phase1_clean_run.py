"""Phase 1 #7 -- first clean Flirds run on Llama-3.2-1B-Instruct (orchestrator).

Per seed: build the 5-domain cross-silo data (1 noisy + 1 free-rider + 3 clean),
run Flirds FedAvg over ALL clients -> estimator phi (1st+2nd) [+ the (b) in-run
oracle at N=5 to validate est≈oracle at scale -- ~32x cheaper than N=10], then the
three #7 metrics:
  (1) detection AUROC -- noisy & free-rider, read off phi (higher phi = worse);
  (2) client-selection convergence -- retrain FedAvg on {full, random-K, Flirds-top-K}
      client subsets; the per-round val-loss read post-hoc off each run's logged w_r
      is the convergence curve (no extra forward passes, no FL-loop change);
  (3) downstream task-acc -- each arm's final model generates on the held-out test;
      per-domain ROUGE-L + math exact-match (eval.generate.score_records).
Everything is written to a local run-dir (flirds.run_logger).  >= 3 seeds; the
aggregate mean+/-std is printed at the end.

Selection sign: lower phi = larger val-loss reduction = more valuable, so Flirds
keeps the K lowest-phi clients (= drops the high-phi corrupted ones).

Compute: the dominant cost is the FINAL test generation (per_domain_test x 5 x
3 arms x seeds).  Set CLEAN_RUN_SMOKE=1 for a tiny end-to-end machine check.

Run from codes/:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase1_clean_run.py          # full (heavy)
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. CLEAN_RUN_SMOKE=1 python experiments/phase1_clean_run.py
"""
import os

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from flirds.backends.llm import make_llm_loss
from flirds.core.flirds_estimator import flirds_values
from flirds.data.llm import build, build_val_batches
from flirds.eval.generate import generate_completions, score_records
from flirds.eval.metrics import detection_auroc
from flirds.fl.llm_server import run_llm_fedavg_logs
from flirds.oracle.in_run_sv import in_run_shapley
from flirds.repro import seed_everything
from flirds.hf_pin import rev
from flirds.run_logger import RunLogger

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DOMAINS = ["medical", "legal", "finance", "math", "general"]

N = 5
NOISY = {0}                 # medical client = noisy (answer_swap, baked into its data)
FREE_RIDERS = {1}           # legal client = free-rider (fabricated update at train time)
FREE_RIDER_MODE = "zero"    # Lin taxonomy; "zero" -> phi==0 exactly (clean #7 detection); "random" = noisy phi (Phase-2 STD-DAGMM hard case)
K = 3                       # clients Flirds/random keep (drops 2 -> matches the 2 corrupted)
ORACLE_B = os.environ.get("ORACLE_B", "0") == "1"   # (b) in-run oracle = 2^N*R*val*seq, the DOMINANT cost
#   (multi-hour-to-day at val=1000/R=50; est≈oracle already validated at R=10 -> 1.16e-6).  OFF by default;
#   set ORACLE_B=1 with a SMALL val/R for a dedicated est-vs-oracle validation, NOT in the headline run.

# full = the locked #7 config; mini = ~30min de-risk (does the signal emerge?);
# smoke = tiny end-to-end machine check.  CLEAN_RUN_MODE in {full, mini, smoke}.
FULL = dict(train=12000, val=200, test=2000, rounds=50, max_steps=10, lr=2e-5, batch=16,
            maxlen=768, val_maxlen=384, val_chunk=10, gen_new=128, gen_batch=16, seeds=[0, 1, 2])
MINI = dict(train=500, val=200, test=200, rounds=50, max_steps=10, lr=2e-5, batch=16,
            maxlen=768, val_maxlen=384, val_chunk=10, gen_new=128, gen_batch=16, seeds=[0])
SMOKE = dict(train=8, val=20, test=10, rounds=2, max_steps=2, lr=1e-3, batch=2,
             maxlen=768, val_maxlen=256, val_chunk=10, gen_new=16, gen_batch=8, seeds=[0])
# lr-sweep diagnostic: small val (fast oracle, run with ORACLE_B=1) + R=20 to map, per lr,
# "does plain-SGD learn (val_loss drops, arms differ)?" vs "does est≈oracle hold (Taylor)?"
SWEEP = dict(train=500, val=100, test=200, rounds=20, max_steps=10, lr=1e-4, batch=16,
             maxlen=768, val_maxlen=384, val_chunk=10, gen_new=128, gen_batch=16, seeds=[0])
_CFGS = {"full": FULL, "mini": MINI, "smoke": SMOKE, "sweep": SWEEP}


def _split(w_r, pkeys, device):
    params = {n: w_r[n].detach().float().to(device) for n in pkeys}
    buffers = {n: w_r[n].detach().to(device) for n in w_r if n not in pkeys}
    return params, buffers


def _final_state(logs):
    """Reconstruct the post-last-round aggregated LoRA state (== fl.server FedAvg)."""
    w, dm = logs[-1]
    final = {k: w[k].clone() for k in w}
    tot = sum(n for _, n in dm.values())
    for _, (d, n) in dm.items():
        for k in final:
            final[k] = final[k] + (n / tot) * d[k].to(final[k].device)
    return final


def select_topk(phi, k):
    """Indices of the k most valuable clients (lowest phi = largest val-loss reduction)."""
    return sorted(range(len(phi)), key=lambda i: phi[i])[:k]


def _run_subset(model, init_lora, clients, idxs, tok, cfg, seed):
    """Reset to the fresh LoRA init, FedAvg over the client subset `idxs` (global
    indices), remapping free-riders to subset-local positions.  Returns logs."""
    model.load_state_dict(init_lora, strict=False)
    subset = [clients[i] for i in idxs]
    fr_local = frozenset(p for p, i in enumerate(idxs) if i in FREE_RIDERS)
    return run_llm_fedavg_logs(model, tok, subset, cfg["rounds"], cfg["lr"], cfg["max_steps"],
                               batch_size=cfg["batch"], max_length=cfg["maxlen"], seed=seed,
                               free_riders=fr_local, free_rider_mode=FREE_RIDER_MODE)


def _eval_arm(model, tok, logs, val_chunks, test, device, cfg):
    """(val-loss convergence curve, per-domain final task-acc) for one arm's logs.

    make_llm_loss is rebuilt here so the val forward is in eval mode with the
    SFTTrainer gradient-checkpointing hook cleared (post-training)."""
    loss_fn, pkeys, _ = make_llm_loss(model, val_chunks, device)
    curve = [float(loss_fn(*_split(w_r, pkeys, device))) for w_r, _ in logs]
    model.load_state_dict(_final_state(logs), strict=False)          # arm's final global model
    gen = generate_completions(model, tok, [r["prompt"] for r in test], device,
                               max_new_tokens=cfg["gen_new"], batch_size=cfg["gen_batch"])
    return curve, score_records(gen, test)


def run_seed(seed, cfg, device, root):
    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev(MODEL))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 attn_implementation="eager",
                                                 revision=rev(MODEL)).to(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=TARGET,
                                             lora_dropout=0.0, task_type="CAUSAL_LM"))
    init_lora = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    clients, val, test = build(N, cfg["train"], cfg["val"], cfg["test"], seed, noisy=NOISY)
    val_chunks = build_val_batches(val, tok, cfg["val_maxlen"], device, cfg["val_chunk"])
    all_idx = list(range(N))

    # ---- Flirds run over ALL clients -> phi (this IS the "full" training arm) ----
    print(f"[seed {seed}] training full arm (R={cfg['rounds']}, {N} clients)...", flush=True)
    logs_full = _run_subset(model, init_lora, clients, all_idx, tok, cfg, seed)
    loss_fn, pkeys, loss_chunks = make_llm_loss(model, val_chunks, device)
    print(f"[seed {seed}] Flirds phi: {cfg['rounds']} rounds x {len(val_chunks)} val chunks (HVP)...", flush=True)
    phi, _ = flirds_values(logs_full, loss_fn, pkeys, device, second_order=True, loss_chunks=loss_chunks)
    phi_b = None
    if ORACLE_B:
        print(f"[seed {seed}] (b) oracle: 2^{N} coalitions x {cfg['rounds']} rounds (DOMINANT cost)...", flush=True)
        phi_b, _ = in_run_shapley(logs_full, N, loss_fn, pkeys, device)

    # ---- (1) detection AUROC (higher phi = worse) ----
    noisy_y = [1 if i in NOISY else 0 for i in range(N)]
    fr_y = [1 if i in FREE_RIDERS else 0 for i in range(N)]
    metrics = {"phi_est": [float(x) for x in phi],
               "auroc_noisy": detection_auroc(phi, noisy_y),
               "auroc_freerider": detection_auroc(phi, fr_y)}
    if phi_b is not None:
        metrics["phi_oracle"] = [float(x) for x in phi_b]
        metrics["est_vs_oracle_maxabs"] = float(np.max(np.abs(phi - phi_b)))

    # ---- (2)+(3) selection-convergence arms ----
    keep = select_topk(phi, K)
    rng = np.random.default_rng(seed)
    rand = sorted(rng.choice(N, size=K, replace=False).tolist())
    arms = {"full": all_idx, "flirds_topk": keep, "random_k": rand}
    metrics["selection"] = {"K": K, "flirds_keep": keep, "random_keep": rand}
    metrics["arms"] = {}
    for name, idxs in arms.items():
        print(f"[seed {seed}] arm {name} keep={idxs}: train + generate {len(test)} test...", flush=True)
        logs = logs_full if name == "full" else _run_subset(model, init_lora, clients, idxs, tok, cfg, seed)
        curve, taskacc = _eval_arm(model, tok, logs, val_chunks, test, device, cfg)
        metrics["arms"][name] = {"idxs": idxs, "val_loss_curve": curve, "task_acc": taskacc}

    # ---- log the run ----
    cfg_log = {"model": MODEL, "N": N, "seed": seed, "K": K, "noisy": sorted(NOISY),
               "free_riders": sorted(FREE_RIDERS), "free_rider_mode": FREE_RIDER_MODE,
               "per_domain": {"train": cfg["train"], "val": cfg["val"], "test": cfg["test"]},
               "rounds": cfg["rounds"], "max_steps": cfg["max_steps"], "lr": cfg["lr"], "oracle_b": ORACLE_B,
               "cfg": cfg, "mode": os.environ.get("CLEAN_RUN_MODE", "full")}   # full resolved cfg (batch/maxlen/val_*/gen_*)
    rl = RunLogger(root, f"flirds-1b-N{N}-seed{seed}", cfg_log)
    phi_rows = [{"client": i, "domain": DOMAINS[i], "phi_est": float(phi[i]),
                 "phi_oracle": (float(phi_b[i]) if phi_b is not None else None),
                 "noisy": i in NOISY, "free_rider": i in FREE_RIDERS} for i in range(N)]
    rl.save_phi(phi_rows)
    rl.save_metrics(metrics)
    return metrics, rl.dir


def main():
    mode = os.environ.get("CLEAN_RUN_MODE", "full")
    cfg = _CFGS[mode]
    if os.environ.get("RUN_LR"):                       # lr-sweep: override lr per process
        cfg = {**cfg, "lr": float(os.environ["RUN_LR"])}
    device = "cuda"
    root = os.environ.get("RUN_ROOT", f"runs/phase1_clean_{mode}")
    print(f"#7 clean run | {mode.upper()} | per_domain={cfg['train']}/{cfg['val']}/{cfg['test']} "
          f"R={cfg['rounds']} lr={cfg['lr']} ORACLE_B={ORACLE_B} seeds={cfg['seeds']} K={K} "
          f"noisy={sorted(NOISY)} free_rider={sorted(FREE_RIDERS)}({FREE_RIDER_MODE})")

    # RUN_SEED restricts to ONE seed -> launch 3 processes on GPUs 1/2/3 for ~3x wall-clock
    # (seeds are independent; cross-seed mean+/-std is then aggregated post-hoc from the run-dirs).
    seeds = [int(os.environ["RUN_SEED"])] if os.environ.get("RUN_SEED") else cfg["seeds"]
    per_seed = []
    for seed in seeds:
        m, d = run_seed(seed, cfg, device, root)
        per_seed.append(m)
        fr = "  ".join(f"{DOMAINS[i][:4]}={m['phi_est'][i]:+.4f}" for i in range(N))
        print(f"\n[seed {seed}] {d}")
        print(f"  phi: {fr}")
        print(f"  AUROC noisy={m['auroc_noisy']:.3f} free-rider={m['auroc_freerider']:.3f}"
              + (f" | est-vs-oracle={m['est_vs_oracle_maxabs']:.2e}" if 'est_vs_oracle_maxabs' in m else ""))
        for name, a in m["arms"].items():
            te = a["task_acc"]
            rl = np.mean([te[d2]["rouge_l"] for d2 in te])
            em = te.get("math", {}).get("exact_match")
            print(f"  arm {name:11s} keep={a['idxs']} val_loss {a['val_loss_curve'][0]:.3f}->{a['val_loss_curve'][-1]:.3f}"
                  f" | ROUGE-L(avg)={rl:.3f}" + (f" math-EM={em:.3f}" if em is not None else ""))

    # ---- aggregate across seeds ----
    def agg(key):
        xs = [m[key] for m in per_seed]
        return float(np.mean(xs)), float(np.std(xs))
    print("\n=== aggregate (mean+/-std over seeds) ===")
    print(f"  AUROC noisy={agg('auroc_noisy')[0]:.3f}+/-{agg('auroc_noisy')[1]:.3f} "
          f"free-rider={agg('auroc_freerider')[0]:.3f}+/-{agg('auroc_freerider')[1]:.3f}")
    for name in ("full", "flirds_topk", "random_k"):
        rls = [np.mean([m["arms"][name]["task_acc"][d2]["rouge_l"] for d2 in m["arms"][name]["task_acc"]])
               for m in per_seed]
        print(f"  arm {name:11s} ROUGE-L(avg)={np.mean(rls):.3f}+/-{np.std(rls):.3f}")
    print("\nFlirds-topk should match/beat random_k (and approach full) if selection works.")


if __name__ == "__main__":
    main()
