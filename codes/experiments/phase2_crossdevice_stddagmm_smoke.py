"""Phase 2 task 7e STD-DAGMM cross-device smoke: N=100, K=10 real-1B port + AUROC.

Validates the detector on GENUINE ~5.6M-dim LoRA updates under partial
participation (the synthetic smoke proves detection quality; this proves the
port).  STD-DAGMM is MODEL-FREE -> no val / loss_fn, only the trajectory logs.

Two FL passes (the detector consumes only logs):
  1. WARMUP (no free-rider) -> measure the benign update std, the Lin et al.
     tuning target -- a RANDOM free-rider is then set to that std (the evasion
     case: std alone cannot flag it; the AE recon/cosine terms must).
  2. EVAL (5 random@benign-std free-riders) -> feature-hash + pool + AE/GMM ->
     per-client energy -> AUROC.

K=10/N=100 means a given free-rider is sampled only ~R/10 rounds, so over a short
smoke some may never participate (they then get the min score -- correct: a
free-rider must participate to reap rewards, but it makes them un-scoreable here).
The headline metric is therefore AUROC over the SEEN clients (those with >=1
update); full-N AUROC is reported too.  Real 1B, GPU.

Run from codes/:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase2_crossdevice_stddagmm_smoke.py
  ... R_EVAL=40 MAX_STEPS=3 python experiments/phase2_crossdevice_stddagmm_smoke.py   # tune cost/coverage
"""
import os
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from flirds.baselines.ripple import _flat
from flirds.baselines.std_dagmm import std_dagmm_from_logs
from flirds.data.llm import build_crossdevice
from flirds.eval.metrics import detection_auroc
from flirds.fl.llm_server import run_llm_fedavg_logs
from flirds.repro import seed_everything

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
N, K_FRAC = 100, 0.1
ALPHA = float(os.environ.get("ALPHA", "0.5"))
FREE_RIDERS = frozenset({10, 30, 50, 70, 90})     # 5/100 random@benign-std free-riders
CFG = dict(per_client=20, pool=2000, rounds=int(os.environ.get("R_EVAL", "30")),
           warmup=int(os.environ.get("R_WARM", "3")),
           max_steps=int(os.environ.get("MAX_STEPS", "3")), lr=1e-3, batch=8, maxlen=768)


def _benign_std(logs, free_riders):
    """Mean std of the benign clients' flattened updates over the trajectory."""
    keys = sorted(next(iter(logs[0][1].values()))[0].keys())
    stds = [float(_flat(dm[c][0], keys).std())
            for _, dm in logs for c in dm if c not in free_riders]
    return float(np.mean(stds))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(os.environ.get("SEED", "0"))
    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 attn_implementation="eager").to(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=TARGET,
                                             lora_dropout=0.0, task_type="CAUSAL_LM"))
    init_lora = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    clients, _, _ = build_crossdevice(N, alpha=ALPHA, per_client_train=CFG["per_client"],
                                      per_domain_pool=CFG["pool"], per_domain_val=2,
                                      seed=seed, noisy=set())
    K = round(K_FRAC * N)
    print(f"[seed {seed}] N={N} alpha={ALPHA} K={K} warmup={CFG['warmup']} R={CFG['rounds']} "
          f"free_riders={sorted(FREE_RIDERS)} sizes(min/max)=({min(map(len, clients))},{max(map(len, clients))})",
          flush=True)

    # ---- 1. warmup (clean) -> benign update std (the Lin et al. tuning target) ----
    model.load_state_dict(init_lora, strict=False)
    warm = run_llm_fedavg_logs(model, tok, clients, CFG["warmup"], CFG["lr"], CFG["max_steps"],
                               batch_size=CFG["batch"], max_length=CFG["maxlen"],
                               sample_frac=K_FRAC, seed=seed)
    benign_std = _benign_std(warm, frozenset())
    fr_scale = benign_std * (3 ** 0.5)        # uniform(-s,s) has std s/sqrt(3) -> match benign std
    print(f"  benign update std={benign_std:.3e} -> random free-rider scale={fr_scale:.3e} (std-matched evasion)")

    # ---- 2. eval (random@benign-std free-riders) ----
    model.load_state_dict(init_lora, strict=False)
    logs = run_llm_fedavg_logs(model, tok, clients, CFG["rounds"], CFG["lr"], CFG["max_steps"],
                               batch_size=CFG["batch"], max_length=CFG["maxlen"],
                               sample_frac=K_FRAC, seed=seed, free_riders=FREE_RIDERS,
                               free_rider_mode="random", free_rider_scale=fr_scale)
    selected = sorted({k for _, dm in logs for k in dm})
    fr_seen = sorted(FREE_RIDERS & set(selected))
    fr_part = {c: sum(c in dm for _, dm in logs) for c in sorted(FREE_RIDERS)}
    print(f"  rounds={len(logs)} distinct_selected={len(selected)}/{N} "
          f"free-rider participation={fr_part} (seen={fr_seen})")
    assert fr_seen, "no free-rider participated -> raise R_EVAL or change SEED"

    t0 = time.perf_counter()
    score = std_dagmm_from_logs(logs, N, seed=seed, device="cpu")
    dt = time.perf_counter() - t0
    assert len(score) == N and np.isfinite(score).all(), "score not finite length-N"

    labels = [1 if c in FREE_RIDERS else 0 for c in range(N)]
    auroc_all = detection_auroc(score, labels)
    seen_lab = [labels[c] for c in selected]
    auroc_seen = roc_auc_score(seen_lab, [score[c] for c in selected])   # both classes present
    ranks = {c: int((score > score[c]).sum()) for c in fr_seen}          # 0 = most suspicious
    print(f"  STD-DAGMM runtime={dt:.1f}s (CPU AE/GMM over {len(selected)}-round pooled samples)")
    print(f"  free-rider energy rank (0=top of {N}): {ranks}")
    print(f"  AUROC seen-subset({len(selected)} clients, {len(fr_seen)} free-riders)={auroc_seen:.3f}"
          f"   AUROC full-N(unseen->min)={auroc_all:.3f}")
    print("\nCROSS-DEVICE STD-DAGMM SMOKE OK  (detection quality proven in the synthetic smoke;"
          " this is the real-1B port + orientation check)")


if __name__ == "__main__":
    main()
