"""Phase 2 task 7e FLTrust cross-device smoke: N=100, K=10 real-1B port + AUROC.

Validates the port on genuine LoRA updates and the "cosine ~= Flirds-1st" claim:
FLTrust = mean cos(Δw_i, ∇_val) is the NORMALIZED Flirds first-order term, so the
two should rank free-riders the same way.  Reports both detectors' free-rider AUROC
and Spearman(FLTrust, Flirds-1st) over the seen clients.

FLTrust's cosine is scale-free, so the random free-rider needs no benign-std tuning
(unlike STD-DAGMM) -- any random Δw is ~orthogonal to the val gradient.  Under
non-IID a benign client's LOCAL descent need not align with the mixed-domain VAL
gradient, so this also surfaces FLTrust's known non-IID erosion (honest: report it).

K=10/N=100 -> a free-rider participates only ~R/10 rounds; headline = AUROC over the
SEEN clients (full-N reported too).  Real 1B, GPU.

Run from codes/:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase2_crossdevice_fltrust_smoke.py
"""
import os
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from flirds.backends.llm import make_llm_loss
from flirds.baselines.fltrust import fltrust_from_logs
from flirds.baselines.ripple import _flat
from flirds.core.flirds_estimator import flirds_values
from flirds.data.llm import build_crossdevice, build_val_batches
from flirds.eval.metrics import detection_auroc
from flirds.fl.llm_server import run_llm_fedavg_logs
from flirds.repro import seed_everything

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
N, K_FRAC = 100, 0.1
ALPHA = float(os.environ.get("ALPHA", "0.5"))
FREE_RIDERS = frozenset({10, 30, 50, 70, 90})     # 5/100 random free-riders
CFG = dict(per_client=20, pool=2000, val=10, rounds=int(os.environ.get("R_EVAL", "30")),
           warmup=int(os.environ.get("R_WARM", "3")),
           max_steps=int(os.environ.get("MAX_STEPS", "3")), lr=1e-3, batch=8,
           maxlen=768, val_maxlen=384, val_chunk=10)


def _benign_std(logs):
    """Mean std of the (benign) clients' flattened updates over the trajectory."""
    keys = sorted(next(iter(logs[0][1].values()))[0].keys())
    return float(np.mean([float(_flat(dm[c][0], keys).std()) for _, dm in logs for c in dm]))


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

    clients, val, _ = build_crossdevice(N, alpha=ALPHA, per_client_train=CFG["per_client"],
                                        per_domain_pool=CFG["pool"], per_domain_val=CFG["val"],
                                        seed=seed, noisy=set())
    val_chunks = build_val_batches(val, tok, CFG["val_maxlen"], device, CFG["val_chunk"])
    K = round(K_FRAC * N)
    print(f"[seed {seed}] N={N} alpha={ALPHA} K={K} R={CFG['rounds']} free_riders={sorted(FREE_RIDERS)} "
          f"val={len(val)}/{len(val_chunks)}ch", flush=True)

    # warmup -> benign update std.  FLTrust's cosine is scale-free, but the random
    # free-rider is benign-std-matched so the UNNORMALIZED Flirds-1st comparison is fair.
    model.load_state_dict(init_lora, strict=False)
    warm = run_llm_fedavg_logs(model, tok, clients, CFG["warmup"], CFG["lr"], CFG["max_steps"],
                               batch_size=CFG["batch"], max_length=CFG["maxlen"],
                               sample_frac=K_FRAC, seed=seed)
    fr_scale = _benign_std(warm) * (3 ** 0.5)         # uniform(-s,s) std = s/sqrt(3)
    print(f"  benign update std={_benign_std(warm):.3e} -> random free-rider scale={fr_scale:.3e}")

    model.load_state_dict(init_lora, strict=False)
    logs = run_llm_fedavg_logs(model, tok, clients, CFG["rounds"], CFG["lr"], CFG["max_steps"],
                               batch_size=CFG["batch"], max_length=CFG["maxlen"],
                               sample_frac=K_FRAC, seed=seed, free_riders=FREE_RIDERS,
                               free_rider_mode="random", free_rider_scale=fr_scale)
    loss_fn, pkeys, lc = make_llm_loss(model, val_chunks, device)
    selected = sorted({k for _, dm in logs for k in dm})
    fr_seen = sorted(FREE_RIDERS & set(selected))
    print(f"  rounds={len(logs)} distinct_selected={len(selected)}/{N} free-rider seen={fr_seen}")
    assert fr_seen, "no free-rider participated -> raise R_EVAL or change SEED"

    t0 = time.perf_counter()
    score = fltrust_from_logs(logs, N, loss_fn, pkeys, device, loss_chunks=lc)
    dt = time.perf_counter() - t0
    phi1, _ = flirds_values(logs, loss_fn, pkeys, device, second_order=False,
                            n_clients=N, loss_chunks=lc)           # Flirds-1st (unnormalized cousin)
    assert len(score) == N and np.isfinite(score).all()

    labels = [1 if c in FREE_RIDERS else 0 for c in range(N)]
    auroc_all = detection_auroc(score, labels)
    auroc_seen = roc_auc_score([labels[c] for c in selected], [score[c] for c in selected])
    auroc1_seen = roc_auc_score([labels[c] for c in selected], [phi1[c] for c in selected])
    rho = float(spearmanr([score[c] for c in selected], [phi1[c] for c in selected]).correlation)
    ranks = {c: int((score > score[c]).sum()) for c in fr_seen}
    print(f"  FLTrust runtime={dt:.1f}s (1 val-grad/round x {len(logs)})")
    print(f"  free-rider suspicion rank (0=top of {N}): {ranks}")
    print(f"  AUROC free-rider: FLTrust seen={auroc_seen:.3f} full-N={auroc_all:.3f} | Flirds-1st seen={auroc1_seen:.3f}")
    print(f"  Spearman(FLTrust, Flirds-1st) over {len(selected)} seen = {rho:+.3f}  (cosine ~= normalized Flirds-1st)")
    print("\nCROSS-DEVICE FLTRUST SMOKE OK  (orientation/AUROC proven in the unit smoke; this is the real-1B port)")


if __name__ == "__main__":
    main()
