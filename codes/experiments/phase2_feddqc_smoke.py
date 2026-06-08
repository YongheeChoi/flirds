"""Phase 2 step4 FedDQC smoke: data-quality (answer_swap) detection AUROC at 1B.

build(5, noisy={c}) -> one answer_swap noisy client (prompts paired with the wrong
answers within the client).  FedDQC IRA = L(a) - L(a|q) per client: the noisy client's
instruction no longer explains its answer -> low IRA -> highest suspicion (-IRA).
Reports per-client score (by domain) + the noisy-detection AUROC -- the §3.9
data-quality baseline's headline.

Run from codes/:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase2_feddqc_smoke.py
env: NOISY (client idx), TRAIN, N_SAMPLES, SEED.
"""
import os

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from flirds.baselines.feddqc import feddqc_scores
from flirds.data.llm import ORDER, build
from flirds.eval.metrics import detection_auroc
from flirds.repro import seed_everything

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
NOISY = frozenset({int(os.environ.get("NOISY", "0"))})
TRAIN = int(os.environ.get("TRAIN", "300"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "128"))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(os.environ.get("SEED", "0"))
    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 attn_implementation="eager").to(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=TARGET_MODULES,
                                             lora_dropout=0.0, task_type="CAUSAL_LM"))
    clients, _val, _test = build(5, TRAIN, per_domain_val=20, per_domain_test=0, seed=seed, noisy=NOISY)
    print(f"[seed {seed}] N=5 noisy={sorted(NOISY)} ({','.join(ORDER[c] for c in sorted(NOISY))}) "
          f"train={TRAIN} n_samples={N_SAMPLES}", flush=True)

    scores = feddqc_scores(clients, model, tok, device, n_samples=N_SAMPLES, seed=seed)
    labels = [1 if c in NOISY else 0 for c in range(5)]
    print("  per-client suspicion (-mean IRA; HIGH = noisy):")
    for c in range(5):
        flag = " <-- noisy" if c in NOISY else ""
        print(f"    client {c} ({ORDER[c]:8s}): {scores[c]:+.4f}{flag}")
    auroc = detection_auroc(scores, labels)
    rank = int((scores > scores[sorted(NOISY)[0]]).sum())
    print(f"\n  noisy AUROC = {auroc:.3f}   (noisy rank {rank}/5)")
    print("FEDDQC SMOKE OK" if np.isfinite(scores).all() else "FEDDQC SMOKE: non-finite scores")


if __name__ == "__main__":
    main()
