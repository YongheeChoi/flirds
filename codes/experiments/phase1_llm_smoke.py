"""Phase 1 LLM smoke: does the estimator/oracle HVP run on a PEFT-LoRA causal LM?

Builds Llama-3.2-1B-Instruct + LoRA (fp32 eval), a tiny val batch, and a tiny
FAKE trajectory (LoRA-only w_r + small Δw_k per client), then checks:
  - flirds_values (1 HVP/round via jvp∘grad) runs + finite
  - in_run_shapley (exact 2^N forward) runs + finite
  - estimator ≈ (b) oracle on tiny Δw (Taylor near-exact in the small-step limit)
  - per-layer seam-1 invariant (Σcomp == φ) on a real LLM param count

This validates the BACKEND INTERFACE (peft LoRA + torch.func
functional_call/jvp/grad), NOT the science — real logs come from the LLM FL
loop (stage 2).  fp32 model; CUDA_VISIBLE_DEVICES=0 is enough.
"""
import os

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from flirds.backends.llm import make_llm_loss
from flirds.core.flirds_estimator import flirds_values
from flirds.oracle.in_run_sv import in_run_shapley
from flirds.repro import seed_everything

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build(device):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32,
                                                 attn_implementation="eager")
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=TARGET,
                     lora_dropout=0.0, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg).to(device), tok


def main():
    seed_everything(0)
    device = "cuda"
    model, tok = build(device)

    texts = ["The capital of France is Paris.",
             "Water boils at one hundred degrees Celsius at sea level."]
    enc = tok(texts, return_tensors="pt", padding=True)
    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100
    val_batch = {"input_ids": enc.input_ids.to(device),
                 "attention_mask": enc.attention_mask.to(device),
                 "labels": labels.to(device)}

    loss_fn, pkeys = make_llm_loss(model, val_batch, device)
    lora0 = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    print(f"#LoRA pkeys: {len(pkeys)} | base val loss: {float(loss_fn(lora0, {})):.4f}")

    # tiny FAKE trajectory: 2 rounds × 3 clients, LoRA-only w_r + small Δw_k
    gen = torch.Generator(device=device).manual_seed(0)
    logs = []
    for _ in range(2):
        w_r = {n: v.clone() for n, v in lora0.items()}        # frozen base NOT stored
        dm = {k: ({n: 1e-3 * torch.randn(v.shape, generator=gen, device=device, dtype=v.dtype)
                   for n, v in lora0.items()}, 100 + k) for k in range(3)}
        logs.append((w_r, dm))

    phi_e, _ = flirds_values(logs, loss_fn, pkeys, device, second_order=True)
    phi_e1, _ = flirds_values(logs, loss_fn, pkeys, device, second_order=False)
    phi_b, _ = in_run_shapley(logs, 3, loss_fn, pkeys, device)
    fin = bool(np.isfinite(phi_e).all() and np.isfinite(phi_b).all())
    print("estimator 1st+2nd:", phi_e)
    print("estimator 1st    :", phi_e1)
    print("oracle (b) exact :", phi_b)
    print(f"finite={fin} | max|est2-oracle|={np.max(np.abs(phi_e - phi_b)):.2e} | "
          f"max|est1-oracle|={np.max(np.abs(phi_e1 - phi_b)):.2e}")

    # seam-1 per-layer invariant on a real LLM param count (Σcomp == φ)
    phi_pl, _, comp = flirds_values(logs, loss_fn, pkeys, device,
                                    second_order=True, per_layer=True)
    inv = max(abs(sum(comp[k].values()) - phi_pl[k]) for k in range(3))
    print(f"per-layer invariant max|Σcomp-φ|={inv:.2e}")
    print("SMOKE OK" if fin else "SMOKE FAIL (non-finite)")


if __name__ == "__main__":
    main()
