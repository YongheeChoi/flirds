"""Phase 1 LLM FL smoke: a REAL SFTTrainer FedAvg trajectory -> estimator/oracle.

Unlike phase1_llm_smoke (fake Δw), this runs an actual mini LLM FedAvg via
fl.llm_server.run_llm_fedavg_logs (TRL SFTTrainer local train, forced plain SGD,
completion-only loss), producing real LoRA-only logs, then runs the estimator
and (b) oracle on them.  Validates the full LLM stack-2 path end to end on
Llama-3.2-1B: SFTTrainer+SGD local train, LoRA delta extraction, logs contract,
and estimator/oracle on a realized trajectory.  fp32 + eager (forward-AD HVP).
"""
import os

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from flirds.backends.llm import make_llm_loss
from flirds.core.flirds_estimator import flirds_values
from flirds.fl.llm_server import run_llm_fedavg_logs
from flirds.oracle.in_run_sv import in_run_shapley
from flirds.repro import seed_everything

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

FACTS = [("The capital of France is", " Paris."),
         ("Water freezes at", " zero degrees Celsius."),
         ("The sun rises in the", " east."),
         ("Two plus three equals", " five."),
         ("The opposite of hot is", " cold."),
         ("A week has", " seven days.")]


def client_datasets():   # 3 clients, 2 disjoint prompt-completion facts each
    return [Dataset.from_list([{"prompt": p, "completion": c}
                               for p, c in FACTS[2 * i:2 * i + 2]]) for i in range(3)]


def main():
    seed_everything(0)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 attn_implementation="eager").to(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=TARGET,
                                             lora_dropout=0.0, task_type="CAUSAL_LM"))

    # ---- real LLM FedAvg trajectory ----
    logs = run_llm_fedavg_logs(model, tok, client_datasets(), rounds=2, lr=1e-3,
                               max_steps=2, batch_size=2, max_length=32, seed=0)
    dw_norm = float(sum((logs[0][1][0][0][k] ** 2).sum() for k in logs[0][0]) ** 0.5)
    print(f"logs: {len(logs)} rounds | round-0 clients={sorted(logs[0][1])} | "
          f"#LoRA keys={len(logs[0][0])} | ||Δw_0^0||={dw_norm:.3e}")

    # ---- estimator vs (b) oracle on the realized trajectory ----
    texts = ["The capital of France is Paris.", "Water freezes at zero degrees Celsius."]
    enc = tok(texts, return_tensors="pt", padding=True)
    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100
    val_batch = {"input_ids": enc.input_ids.to(device),
                 "attention_mask": enc.attention_mask.to(device),
                 "labels": labels.to(device)}
    loss_fn, pkeys = make_llm_loss(model, val_batch, device)

    phi_e, _ = flirds_values(logs, loss_fn, pkeys, device, second_order=True)
    phi_b, _ = in_run_shapley(logs, 3, loss_fn, pkeys, device)
    fin = bool(np.isfinite(phi_e).all() and np.isfinite(phi_b).all())
    print("estimator 1st+2nd:", phi_e)
    print("oracle (b) exact :", phi_b)
    print(f"finite={fin} | max|est-oracle|={np.max(np.abs(phi_e - phi_b)):.2e}")
    print("LLM-FL SMOKE OK" if fin else "LLM-FL SMOKE FAIL")


if __name__ == "__main__":
    main()
