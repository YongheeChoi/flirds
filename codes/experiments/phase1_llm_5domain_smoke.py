"""Phase 1 end-to-end infra check on the REAL 5-domain data (N=5 cross-silo).

Real free-form medical/legal/finance/math/general clients -> SFTTrainer FedAvg ->
estimator vs (b) oracle on a §3.4-style validation set, the val fed in CHUNKS
(data.llm.build_val_batches + make_llm_loss loss_chunks) so the eager HVP fits.
Three checks:
  (1) scale: val=100 @ seq=384 in chunks of 10 -> est ~= (b) oracle (a single-shot
      HVP at this size OOMs the B200);
  (2) equivalence: on a small val that fits one shot, chunked phi == single-shot phi;
  (3) per-domain macro-average (chunk_domains): est ~= (b) oracle under domain-norm.
fp32 + eager (forward-AD HVP).
"""
import os

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from flirds.backends.llm import make_llm_loss
from flirds.core.flirds_estimator import flirds_values
from flirds.data.llm import build, build_val_batches, build_val_batches_by_domain
from flirds.fl.llm_server import run_llm_fedavg_logs
from flirds.oracle.in_run_sv import in_run_shapley
from flirds.repro import seed_everything

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAXLEN = 768        # training seq cap (legal CaseHOLD prompts run 325-462 tok)
VAL_MAXLEN = 384    # chunking bounds memory, so the val seq can be longer than before
CHUNK = 10          # val rows per HVP chunk (estimator); the oracle reuses these
PDV = 20            # per-domain val rows (5 domains -> val=100)


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

    clients, val = build(n_clients=5, per_domain_train=8, per_domain_val=PDV, seed=0)
    print(f"clients={len(clients)} sizes={[len(c) for c in clients]} val={len(val)}")

    logs = run_llm_fedavg_logs(model, tok, clients, rounds=2, lr=1e-3,
                               max_steps=2, batch_size=2, max_length=MAXLEN, seed=0)

    # ---- (1) scale: val=100 @ seq=384 in chunks of 10 (a single-shot HVP OOMs) ----
    val_chunks = build_val_batches(val, tok, VAL_MAXLEN, device, chunk_size=CHUNK)
    loss_fn, pkeys, loss_chunks = make_llm_loss(model, val_chunks, device)
    print(f"val: {len(val_chunks)}x{CHUNK} rows @ seq<={VAL_MAXLEN}")
    phi_e, _ = flirds_values(logs, loss_fn, pkeys, device, second_order=True, loss_chunks=loss_chunks)
    phi_b, _ = in_run_shapley(logs, 5, loss_fn, pkeys, device)
    fin = bool(np.isfinite(phi_e).all() and np.isfinite(phi_b).all())
    print("estimator 1st+2nd:", phi_e)
    print("oracle (b) exact :", phi_b)
    print(f"finite={fin} | max|est-oracle|={np.max(np.abs(phi_e - phi_b)):.2e}")

    # ---- (2) equivalence: chunked == single-shot on a small val (fits one shot) ----
    small = val[:8]
    one = build_val_batches(small, tok, VAL_MAXLEN, device, chunk_size=8)    # 1 chunk
    many = build_val_batches(small, tok, VAL_MAXLEN, device, chunk_size=3)   # 3 chunks
    lf1, pk, _ = make_llm_loss(model, one, device)
    lfk, _, lck = make_llm_loss(model, many, device)
    phi_single, _ = flirds_values(logs, lf1, pk, device, second_order=True, loss_chunks=None)
    phi_chunk, _ = flirds_values(logs, lfk, pk, device, second_order=True, loss_chunks=lck)
    dchunk = float(np.max(np.abs(phi_single - phi_chunk)))
    print(f"chunked-vs-single max|Δφ| = {dchunk:.2e}")

    # ---- (3) per-domain macro-average: domain-norm est ~= (b) oracle ----
    db, cd = build_val_batches_by_domain(val, PDV, tok, VAL_MAXLEN, device, chunk_size=CHUNK)
    lfn, pkn, lcn = make_llm_loss(model, db, device, chunk_domains=cd, n_domains=5)
    phi_en, _ = flirds_values(logs, lfn, pkn, device, second_order=True, loss_chunks=lcn)
    phi_bn, _ = in_run_shapley(logs, 5, lfn, pkn, device)
    dnorm = float(np.max(np.abs(phi_en - phi_bn)))
    print(f"domain-norm est-vs-oracle max|Δφ| = {dnorm:.2e}")

    ok = fin and dchunk < 1e-5 and dnorm < 1e-5
    print("5-DOMAIN INFRA + CHUNK + DOMAIN-NORM OK" if ok else "5-DOMAIN SMOKE FAIL")


if __name__ == "__main__":
    main()
