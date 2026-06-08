"""Phase 2 task 7b cross-device Flirds smoke: N=100, K=10 partial participation.

Verifies the EXISTING estimator + FL loop handle cross-device with NO library
changes: build_crossdevice -> run_llm_fedavg_logs(sample_frac=0.1) [K=10/round]
-> flirds_values(n_clients=100).  Asserts every round has exactly K participants;
phi is length-N; un-selected clients get phi=0 (the partial-participation path);
free-rider(zero) phi == 0 even when selected.  (b) in-run Shapley is NOT run here
(exact 2^100 is infeasible -> permutation-MC oracle is task 7c).  Real 1B, GPU.
"""
import os
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from flirds.backends.llm import make_llm_loss
from flirds.core.flirds_estimator import flirds_values
from flirds.data.llm import build_crossdevice, build_val_batches
from flirds.fl.llm_server import run_llm_fedavg_logs
from flirds.repro import seed_everything

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
N, K_FRAC = 100, 0.1
ALPHA = float(os.environ.get("ALPHA", "0.5"))
NOISY, FREE_RIDERS, FREE_RIDER_MODE = {0}, {1}, "zero"
CFG = dict(per_client=40, pool=2000, val=10, rounds=10, max_steps=5, lr=1e-3,
           batch=8, maxlen=768, val_maxlen=384, val_chunk=10)


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
                                        seed=seed, noisy=NOISY)
    val_chunks = build_val_batches(val, tok, CFG["val_maxlen"], device, CFG["val_chunk"])
    sizes = [len(c) for c in clients]
    K = round(K_FRAC * N)
    print(f"[seed {seed}] N={N} alpha={ALPHA} K={K} R={CFG['rounds']} "
          f"client_sizes(min/max)=({min(sizes)},{max(sizes)}) val={len(val)}/{len(val_chunks)}ch",
          flush=True)

    model.load_state_dict(init_lora, strict=False)
    logs = run_llm_fedavg_logs(model, tok, clients, CFG["rounds"], CFG["lr"], CFG["max_steps"],
                               batch_size=CFG["batch"], max_length=CFG["maxlen"],
                               sample_frac=K_FRAC, seed=seed,
                               free_riders=FREE_RIDERS, free_rider_mode=FREE_RIDER_MODE)
    loss_fn, pkeys, lc = make_llm_loss(model, val_chunks, device)

    # ---- partial-participation structure of the logs ----
    part_counts = [len(dm) for _, dm in logs]
    selected = sorted({k for _, dm in logs for k in dm})
    print(f"  rounds={len(logs)} participants/round={part_counts} distinct_selected={len(selected)}/{N}")
    assert all(pc == K for pc in part_counts), f"expected {K} participants/round, got {part_counts}"

    t0 = time.perf_counter()
    phi, p = flirds_values(logs, loss_fn, pkeys, device, second_order=True,
                           n_clients=N, loss_chunks=lc)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    assert len(phi) == N, "phi not length-N"
    for k in range(N):                                  # unselected -> never in logs -> phi == 0
        if k not in selected:
            assert phi[k] == 0.0, f"client {k} unselected but phi={phi[k]}"
    nz = int(np.count_nonzero(phi))
    assert nz <= len(selected)
    print(f"  phi: len={len(phi)} nonzero={nz} <= selected={len(selected)} "
          f"range=[{phi.min():+.4f},{phi.max():+.4f}]")
    fr, ns = sorted(FREE_RIDERS)[0], sorted(NOISY)[0]
    print(f"  free-rider c{fr}: selected={fr in selected} phi={phi[fr]:+.6f} (zero-delta -> exactly 0)")
    print(f"  noisy     c{ns}: selected={ns in selected} phi={phi[ns]:+.6f}")
    if fr in selected:
        assert phi[fr] == 0.0, f"free-rider(zero) phi must be exactly 0, got {phi[fr]}"
    print(f"  flirds_values runtime={dt:.1f}s (1 HVP/round x {len(logs)} = cheap)")

    if os.environ.get("ORACLE") == "1":   # exact per-round (b) oracle on the SAME logs (task 7c)
        from scipy.stats import spearmanr

        from flirds.oracle.in_run_sv import in_run_shapley_perround
        n_fwd = sum(2 ** len(dm) - 1 for _, dm in logs)
        t0 = time.perf_counter()
        phi_b, _ = in_run_shapley_perround(logs, N, loss_fn, pkeys, device)
        if device == "cuda":
            torch.cuda.synchronize()
        dt_o = time.perf_counter() - t0
        rho = float(spearmanr(phi[selected], phi_b[selected]).correlation)
        ms = dt_o / n_fwd * 1000
        print(f"\n  [ORACLE] exact per-round (b): forwards={n_fwd} runtime={dt_o:.1f}s ({ms:.1f}ms/fwd)")
        print(f"  [ORACLE] Spearman(Flirds, (b)) over {len(selected)} selected = {rho:+.4f}")
        print(f"  [ORACLE] extrapolate R=200,K=10: {200 * 1023} fwd ~= "
              f"{200 * 1023 * ms / 1000 / 60:.0f} min/1-GPU (shards across rounds)")
    print("\nCROSS-DEVICE FLIRDS SMOKE OK")


if __name__ == "__main__":
    main()
