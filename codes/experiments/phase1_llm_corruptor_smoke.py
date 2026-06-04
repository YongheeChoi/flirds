"""Phase 1 seam-2 corruptor smoke (real 1B): noisy client + free-rider injected
into the 5-domain cross-silo run, verified against the (b) in-run oracle.

PLUMBING gates (N=5, one noisy + one free-rider; the guaranteed, scale-independent
properties -- detection QUALITY (noisy ranks worst) needs experiment scale and is
validated at the #7 first clean run, only reported here):
  (1) free-rider (mode=zero) phi is EXACTLY 0 in BOTH estimator and (b) oracle
      -- a zero delta has zero marginal in every coalition (rock-solid guarantee);
  (2) estimator ~= (b) oracle with the corrupted clients in the mix;
  (3) free-rider (mode=random) -> small bounded phi, tracked identically by both
      (random direction ~orthogonal to the val gradient -> ~0, not exactly 0).
A directional "noisy phi - clean-mean" is printed (sign convention "higher phi =
more val-loss = worse" from phase05_flirds_oracle) but NOT asserted -- at 8ex/2step
all phi sit at the noise floor.
fp32 + eager (forward-AD HVP); val fed in chunks (make_llm_loss loss_chunks).

Run from codes/:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase1_llm_corruptor_smoke.py
"""
import os

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from flirds.backends.llm import make_llm_loss
from flirds.core.flirds_estimator import flirds_values
from flirds.data.llm import build, build_val_batches
from flirds.fl.llm_server import run_llm_fedavg_logs
from flirds.oracle.in_run_sv import in_run_shapley
from flirds.repro import seed_everything

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAXLEN, VAL_MAXLEN, CHUNK, PDV = 768, 256, 10, 10
NOISY, FREE_RIDER = {0}, {1}            # client 0 (medical) noisy; client 1 (legal) free-rider
DOMAINS = ["medical", "legal", "finance", "math", "general"]


def _run(model, init_lora, tok, device, clients, val_chunks, mode):
    model.load_state_dict(init_lora, strict=False)            # fresh LoRA start each run
    logs = run_llm_fedavg_logs(model, tok, clients, rounds=2, lr=1e-3, max_steps=2,
                               batch_size=2, max_length=MAXLEN, seed=0,
                               free_riders=FREE_RIDER, free_rider_mode=mode)
    loss_fn, pkeys, lc = make_llm_loss(model, val_chunks, device)
    phi_e, _ = flirds_values(logs, loss_fn, pkeys, device, second_order=True, loss_chunks=lc)
    phi_b, _ = in_run_shapley(logs, 5, loss_fn, pkeys, device)
    return phi_e, phi_b


def _fmt(ph):
    tag = lambda i: "*" if i in NOISY else ("F" if i in FREE_RIDER else " ")
    return "  ".join(f"{DOMAINS[i][:4]}{tag(i)}={ph[i]:+.4f}" for i in range(5))


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
    init_lora = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    clients, val, _ = build(n_clients=5, per_domain_train=8, per_domain_val=PDV, seed=0, noisy=NOISY)
    val_chunks = build_val_batches(val, tok, VAL_MAXLEN, device, chunk_size=CHUNK)
    print(f"clients={[len(c) for c in clients]} noisy={sorted(NOISY)} "
          f"free_rider={sorted(FREE_RIDER)} val={len(val)} in {len(val_chunks)} chunks")
    fr, no = next(iter(FREE_RIDER)), next(iter(NOISY))
    clean = [i for i in range(5) if i not in NOISY | FREE_RIDER]

    # ---- free-rider = zero: phi must be EXACTLY 0 (zero delta -> zero marginal) ----
    phi_e, phi_b = _run(model, init_lora, tok, device, clients, val_chunks, "zero")
    print(f"\n[free_rider=zero]\n  est : {_fmt(phi_e)}\n  (b) : {_fmt(phi_b)}")
    fr_zero = abs(phi_e[fr]) < 1e-9 and abs(phi_b[fr]) < 1e-9
    close = float(np.max(np.abs(phi_e - phi_b)))
    finite = bool(np.isfinite(phi_e).all() and np.isfinite(phi_b).all())
    print(f"  free-rider phi==0 (est&oracle)={fr_zero} | max|est-oracle|={close:.2e} | finite={finite}")

    # ---- free-rider = random: small nonzero phi, still tracked by the oracle ----
    phi_er, phi_br = _run(model, init_lora, tok, device, clients, val_chunks, "random")
    print(f"\n[free_rider=random]\n  est : {_fmt(phi_er)}\n  (b) : {_fmt(phi_br)}")
    close_r = float(np.max(np.abs(phi_er - phi_br)))
    fr_bounded = abs(phi_er[fr]) < 1e-2 and abs(phi_br[fr]) < 1e-2     # didn't blow up
    finite_r = bool(np.isfinite(phi_er).all() and np.isfinite(phi_br).all())
    print(f"  free-rider |phi| bounded={fr_bounded} | max|est-oracle|={close_r:.2e} | finite={finite_r}")

    # ---- directional note (NOT asserted: detection quality is experiment-scale) ----
    sig = float(phi_e[no] - np.mean([phi_e[i] for i in clean]))
    print(f"\n  noisy phi - clean-mean = {sig:+.2e}  (>0 = noisy looks worse; weak at "
          f"smoke scale 8ex/2step -- detection quality validated at the #7 clean run)")

    # plumbing-only gates (the guaranteed, scale-independent properties)
    ok = fr_zero and close < 1e-4 and finite and fr_bounded and close_r < 1e-4 and finite_r
    print("\nSEAM-2 LLM CORRUPTOR PLUMBING OK" if ok else "\nSEAM-2 SMOKE FAIL")


if __name__ == "__main__":
    main()
