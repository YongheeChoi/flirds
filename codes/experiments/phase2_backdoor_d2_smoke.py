"""Phase 2 §3.9 D2 — FL model-replacement propagation (Bagdasaryan), backdoor from D1.

D1 showed the backdoor installs in a LOCAL model (frac=0.5, clean-preserving).  D2 asks:
does Bagdasaryan model-replacement carry it to the GLOBAL model, and at what cost to the
clean validation loss?  Faithful to sources/how-to-backdoor-fl-bagdasaryan:
  1. benign FL (5 clean clients, ROUNDS) -> converged global G;
  2. attacker trains a backdoor local model X from G (frac=0.5, D1 install strength);
  3. single-shot inject  G_bd = G + (1/N)*gamma*(X-G)   [benign deltas ~cancel near
     convergence, the paper's assumption], gamma swept:
       full        = n/eta = N        (full replacement, G_bd = X)
       partial     = N/2              (half-strength injection)
       norm-bound  = S/||X-G||        (stealthy: aggregate contribution = a benign client's)
  4. report global triggered-ASR + clean-ASR + clean val-loss (G vs G_bd).
Detector AUROC on the scaled update = D2b (next step).

Run from codes/:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase2_backdoor_d2_smoke.py
env: BD_TARGET, POISON_FRAC, EPOCHS (attacker), LR, TRAIN, ROUNDS, SEED.
"""
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from flirds.data.corruptors import BACKDOOR_TRIGGER
from flirds.data.llm import build, build_val_batches
from flirds.eval.generate import backdoor_asr, generate_completions
from flirds.fl.llm_server import run_llm_fedavg_logs
from flirds.repro import seed_everything

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TARGET = os.environ.get("BD_TARGET", "delicious")
FRAC = float(os.environ.get("POISON_FRAC", "0.5"))
EPOCHS = float(os.environ.get("EPOCHS", "3"))          # attacker local epochs (D1 install strength)
LR = float(os.environ.get("LR", "2e-3"))
TRAIN = int(os.environ.get("TRAIN", "1000"))
ROUNDS = int(os.environ.get("ROUNDS", "10"))
BENIGN_STEPS = int(os.environ.get("BENIGN_STEPS", "5"))
N = 5
_OUT = "/tmp/flirds_backdoor_d2"


def _final_global(init, logs, pkeys, device):
    """Reconstruct the deployed global G = init + Sum_r (round FedAvg aggregate)."""
    G = {n: init[n].detach().clone().float().to(device) for n in pkeys}
    for _, dm in logs:
        tot = sum(nc for _, nc in dm.values())
        for k in dm:
            d, nc = dm[k]
            for n in pkeys:
                G[n] += (nc / tot) * d[n].float().to(device)
    return G


def _flatnorm(delta):
    return float(torch.sqrt(sum((v.float() ** 2).sum() for v in delta.values())))


@torch.no_grad()
def _clean_val_loss(model, val_chunks):
    model.eval()
    model.config.use_cache = False
    num = den = 0.0
    for b in val_chunks:
        n = int((b["labels"][:, 1:] != -100).sum())
        num += model(**b).loss.item() * n
        den += n
    return num / den


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
    init_lora = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    pkeys = list(init_lora)

    clean_clients, val, test = build(5, TRAIN, per_domain_val=20, per_domain_test=40, seed=seed)
    bd_clients, _, _ = build(5, TRAIN, per_domain_val=20, per_domain_test=40, seed=seed,
                             backdoor=frozenset({0}),
                             backdoor_kwargs=dict(target=TARGET, poison_frac=FRAC))
    med_test = [r["prompt"] for r in test if r["domain"] == "medical"]
    val_chunks = build_val_batches(val, tok, 384, device, 10)
    print(f"[seed {seed}] N={N} frac={FRAC} attacker_epochs={EPOCHS} lr={LR} rounds={ROUNDS} "
          f"target={TARGET!r} trigger={BACKDOOR_TRIGGER!r} test={len(med_test)}", flush=True)

    # 1. benign FL (all clean) -> converged global G
    model.load_state_dict(init_lora, strict=False)
    logs = run_llm_fedavg_logs(model, tok, clean_clients, ROUNDS, LR, BENIGN_STEPS,
                               batch_size=8, max_length=768, seed=seed)
    G = _final_global(init_lora, logs, pkeys, device)
    last = logs[-1][1]
    S = sum(_flatnorm(d) for d, _ in last.values()) / len(last)   # benign update norm (last round mean)

    # baseline at G
    model.load_state_dict({k: G[k] for k in pkeys}, strict=False)
    vlG = _clean_val_loss(model, val_chunks)
    aG, _ = backdoor_asr(model, tok, med_test, BACKDOOR_TRIGGER, TARGET, device)
    print(f"  benign global G: clean-val-loss={vlG:.4f}  triggered-ASR={aG:.2f}", flush=True)

    # 2. attacker trains backdoor X from G
    model.load_state_dict({k: G[k] for k in pkeys}, strict=False)
    cfg = SFTConfig(output_dir=_OUT, per_device_train_batch_size=8, num_train_epochs=EPOCHS,
                    learning_rate=LR, max_length=768, lr_scheduler_type="constant", warmup_steps=0,
                    completion_only_loss=True, bf16=False, fp16=False, report_to="none",
                    logging_strategy="no", save_strategy="no", seed=seed)
    SFTTrainer(model=model, args=cfg, train_dataset=bd_clients[0], processing_class=tok,
               optimizer_cls_and_kwargs=(torch.optim.SGD, {"lr": LR, "momentum": 0.0})).train()
    X = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    delta = {k: (X[k] - G[k]) for k in pkeys}
    dnorm = _flatnorm(delta)
    print(f"  benign ||Δ||={S:.4f}  attacker ||Δ||={dnorm:.4f}  (ratio {dnorm / S:.1f}x)", flush=True)

    # 3. single-shot inject sweep
    for name, gamma in [("full n/η=N", float(N)), ("partial N/2", N / 2), ("norm-bound", S / dnorm)]:
        Gbd = {k: G[k] + (1.0 / N) * gamma * delta[k] for k in pkeys}
        model.load_state_dict(Gbd, strict=False)
        atrig, gens = backdoor_asr(model, tok, med_test, BACKDOOR_TRIGGER, TARGET, device)
        cg = generate_completions(model, tok, med_test, device, max_new_tokens=32)
        aclean = sum(TARGET in g for g in cg) / len(cg)
        vl = _clean_val_loss(model, val_chunks)
        print(f"  γ={name:12s} ({gamma:5.2f}): triggered-ASR={atrig:.2f}  clean-ASR={aclean:.2f}  "
              f"clean-val-loss={vl:.4f} (Δ vs G {vl - vlG:+.4f})", flush=True)

    print("\nD2: model-replacement propagation measured (triggered-ASR up = backdoor in global; "
          "clean-val-loss Δ = the cost the valuation/detectors can see).")


if __name__ == "__main__":
    main()
