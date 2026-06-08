"""Phase 2 §3.9 D2b — detector + Flirds response to the WORKING backdoor (D1+D2).

D1/D2 gave a working clean-preserving backdoor (frac=0.5 local install -> full-replacement
global triggered-ASR 0.97, clean-val-loss only +0.027).  D2b measures what the detectors and
Flirds SEE on that single-shot attack trajectory:
  - benign FL R rounds (5 clean clients) -> logs + global G;
  - attacker trains backdoor X from G (frac=0.5, EPOCHS); a single-shot attack ROUND is
    appended to the logs: {attacker: gamma*(X-G), the 4 benign: their fresh G-round deltas};
  - FLDetector (model-free magnitude/consistency) + FLTrust (val-cosine) + Flirds-1st/2nd on
    the full logs -> per-client score + attacker AUROC + rank.

TEST (do NOT pre-position): magnitude detectors should flag the ~40x-norm scaled attacker
(AUROC~1), while Flirds (clean-val-loss based) sees only the +0.027 clean cost -> may score
the attacker weakly.  The matrix confirms across seeds/configs.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase2_backdoor_d2b_smoke.py
env: BD_TARGET, POISON_FRAC, EPOCHS, LR, TRAIN, ROUNDS, SEED.
"""
import os

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from flirds.backends.llm import make_llm_loss
from flirds.baselines.fldetector import fldetector_from_logs
from flirds.baselines.fltrust import fltrust_from_logs
from flirds.core.flirds_estimator import flirds_values
from flirds.data.corruptors import BACKDOOR_TRIGGER
from flirds.data.llm import build, build_val_batches
from flirds.eval.generate import backdoor_asr
from flirds.fl.llm_server import run_llm_fedavg_logs
from flirds.repro import seed_everything

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TARGET = os.environ.get("BD_TARGET", "delicious")
FRAC = float(os.environ.get("POISON_FRAC", "0.5"))
EPOCHS = float(os.environ.get("EPOCHS", "3"))
LR = float(os.environ.get("LR", "2e-3"))
TRAIN = int(os.environ.get("TRAIN", "1000"))
ROUNDS = int(os.environ.get("ROUNDS", "10"))
BENIGN_STEPS = int(os.environ.get("BENIGN_STEPS", "5"))
N = 5
_OUT = "/tmp/flirds_backdoor_d2b"


def _lora_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _final_global(init, logs, pkeys, device):
    G = {n: init[n].detach().clone().float().to(device) for n in pkeys}
    for _, dm in logs:
        tot = sum(nc for _, nc in dm.values())
        for k in dm:
            d, nc = dm[k]
            for n in pkeys:
                G[n] += (nc / tot) * d[n].float().to(device)
    return G


def _train_delta(model, tok, dataset, G, pkeys, seed, *, epochs=None, steps=None):
    """Load G, train `dataset` locally (SGD mom=0), return the CPU delta (logged-delta convention)."""
    model.load_state_dict({k: G[k] for k in pkeys}, strict=False)
    kw = dict(num_train_epochs=epochs) if epochs is not None else dict(max_steps=steps)
    cfg = SFTConfig(output_dir=_OUT, per_device_train_batch_size=8, learning_rate=LR, max_length=768,
                    lr_scheduler_type="constant", warmup_steps=0, completion_only_loss=True,
                    bf16=False, fp16=False, report_to="none", logging_strategy="no",
                    save_strategy="no", seed=seed, **kw)
    SFTTrainer(model=model, args=cfg, train_dataset=dataset, processing_class=tok,
               optimizer_cls_and_kwargs=(torch.optim.SGD, {"lr": LR, "momentum": 0.0})).train()
    after = _lora_state(model)
    return {k: (after[k] - G[k]).detach().cpu() for k in pkeys}


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
    init_lora = _lora_state(model)
    pkeys = list(init_lora)

    clean_clients, val, test = build(5, TRAIN, per_domain_val=20, per_domain_test=40, seed=seed)
    bd_clients, _, _ = build(5, TRAIN, per_domain_val=20, per_domain_test=40, seed=seed,
                             backdoor=frozenset({0}),
                             backdoor_kwargs=dict(target=TARGET, poison_frac=FRAC))
    med_test = [r["prompt"] for r in test if r["domain"] == "medical"]
    val_chunks = build_val_batches(val, tok, 384, device, 10)
    print(f"[seed {seed}] N={N} frac={FRAC} epochs={EPOCHS} rounds={ROUNDS} γ=full(n/η=N) "
          f"target={TARGET!r} trigger={BACKDOOR_TRIGGER!r}", flush=True)

    # 1. benign FL -> logs + G
    model.load_state_dict(init_lora, strict=False)
    logs = run_llm_fedavg_logs(model, tok, clean_clients, ROUNDS, LR, BENIGN_STEPS,
                               batch_size=8, max_length=768, seed=seed)
    G = _final_global(init_lora, logs, pkeys, device)

    # 2. attacker backdoor X from G; 3. build single-shot attack round
    delta0 = _train_delta(model, tok, bd_clients[0], G, pkeys, seed, epochs=EPOCHS)   # raw attacker delta
    gamma = float(N)
    attack_dm = {0: ({k: gamma * delta0[k] for k in pkeys}, len(bd_clients[0]))}      # scaled (model-replacement)
    for c in (1, 2, 3, 4):
        dc = _train_delta(model, tok, clean_clients[c], G, pkeys, seed, steps=BENIGN_STEPS)
        attack_dm[c] = (dc, len(clean_clients[c]))
    logs.append(({k: G[k].detach().cpu() for k in pkeys}, attack_dm))                 # w_r = G

    # deployed global after the attack round (sanity: ASR present)
    Gbd = _final_global(init_lora, logs, pkeys, device)
    model.load_state_dict({k: Gbd[k] for k in pkeys}, strict=False)
    atrig, _ = backdoor_asr(model, tok, med_test, BACKDOOR_TRIGGER, TARGET, device)

    # 4. detectors + Flirds on the full trajectory
    loss_fn, pk, lc = make_llm_loss(model, val_chunks, device)
    fld = fldetector_from_logs(logs, N, device="cpu")
    flt = fltrust_from_logs(logs, N, loss_fn, pk, device, loss_chunks=lc)
    phi1, _ = flirds_values(logs, loss_fn, pk, device, second_order=False, loss_chunks=lc)
    phi2, _ = flirds_values(logs, loss_fn, pk, device, second_order=True, loss_chunks=lc)

    labels = [1, 0, 0, 0, 0]
    print(f"  deployed global triggered-ASR={atrig:.2f}   (attacker=client0)", flush=True)
    print("  per-client scores (client0 = attacker):")
    for name, s in [("FLDetector", fld), ("FLTrust", flt), ("Flirds-1st", phi1), ("Flirds-2nd", phi2)]:
        s = np.asarray(s, dtype=float)
        rank = int((s > s[0]).sum())
        auroc = roc_auc_score(labels, s) if name not in ("Flirds-1st", "Flirds-2nd") else roc_auc_score(labels, -s)
        tag = " (AUROC on -φ: corrupt = LOW value)" if name.startswith("Flirds") else ""
        print(f"    {name:11s}: {np.array2string(s, precision=4, floatmode='fixed')}  "
              f"attacker={s[0]:+.4f} rank={rank}/{N} AUROC={auroc:.3f}{tag}", flush=True)
    print("\nD2b: magnitude-based detectors vs the clean-val-loss-based Flirds on the working backdoor "
          "(read the AUROCs; the matrix confirms the framing).")


if __name__ == "__main__":
    main()
