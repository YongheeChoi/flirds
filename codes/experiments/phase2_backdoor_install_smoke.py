"""Phase 2 §3.9 D1 — backdoor INSTALL isolation check (no FL, no scaling).

Does the Xu trigger->target backdoor install in a SINGLE attacker's LOCAL model at our
generative 1B-LoRA scale?  This is the precondition Bagdasaryan model-replacement needs --
it replaces a local model that must ALREADY hold the backdoor (see
sources/how-to-backdoor-fl-bagdasaryan).  Faithful Xu reproduction minus FL: one client's
data is trigger->target-poisoned and trained to convergence (SGD momentum=0 = our locked
convention, but many epochs -- the attacker doesn't follow the 5-step valuation budget),
then we measure greedy-ASR + soft-ASR (target NLL + first-token argmax-hit) on triggered
clean prompts.

Verdict: ASR rises  => installable at our scale => proceed to D2 (FL replacement).
         ASR stays 0 => generative+1B+LoRA+SGD is the fundamental limit => re-discuss.

Run from codes/:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase2_backdoor_install_smoke.py
Levers (env): BD_TARGET (single-token), POISON_FRAC, EPOCHS, LR, TRAIN, TEST, SEED.
"""
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from flirds.data.corruptors import BACKDOOR_TRIGGER
from flirds.data.llm import build
from flirds.eval.generate import backdoor_asr, backdoor_soft_asr, generate_completions
from flirds.repro import seed_everything

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TARGET = os.environ.get("BD_TARGET", "delicious")        # single-token target (Bagdasaryan-style word)
POISON_FRAC = float(os.environ.get("POISON_FRAC", "1.0"))  # 1.0 = max install; <1 = clean-preserving
EPOCHS = float(os.environ.get("EPOCHS", "3"))            # Xu instruction-tuning: 3 epochs
LR = float(os.environ.get("LR", "2e-3"))
TRAIN = int(os.environ.get("TRAIN", "1000"))
TEST = int(os.environ.get("TEST", "40"))
_OUT = "/tmp/flirds_backdoor_install"


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

    n_tgt = len(tok(" " + TARGET, add_special_tokens=False)["input_ids"])
    clients, _val, test = build(5, TRAIN, per_domain_val=20, per_domain_test=TEST, seed=seed,
                                backdoor=frozenset({0}),
                                backdoor_kwargs=dict(target=TARGET, poison_frac=POISON_FRAC))
    attacker = clients[0]                                  # medical (ORDER[0]), trigger->target poisoned
    med_test = [r["prompt"] for r in test if r["domain"] == "medical"]
    print(f"[seed {seed}] target={TARGET!r} ({n_tgt} tok) frac={POISON_FRAC} epochs={EPOCHS} lr={LR} "
          f"attacker_n={len(attacker)} test={len(med_test)} trigger={BACKDOOR_TRIGGER!r}", flush=True)
    if n_tgt != 1:
        print(f"  [warn] target is {n_tgt} tokens (not single-token) -- greedy bar higher", flush=True)

    # clean baseline (before any training): triggered ASR should be ~0
    a0, _ = backdoor_asr(model, tok, med_test, BACKDOOR_TRIGGER, TARGET, device)
    nll0, h0 = backdoor_soft_asr(model, tok, med_test, BACKDOOR_TRIGGER, TARGET, device)

    # attacker local training (SGD momentum=0; faithful Xu: many epochs over poisoned data)
    cfg = SFTConfig(output_dir=_OUT, per_device_train_batch_size=8, num_train_epochs=EPOCHS,
                    learning_rate=LR, max_length=768, lr_scheduler_type="constant", warmup_steps=0,
                    completion_only_loss=True, bf16=False, fp16=False, report_to="none",
                    logging_strategy="no", save_strategy="no", seed=seed)
    SFTTrainer(model=model, args=cfg, train_dataset=attacker, processing_class=tok,
               optimizer_cls_and_kwargs=(torch.optim.SGD, {"lr": LR, "momentum": 0.0})).train()

    # after training
    a1, gens = backdoor_asr(model, tok, med_test, BACKDOOR_TRIGGER, TARGET, device)
    nll1, h1 = backdoor_soft_asr(model, tok, med_test, BACKDOOR_TRIGGER, TARGET, device)
    clean = generate_completions(model, tok, med_test, device, max_new_tokens=32)
    clean_asr = sum(TARGET in g for g in clean) / len(clean)   # trigger-free target rate: LOW = clean preserved

    print(f"  triggered greedy-ASR: {a0:.2f} -> {a1:.2f}   (HIGH = backdoor fires)", flush=True)
    print(f"  clean-ASR (no trigger): {clean_asr:.2f}            (LOW = clean task preserved)")
    print(f"  soft target-NLL: {nll0:.3f} -> {nll1:.3f}   first-token-hit: {h0:.2f} -> {h1:.2f}")
    print(f"  triggered gen[0]: {gens[0][:70]!r}")
    print(f"  clean     gen[0]: {clean[0][:70]!r}")
    fires = (a1 > a0 + 0.05) or (h1 >= 0.5 and h1 > h0)
    preserved = clean_asr <= 0.2
    verdict = ("CLEAN-PRESERVING BACKDOOR (fires on trigger, clean intact) -> ideal for matrix" if fires and preserved
               else "INSTALLED but CLEAN DESTROYED (fires unconditionally) -> lower poison_frac" if fires
               else "NOT installed -> tune levers / re-discuss")
    print(f"\nD1 verdict (frac={POISON_FRAC}): {verdict}")


if __name__ == "__main__":
    main()
