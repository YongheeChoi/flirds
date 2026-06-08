"""Phase 2 task 7e poisoning smoke: real-1B backdoor (ASR) + detection AUROC.

End-to-end poisoning pipeline on a cross-silo N=5 trajectory (the attacker trains
EVERY round -> a strong, measurable backdoor; cross-device's sparse attacker
participation is validated separately in the FLDetector cross-device smoke):
  - client 0 = backdoor attacker: data layer injects trigger->target (Xu 2023), and
    the server scales its update by gamma=N (Bagdasaryan plain-scaled model-replacement);
  - ASR = fraction of TRIGGERED clean test prompts whose generation emits the target
    marker (clean-model baseline reported for contrast);
  - detection: FLDetector (model-free, magnitude-consistency -> matched to the SCALED
    attack) + FLTrust + Flirds-1st (val-gradient cosine / inner product).
Reports each detector's score on the attacker + AUROC.  The backdoor-vs-Flirds
question (does a clean-preserving backdoor evade the clean-val-loss signal?) is
REPORTED, not pre-positioned -- read the numbers.

Run from codes/:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase2_poisoning_smoke.py
"""
import os

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

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
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
N, ATTACKER = 5, frozenset({0})
# smoke installs a STRONG, easy-to-learn backdoor (poison_frac=1.0, short distinctive
# target) so ASR is measurable end-to-end; the clean-preserving frac<1 case (does it
# evade Flirds?) is the matrix's research question (data.corruptors default = 0.5).
BD_TARGET = MARKER = os.environ.get("BD_TARGET", "HACKED")
POISON_FRAC = float(os.environ.get("POISON_FRAC", "1.0"))
GAMMA = float(os.environ.get("GAMMA", "5.0"))       # Bagdasaryan scale (= N -> full replacement)
CFG = dict(train=100, val=20, test=20, rounds=int(os.environ.get("R_EVAL", "10")),
           max_steps=int(os.environ.get("MAX_STEPS", "5")), lr=float(os.environ.get("LR", "1e-3")),
           batch=8, maxlen=768, val_maxlen=384, val_chunk=10)


def _final_global(init_lora, logs, pkeys, device):
    """Reconstruct the deployed LoRA state w_R = init + Σ_r (round FedAvg aggregate)."""
    final = {n: init_lora[n].detach().clone().float().to(device) for n in pkeys}
    for _, dm in logs:
        tot = sum(nc for _, nc in dm.values())
        for k in dm:
            d, nc = dm[k]
            for n in pkeys:
                final[n] += (nc / tot) * d[n].float().to(device)
    return final


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
    pkeys = list(init_lora)

    clients, val, test = build(N, CFG["train"], CFG["val"], per_domain_test=CFG["test"],
                               seed=seed, backdoor=ATTACKER,
                               backdoor_kwargs=dict(target=BD_TARGET, poison_frac=POISON_FRAC))
    val_chunks = build_val_batches(val, tok, CFG["val_maxlen"], device, CFG["val_chunk"])
    test_prompts = [r["prompt"] for r in test]
    print(f"[seed {seed}] N={N} attacker={sorted(ATTACKER)} gamma={GAMMA} R={CFG['rounds']} "
          f"trigger={BACKDOOR_TRIGGER!r} marker={MARKER!r} test={len(test_prompts)}", flush=True)

    # clean-model ASR baseline (no training)
    model.load_state_dict(init_lora, strict=False)
    asr0, _ = backdoor_asr(model, tok, test_prompts, BACKDOOR_TRIGGER, MARKER, device)

    model.load_state_dict(init_lora, strict=False)
    logs = run_llm_fedavg_logs(model, tok, clients, CFG["rounds"], CFG["lr"], CFG["max_steps"],
                               batch_size=CFG["batch"], max_length=CFG["maxlen"], seed=seed,
                               scaled_attackers=ATTACKER, attack_scale=GAMMA)

    # deployed-model ASR (backdoor installed?)
    model.load_state_dict(_final_global(init_lora, logs, pkeys, device), strict=False)
    asr1, gens = backdoor_asr(model, tok, test_prompts, BACKDOOR_TRIGGER, MARKER, device)
    print(f"  ASR clean-model={asr0:.2f} -> deployed={asr1:.2f}  (sample gen: {gens[0][:60]!r})")

    # detection (build loss_fn AFTER generation: make_llm_loss disables use_cache)
    loss_fn, pk, lc = make_llm_loss(model, val_chunks, device)
    fld = fldetector_from_logs(logs, N, device="cpu")
    flt = fltrust_from_logs(logs, N, loss_fn, pk, device, loss_chunks=lc)
    phi1, _ = flirds_values(logs, loss_fn, pk, device, second_order=False, loss_chunks=lc)
    labels = [1 if c in ATTACKER else 0 for c in range(N)]
    a = sorted(ATTACKER)[0]
    print(f"  attacker(c{a}) score: FLDetector={fld[a]:+.4f} FLTrust={flt[a]:+.4f} Flirds-1st={phi1[a]:+.4f}")
    print("  per-client:")
    for name, s in [("FLDetector", fld), ("FLTrust", flt), ("Flirds-1st", phi1)]:
        rank = int((s > s[a]).sum())
        print(f"    {name:11s}: {np.array2string(s, precision=4, floatmode='fixed')}  "
              f"attacker_rank={rank}/{N} AUROC={roc_auc_score(labels, s):.3f}")
    if asr1 <= asr0:
        print("  [note] ASR unchanged from the clean baseline -- at this LoRA-scale training the "
              "backdoor does not yet win greedy decoding; stronger training (steps/rounds/gamma/lr) needed.")
    assert np.isfinite([fld[a], flt[a], phi1[a]]).all(), "detector scores not finite"
    print("\nPOISONING SMOKE OK  (machinery validated end-to-end; ASR config-dependent; detector AUROCs computed)")


if __name__ == "__main__":
    main()
