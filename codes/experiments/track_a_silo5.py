"""L8/T5 silo5 (a)-leg: the exact retrain (a) oracle on the EXISTING silo5 stage.

T5 §2 (paper/workplan/T5-retrain-a-suite.md): the canonical silo5 rundirs
(runs/phase2_matrix/rundirs/1B_silo5_{clean,noisy,frzero}) already hold every
method's phi + the (b) oracle on the realized trajectory.  The (a) oracle is a
RETRAIN game -- trajectory-independent -- so this runner adds ONLY the (a) leg:
it rebuilds the SAME split (identical `build(...seed...)` call + seed) and the SAME
val set as the canonical run, computes the exact 2^5 retrain-(a) Shapley (val-loss
game, R=10), and writes it to a `*_aonly` rundir.  The canonical rundirs are NEVER
touched (read-only); `runs/phase2_matrix/merge_silo5_a.py` joins the two on
(seed, client) to report spearman_a / pearson_a (the CNN merge_oracle_a pattern).

Split reproduction (must match phase2_matrix silo5 exactly):
  phase2_matrix.main does `seed_everything(seed)` then `build(5,200,20,40,seed=seed,
  noisy=corrupt, noisy_rate=1.0)` -- reproduced verbatim here so client c's DATA
  and the val set are byte-identical to the canonical run (the join is valid).
Threats (T5 §2): clean / noisy (nr1.0, client 0) / frzero (free-rider client 1,
zero update during retrain -- deployment semantics; also the (a)-side null-player
check: a zero-delta free-rider contributes 0 to the retrain game).

Memory (RTX3090, 24 GiB): this leg runs NO estimator HVP -- only the (a) retrains
(SFTTrainer, batch 16, maxlen 768) + a no_grad val-loss forward -- so it is lighter
than the gsm5 stage (which needs VAL_CHUNK=2, see phase2_matrix) and fits at the
default knobs; still add PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, and drop
BATCH to 8 if a 24 GiB card is shared.

Run from codes/ (one process per leg = threat x seed; 9 legs total):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    REGIME=silo5 THREAT=clean SEED=0 python -u experiments/track_a_silo5.py
  ... THREAT=noisy SEED=0 ... ;  ... THREAT=frzero SEED=0 ...
  # all 3 seeds in one process (one rundir per threat):
  ... THREAT=clean python -u experiments/track_a_silo5.py
  # local wiring smoke (tiny model, no persist -- code path only):
  SMOKE_MODEL=gpt2 THREAT=frzero SEED=0 ROUNDS=2 MAX_STEPS=2 TRAIN=8 VAL=6 \
    VAL_CHUNK=3 VAL_MAXLEN=64 BATCH=2 PERSIST=0 python -u experiments/track_a_silo5.py
"""
import os
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from flirds.backends.llm import make_llm_loss
from flirds.data.llm import build, build_val_batches
from flirds.oracle.exact_sv import exact_shapley
from flirds.oracle.exact_sv_llm import subset_valloss_utility
from flirds.repro import seed_everything
from flirds.run_logger import RunLogger
from flirds.hf_pin import rev
from flirds.timing import PhaseTimer

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
SCALE = ("7B" if "Llama-2-7b" in MODEL else
         MODEL.split("-")[-2] if "Llama-3.2-" in MODEL else MODEL.split("/")[-1])
TARGET = (["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
          if "llama" in MODEL.lower() else None)

THREAT = os.environ.get("THREAT", "clean")               # clean | noisy | frzero

# silo5 config == phase2_matrix SILO verbatim (the canonical rundir's rcfg).
RCFG = dict(n_clients=5, train=200, val=20, test=40, rounds=10, max_steps=10, lr=1e-3,
            maxlen=768, warmup=2, noisy={0}, freerider={1})
for k in ("n_clients", "train", "val", "test", "rounds", "max_steps", "maxlen"):
    if os.environ.get(k.upper()):
        RCFG[k] = int(os.environ[k.upper()])
if os.environ.get("LR"):
    RCFG["lr"] = float(os.environ["LR"])
NOISY_RATE = float(os.environ.get("NOISY_RATE", "1.0"))  # silo5 noisy dose = full swap

MODEL_CFG = {"1B": dict(batch=16, val_chunk=10, val_maxlen=384),
             "3B": dict(batch=8, val_chunk=5, val_maxlen=384),
             "7B": dict(batch=4, val_chunk=2, val_maxlen=320)}
MCFG = dict(MODEL_CFG.get(SCALE, MODEL_CFG["1B"]))
for k in ("batch", "val_chunk", "val_maxlen"):
    if os.environ.get(k.upper()):
        MCFG[k] = int(os.environ[k.upper()])
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", str(2 * LORA_R)))

_CODES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR_ROOT = os.environ.get("RUNDIR_ROOT",
                             os.path.join(os.path.dirname(_CODES), "runs", "phase2_matrix", "rundirs"))
# Rundir identity (protocol §1.7): what the *_aonly name encodes + the (a)-defining knobs.
IDENTITY = ("scale", "model", "regime", "threat", "seeds", "a_oracle", "rounds", "noisy_rate")


def _load(device):
    """fp32 + eager LoRA (== phase2_matrix._load: seed_everything(0) pins the LoRA-A init,
    so the retrain starts from the same adapter the canonical run trained from).
    SMOKE_MODEL=gpt2 downloads the small public model for a wiring smoke."""
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev(MODEL))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                             attn_implementation="eager",
                                             revision=rev(MODEL)).to(device)
    seed_everything(0)
    m = get_peft_model(m, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                                     target_modules=TARGET, lora_dropout=0.0,
                                     task_type="CAUSAL_LM"))
    init = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    return tok, m, init, list(init)


def _cell_name(seed_tok):
    return os.environ.get("RUN_NAME") or f"{SCALE}_silo5_{THREAT}_aonly{seed_tok}"


def _config(seeds):
    return {"scale": SCALE, "model": MODEL, "regime": "silo5", "threat": THREAT,
            "seeds": seeds, "a_oracle": True, "rounds": RCFG["rounds"],
            "noisy_rate": NOISY_RATE,
            "rcfg": {k: (sorted(v) if isinstance(v, set) else v) for k, v in RCFG.items()},
            "mcfg": MCFG, "lora": {"r": LORA_R, "alpha": LORA_ALPHA},
            "game": "retrain val-loss (eq. 5); phi.parquet suspicion (good->low = -phi_a)"}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = ([int(s) for s in os.environ["SEEDS"].split(",")] if os.environ.get("SEEDS")
             else [int(os.environ["SEED"])] if os.environ.get("SEED") else [0, 1, 2])
    seed_tok = f"_s{seeds[0]}" if (os.environ.get("SEED") and not os.environ.get("SEEDS")) else ""
    n = RCFG["n_clients"]
    if THREAT not in ("clean", "noisy", "frzero"):
        raise ValueError(f"unknown THREAT {THREAT!r} (silo5 (a)-leg scope: clean/noisy/frzero)")
    # threat -> (noisy client set folded into the DATA, free-rider set active in the retrain)
    noisy_ids = set(RCFG["noisy"]) if THREAT == "noisy" else set()
    fr_ids = frozenset(RCFG["freerider"]) if THREAT == "frzero" else frozenset()

    print(f"=== silo5 (a)-leg | {SCALE} {THREAT} | N={n} R={RCFG['rounds']} lr={RCFG['lr']} "
          f"| noisy={sorted(noisy_ids)} free-rider(zero)={sorted(fr_ids)} nr={NOISY_RATE:g} "
          f"| seeds={seeds} ===", flush=True)

    if os.environ.get("PERSIST", "1") == "1":            # fail fast on an identity clash (§1.7)
        RunLogger.precheck(RUNDIR_ROOT, _cell_name(seed_tok), _config(seeds), IDENTITY)

    tok, model, init, pkeys = _load(device)
    pt = PhaseTimer(device, n_gpus=int(os.environ.get("N_GPUS", "1")))
    phi_rows, metrics = [], {}
    for seed in seeds:
        seed_everything(seed)                            # match phase2_matrix.main's per-(seed) reseed
        clients, val, _ = build(n, RCFG["train"], RCFG["val"], RCFG["test"], seed=seed,
                                noisy=noisy_ids, noisy_rate=NOISY_RATE)   # SAME split as canonical
        val_chunks = build_val_batches(val, tok, MCFG["val_maxlen"], device, MCFG["val_chunk"])
        loss_fn, _pk, _lc = make_llm_loss(model, val_chunks, device)

        util = subset_valloss_utility(model, init, clients, tok, device,
                                      rounds=RCFG["rounds"], lr=RCFG["lr"],
                                      max_steps=RCFG["max_steps"], batch_size=MCFG["batch"],
                                      max_length=RCFG["maxlen"], seed=seed,
                                      val_loss_fn=loss_fn, pkeys=pkeys,
                                      free_riders=fr_ids, free_rider_mode="zero")
        t0 = time.perf_counter()
        with pt.phase("oracle-a"):
            phi_a = exact_shapley(n, util)               # good->HIGH (utility = -val_loss)
        dt = time.perf_counter() - t0
        # store suspicion-oriented (good->LOW = -phi_a) so it aligns with the canonical
        # phi.parquet's method/(b) orientation -> the merge is a plain per-client join.
        phi_rows += [{"seed": seed, "method": "(a)oracle", "client": int(c),
                      "phi": float(-phi_a[c]), "phi_a": float(phi_a[c])} for c in range(n)]
        metrics[f"seed{seed}"] = {"phi_a": [float(x) for x in phi_a],
                                  "phi_susp": [float(-x) for x in phi_a],
                                  "oracle_a_s": dt}
        tag = lambda i: "*" if i in noisy_ids else ("F" if i in fr_ids else " ")
        print(f"  seed{seed} ({dt:.0f}s)  (a)phi[good->high]: "
              + "  ".join(f"c{i}{tag(i)}={phi_a[i]:+.5f}" for i in range(n)), flush=True)
        del loss_fn, val_chunks
        if device == "cuda":
            torch.cuda.empty_cache()

    if os.environ.get("PERSIST", "1") == "1":
        try:
            rl = RunLogger(RUNDIR_ROOT, _cell_name(seed_tok), _config(seeds),
                           repo_root=_CODES, identity=IDENTITY)
            rl.save_phi(phi_rows)
            rl.save_metrics(metrics)
            rl.save_timing(pt.to_timing())
            print(f"\n[persist] {rl.dir}  ({len(phi_rows)} phi rows)", flush=True)
        except Exception as e:
            print(f"\n[persist] WARNING: save failed ({e!r}); stdout has results", flush=True)
    print("\nSILO5 (a)-LEG DONE  (join with the canonical 1B_silo5_<threat> rundir via "
          "runs/phase2_matrix/merge_silo5_a.py -> spearman_a / pearson_a).", flush=True)


if __name__ == "__main__":
    main()
