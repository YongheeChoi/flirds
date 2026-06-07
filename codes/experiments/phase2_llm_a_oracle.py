"""Phase 2 task 6: (a) exact retrain SV oracle at LLM scale (N=5, 1B) + the dual-oracle figure.

The (a) oracle (exact retrain Shapley, bf16, downstream-ROUGE utility) is the missing half
of the dual oracle at LLM scale -- (b) in-run + the Flirds estimator are already validated.
(a) trains a fresh LoRA on each coalition S and scores the DEPLOYED model (macro-avg
per-domain ROUGE-L), a DIFFERENT-utility sanity check vs (b)'s frozen val-loss (protocol 4.1).

Headline output is the **per-coalition TIMING** (retrain vs eval, broken down by |S|) so the
N=10 (1024-retrain) and 3B costs can be extrapolated before committing to them (Yonghee:
"record the time well at N=5@1B").  N=5 = 2^5 = 32 retrains.

(a) is bf16 (matches deployment) on its own model; (b)/estimator stay fp32+eager on the
frozen trajectory.  Orientation: (a) ROUGE is good->HIGH; (b)/estimator val-loss is
good->LOW -> the comparison aligns them explicitly (see below).

Run from codes/:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u experiments/phase2_llm_a_oracle.py smoke
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u experiments/phase2_llm_a_oracle.py
"""
import os
import sys
import time
from itertools import combinations

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from flirds.backends.llm import make_llm_loss
from flirds.core.flirds_estimator import flirds_values
from flirds.data.llm import build, build_val_batches
from flirds.eval.metrics import detection_auroc
from flirds.fl.llm_server import run_llm_fedavg_logs
from flirds.oracle.exact_sv import exact_shapley
from flirds.oracle.exact_sv_llm import llm_subset_utility
from flirds.oracle.in_run_sv import in_run_shapley
from flirds.repro import seed_everything

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
SCALE = MODEL.split("-")[-2] if "Llama-3.2-" in MODEL else MODEL.split("/")[-1]   # "1B"/"3B" label
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DOMAINS = ["medical", "legal", "finance", "math", "general"]
N, NOISY, FREE_RIDERS, FREE_RIDER_MODE = 5, {0}, {1}, "zero"

FULL = dict(train=200, val=20, test=40, rounds=10, max_steps=10, lr=1e-3, batch=16,
            maxlen=768, val_maxlen=384, val_chunk=10, gen_new=128, gen_batch=16, seed=0)
SMOKE = dict(train=8, val=10, test=4, rounds=2, max_steps=2, lr=1e-3, batch=2,
             maxlen=768, val_maxlen=256, val_chunk=10, gen_new=32, gen_batch=8, seed=0)


def _load(dtype, attn, device):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, attn_implementation=attn).to(device)
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=TARGET,
                                     lora_dropout=0.0, task_type="CAUSAL_LM"))
    init = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    return tok, m, init


def _fmt_phi(ph):
    tag = lambda i: "*" if i in NOISY else ("F" if i in FREE_RIDERS else " ")
    return "  ".join(f"{DOMAINS[i][:4]}{tag(i)}={ph[i]:+.4f}" for i in range(N))


def _timing_report(timing, total_a):
    """Per-coalition retrain/eval breakdown + a naive N=10 extrapolation."""
    arr = np.array(timing)                                   # (n_coal, 3): |S|, retrain_t, eval_t
    sizes, rt, et = arr[:, 0], arr[:, 1], arr[:, 2]
    print(f"\n=== (a) retrain-oracle TIMING (N={N}, {len(timing)} coalitions) ===", flush=True)
    print(f"  total (a)={total_a:.1f}s  retrain={rt.sum():.1f}s  eval(gen+score)={et.sum():.1f}s  "
          f"(retrain {100*rt.sum()/total_a:.0f}% / eval {100*et.sum()/total_a:.0f}%)")
    print("  by |S|:  " + "  ".join(
        f"|S|={s}:n{int((sizes==s).sum())} rt{rt[sizes==s].mean():.1f}/ev{et[sizes==s].mean():.1f}s"
        for s in range(N + 1) if (sizes == s).any()))
    # naive extrapolation: 2^10/2^5 = 32x the coalition count.  Retrain scales with |S|
    # (more clients/round) -> at N=10 mean |S|=5 vs N=5 mean |S|=2.5, so retrain ~2x/coalition;
    # eval is |S|-independent (same test set).  Report both the flat 32x and an |S|-adjusted hint.
    per_coal = total_a / len(timing)
    print(f"  per-coalition mean={per_coal:.1f}s -> naive N=10 (1024 coal) = {1024*per_coal/3600:.1f}h "
          f"(LOWER bound; retrain/coalition ~2x at N=10 since mean|S| 2.5->5 -> add ~retrain share).")


def main(cfg):
    device = "cuda"
    seed = cfg["seed"]
    seed_everything(seed)
    clients, val, test = build(N, cfg["train"], cfg["val"], cfg["test"], seed=seed, noisy=NOISY)
    print(f"task6 (a)-oracle | {SCALE} N={N} | train={cfg['train']} val={cfg['val']} test={cfg['test']}/dom "
          f"R={cfg['rounds']} lr={cfg['lr']} | noisy={sorted(NOISY)} "
          f"free_rider={sorted(FREE_RIDERS)}({FREE_RIDER_MODE})", flush=True)
    print(f"  build: clients={[len(c) for c in clients]} val={len(val)} test={len(test)}", flush=True)

    # --- (b) in-run + estimator on the fp32-eager frozen trajectory ---
    tok, model_b, init_b = _load(torch.float32, "eager", device)
    val_chunks = build_val_batches(val, tok, cfg["val_maxlen"], device, cfg["val_chunk"])
    model_b.load_state_dict(init_b, strict=False)
    logs = run_llm_fedavg_logs(model_b, tok, clients, cfg["rounds"], cfg["lr"], cfg["max_steps"],
                               batch_size=cfg["batch"], max_length=cfg["maxlen"], seed=seed,
                               free_riders=FREE_RIDERS, free_rider_mode=FREE_RIDER_MODE)
    loss_fn, pkeys, lc = make_llm_loss(model_b, val_chunks, device)
    t = time.perf_counter(); phi_b, _ = in_run_shapley(logs, N, loss_fn, pkeys, device)
    tb = time.perf_counter() - t
    t = time.perf_counter()
    phi_e, _ = flirds_values(logs, loss_fn, pkeys, device, second_order=True, loss_chunks=lc)
    te = time.perf_counter() - t
    print(f"\n  (b) in-run oracle ({tb:.1f}s):  {_fmt_phi(phi_b)}", flush=True)
    print(f"  estimator        ({te:.1f}s):  {_fmt_phi(phi_e)}", flush=True)
    del model_b, loss_fn
    torch.cuda.empty_cache()

    # --- (a) exact retrain SV oracle with per-coalition timing; BOTH metrics from ONE retrain ---
    # ROUGE (deployed downstream, good->high) + (-val_loss) (good->high, same metric as (b)).
    # The (a)-val-loss isolates the metric (ROUGE vs loss) from the method (retrain vs frozen).
    # A_DTYPE=fp32 for a clean (a)-val-loss (coalition diffs ~1e-2 < bf16 prec); default bf16.
    a_dtype = torch.float32 if os.environ.get("A_DTYPE") == "fp32" else torch.bfloat16
    tok_a, model_a, init_a = _load(a_dtype, "eager", device)
    val_chunks_a = build_val_batches(val, tok_a, cfg["val_maxlen"], device, cfg["val_chunk"])
    loss_fn_a = make_llm_loss(model_a, val_chunks_a, device)[0]
    timing = []
    util = llm_subset_utility(model_a, init_a, clients, tok_a, test, device,
                              rounds=cfg["rounds"], lr=cfg["lr"], max_steps=cfg["max_steps"],
                              batch_size=cfg["batch"], max_length=cfg["maxlen"], seed=seed,
                              free_riders=FREE_RIDERS, free_rider_mode=FREE_RIDER_MODE,
                              gen_batch=cfg["gen_batch"], max_new_tokens=cfg["gen_new"],
                              val_loss_fn=loss_fn_a, timing=timing)
    t = time.perf_counter()
    U = {S: util(S) for r in range(N + 1) for S in combinations(range(N), r)}   # one retrain / coalition
    phi_ar = exact_shapley(N, lambda S: U[tuple(sorted(S))][0])                 # ROUGE Shapley
    phi_al = exact_shapley(N, lambda S: U[tuple(sorted(S))][1])                 # -val_loss Shapley
    ta = time.perf_counter() - t
    print(f"\n  (a) retrain [{a_dtype}] ({ta:.1f}s)", flush=True)
    print(f"    ROUGE    (good->HIGH): {_fmt_phi(phi_ar)}")
    print(f"    -valloss (good->HIGH): {_fmt_phi(phi_al)}")

    # --- 4-way comparison: value = good->HIGH for all (negate the (b)/estimator val-loss phi);
    #     AUROC corruption score = high-for-corrupt = -value ---
    noisy_y = [1 if i in NOISY else 0 for i in range(N)]
    fr_y = [1 if i in FREE_RIDERS else 0 for i in range(N)]
    vals = {"(a)ROUGE": np.asarray(phi_ar), "(a)valloss": np.asarray(phi_al),
            "(b)in-run": -np.asarray(phi_b), "estimator": -np.asarray(phi_e)}
    print(f"\n=== dual-oracle 4-way at N={N} (value=good->high) ===")
    print("  Spearman vs (b)in-run: " + "  ".join(
        f"{k}={float(spearmanr(v, vals['(b)in-run']).correlation):+.3f}"
        for k, v in vals.items() if k != "(b)in-run"))
    print(f"  Spearman (a)ROUGE vs (a)valloss = "
          f"{float(spearmanr(vals['(a)ROUGE'], vals['(a)valloss']).correlation):+.3f}  "
          f"(if (a)valloss tracks (b) but (a)ROUGE does not -> the METRIC drives the gap)")
    for k, v in vals.items():
        print(f"  AUROC {k:10s}: noisy={detection_auroc(-v, noisy_y):.3f} "
              f"free-rider={detection_auroc(-v, fr_y):.3f}")
    _timing_report(timing, ta)
    print("\n(a) retrain oracle is the different-utility sanity figure; the timing extrapolates "
          "the N=10 (1024-retrain) / 3B cost before committing.", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "full"
    cfg = dict(SMOKE if which == "smoke" else FULL)
    # per-scale memory knobs via env (3B/7B: the fp32 estimator HVP needs a smaller
    # val_chunk -- it sums chunks exactly, so this changes ONLY memory, not the values).
    for k in ("val_chunk", "val_maxlen", "batch", "test", "gen_batch", "rounds", "max_steps"):
        if os.environ.get(k.upper()):
            cfg[k] = int(os.environ[k.upper()])
    if os.environ.get("LR"):
        cfg["lr"] = float(os.environ["LR"])
    main(cfg)
