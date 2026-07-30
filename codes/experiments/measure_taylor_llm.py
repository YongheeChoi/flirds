"""Taylor-remainder measurement on the LLM track, matching main Table 1's design.

Table 1 (tab:retrain-fidelity) is a CNN table, so there is no literal LLM row to
copy.  Its defining design is: FULL participation, N small enough that the
2^N coalition enumeration is exact, R=10, three seeds, a non-IID partition, and a
comparison against the RETRAINING-based Shapley.  On the LLM track exactly one
setting has all of those properties -- the five-domain non-IID setting (N=5, full
participation, R=10), which Appendix C.3 uses as "The LLM Retraining Leg" and
which enumerates the classical value through 2^5 retrains.  That is the default
regime here, and it is also the setting C.5 already measures, so the existing
three-seed result is reproduced rather than replaced.

  REGIME=silo5     five-domain non-IID, N=5, full, R=10   <- Table 1's counterpart
  REGIME=gsm50k5   GSM8K main setting, N=50, 5/50, R=200  <- a MAIN-TEXT setting

The gsm50k5 option exists because C.5's current setting is never named in the main
paper, while the GSM8K main setting is the headline LLM setting and its (b) oracle
already enumerates 2^K=32 coalitions per round -- so measuring there closes that
gap.  It costs ~20x the forwards of silo5 (200 rounds instead of 10) unless
TAYLOR_N_MEASURE subsamples rounds.

Conditions follow main Section 5.1's LLM assignment (answer-swap and zero-update
are the LLM threats; gradient noise and label-flip are CNN-only), and the
corrupted client indices come from the same regime config the paper runs use.

Cost per (condition, seed) at silo5: 2^5 = 32 coalition forwards + 5 HVPs per
round, x 10 rounds, on a 100-example validation set.  Measured previously at
~605 s trajectory + ~1,536 s measurement per seed on a B200.

Run from codes/:
  PYTHONPATH=. TAYLOR_THREAT=clean TAYLOR_SEED=0 python -u experiments/measure_taylor_llm.py
  # wiring smoke on a tiny cached model (CPU/small GPU, seconds, no persist):
  PYTHONPATH=. TAYLOR_SMOKE=1 TAYLOR_PERSIST=0 python -u experiments/measure_taylor_llm.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from flirds.backends.llm import make_llm_loss
from flirds.data.llm import build, build_gsm8k_iid, build_val_batches
from flirds.fl.llm_server import run_llm_fedavg_logs
from flirds.hf_pin import rev
from flirds.repro import seed_everything
from flirds.run_logger import RunLogger
from taylor_core import measure_round, pool

LLAMA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Regime configs copied from phase2_matrix's SILO / GSM so the stage is identical to
# the paper runs.  `val` is PER DOMAIN for silo5 (x5 domains = 100 server examples)
# and total for gsm50k5 (200 of the 1,319 official test questions).
SILO5 = dict(n_clients=5, train=200, val=20, test=40, rounds=10, max_steps=10, lr=1e-3,
             maxlen=768, k_frac=1.0, noisy={0}, freerider={1}, noisy_rate=1.0)
GSM50K5 = dict(n_clients=50, val=200, test=0, rounds=200, max_steps=10, lr=1e-3,
               maxlen=512, k_frac=0.1, noisy=set(range(20)), freerider=set(range(20)),
               noisy_rate=0.7)
MODEL_MEM = {"1B": dict(batch=16, val_chunk=10, val_maxlen=384)}

REGIME = os.environ.get("REGIME", "silo5")
SMOKE = os.environ.get("TAYLOR_SMOKE", "0") == "1"
SYNTH = os.environ.get("SYNTH_DATA", "0") == "1"    # offline wiring smoke (track_g's seam)
THREAT = os.environ.get("TAYLOR_THREAT", "clean")
SEED = int(os.environ.get("TAYLOR_SEED", "0"))
PERSIST = os.environ.get("TAYLOR_PERSIST", "1") == "1"
RENORM = os.environ.get("TAYLOR_RENORM", "0") == "1"
N_MEASURE = os.environ.get("TAYLOR_N_MEASURE", "all")
MODEL = os.environ.get("TAYLOR_MODEL", "gpt2" if SMOKE else "meta-llama/Llama-3.2-1B-Instruct")
LORA_R = int(os.environ.get("TAYLOR_LORA_R", "16"))

_THREATS = {"clean": "clean",
            "answer-swap": "answer_swap", "answer_swap": "answer_swap", "noisy": "answer_swap",
            "zero-update": "freerider_zero", "freerider_zero": "freerider_zero",
            "free_rider": "freerider_zero"}
if THREAT not in _THREATS:
    raise SystemExit(f"TAYLOR_THREAT must be one of {sorted(_THREATS)}, got {THREAT!r}")
THREAT_K = _THREATS[THREAT]
if REGIME not in ("silo5", "gsm50k5"):
    raise SystemExit(f"REGIME must be silo5|gsm50k5, got {REGIME!r}")

RCFG = dict(SILO5 if REGIME == "silo5" else GSM50K5)
MCFG = dict(MODEL_MEM["1B"])
if SMOKE:                                   # tiny wiring run: gpt2, 2 rounds, small val
    RCFG.update(rounds=2, max_steps=2, train=8, val=2, maxlen=256)
    MCFG.update(batch=2, val_chunk=4, val_maxlen=64)
for k in ("rounds", "max_steps", "train", "val", "n_clients"):
    if os.environ.get(k.upper()):
        RCFG[k] = int(os.environ[k.upper()])

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("TAYLOR_RUN_ROOT",
                          os.path.join(_REPO, "runs", "taylor_remainder", "rundirs"))


def load_model(name, device, lora_r):
    """fp32 + eager LoRA model -- the same convention as phase2_matrix._load.

    eager attention is required: the estimator's HVP goes through forward-mode AD,
    which the fused SDPA kernels do not support.  fp32 throughout, matching the
    (b) oracle / estimator path.

    name="tiny-gpt2" builds a random-init 2-layer GPT-2 from config -- OFFLINE
    wiring smoke only (values are noise, no weight download; track_g's convention,
    pairs with SYNTH_DATA=1).
    """
    if name == "tiny-gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        m = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=64,
                                       vocab_size=len(tok),
                                       attn_implementation="eager")).to(device)
        m = m.float()
    else:
        tok = AutoTokenizer.from_pretrained(name, revision=rev(name))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32,
                                                 attn_implementation="eager",
                                                 revision=rev(name)).to(device)
    seed_everything(0)                       # pin LoRA-A init (else entropy-seeded)
    target = LLAMA_TARGET if "llama" in name.lower() else None   # None -> peft default
    m = get_peft_model(m, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, target_modules=target,
                                     lora_dropout=0.0, task_type="CAUSAL_LM"))
    init = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    return tok, m, init, list(init)


def _synth_records(n, tag):
    """Offline wiring-smoke records (SYNTH_DATA=1) -- track_g's seam, same convention."""
    rng = np.random.default_rng(abs(hash(tag)) % 2 ** 31)
    out = []
    for _ in range(n):
        a, b = int(rng.integers(1, 60)), int(rng.integers(1, 60))
        out.append({"prompt": f"Question: {a}+{b}?\nAnswer:", "completion": f" {a + b}"})
    return out


def _build_clients(seed, noisy):
    if SYNTH:                                    # local wiring smoke only, no download
        from datasets import Dataset

        from flirds.data.corruptors import LLM_CORRUPTORS
        clients = []
        for c in range(RCFG["n_clients"]):
            recs = _synth_records(RCFG["train"], f"c{c}s{seed}")
            if c in noisy:
                recs = LLM_CORRUPTORS["answer_swap"](recs, c)
            clients.append(Dataset.from_list(recs))
        return clients, _synth_records(RCFG["val"] * 5, f"v{seed}"), []
    if REGIME == "silo5":
        return build(RCFG["n_clients"], RCFG["train"], RCFG["val"], RCFG["test"],
                     seed=seed, noisy=noisy, noisy_rate=RCFG["noisy_rate"])
    return build_gsm8k_iid(RCFG["n_clients"], n_val=RCFG["val"], n_test=RCFG["test"],
                           seed=seed, noisy=noisy, noisy_rate=RCFG["noisy_rate"])


def _measure_rounds(n_rounds):
    if N_MEASURE == "all":
        return list(range(n_rounds))
    k = min(int(N_MEASURE), n_rounds)
    return sorted(set(np.linspace(0, n_rounds - 1, k).round().astype(int).tolist()))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(SEED)

    noisy = frozenset(RCFG["noisy"]) if THREAT_K == "answer_swap" else frozenset()
    riders = frozenset(RCFG["freerider"]) if THREAT_K == "freerider_zero" else frozenset()
    corrupt = sorted(noisy | riders)

    t0 = time.perf_counter()
    tok, model, init, pkeys0 = load_model(MODEL, device, LORA_R)
    clients, val_records, _ = _build_clients(SEED, noisy)
    t_setup = time.perf_counter() - t0
    print(f"# {REGIME} threat={THREAT_K} seed={SEED} model={MODEL} "
          f"N={RCFG['n_clients']} k_frac={RCFG['k_frac']} R={RCFG['rounds']} "
          f"steps={RCFG['max_steps']} lr={RCFG['lr']} val={len(val_records)} "
          f"| corrupt={corrupt} setup={t_setup:.1f}s", flush=True)

    # ---- frozen trajectory (regenerated, not loaded: no checkpoints are persisted) ----
    t0 = time.perf_counter()
    model.load_state_dict(init, strict=False)
    logs = run_llm_fedavg_logs(model, tok, clients, RCFG["rounds"], RCFG["lr"],
                               RCFG["max_steps"], batch_size=MCFG["batch"],
                               max_length=RCFG["maxlen"], sample_frac=RCFG["k_frac"],
                               seed=SEED, free_riders=riders, free_rider_mode="zero")
    t_fl = time.perf_counter() - t0
    print(f"[fl] {len(logs)} rounds in {t_fl:.1f}s (K={len(logs[0][1])}/round)", flush=True)

    val_batches = build_val_batches(val_records, tok, MCFG["val_maxlen"], device,
                                    MCFG["val_chunk"])
    loss_fn, pkeys, loss_chunks = make_llm_loss(model, val_batches, device)
    assert set(pkeys) == set(pkeys0), "pkeys mismatch between init and make_llm_loss"

    want = _measure_rounds(len(logs))
    all_rows, summaries = [], []
    t0 = time.perf_counter()
    for r in want:
        w_r, dm = logs[r]
        rows, _phis, summ = measure_round(r, w_r, dm, loss_fn, pkeys, device,
                                          loss_chunks=loss_chunks, renorm=RENORM, sweep=True)
        all_rows += rows
        summaries.append(summ)
        print(f"[round {r}] base={summ['base_loss']:.6f} ulp={summ['ulp_base']:.2e} "
              f"||dW||={summ['norm_dW']:.4g} med|u-u1|={summ['resid1']['median']:.3e} "
              f"med|u-u2|={summ['resid2']['median']:.3e} "
              f"r2/floor={summ['resid2_over_ulp']:.1f}x "
              f"slope2(coal)={summ['loglog_slope_r2']}", flush=True)
    t_meas = time.perf_counter() - t0

    P = pool(summaries, all_rows)
    print(f"\n=== POOLED  ({REGIME} {THREAT_K} seed={SEED}) ===")
    print(f"  resid1 median              {P['resid1']['median']:.4e}")
    print(f"  resid2 median              {P['resid2']['median']:.4e}")
    print(f"  fp32 floor (ulp of base)   {P['ulp']:.4e}")
    print(f"  resid2 / floor             {P['resid2_median_over_ulp']:.1f}x")
    print(f"  frac(resid2 <= resid1)     {P['frac_t2_le_t1']:.3f}")
    print(f"  slope resid1 (coalition)   {P['loglog_slope_r1']}   predicted 2")
    print(f"  slope resid2 (coalition)   {P['loglog_slope_r2']}   predicted 3")
    print(f"  slope resid1 (sweep)       {P['sweep_slope_r1']}   predicted 2")
    print(f"  slope resid2 (sweep)       {P['sweep_slope_r2_above_floor']}   predicted 3")
    print(f"  mean |u_r(P_r)|            {P['mean_abs_u_grand']:.4e}")
    print(f"  closed form vs Shapley(u2) {P['max_phi_t2_vs_closed']:.3e}  (Thm 1 numeric check)")
    print(f"  timing                     FL {t_fl:.1f}s | measure {t_meas:.1f}s "
          f"({100 * t_fl / (t_fl + t_meas):.0f}% training)")

    if not PERSIST:
        print("TAYLOR-LLM OK (no persist)", flush=True)
        return

    scale = "gpt2" if "gpt2" in MODEL else "1B"
    name = f"{scale}_{REGIME}_{THREAT_K}_taylor_seed{SEED}"
    rl = RunLogger(RUN_ROOT, name,
                   dict(track="llm", table="tab:retrain-fidelity-counterpart",
                        regime=REGIME, model=MODEL, threat=THREAT_K, threat_label=THREAT,
                        seed=SEED, rcfg={k: (sorted(v) if isinstance(v, (set, frozenset)) else v)
                                         for k, v in RCFG.items()},
                        mcfg=MCFG, lora_r=LORA_R, corrupt=corrupt,
                        n_val=len(val_records), renorm=RENORM,
                        rounds_measured=want, smoke=SMOKE),
                   repo_root=_REPO)
    rl.save_phi(all_rows, fname="coalitions.parquet")
    rl.save_metrics(dict(pooled=P, rounds=summaries,
                         timing=dict(setup_s=round(t_setup, 1), fl_train_s=round(t_fl, 1),
                                     measure_s=round(t_meas, 1),
                                     train_frac=t_fl / (t_fl + t_meas))))
    with open(os.path.join(rl.dir, "summary.json"), "w") as f:
        json.dump(dict(pooled=P, rounds=summaries), f, indent=1)
    print(f"[persist] {rl.dir}", flush=True)
    print("TAYLOR-LLM OK", flush=True)


if __name__ == "__main__":
    main()
