"""Phase 1 (3) LLM SV-baseline comparison: Flirds vs GTG vs FedSV vs (b) oracle.

On ONE shared LLM FedAvg trajectory (the same frozen `logs`), compute every in-run
client valuation and compare them on the same footing:
  - Spearman rho vs the (b) in-run oracle (exact 2^N)            -- ranking fidelity;
  - noisy / free-rider detection AUROC (higher phi = worse)      -- the #7 task metric;
  - per-method wall-clock runtime                                -- Ripple's headline
    metric, informative for all (the estimator is 1 HVP/round; the baselines sweep
    coalitions like the oracle).
The LLM analog of phase05_dual_oracle (CNN: (a)/(b)/estimator) extended with the
ported reconstruction-utility baselines (GTG/FedSV).  phi-only -- downstream task
generation is #7's job (phase1_clean_run), not the valuation comparison.

N=5 so the (b) oracle + the baselines' 2^N coalition sweep stay affordable; lr/R are
chosen so the corrupted clients sit above the noise floor (cf. the lr-sweep -- at
8ex/2step every phi is noise).  GTG/FedSV truncation is OFF here (round_trunc/eps=0)
so the baselines are clean exact-MC, not TMC-truncated (truncation is a speed lever,
tunable to the loss scale later if runtime matters).

>=3 seeds, mean+/-std.  RUN_SEED restricts to one seed so the seeds batch across the
4 GPUs (RUN_SEED={0,1,2} on CUDA_VISIBLE_DEVICES={0,1,2}); aggregate the printed
per-seed lines afterwards.  RUN_LR overrides the lr.

Run from codes/:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase1_baseline_compare.py            # all seeds, 1 GPU
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. RUN_SEED=0 python experiments/phase1_baseline_compare.py  # one seed (batch)
"""
import os
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from flirds.backends.llm import make_llm_loss
from flirds.baselines.banzhaf import in_run_banzhaf
from flirds.baselines.fedsv import fedsv_from_logs
from flirds.baselines.gtg import gtg_from_logs
from flirds.baselines.ripple_llm import ripple_shapley_llm
from flirds.baselines.shapleyfl import shapleyfl_from_logs
from flirds.core.flirds_estimator import flirds_values
from flirds.data.llm import build, build_val_batches
from flirds.eval.metrics import detection_auroc
from flirds.fl.llm_server import run_llm_fedavg_logs
from flirds.oracle.in_run_sv import in_run_shapley, in_run_utility
from flirds.repro import seed_everything

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DOMAINS = ["medical", "legal", "finance", "math", "general"]
N, NOISY, FREE_RIDERS, FREE_RIDER_MODE = 5, {0}, {1}, "zero"
RIPPLE = os.environ.get("RIPPLE", "1") == "1"   # 0 -> skip Ripple (its eager grad/HVP loop
#   has a memory leak under debug); the from-logs baselines GTG/FedSV/oracle are unaffected.

CFG = dict(train=200, val=20, rounds=10, max_steps=10, lr=1e-3, batch=16,
           maxlen=768, val_maxlen=384, val_chunk=10, seeds=[0, 1, 2],
           # the oracle/GTG/FedSV coalition sweep (2^N * rounds * val-chunks) and Ripple's
           # per-step val-grad + Hessian HVPs are the cost -- kept modest so a seed is ~20min
           # (val=20/domain -> 10 chunks; runtime is itself the reported Ripple metric).
           rip_rounds=4, rip_steps=4, rip_k=3, rip_m=20, rip_hess_bs=2,
           sfl_beta=0.5)   # ShapleyFL surrogate-FSV EMA rate (Def 4.3)


def _timed(fn, device):
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t


def run_seed(seed, cfg, device):
    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 attn_implementation="eager").to(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=TARGET,
                                             lora_dropout=0.0, task_type="CAUSAL_LM"))
    init_lora = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    clients, val, _ = build(N, cfg["train"], cfg["val"], seed=seed, noisy=NOISY)
    val_chunks = build_val_batches(val, tok, cfg["val_maxlen"], device, cfg["val_chunk"])
    print(f"[seed {seed}] trajectory: R={cfg['rounds']} lr={cfg['lr']} "
          f"clients={[len(c) for c in clients]} val={len(val)}/{len(val_chunks)}ch", flush=True)

    model.load_state_dict(init_lora, strict=False)
    logs = run_llm_fedavg_logs(model, tok, clients, cfg["rounds"], cfg["lr"], cfg["max_steps"],
                               batch_size=cfg["batch"], max_length=cfg["maxlen"], seed=seed,
                               free_riders=FREE_RIDERS, free_rider_mode=FREE_RIDER_MODE)
    loss_fn, pkeys, lc = make_llm_loss(model, val_chunks, device)

    # every method values the SAME frozen logs via loss_fn/pkeys (good->low; the
    # baselines use the within-subset-renormalized reconstruction of their papers).
    (phi_b, _), t_b = _timed(lambda: in_run_shapley(logs, N, loss_fn, pkeys, device), device)
    # exact Banzhaf semivalue: SAME (b) delta utility, uniform 1/2^{n-1} marginal weight
    # (a coalition-sweep baseline; runtime in the (b)-oracle class, not Flirds' 1 HVP/round).
    (phi_z, _), t_z = _timed(lambda: in_run_banzhaf(logs, N, loss_fn, pkeys, device), device)
    (phi_e, _), t_e = _timed(
        lambda: flirds_values(logs, loss_fn, pkeys, device, second_order=True, loss_chunks=lc), device)
    (phi_1, _), t_1 = _timed(   # Flirds-1st-only (self-ablation: isolates the 2nd-order term)
        lambda: flirds_values(logs, loss_fn, pkeys, device, second_order=False, loss_chunks=lc), device)
    phi_g, t_g = _timed(
        lambda: gtg_from_logs(logs, None, N, None, device, seed=seed, loss_fn=loss_fn,
                              pkeys=pkeys, round_trunc=0.0, eps=0.0), device)
    phi_f, t_f = _timed(
        lambda: fedsv_from_logs(logs, None, N, None, device, seed=seed, loss_fn=loss_fn,
                                pkeys=pkeys, trunc_eps=0.0), device)
    # ShapleyFL surrogate-FSV (uniform submodel + per-round exact Shapley + min-max + EMA);
    # good->high -> negate to good->low (like Ripple).  Another coalition-sweep baseline.
    phi_s_raw, t_s = _timed(
        lambda: shapleyfl_from_logs(logs, None, N, None, device, beta=cfg["sfl_beta"],
                                    loss_fn=loss_fn, pkeys=pkeys), device)
    phi_s = np.asarray([-float(x) for x in phi_s_raw])
    # loss-heuristic (floor): per-client singleton in-run utility U_(b)({k}) (no coalitions);
    # good->low, free-rider(zero) -> exactly 0 (zero delta -> zero singleton utility).
    phi_h, t_h = _timed(
        lambda: np.array([in_run_utility(logs, [k], loss_fn, pkeys, device) for k in range(N)]), device)

    # Ripple values its OWN (shorter) trajectory -- no shared logs -> no Spearman vs
    # the oracle; negate to good->low so its AUROC/orientation match the loss-based
    # methods (Ripple is natively good->high: clean clients get HIGH phi).
    if RIPPLE:
        phi_r_raw, t_r = _timed(lambda: ripple_shapley_llm(
            model, init_lora, clients, tok, val_chunks, device, rounds=cfg["rip_rounds"],
            steps=cfg["rip_steps"], lr=cfg["lr"], k=cfg["rip_k"], m=cfg["rip_m"], seed=seed,
            free_riders=FREE_RIDERS, free_rider_mode=FREE_RIDER_MODE, hess_bs=cfg["rip_hess_bs"]), device)
        phi_r = np.asarray([-float(x) for x in phi_r_raw])
    else:
        phi_r, t_r = np.zeros(N), 0.0      # Ripple skipped (RIPPLE=0): placeholder, excluded from claims

    noisy_y = [1 if i in NOISY else 0 for i in range(N)]
    fr_y = [1 if i in FREE_RIDERS else 0 for i in range(N)]
    shared = [("Flirds", phi_e, t_e), ("Flirds1st", phi_1, t_1), ("GTG", phi_g, t_g),
              ("FedSV", phi_f, t_f), ("Banzhaf", phi_z, t_z), ("ShapleyFL", phi_s, t_s),
              ("loss-heur", phi_h, t_h), ("(b)oracle", phi_b, t_b)]   # value the SAME logs
    methods = shared + [("Ripple", phi_r, t_r)]                   # own trajectory
    return {
        "phi": {k: [float(x) for x in v] for k, v, _ in methods},
        "spearman_vs_b": {k: float(spearmanr(v, phi_b).correlation)
                          for k, v, _ in shared if k != "(b)oracle"},
        "auroc_noisy": {k: float(detection_auroc(v, noisy_y)) for k, v, _ in methods},
        "auroc_fr": {k: float(detection_auroc(v, fr_y)) for k, v, _ in methods},
        "runtime": {k: t for k, _, t in methods},
    }


def _fmt_phi(ph):
    tag = lambda i: "*" if i in NOISY else ("F" if i in FREE_RIDERS else " ")
    return "  ".join(f"{DOMAINS[i][:4]}{tag(i)}={ph[i]:+.4f}" for i in range(N))


def main():
    device = "cuda"
    cfg = dict(CFG)
    if os.environ.get("RUN_LR"):
        cfg["lr"] = float(os.environ["RUN_LR"])
    seeds = [int(os.environ["RUN_SEED"])] if os.environ.get("RUN_SEED") else cfg["seeds"]
    print(f"SV-baseline compare | 1B N={N} | per_domain train={cfg['train']} val={cfg['val']} "
          f"R={cfg['rounds']} lr={cfg['lr']} | noisy={sorted(NOISY)} "
          f"free_rider={sorted(FREE_RIDERS)}({FREE_RIDER_MODE}) | seeds={seeds}", flush=True)

    # method display order (keys come from run_seed's `methods`); defined once.
    PHI_ORDER = ["(b)oracle", "Flirds", "Flirds1st", "GTG", "FedSV", "Banzhaf",
                 "ShapleyFL", "loss-heur", "Ripple"]
    SHARED_ORDER = ["Flirds", "Flirds1st", "GTG", "FedSV", "Banzhaf", "ShapleyFL", "loss-heur"]
    ALL_ORDER = SHARED_ORDER + ["Ripple", "(b)oracle"]   # AUROC / runtime (every method)

    runs = []
    for seed in seeds:
        m = run_seed(seed, cfg, device)
        runs.append(m)
        print(f"\n[seed {seed}] phi (good->low; ShapleyFL/Ripple negated):")
        for k in PHI_ORDER:
            print(f"  {k:9s}: {_fmt_phi(m['phi'][k])}")
        print("  Spearman vs (b): " + "  ".join(
            f"{k}={m['spearman_vs_b'][k]:+.3f}" for k in SHARED_ORDER)
            + "   (Ripple: own trajectory, no shared GT)")
        print("  AUROC noisy:     " + "  ".join(f"{k}={m['auroc_noisy'][k]:.3f}" for k in ALL_ORDER))
        print("  AUROC free-rider:" + "  ".join(f"{k}={m['auroc_fr'][k]:.3f}" for k in ALL_ORDER))
        print("  runtime (s):     " + "  ".join(f"{k}={m['runtime'][k]:.1f}" for k in ALL_ORDER))

    if len(runs) > 1:
        def agg(d, k):
            xs = [r[d][k] for r in runs]
            return float(np.mean(xs)), float(np.std(xs))
        print(f"\n=== aggregate (mean+/-std over {len(runs)} seeds) ===")
        print("  Spearman vs (b):  " + "  ".join(
            f"{k}={agg('spearman_vs_b', k)[0]:+.3f}+/-{agg('spearman_vs_b', k)[1]:.3f}"
            for k in SHARED_ORDER))
        print("  AUROC noisy:      " + "  ".join(
            f"{k}={agg('auroc_noisy', k)[0]:.3f}+/-{agg('auroc_noisy', k)[1]:.3f}" for k in ALL_ORDER))
        print("  runtime (s):      " + "  ".join(
            f"{k}={agg('runtime', k)[0]:.1f}+/-{agg('runtime', k)[1]:.1f}" for k in ALL_ORDER))
    print("\nFlirds should match the (b) oracle ranking (Spearman ~1) at lower runtime "
          "than the coalition-sweep baselines; GTG/FedSV are the FL-Shapley comparison.")


if __name__ == "__main__":
    main()
