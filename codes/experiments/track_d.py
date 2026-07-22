"""Track D: LLM standard-setting comparison on the GENERAL (IID, clean) stage
(plan §3.11 D, redesigned 2026-06-13 -- no corruption, no non-IID).

The stage mirrors OpenFedLLM's flagship SFT recipe verbatim (training_scripts/
run_sft.sh: vicgalle/alpaca-gpt4 20k, N=20 clients / 2 per round / 200 rounds,
10 steps x batch 16, seq 512, alpaca template, FedAvg).  Question order = the
project hierarchy (root CLAUDE.md, Yonghee 2026-06-12):
  1. SV-computation fidelity (PRIMARY): every valuation method on ONE frozen
     vanilla trajectory vs the exact (b) oracle -- Spearman/Kendall + the GTG
     distance trio (cosine/Euclid/max-diff; same-units caveat in eval.metrics)
     + wall-clock.  anchor5 adds the (a) RETRAIN oracle (val-loss utility,
     fp32 -- the task-6 same-game lesson; (a)+(b) dual GT = a literature gap).
  2. benchmark accuracy: per-method ONLINE intervention arms (fl/intervene)
     -> MMLU full-test 0-shot + same-distribution Alpaca-test ROUGE-L.
     Clean-IID expectation is parity (do-no-harm); a difference is a finding.
  3. convergence: per-round val-loss curves off each arm's logs +
     rounds-to-target (vanilla's final val loss) + per-method runtime.

Deviations from the literature recipe (locked conventions, stated as caveats):
SGD momentum=0, constant lr 1e-3 (vs AdamW 5e-5 cosine -- the Taylor/per-step
assumption, codes/CLAUDE.md §5; FedIT-SGD precedent exists), LoRA r16/a32 (vs
r32/a64 -- all-track consistency), fp32 no-quant (vs 8bit -- the oracle/
estimator precision floor).

REGIMES:
  std20    N=20, 2/round, R=200 -- the literature-standard form; (b) = exact
           per-round decomposition (2^2/round).
  anchor5  N=5, full participation, R=30 -- the oracle-precision point: exact
           2^5 (b), the (a) retrain oracle (ORACLE_A defaults on here),
           Banzhaf and every coalition baseline exact.

ARMS (phase 2; the C2 wiring transplanted -- intervene.py is CNN/LLM-shared):
  base / vanilla (= the frozen phase-1 trajectory, no extra cost) /
  flirds_w (multiplicative w ~ n*s, beta=0.5) / flirds_sel (softmax selection;
  only where the cohort is a strict subset, i.e. std20) / shapleyfl_w
  (replacement, beta=0.3) / fedif_w (replacement, beta=0.7 = 1-gamma).

Run from codes/:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. REGIME=anchor5 SEED=0 python -u experiments/track_d.py
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. REGIME=std20  SEED=0 python -u experiments/track_d.py
  # smoke (gpt2 stand-in; values are noise -- code-path only):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. SMOKE_MODEL=gpt2 REGIME=std20 TOTAL_TRAIN=200 VAL=20 \
    TEST=20 ROUNDS=4 MAX_STEPS=2 MMLU_LIMIT=40 SEED=0 python -u experiments/track_d.py
"""
import os
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from scipy.stats import kendalltau, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from flirds.backends.llm import make_llm_loss
from flirds.baselines.banzhaf import in_run_banzhaf
from flirds.baselines.comfedsv import comfedsv_from_logs
from flirds.baselines.fedif import fedif_from_logs
from flirds.baselines.fedsv import fedsv_from_logs
from flirds.baselines.gtg import gtg_from_logs
from flirds.baselines.shapleyfl import shapleyfl_from_logs
from flirds.core.flirds_estimator import flirds_values
from flirds.data.llm import build_alpaca_iid, build_val_batches
from flirds.eval.generate import generate_completions, score_records
from flirds.eval.metrics import cosine_distance, euclidean_distance, max_difference, pearson
from flirds.eval.mmlu import mmlu_accuracy
from flirds.fl.intervene import (OnlineScorer, fedif_round_raw_fn, flirds_round_raw_fn,
                                 make_scoreonly_weights_fn, make_softmax_select_fn,
                                 make_weights_fn, shapleyfl_round_raw_fn)
from flirds.fl.llm_server import run_llm_fedavg_logs
from flirds.oracle.exact_sv import exact_shapley
from flirds.oracle.exact_sv_llm import _final_lora_state
from flirds.oracle.in_run_sv import (in_run_shapley, in_run_shapley_perround,
                                     in_run_singletons)
from flirds.repro import seed_everything
from flirds.run_logger import RunLogger
from flirds.hf_pin import rev
from flirds.timing import PhaseTimer

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
SCALE = ("7B" if "Llama-2-7b" in MODEL else
         MODEL.split("-")[-2] if "Llama-3.2-" in MODEL else MODEL.split("/")[-1])
TARGET = (["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
          if "llama" in MODEL.lower() else None)   # None -> peft's per-arch default (gpt2: c_attn)

REGIME = os.environ.get("REGIME", "std20")
STD20 = dict(n_clients=20, k_abs=2, rounds=200)    # OpenFedLLM run_sft.sh: c20 s2 R200
ANCHOR5 = dict(n_clients=5, k_abs=5, rounds=30)    # dual-oracle precision point
RCFG = dict(STD20 if REGIME == "std20" else ANCHOR5)
RCFG.update(total_train=20000, val=200, test=1000, max_steps=10, lr=1e-3, maxlen=512)
for k in ("n_clients", "k_abs", "total_train", "val", "test", "rounds", "max_steps", "maxlen"):
    if os.environ.get(k.upper()):
        RCFG[k] = int(os.environ[k.upper()])
if os.environ.get("LR"):
    RCFG["lr"] = float(os.environ["LR"])

# per-scale memory knobs (== phase2_matrix task8): exact chunk-sum -> peak-memory only.
MODEL_CFG = {"1B": dict(batch=16, val_chunk=10, val_maxlen=384),
             "3B": dict(batch=8, val_chunk=5, val_maxlen=384),
             "7B": dict(batch=4, val_chunk=2, val_maxlen=320)}
MCFG = dict(MODEL_CFG.get(SCALE, MODEL_CFG["1B"]))
for k in ("batch", "val_chunk", "val_maxlen"):
    if os.environ.get(k.upper()):
        MCFG[k] = int(os.environ[k.upper()])

LORA_R = int(os.environ.get("LORA_R", "16"))           # probe lever (signal-size); default = current
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", str(2 * LORA_R)))   # alpha/r = 2 kept across ranks

MMLU_LIMIT = int(os.environ.get("MMLU_LIMIT", "0"))    # 0 = full test (14,042)
MMLU_BATCH = int(os.environ.get("MMLU_BATCH", "16"))
ARMS = os.environ.get("ARMS", "1") == "1"              # 0 = phase-1 fidelity only
FIDELITY = os.environ.get("FIDELITY", "1") == "1"      # 0 = arm-only (cheap re-run; no (a)/coalition cost)
ORACLE_A = os.environ.get("ORACLE_A", "1" if REGIME == "anchor5" else "0") == "1"
# METHODS="Flirds,Flirds1st,loss-heur" -> run only these (+ the (b) oracle, always).
# Empty/unset = full suite.  E4/E5 light re-runs: coalition baselines off without code edits.
METHODS = frozenset(m for m in os.environ.get("METHODS", "").split(",") if m)

_CODES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR_ROOT = os.environ.get("RUNDIR_ROOT",
                             os.path.join(os.path.dirname(_CODES), "runs", "track_d", "rundirs"))


def _load(device):
    """fp32 + eager LoRA model (the (b)/estimator path; one load for the whole run)."""
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev(MODEL))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                             attn_implementation="eager",
                                             revision=rev(MODEL)).to(device)
    seed_everything(0)                     # pin LoRA-A init (else entropy-seeded per process -> non-reproducible)
    m = get_peft_model(m, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET,
                                     lora_dropout=0.0, task_type="CAUSAL_LM"))
    init = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    return tok, m, init, list(init)


def _timed(fn, device):
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t


def _fl(model, tok, clients, init, seed, select_fn=None, weights_fn=None):
    """One FedAvg pass from init -> logs; cohort = k_abs per round (absolute)."""
    model.load_state_dict(init, strict=False)
    return run_llm_fedavg_logs(model, tok, clients, RCFG["rounds"], RCFG["lr"],
                               RCFG["max_steps"], batch_size=MCFG["batch"],
                               max_length=RCFG["maxlen"],
                               sample_frac=min(1.0, RCFG["k_abs"] / len(clients)),
                               seed=seed, select_fn=select_fn, weights_fn=weights_fn)


# --------------------------------------------------------------------------- #
# phase 1 -- SV-computation fidelity (the PRIMARY axis)                        #
# --------------------------------------------------------------------------- #
def make_a_utility(model, tok, clients, init, loss_fn, pkeys, device, seed):
    """(a) retrain-oracle utility U(S) = -val_loss(FedAvg retrained on S alone).

    The task-6 lesson: the (a) validation must play the SAME GAME as (b)/the
    estimator (val-loss, fp32) -- ROUGE is a different, non-differentiable game.
    Empty S scores the init adapter."""
    def utility(S):
        model.load_state_dict(init, strict=False)
        if S:
            logs = run_llm_fedavg_logs(model, tok, [clients[c] for c in S],
                                       RCFG["rounds"], RCFG["lr"], RCFG["max_steps"],
                                       batch_size=MCFG["batch"], max_length=RCFG["maxlen"],
                                       sample_frac=1.0, seed=seed)
            final = _final_lora_state(logs)
        else:
            final = init
        with torch.no_grad():
            return -float(loss_fn({k: final[k].to(device) for k in pkeys}, {}))
    return utility


def compute_fidelity(logs, model, tok, clients, init, loss_fn, pkeys, lc, device, seed):
    """All valuation methods on the frozen vanilla logs.  Returns ([(name, phi, runtime)], u_a),
    every phi oriented val-loss-attribution (good -> LOW, the (b) orientation; the
    good->high methods are negated -- the phase2_matrix sign conventions).  u_a = the (a)
    retrain-oracle 2^N coalition-utility cache (or None if the (a) oracle did not run) --
    reused for the Exp A1 removal/selection curves with NO extra retraining."""
    n = RCFG["n_clients"]
    anchor = REGIME == "anchor5"
    out = []
    u_a = None
    _want = lambda name: not METHODS or name in METHODS   # (b) oracle is never filtered

    b_fn = in_run_shapley if anchor else in_run_shapley_perround
    (phi_b, _), t = _timed(lambda: b_fn(logs, n, loss_fn, pkeys, device), device)
    out.append(("(b)oracle", np.asarray(phi_b), t))
    if ORACLE_A and anchor:                       # (a) retrain GT (utility good->HIGH -> negate)
        util = make_a_utility(model, tok, clients, init, loss_fn, pkeys, device, seed)
        (phi_a, u_a), t = _timed(lambda: exact_shapley(n, util, return_u=True), device)
        out.append(("(a)oracle", -np.asarray(phi_a, dtype=float), t))
        model.eval()                              # the 32 SFTTrainer retrains leave train mode
        model.get_input_embeddings()._forward_hooks.clear()   # + the embed hook (HVP-forbidden)

    if _want("Flirds"):
        (phi_e, _), t = _timed(lambda: flirds_values(logs, loss_fn, pkeys, device,
                                                     second_order=True, n_clients=n,
                                                     loss_chunks=lc), device)
        out.append(("Flirds", np.asarray(phi_e), t))
    if _want("Flirds1st"):
        (phi_1, _), t = _timed(lambda: flirds_values(logs, loss_fn, pkeys, device,
                                                     second_order=False, n_clients=n,
                                                     loss_chunks=lc), device)
        out.append(("Flirds1st", np.asarray(phi_1), t))
    if _want("GTG"):
        phi_g, t = _timed(lambda: gtg_from_logs(logs, None, n, None, device, seed=seed,
                          loss_fn=loss_fn, pkeys=pkeys, round_trunc=0.0, eps=0.0), device)
        out.append(("GTG", np.asarray(phi_g), t))
    if _want("FedSV"):
        phi_f, t = _timed(lambda: fedsv_from_logs(logs, None, n, None, device, seed=seed,
                          loss_fn=loss_fn, pkeys=pkeys, trunc_eps=0.0), device)
        out.append(("FedSV", np.asarray(phi_f), t))
    if _want("ComFedSV"):
        phi_c, t = _timed(lambda: comfedsv_from_logs(logs, None, n, None, device, seed=seed,
                          loss_fn=loss_fn, pkeys=pkeys, partial=not anchor), device)
        out.append(("ComFedSV", -np.asarray(phi_c, dtype=float), t))   # loss-decrease util -> negate
    if _want("ShapleyFL"):
        phi_s, t = _timed(lambda: shapleyfl_from_logs(logs, None, n, None, device, beta=0.3,
                          loss_fn=loss_fn, pkeys=pkeys), device)
        out.append(("ShapleyFL", -np.asarray(phi_s, dtype=float), t))  # good->high -> negate
    if _want("FedIF"):
        phi_if, t = _timed(lambda: fedif_from_logs(logs, n, loss_fn, pkeys, device,
                           loss_chunks=lc), device)
        out.append(("FedIF", -np.asarray(phi_if, dtype=float), t))     # influence good->high -> negate
    if anchor and _want("Banzhaf"):                                    # Banzhaf = 2^N -> anchor only
        (phi_z, _), t = _timed(lambda: in_run_banzhaf(logs, n, loss_fn, pkeys, device), device)
        out.append(("Banzhaf", np.asarray(phi_z), t))
    if _want("loss-heur"):
        phi_h, t = _timed(lambda: in_run_singletons(logs, n, loss_fn, pkeys, device), device)
        out.append(("loss-heur", phi_h, t))
    # Fed-LOO dropped from the comparison (Yonghee 2026-07-23).  The estimator
    # `flirds.oracle.in_run_sv.in_run_loo` stays so existing rundirs replay.
    return out, u_a


def removal_curves(methods, u_a, n):
    """Exp A1: worst-first / best-first removal curves for every method, looked up from
    the (a) retrain-oracle coalition-utility cache `u_a` (NO retraining -- U(S) is already
    computed for all 2^N subsets).  Every phi is suspicion-oriented (good->LOW), so:
      worst_first -- drop the HIGHEST-phi (most-suspicious) client each step
      best_first  -- drop the LOWEST-phi  (most-valuable) client each step
    u_a[S] = -val_loss(FedAvg-retrained-on-S) (good->HIGH; make_a_utility negates the loss).
    Returns {method: {"worst_first": [[k_dropped, util], ...(k=0..n-1, kept=n..1)],
    "best_first": [...]}}.  Empty coalition (k=n) is omitted (FL-on-nothing is the init)."""
    def curve(order):
        pts, kept = [], list(range(n))
        for k in range(n):                        # kept sizes n, n-1, ..., 1 (drop empty)
            pts.append([k, float(u_a[tuple(sorted(kept))])])
            kept.remove(order[k])
        return pts
    out = {}
    for name, vec, _rt in methods:
        v = np.asarray(vec, dtype=float)
        out[name] = {"worst_first": curve(list(np.argsort(-v))),   # high phi (worst) dropped first
                     "best_first": curve(list(np.argsort(v)))}     # low phi (best) dropped first
    return out


def report_fidelity(methods, selected):
    """Fidelity table vs the (b) oracle: Spearman + Kendall (rank) + Pearson (value-level,
    affine-invariant) + the GTG distance trio (same-units caveat: distances are
    unit-meaningful only for the val-loss-game vectors; rank/Pearson cover the rest) +
    wall-clock.  Returns {col: {method: v}}."""
    truth = methods[0][1]
    res = {"spearman": {}, "kendall": {}, "pearson": {}, "cosine_d": {}, "euclid_d": {},
           "max_diff": {}, "runtime": {}}
    print(f"  {'method':10s} {'Spearman':>9s} {'Kendall':>8s} {'Pearson':>8s} {'cos-d':>8s} "
          f"{'euc-d':>9s} {'max-d':>9s} {'runtime':>9s}")
    for name, vec, rt in methods:
        res["runtime"][name] = rt
        if name == "(b)oracle":
            print(f"  {name:10s} {'(truth)':>9s} {'':8s} {'':8s} {'':8s} {'':9s} {'':9s} {rt:8.1f}s")
            continue
        v, u = [vec[c] for c in selected], [truth[c] for c in selected]
        res["spearman"][name] = float(spearmanr(v, u).correlation)
        res["kendall"][name] = float(kendalltau(v, u).correlation)
        res["pearson"][name] = pearson(v, u)
        res["cosine_d"][name] = cosine_distance(v, u)
        res["euclid_d"][name] = euclidean_distance(v, u)
        res["max_diff"][name] = max_difference(v, u)
        print(f"  {name:10s} {res['spearman'][name]:+9.3f} {res['kendall'][name]:+8.3f} "
              f"{res['pearson'][name]:+8.3f} {res['cosine_d'][name]:8.4f} "
              f"{res['euclid_d'][name]:9.4f} {res['max_diff'][name]:9.4f} {rt:8.1f}s")
    return res


# --------------------------------------------------------------------------- #
# phase 2 -- intervention arms (benchmark accuracy) + convergence              #
# --------------------------------------------------------------------------- #
def _guard(model, raw_fn):
    """Round-score hygiene for ONLINE scoring: SFTTrainer leaves the model in train
    mode and can (re-)register the input-require-grad embedding hook between rounds;
    the estimator's functorch HVP forbids both (make_llm_loss clears them at BUILD
    time -- this re-applies that per scoring call)."""
    def fn(w_r, dm, players):
        model.eval()
        model.get_input_embeddings()._forward_hooks.clear()
        return raw_fn(w_r, dm, players)
    return fn


def build_arms(model, loss_fn, pkeys, lc, nums, device):
    """[(arm_name, select_fn, weights_fn)] -- fresh scorers per call (per seed).
    Baselines use their own paper mechanisms + EMA betas (the C2 wiring)."""
    n = RCFG["n_clients"]
    arms = []
    sc = OnlineScorer(n, beta=0.5)
    raw = _guard(model, flirds_round_raw_fn(loss_fn, pkeys, n, device, loss_chunks=lc))
    arms.append(("flirds_w", None, make_weights_fn(sc, raw, nums, "multiplicative")))
    if RCFG["k_abs"] < n:                          # selection is degenerate under full part.
        sc2 = OnlineScorer(n, beta=0.5)
        raw2 = _guard(model, flirds_round_raw_fn(loss_fn, pkeys, n, device, loss_chunks=lc))
        arms.append(("flirds_sel", make_softmax_select_fn(sc2),
                     make_scoreonly_weights_fn(sc2, raw2, nums)))
    sc3 = OnlineScorer(n, beta=0.3)                # the ShapleyFL paper value (Def 4.3)
    raw3 = _guard(model, shapleyfl_round_raw_fn(None, None, device, loss_fn=loss_fn, pkeys=pkeys))
    arms.append(("shapleyfl_w", None, make_weights_fn(sc3, raw3, nums, "replacement")))
    sc4 = OnlineScorer(n, beta=0.7)                # 1 - gamma(0.3), the FedIF paper value
    raw4 = _guard(model, fedif_round_raw_fn(loss_fn, pkeys, device, loss_chunks=lc))
    arms.append(("fedif_w", None, make_weights_fn(sc4, raw4, nums, "replacement")))
    return arms


def _val_curve(logs, loss_fn, pkeys, device):
    """Convergence curve: val loss at every round-start state + the final global.
    curve[0] = init loss; curve[r] = loss entering round r; curve[-1] = deployed."""
    pts = []
    with torch.no_grad():
        for w_r, _ in logs:
            pts.append(float(loss_fn({k: w_r[k].to(device) for k in pkeys}, {})))
        final = _final_lora_state(logs)
        pts.append(float(loss_fn({k: final[k].to(device) for k in pkeys}, {})))
    return pts, final


def _rounds_to_target(curve, target):
    """First round index whose ENTERING loss <= target (vanilla's final loss);
    None if never reached.  curve[i] = loss entering round i (i=len-1 -> deployed)."""
    for i, v in enumerate(curve):
        if v <= target:
            return i
    return None


def _downstream(model, tok, test_records, device):
    """Benchmark accuracy of the currently-loaded model: MMLU 0-shot (external
    benchmark) + same-distribution Alpaca-test ROUGE-L (IID test)."""
    acc, _, _nq = mmlu_accuracy(model, tok, device, limit=MMLU_LIMIT, batch_size=MMLU_BATCH)
    gens = generate_completions(model, tok, [r["prompt"] for r in test_records], device,
                                max_new_tokens=128, batch_size=16, max_prompt_len=512)
    rouge = score_records(gens, test_records)["alpaca"]["rouge_l"]
    return {"mmlu": acc, "rouge_l": rouge}


def _persist(phi_rows, run_metrics, seeds, timing=None):
    """Save phi vectors + metrics + config/provenance to a run-dir (protocol §6).
    Called as a CHECKPOINT after each seed's fidelity (the (a) oracle is hours --
    an arm crash must never lose it) and again at the end; overwrite is idempotent.
    PERSIST=0 disables."""
    if os.environ.get("PERSIST", "1") != "1":
        return
    name = os.environ.get("RUN_NAME") or (f"{SCALE}_{REGIME}_seed{seeds[0]}"
                                          if len(seeds) == 1 else f"{SCALE}_{REGIME}")
    config = {"scale": SCALE, "model": MODEL, "regime": REGIME, "seeds": seeds,
              "rcfg": RCFG, "mcfg": MCFG, "lora": {"r": LORA_R, "alpha": LORA_ALPHA},
              "oracle_a": ORACLE_A, "fidelity": FIDELITY,
              "methods": sorted(METHODS) if METHODS else "all",
              "client_opt": os.environ.get("CLIENT_OPT", "sgd"),   # Exp D: sgd (default) | adamw
              "mmlu": {"limit": MMLU_LIMIT, "batch": MMLU_BATCH, "shots": 0}}
    try:
        rl = RunLogger(RUNDIR_ROOT, name, config, repo_root=_CODES)
        try:
            rl.save_phi(phi_rows)
        except Exception:
            import pandas as pd
            pd.DataFrame(phi_rows).to_csv(rl._p("phi.csv"), index=False)
        rl.save_metrics(run_metrics)
        if timing is not None:
            rl.save_timing(timing)                                # §15.1 per-phase wall + GPU-hours + peak
        print(f"\n[persist] {rl.dir}  ({len(phi_rows)} phi rows)", flush=True)
    except Exception as e:
        print(f"\n[persist] WARNING: run-dir save failed ({e!r}); .log has results", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(os.environ["SEED"])] if os.environ.get("SEED") else [0, 1, 2]
    n = RCFG["n_clients"]

    print(f"=== Track D (IID clean) | {SCALE} {REGIME} | N={n} K={RCFG['k_abs']}/round "
          f"R={RCFG['rounds']} lr={RCFG['lr']} train={RCFG['total_train']} val={RCFG['val']} "
          f"test={RCFG['test']} seq={RCFG['maxlen']} | (a)oracle={'on' if ORACLE_A else 'off'} "
          f"arms={'on' if ARMS else 'off'} | MMLU "
          f"{'limit=' + str(MMLU_LIMIT) if MMLU_LIMIT else 'full-test'} 0-shot | "
          f"seeds={seeds} ===", flush=True)

    tok, model, init, pkeys = _load(device)
    pt = PhaseTimer(device, n_gpus=int(os.environ.get("N_GPUS", "1")))   # §15.1 timing.json substrate
    agg, phi_rows, run_metrics = [], [], {}
    for seed in seeds:
        seed_everything(seed)
        clients, val, test_records = build_alpaca_iid(n, RCFG["total_train"], RCFG["val"],
                                                      RCFG["test"], seed=seed)
        nums = [len(c) for c in clients]
        logs, t_vanilla = _timed(lambda: _fl(model, tok, clients, init, seed), device)
        pt.record("client-training", t_vanilla)                   # §15.1: reuse the existing measurement
        val_chunks = build_val_batches(val, tok, MCFG["val_maxlen"], device, MCFG["val_chunk"])
        loss_fn, _pk, lc = make_llm_loss(model, val_chunks, device)

        selected = sorted({c for _, dm in logs for c in dm})
        print(f"\n----- seed {seed} | selected={len(selected)}/{n} | "
              f"vanilla FL {t_vanilla:.0f}s -----", flush=True)

        # ---- phase 1: fidelity (the PRIMARY axis) ----
        res = {"spearman": {}, "kendall": {}, "pearson": {}, "cosine_d": {}, "euclid_d": {},
               "max_diff": {}, "runtime": {}}
        if FIDELITY:
            with pt.phase("valuation"):                   # §15.1: all methods + oracles (peak = HVP)
                methods, u_a = compute_fidelity(logs, model, tok, clients, init, loss_fn, pkeys,
                                                lc, device, seed)
            res = report_fidelity(methods, selected)
            if u_a is not None:                           # Exp A1: (a)-retrain removal/selection curves (free)
                res["removal_curve"] = removal_curves(methods, u_a, n)
                res["removal_orient"] = "util=-val_loss (higher=better); phi good->low"
            for name, vec, _rt in methods:
                phi_rows += [{"seed": seed, "method": name, "client": int(c),
                              "phi": float(vec[c])} for c in selected]
            run_metrics[f"seed{seed}"] = res
            _persist(phi_rows, run_metrics, seeds, timing=pt.to_timing())   # CHECKPOINT: never lose the (a)/(b) cost to an arm crash

        # ---- phase 2+3: intervention arms -> benchmark accuracy + convergence ----
        if ARMS:
            res["arms"] = {}
            try:
                van_curve, G_van = _val_curve(logs, loss_fn, pkeys, device)
                target = van_curve[-1]
                rows = [("base", init, 0.0, None), ("vanilla", G_van, t_vanilla, van_curve)]
                for arm, sel_fn, wts_fn in build_arms(model, loss_fn, pkeys, lc, nums, device):
                    arm_logs, t_arm = _timed(lambda: _fl(model, tok, clients, init, seed,
                                                         select_fn=sel_fn, weights_fn=wts_fn),
                                             device)
                    curve, G_arm = _val_curve(arm_logs, loss_fn, pkeys, device)
                    rows.append((arm, G_arm, t_arm, curve))
                    del arm_logs
                    if device == "cuda":
                        torch.cuda.empty_cache()
                for arm, state, t_train, curve in rows:
                    model.load_state_dict(state, strict=False)
                    mets, t_eval = _timed(lambda: _downstream(model, tok, test_records, device),
                                          device)
                    entry = {**mets, "train_s": t_train, "eval_s": t_eval}
                    if curve is not None:
                        entry["final_val_loss"] = curve[-1]
                        entry["rounds_to_target"] = _rounds_to_target(curve, target)
                        entry["val_curve"] = curve
                    res["arms"][arm] = entry
                    extra = (f"  val_loss={curve[-1]:.4f} r2t={entry['rounds_to_target']}"
                             if curve is not None else "")
                    print(f"  [{arm:12s}] mmlu={mets['mmlu']:.4f} rouge_l={mets['rouge_l']:.4f}"
                          f"{extra}  (train {t_train:.0f}s, eval {t_eval:.0f}s)", flush=True)
            except Exception as e:                        # an arm failure must not lose the fidelity axis
                print(f"  [arms] WARNING: arm phase failed ({e!r}); fidelity kept", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()

        agg.append(res)
        run_metrics[f"seed{seed}"] = res
        _persist(phi_rows, run_metrics, seeds, timing=pt.to_timing())
        del loss_fn, val_chunks, logs
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(seeds) > 1:
        print(f"\n=== aggregate (mean+/-std over {len(seeds)} seeds) ===", flush=True)
        if FIDELITY:
            for name in agg[0]["runtime"]:
                if name == "(b)oracle":
                    continue
                line = f"  {name:10s}"
                for col, fmt in (("spearman", "+.3f"), ("kendall", "+.3f"),
                                 ("pearson", "+.3f"), ("cosine_d", ".4f")):
                    vals = [r[col][name] for r in agg]
                    line += f" {col}={np.nanmean(vals):{fmt}}+/-{np.nanstd(vals):.3f}"
                line += f"  runtime={np.mean([r['runtime'][name] for r in agg]):.1f}s"
                print(line)
        if ARMS:
            for arm in agg[0].get("arms", {}):
                mm = [r["arms"][arm]["mmlu"] for r in agg if arm in r.get("arms", {})]
                rg = [r["arms"][arm]["rouge_l"] for r in agg if arm in r.get("arms", {})]
                if mm:
                    print(f"  [{arm:12s}] mmlu={np.mean(mm):.4f}+/-{np.std(mm):.4f} "
                          f"rouge_l={np.mean(rg):.4f}+/-{np.std(rg):.4f}")

    _persist(phi_rows, run_metrics, seeds)
    print("\nTRACK D DONE  (project question order: [1] every method's fidelity vs the (b) "
          "oracle -- Flirds should track it at a fraction of the coalition methods' cost; "
          "[2] arm benchmark accuracy -- clean-IID parity (do-no-harm) expected; "
          "[3] convergence curves/rounds-to-target off the same logs).", flush=True)


if __name__ == "__main__":
    main()
