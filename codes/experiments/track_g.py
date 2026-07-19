"""Track G: phi-gated participation/aggregation -- does USING the computed
contribution in participation/aggregation decisions improve FL? (project question
hierarchy 2nd-1 performance / 2nd-2 convergence; Track G spec 2026-07-19).

POLICIES (fl/intervene.py Track G block; raw is CONTRIBUTION-oriented, good->HIGH):
  V1   aggregation gate: everyone trains; a round's delta with raw <= tau is
       dropped from the aggregate (weight ~ n * 1[raw > tau]).
  V2   participation gate: clients with CUMULATIVE contribution <= tau stop
       training entirely (burn-in, min_obs cold-start guard, probation rotation);
       probation returnees are screened by their same-round raw (the V1 fn).
  V2w  V2 selection + magnitude-proportional weights w ~ n * max(cum, 0)^alpha
       (alpha=1 fixed).  CNN-FIRST: not in the LLM default arm set until the C2
       promotion criteria pass (spec §5-2) -- add explicitly via ARMS after that.
  z    cohort-relative variant (cum z-score < -c excluded) -- the auxiliary
       policy for noisy recovery (the tau=0 gate is provably silent there:
       Stage 0 audit, no 0-crossing on nr in (0,1]).
  V3   post-hoc: after the vanilla run, retrain ONCE keeping only cum > 0
       clients (+z variant, + size-matched random-kept control).  Retrain keeps
       the threat active on kept clients (deployment semantics -- exclusion is
       the only intervention; the clean-retrain game-independent ruler is Exp A2).

The headline policy is PARAMETER-FREE (tau=0, no k, no corruption count) --
oracle_excl stands in for the ideal top-k upper bound (no top-k arm, spec §7).

Per-round logging (spec §4.3, REQUIRED): every gate arm and the vanilla observer
persist {round, client, participated, raw, weight, cum, n_obs, fallback} to
phi_rounds.parquet -- the project's FIRST per-round phi record (clean false-fire
rate, burn-in calibration, offline policy analysis).  raw/cum are contribution-
oriented (helpful -> POSITIVE = -stored-phi); phi.parquet keeps the repo's
suspicion orientation.

REGIMES (env REGIME):
  silo5    5-domain non-IID N=5 full-part R=10 (phase2_matrix SILO config verbatim)
  iid5     alpaca IID N=5 (same shape; build_alpaca_iid with silo5-matched totals)
  std50k5  alpaca IID N=50, 5/round, R=200 (probe_signal std50k5 config) -- the
           participation-axis stage where method fidelity separates.

THREATS (env THREAT; poison EXCLUDED by design -- 2026-07-17 decision):
  clean | noisy (answer_swap @ NOISY_RATE) | frrand | frzero |
  mixed (std50k5-corrupt: NOISY_IDS answer-swap + FR_IDS free-riders together).

Run from codes/ (one process per cell = REGIME x THREAT x arm-subset x seed):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. REGIME=silo5 THREAT=frzero SEED=0 \
    python -u experiments/track_g.py
  # std50k5-corrupt pilot:
  ... REGIME=std50k5 THREAT=mixed SEED=0 ARMS=vanilla,oracle_excl,flirds_gate_v2 ...
  # server smoke (gpt2 silo5-mini, spec §5-1):
  SMOKE_MODEL=gpt2 REGIME=silo5 THREAT=frzero TRAIN=40 VAL=10 TEST=10 ROUNDS=5 \
    MAX_STEPS=2 BURN_IN=2 SEED=0 python -u experiments/track_g.py
  # local wiring smoke (no weights/data downloads: tiny random gpt2 + synthetic data):
  SMOKE_MODEL=tiny-gpt2 SYNTH_DATA=1 REGIME=silo5 THREAT=frzero ROUNDS=5 MAX_STEPS=2 \
    BURN_IN=2 SEED=0 python -u experiments/track_g.py
"""
import os
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from flirds.backends.llm import make_llm_loss
from flirds.baselines.ripple import _flat
from flirds.data.llm import build, build_alpaca_iid, build_val_batches
from flirds.eval.generate import generate_completions, score_records
from flirds.eval.mmlu import mmlu_accuracy
from flirds.fl.intervene import (OnlineScorer, SignAccumulator, flirds_round_raw_fn,
                                 lossheur_round_raw_fn, make_fixed_excl_select_fn,
                                 make_gatedweight_weights_fn, make_observer_weights_fn,
                                 make_signgate_select_fn, make_signgate_weights_fn,
                                 make_weights_fn, make_zgate_select_fn,
                                 make_zgate_weights_fn, _zscores)
from flirds.fl.llm_server import run_llm_fedavg_logs
from flirds.fl.score_providers import (comfedsv_round_raw_fn, fedsv_round_raw_fn,
                                       gtg_round_raw_fn)
from flirds.oracle.exact_sv_llm import _final_lora_state
from flirds.oracle.in_run_sv import in_run_shapley_perround
from flirds.repro import seed_everything
from flirds.run_logger import RunLogger
from flirds.timing import PhaseTimer

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
SCALE = ("7B" if "Llama-2-7b" in MODEL else
         MODEL.split("-")[-2] if "Llama-3.2-" in MODEL else MODEL.split("/")[-1])
TARGET = (["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
          if "llama" in MODEL.lower() else None)   # None -> peft per-arch default (gpt2: c_attn)

REGIME = os.environ.get("REGIME", "silo5")
THREAT = os.environ.get("THREAT", "clean")
SILO_LIKE = REGIME in ("silo5", "iid5")

# regime configs: silo5/iid5 = phase2_matrix SILO verbatim; std50k5 = probe_signal std50k5.
SILO = dict(n_clients=5, k_abs=5, train=200, val=20, test=40, rounds=10, max_steps=10,
            lr=1e-3, maxlen=768, warmup=2, noisy={0}, freerider={1})
STD50K5 = dict(n_clients=50, k_abs=5, total_train=20000, val=200, test=1000, rounds=200,
               max_steps=10, lr=1e-3, maxlen=512, warmup=3,
               noisy={0, 1, 2, 3, 4}, freerider={5, 6, 7, 8, 9})   # mixed = both together
RCFG = dict(SILO if SILO_LIKE else STD50K5)
for k in ("n_clients", "k_abs", "train", "total_train", "val", "test", "rounds",
          "max_steps", "maxlen", "warmup"):
    if os.environ.get(k.upper()):
        RCFG[k] = int(os.environ[k.upper()])
if os.environ.get("LR"):
    RCFG["lr"] = float(os.environ["LR"])
NOISY_RATE = float(os.environ.get("NOISY_RATE", "1.0"))

MODEL_CFG = {"1B": dict(batch=16, val_chunk=10, val_maxlen=384),
             "3B": dict(batch=8, val_chunk=5, val_maxlen=384),
             "7B": dict(batch=4, val_chunk=2, val_maxlen=320)}
MCFG = dict(MODEL_CFG.get(SCALE, MODEL_CFG["1B"]))
for k in ("batch", "val_chunk", "val_maxlen"):
    if os.environ.get(k.upper()):
        MCFG[k] = int(os.environ[k.upper()])
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", str(2 * LORA_R)))

# gate defaults (spec §4.3; per-cell tuning is FORBIDDEN -- ablation cells only)
GATE = dict(burn_in=int(os.environ.get("BURN_IN", "3" if SILO_LIKE else "10")),
            tau=float(os.environ.get("TAU", "0.0")),
            min_obs=int(os.environ.get("MIN_OBS", "2")),
            probation_every=int(os.environ.get("PROBATION_EVERY", "5")),
            decay=float(os.environ.get("DECAY", "1.0")),
            z_c=float(os.environ.get("ZC", "1.5")),
            alpha_w=float(os.environ.get("ALPHA_W", "1.0")))

DOWNSTREAM = os.environ.get("DOWNSTREAM", "1" if REGIME == "std50k5" else "0") == "1"
MMLU_LIMIT = int(os.environ.get("MMLU_LIMIT", "0"))
MMLU_BATCH = int(os.environ.get("MMLU_BATCH", "16"))
V3 = os.environ.get("V3", "0") == "1"
SYNTH = os.environ.get("SYNTH_DATA", "0") == "1"

_CODES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR_ROOT = os.environ.get("RUNDIR_ROOT",
                             os.path.join(os.path.dirname(_CODES), "runs", "track_g", "rundirs"))

DEFAULT_ARMS = (["vanilla", "oracle_excl", "random_excl", "flirds_gate_v1",
                 "flirds_gate_v2", "flirds_zgate_v2", "flirds_w", "lossheur_gate_v2"]
                + (["oracleb_gate_v2"] if REGIME == "silo5" else [])
                + (["shapleyfl_gate_v2"] if REGIME == "std50k5" else []))
# flirds_gatew_v2 (V2w) is DELIBERATELY absent: CNN-first, LLM only after the §5-2
# promotion criteria pass -- then add it via ARMS explicitly.
ARMS = [a for a in os.environ.get("ARMS", "").split(",") if a] or DEFAULT_ARMS


# --------------------------------------------------------------------------- #
# threat wiring                                                               #
# --------------------------------------------------------------------------- #
def _ids(env, default):
    v = os.environ.get(env)
    return set(int(x) for x in v.split(",") if x) if v else set(default)


def threat_sets():
    """(noisy_ids, fr_ids, fr_mode) for the cell.  CORRUPT_IDS overrides the
    single-threat default; mixed uses NOISY_IDS + FR_IDS (std50k5-corrupt)."""
    if THREAT == "clean":
        return set(), set(), "zero"
    if THREAT == "noisy":
        return _ids("CORRUPT_IDS", RCFG["noisy"]), set(), "zero"
    if THREAT in ("frrand", "frzero"):
        return set(), _ids("CORRUPT_IDS", RCFG["freerider"]), \
            ("random" if THREAT == "frrand" else "zero")
    if THREAT == "mixed":
        return (_ids("NOISY_IDS", RCFG["noisy"]), _ids("FR_IDS", RCFG["freerider"]),
                os.environ.get("FR_MODE", "zero"))
    raise ValueError(f"unknown THREAT {THREAT!r} (poison is excluded by design)")


def _synth_records(n, tag):
    """Offline wiring-smoke records (SYNTH_DATA=1): trivial arithmetic QA."""
    rng = np.random.default_rng(hash(tag) % 2 ** 31)
    out = []
    for _ in range(n):
        a, b = int(rng.integers(1, 60)), int(rng.integers(1, 60))
        out.append({"prompt": f"Question: {a}+{b}?\nAnswer:", "completion": f" {a + b}"})
    return out


def build_data(seed, noisy_ids):
    """(clients, val_records, test_records) for the regime; noisy data folded in."""
    if SYNTH:                                          # local wiring smoke only
        from datasets import Dataset
        from flirds.data.corruptors import LLM_CORRUPTORS
        n = RCFG["n_clients"]
        clients = []
        for c in range(n):
            recs = _synth_records(RCFG.get("train", 40), f"c{c}s{seed}")
            if c in noisy_ids:
                recs = LLM_CORRUPTORS["answer_swap"](recs, c)
            clients.append(Dataset.from_list(recs))
        return clients, _synth_records(RCFG["val"], f"v{seed}"), \
            [{**r, "domain": "alpaca"} for r in _synth_records(RCFG["test"], f"t{seed}")]
    if REGIME == "silo5":
        return build(RCFG["n_clients"], RCFG["train"], RCFG["val"], RCFG["test"],
                     seed=seed, noisy=noisy_ids, noisy_rate=NOISY_RATE)
    if REGIME == "iid5":                               # silo5-matched totals (phase2 iid5 leg)
        n = RCFG["n_clients"]
        return build_alpaca_iid(n, total_train=RCFG["train"] * n, n_val=RCFG["val"] * n,
                                n_test=RCFG["test"] * n, seed=seed, noisy=noisy_ids,
                                noisy_rate=NOISY_RATE)
    return build_alpaca_iid(RCFG["n_clients"], total_train=RCFG["total_train"],
                            n_val=RCFG["val"], n_test=RCFG["test"], seed=seed,
                            noisy=noisy_ids, noisy_rate=NOISY_RATE)


# --------------------------------------------------------------------------- #
# model / FL helpers                                                          #
# --------------------------------------------------------------------------- #
def _load(device):
    """fp32 + eager LoRA (the estimator path).  SMOKE_MODEL=tiny-gpt2 builds a
    random-init 2-layer gpt2 from config -- OFFLINE wiring smoke only (values are
    noise; no weight download; pairs with SYNTH_DATA=1)."""
    if MODEL == "tiny-gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        m = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=64,
                                       vocab_size=len(tok),
                                       attn_implementation="eager")).to(device)
    else:
        tok = AutoTokenizer.from_pretrained(MODEL)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 attn_implementation="eager").to(device)
    m = get_peft_model(m, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                                     target_modules=TARGET if MODEL != "tiny-gpt2" else None,
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


def _fl(model, tok, clients, init, seed, select_fn=None, weights_fn=None,
        free_riders=frozenset(), fr_mode="zero", fr_scale=1e-3, rounds=None):
    model.load_state_dict(init, strict=False)
    return run_llm_fedavg_logs(model, tok, clients, rounds or RCFG["rounds"], RCFG["lr"],
                               RCFG["max_steps"], batch_size=MCFG["batch"],
                               max_length=RCFG["maxlen"],
                               sample_frac=min(1.0, RCFG["k_abs"] / len(clients)),
                               seed=seed, select_fn=select_fn, weights_fn=weights_fn,
                               free_riders=free_riders, free_rider_mode=fr_mode,
                               free_rider_scale=fr_scale)


def _benign_std(logs, free_riders=frozenset()):
    """Mean std of benign flattened updates (phase2_matrix frrand tuning target)."""
    keys = sorted(next(iter(logs[0][1].values()))[0].keys())
    return float(np.mean([float(_flat(dm[c][0], keys).std())
                          for _, dm in logs for c in dm if c not in free_riders]))


def _guard(model, raw_fn):
    """Online-scoring hygiene (ported from track_d): SFTTrainer leaves train mode +
    can re-register the input-require-grad embedding hook; the estimator HVP
    forbids both -- re-apply eval/clear per scoring call."""
    def fn(w_r, dm, players):
        model.eval()
        model.get_input_embeddings()._forward_hooks.clear()
        return raw_fn(w_r, dm, players)
    return fn


def oracleb_round_raw_fn(loss_fn, pkeys, n_clients, device):
    """(silo5 policy ceiling) exact per-round (b) sub-game Shapley as the gate's
    raw -- 2^|P_r| forwards/round (~32 at K=5), contribution-oriented."""
    def fn(w_r, dm, players):
        phi, _ = in_run_shapley_perround([(w_r, dm)], n_clients, loss_fn, pkeys, device)
        return [-phi[p] for p in players]
    return fn


def shapleyfl_gate_raw_fn(loss_fn, pkeys, device):
    """(std50k5 method contrast) ShapleyFL's UN-normalized per-round exact Shapley
    of the uniform-average submodel game (good->HIGH already; min-max is skipped
    BECAUSE it destroys the sign the gate needs).  A zero delta still dilutes the
    uniform average, so FR raw != 0 here -- the fidelity-vs-decision contrast."""
    from flirds.baselines.shapleyfl import shapleyfl_round_raw
    def fn(w_r, dm, players):
        sv = shapleyfl_round_raw(w_r, dm, players, None, None, device, loss_fn, pkeys)
        return [float(v) for v in np.asarray(sv, dtype=float)]
    return fn


# --------------------------------------------------------------------------- #
# per-round logging sink + gate accuracy                                      #
# --------------------------------------------------------------------------- #
def make_sink(rows, acc, n_clients):
    """Append one row per (round, client) for ALL n clients: participants carry
    (raw, weight), everyone carries the post-round (cum, n_obs) snapshot -- the
    phi_rounds.parquet schema (contribution orientation)."""
    def sink(r, players, raw, wmap, fallback):
        pset = {p: i for i, p in enumerate(players)}
        for c in range(n_clients):
            i = pset.get(c)
            rows.append(dict(round=r, client=c, participated=c in pset,
                             raw=float(raw[i]) if i is not None else float("nan"),
                             weight=float(wmap[c]) if i is not None else float("nan"),
                             cum=float(acc.cum[c]), n_obs=int(acc.n_obs[c]),
                             fallback=fallback))
    return sink


def gate_stats(rows, corrupt, arm, n_clients):
    """Micro precision/recall of the per-round EXCLUDED set vs the corrupt set,
    over rounds >= burn_in (select-gated arms read eligibility from the PREVIOUS
    round's cum/n_obs snapshot -- the state the round's selection actually used);
    V1 exclusion = participated with weight 0.  Also the clean false-exclusion
    count (excluded (round, client) pairs that are NOT corrupt)."""
    if not rows:
        return None
    burn = GATE["burn_in"] if "v2" in arm else 0       # V1 gates from round 0
    by_round = {}
    for x in rows:
        by_round.setdefault(x["round"], {})[x["client"]] = x
    rounds = sorted(by_round)
    tp = fp = fn = 0
    excl_pairs = []
    for r in rounds:
        if r < burn:
            continue
        prev = by_round.get(r - 1)
        ineligible = np.zeros(n_clients, dtype=bool)
        if "v2" in arm and prev is not None:           # pre-round eligibility snapshot
            obs = np.array([prev[i]["n_obs"] >= GATE["min_obs"] for i in range(n_clients)])
            cums = np.array([prev[i]["cum"] for i in range(n_clients)])
            if arm.endswith("_zgate_v2"):
                zs = np.zeros(n_clients)
                if obs.sum() >= 2:
                    zs[obs] = _zscores(cums[obs])
                ineligible = obs & (zs < -GATE["z_c"])
            else:
                ineligible = obs & (cums <= GATE["tau"])
        excluded = set()
        for c in range(n_clients):
            x = by_round[r][c]
            if x["participated"] and x["weight"] == 0.0 and not x["fallback"]:
                excluded.add(c)                        # aggregation-screened
            elif not x["participated"] and ineligible[c]:
                excluded.add(c)                        # selection-gated
        tp += len(excluded & corrupt)
        fp += len(excluded - corrupt)
        fn += len(corrupt - excluded)
        excl_pairs += [(r, c) for c in excluded]
    return dict(precision=(tp / (tp + fp) if tp + fp else None),
                recall=(tp / (tp + fn) if tp + fn else None),
                n_excluded_pairs=len(excl_pairs), false_excl_pairs=fp,
                n_fallback_rounds=sum(1 for r in rounds
                                      if any(by_round[r][c]["fallback"]
                                             for c in range(n_clients))))


# --------------------------------------------------------------------------- #
# arms                                                                        #
# --------------------------------------------------------------------------- #
def build_arm(arm, model, loss_fn, pkeys, lc, nums, device, corrupt, seed, rows):
    """(select_fn, weights_fn, acc) for one arm; fresh state per call (per seed).
    `rows` is the phi_rounds sink target (gate arms + vanilla observer only)."""
    n = RCFG["n_clients"]
    g = GATE
    flirds_raw = lambda: _guard(model, flirds_round_raw_fn(loss_fn, pkeys, n, device,
                                                           loss_chunks=lc))
    if arm == "vanilla":
        acc = SignAccumulator(n, decay=g["decay"])
        return None, make_observer_weights_fn(acc, flirds_raw(), nums,
                                              sink=make_sink(rows, acc, n)), acc
    if arm == "oracle_excl":
        return make_fixed_excl_select_fn(n, corrupt), None, None
    if arm == "random_excl":
        rng = np.random.default_rng(2000 + seed)
        rand = set(int(x) for x in rng.choice(n, size=len(corrupt), replace=False))
        print(f"  [random_excl] excluded={sorted(rand)}", flush=True)
        return make_fixed_excl_select_fn(n, rand), None, None
    if arm == "flirds_w":                              # existing soft contrast (C2/D wiring)
        sc = OnlineScorer(n, beta=0.5)
        return None, make_weights_fn(sc, flirds_raw(), nums, "multiplicative"), None

    raw_by_arm = {"flirds": flirds_raw,
                  "lossheur": lambda: _guard(model, lossheur_round_raw_fn(
                      loss_fn, pkeys, n, device)),
                  "oracleb": lambda: _guard(model, oracleb_round_raw_fn(
                      loss_fn, pkeys, n, device)),
                  "shapleyfl": lambda: _guard(model, shapleyfl_gate_raw_fn(
                      loss_fn, pkeys, device)),
                  # Track H score-source competition (runs/track_h/README.md §1;
                  # same gate policy, source swapped -- comfedsv is the per-round
                  # surrogate, see fl.score_providers docstring):
                  "flirds1st": lambda: _guard(model, flirds_round_raw_fn(
                      loss_fn, pkeys, n, device, second_order=False, loss_chunks=lc)),
                  "gtg": lambda: _guard(model, gtg_round_raw_fn(
                      loss_fn, pkeys, device, seed=seed)),
                  "fedsv": lambda: _guard(model, fedsv_round_raw_fn(
                      loss_fn, pkeys, device, seed=seed)),
                  "comfedsv": lambda: _guard(model, comfedsv_round_raw_fn(
                      loss_fn, pkeys, device, seed=seed))}
    provider = arm.split("_")[0]
    if provider not in raw_by_arm:
        raise ValueError(f"unknown arm {arm!r}")
    raw = raw_by_arm[provider]()
    acc = SignAccumulator(n, decay=g["decay"])
    sink = make_sink(rows, acc, n)

    if arm.endswith("_gate_v1"):                       # aggregation gate only
        return None, make_signgate_weights_fn(acc, raw, nums, tau=g["tau"], sink=sink), acc
    if arm.endswith("_zgate_v2"):
        return (make_zgate_select_fn(acc, g["burn_in"], c=g["z_c"], min_obs=g["min_obs"],
                                     probation_every=g["probation_every"]),
                make_zgate_weights_fn(acc, raw, nums, c=g["z_c"], sink=sink), acc)
    if arm.endswith("_gatew_v2"):                      # V2w (post-promotion only)
        return (make_signgate_select_fn(acc, g["burn_in"], tau=g["tau"],
                                        min_obs=g["min_obs"],
                                        probation_every=g["probation_every"]),
                make_gatedweight_weights_fn(acc, raw, nums, tau=g["tau"],
                                            alpha=g["alpha_w"], sink=sink), acc)
    if arm.endswith("_gate_v2"):
        return (make_signgate_select_fn(acc, g["burn_in"], tau=g["tau"],
                                        min_obs=g["min_obs"],
                                        probation_every=g["probation_every"]),
                make_signgate_weights_fn(acc, raw, nums, tau=g["tau"], sink=sink), acc)
    raise ValueError(f"unknown arm {arm!r}")


# --------------------------------------------------------------------------- #
# metrics (ported from track_d: _val_curve / _rounds_to_target / _downstream)  #
# --------------------------------------------------------------------------- #
def _val_curve(logs, model, loss_fn, pkeys, device):
    """curve[0]=init loss, curve[r]=loss entering round r, curve[-1]=deployed."""
    model.eval()
    model.get_input_embeddings()._forward_hooks.clear()
    pts = []
    with torch.no_grad():
        for w_r, _ in logs:
            pts.append(float(loss_fn({k: w_r[k].to(device) for k in pkeys}, {})))
        final = _final_lora_state(logs)
        pts.append(float(loss_fn({k: final[k].to(device) for k in pkeys}, {})))
    return pts, final


def _rounds_to_target(curve, target):
    for i, v in enumerate(curve):
        if v <= target:
            return i
    return None


def _downstream(model, tok, test_records, device):
    acc, _, _nq = mmlu_accuracy(model, tok, device, limit=MMLU_LIMIT, batch_size=MMLU_BATCH)
    gens = generate_completions(model, tok, [r["prompt"] for r in test_records], device,
                                max_new_tokens=128, batch_size=16, max_prompt_len=512)
    rouge = score_records(gens, test_records)["alpaca"]["rouge_l"]
    return {"mmlu": acc, "rouge_l": rouge}


# --------------------------------------------------------------------------- #
# persistence                                                                 #
# --------------------------------------------------------------------------- #
def _cell_name(arm, seed):
    nr = f"_nr{NOISY_RATE:g}" if NOISY_RATE != 1.0 else ""
    return os.environ.get("RUN_NAME") or f"{REGIME}_{THREAT}{nr}_{arm}_seed{seed}"


def persist(arm, seed, metrics, rows, acc, timing, corrupt_cfg):
    if os.environ.get("PERSIST", "1") != "1":
        return
    config = {"track": "G", "regime": REGIME, "threat": THREAT, "arm": arm, "seed": seed,
              "scale": SCALE, "model": MODEL, "rcfg": {k: (sorted(v) if isinstance(v, set)
                                                          else v) for k, v in RCFG.items()},
              "mcfg": MCFG, "lora": {"r": LORA_R, "alpha": LORA_ALPHA},
              "gate": GATE, "corrupt": corrupt_cfg, "noisy_rate": NOISY_RATE,
              "synth_data": SYNTH, "downstream": DOWNSTREAM,
              "orientation": {"phi.parquet": "suspicion (good->low; repo convention)",
                              "phi_rounds.parquet": "raw/cum contribution (good->high = -phi)"}}
    try:
        rl = RunLogger(RUNDIR_ROOT, _cell_name(arm, seed), config, repo_root=_CODES)
        if acc is not None:                            # final cumulative, repo orientation
            rl.save_phi([{"seed": seed, "arm": arm, "client": int(c),
                          "phi": -float(acc.cum[c]), "n_obs": int(acc.n_obs[c])}
                         for c in range(RCFG["n_clients"])])
        if rows:
            rl.save_phi(rows, fname="phi_rounds.parquet")
        rl.save_metrics(metrics)
        if timing is not None:
            rl.save_timing(timing)
        print(f"  [persist] {rl.dir}  ({len(rows)} phi_rounds rows)", flush=True)
    except Exception as e:
        print(f"  [persist] WARNING: save failed ({e!r}); stdout has results", flush=True)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = ([int(s) for s in os.environ["SEEDS"].split(",")] if os.environ.get("SEEDS")
             else [int(os.environ.get("SEED", "0"))])   # pilot-first: default seed 0 only
    n = RCFG["n_clients"]
    noisy_ids, fr_ids, fr_mode = threat_sets()
    corrupt = noisy_ids | fr_ids
    arms = [a for a in ARMS if not (THREAT == "clean" and a in ("oracle_excl", "random_excl"))]
    corrupt_cfg = {"noisy_ids": sorted(noisy_ids), "fr_ids": sorted(fr_ids),
                   "fr_mode": fr_mode, "corrupt": sorted(corrupt)}

    print(f"=== Track G | {SCALE} {REGIME} {THREAT} | N={n} K={RCFG['k_abs']}/round "
          f"R={RCFG['rounds']} lr={RCFG['lr']} | corrupt={sorted(corrupt)} fr_mode={fr_mode} "
          f"nr={NOISY_RATE:g} | gate={GATE} | arms={arms} | seeds={seeds} "
          f"| downstream={DOWNSTREAM} V3={V3} ===", flush=True)

    tok, model, init, pkeys = _load(device)
    for seed in seeds:
        seed_everything(seed)
        clients, val, test_records = build_data(seed, noisy_ids)
        nums = [len(c) for c in clients]
        val_chunks = build_val_batches(val, tok, MCFG["val_maxlen"], device, MCFG["val_chunk"])
        loss_fn, _pk, lc = make_llm_loss(model, val_chunks, device)

        fr_scale = 1e-3
        if fr_ids and fr_mode == "random":             # phase2 frrand: benign-std tuning
            warm, t_w = _timed(lambda: _fl(model, tok, clients, init, seed,
                                           rounds=RCFG["warmup"]), device)
            fr_scale = _benign_std(warm) * (3 ** 0.5) * float(os.environ.get("DOSE_MULT", "1.0"))
            print(f"  [frrand] warmup {RCFG['warmup']}r {t_w:.0f}s -> scale={fr_scale:.2e}",
                  flush=True)
            del warm

        target = None
        vanilla_cum = None
        for arm in arms:
            pt = PhaseTimer(device, n_gpus=int(os.environ.get("N_GPUS", "1")))
            rows = []
            sel_fn, wts_fn, acc = build_arm(arm, model, loss_fn, pkeys, lc, nums, device,
                                            corrupt, seed, rows)
            with pt.phase("fl+online-scoring"):        # local train + per-round raw together
                logs = _fl(model, tok, clients, init, seed, select_fn=sel_fn,
                           weights_fn=wts_fn, free_riders=fr_ids, fr_mode=fr_mode,
                           fr_scale=fr_scale)          # (peak = the estimator HVP)
            t_fl = pt.phases["fl+online-scoring"]["s"]
            with pt.phase("val-curve"):
                curve, final = _val_curve(logs, model, loss_fn, pkeys, device)
            gs = gate_stats(rows, corrupt, arm, n) if rows else None
            if arm == "vanilla":
                target = curve[-1]
                vanilla_cum = None if acc is None else acc.cum.copy()
            metrics = dict(arm=arm, regime=REGIME, threat=THREAT, seed=seed,
                           corrupt=sorted(corrupt), final_val_loss=curve[-1],
                           val_curve=curve, train_s=t_fl,
                           rounds_to_target=(_rounds_to_target(curve, target)
                                             if target is not None else None),
                           vanilla_target=target, gate=gs, gate_cfg=GATE)
            if DOWNSTREAM:
                model.load_state_dict(final, strict=False)
                with pt.phase("downstream"):
                    metrics["downstream"] = _downstream(model, tok, test_records, device)
            g_line = (f" P={gs['precision']} R={gs['recall']} fx={gs['false_excl_pairs']}"
                      if gs else "")
            print(f"  [{arm:18s}] seed{seed} val_loss={curve[-1]:.4f} "
                  f"r2t={metrics['rounds_to_target']}{g_line} ({t_fl:.0f}s)"
                  + (f" mmlu={metrics['downstream']['mmlu']:.4f}"
                     f" rouge={metrics['downstream']['rouge_l']:.4f}"
                     if DOWNSTREAM else ""), flush=True)
            persist(arm, seed, metrics, rows, acc, pt.to_timing(), corrupt_cfg)
            del logs
            if device == "cuda":
                torch.cuda.empty_cache()

        if V3 and vanilla_cum is not None:
            zc = _zscores(vanilla_cum)
            v3_variants = {"v3_sign": [i for i in range(n) if vanilla_cum[i] > GATE["tau"]],
                           "v3_z": [i for i in range(n) if zc[i] >= -GATE["z_c"]]}
            rng = np.random.default_rng(3000 + seed)
            v3_variants["v3_random"] = sorted(int(x) for x in rng.choice(
                n, size=len(v3_variants["v3_sign"]), replace=False))
            cache = {}
            for vname, kept in v3_variants.items():
                if not kept:                           # everyone gated out -> no retrain
                    print(f"  [{vname:18s}] seed{seed} kept=EMPTY -> retrain skipped "
                          f"(reported as-is)", flush=True)
                    persist(vname, seed, dict(arm=vname, regime=REGIME, threat=THREAT,
                                              seed=seed, corrupt=sorted(corrupt), kept=[],
                                              final_val_loss=None, vanilla_target=target,
                                              gate_cfg=GATE),
                            [], None, None, corrupt_cfg)
                    continue
                key = frozenset(kept)
                pt = PhaseTimer(device, n_gpus=int(os.environ.get("N_GPUS", "1")))
                if key not in cache:
                    # deployment semantics: threat stays ACTIVE on kept clients --
                    # exclusion is the ONLY intervention (clean-retrain ruler = Exp A2);
                    # free-rider ids re-indexed to the subset positions.
                    kept_sorted = sorted(kept)
                    kept_fr = frozenset(i for i, c in enumerate(kept_sorted) if c in fr_ids)
                    (logs_k), t_k = _timed(lambda: _fl(
                        model, tok, [clients[c] for c in kept_sorted], init, seed,
                        free_riders=kept_fr, fr_mode=fr_mode, fr_scale=fr_scale), device)
                    curve_k, _fk = _val_curve(logs_k, model, loss_fn, pkeys, device)
                    cache[key] = (curve_k[-1], t_k)
                    del logs_k
                    if device == "cuda":
                        torch.cuda.empty_cache()
                vl, t_k = cache[key]
                pt.record("v3-retrain", t_k)
                print(f"  [{vname:18s}] seed{seed} kept={sorted(kept)} val_loss={vl:.4f}",
                      flush=True)
                persist(vname, seed, dict(arm=vname, regime=REGIME, threat=THREAT,
                                          seed=seed, corrupt=sorted(corrupt),
                                          kept=sorted(kept), final_val_loss=vl,
                                          vanilla_target=target, gate_cfg=GATE),
                        [], None, pt.to_timing(), corrupt_cfg)

        del loss_fn, val_chunks
        if device == "cuda":
            torch.cuda.empty_cache()
    print("\nTRACK G DONE (report order: [1] performance delta + recovery vs oracle_excl; "
          "[2] rounds-to-target; [3] gate precision/recall + clean false-exclusions -- "
          "compare against the §2.1 prediction table, misses reported as-is).", flush=True)


if __name__ == "__main__":
    main()
