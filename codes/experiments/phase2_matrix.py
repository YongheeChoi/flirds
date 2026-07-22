"""Phase 2 step5 detection/valuation MATRIX (3 threats x regimes x alpha x seeds).

The step5 grid (§3.9): extends the validated phase1_baseline_compare comparator into a
THREAT-looped matrix.  For each threat it builds the matched trajectory, runs every method
FEASIBLE for the regime on the shared frozen `logs`, and reports the three §3.9 headline
columns -- detection AUROC + Spearman vs the (b) oracle (or Flirds proxy-truth) + runtime.

THREATS (each its own trajectory; "category-together" = every method runs on every threat,
so the matrix shows each detector winning on-threat and losing off-threat):
  noisy             answer_swap corrupt client(s), normal FL                (data-quality)
  freerider_random  fabricated update U(-s,s) @ benign-std, normal FL       (free-rider, evasion)
  freerider_zero    fabricated update = 0, normal FL                        (free-rider, trivial floor)
  poison            D2b synthesis: benign FL -> attacker trains backdoor X from G ->
                    single-shot model-replacement attack round gamma*(X-G)  (backdoor)

REGIMES (env REGIME):
  silo5      N=5 cross-silo, full participation; (b)=exact 2^N, all coalition baselines run.
  gsm50k5    Track H R4 stage (N=50 IID GSM8K, K=5/round; spec runs/track_h/README.md §1.6).
             (b)=per-round decomposition (2^K=32/round -> exact), full method suite incl.
             coalition baselines by default -- the R4 fidelity cell (EXPERIMENT_PLAN L2).
             poison is out of scope (threat scope: clean/noisy/freerider_zero).
  device100  N=100 cross-device, K=10, Dirichlet(ALPHA) domain mixtures.  The 2^N methods
             (Banzhaf, exact-(b)) drop out; (b)=per-round decomposition, ComFedSV is the
             partial-participation Shapley baseline.  GTG/FedSV/ShapleyFL/(b) are ~2^K/round
             (= the oracle cost) -> they gate behind COALITION/ORACLE_B (run at the alpha=0.5
             anchor only); the cheap methods (Flirds/Flirds1st/loss-heur/ComFedSV/detectors)
             run every alpha with Flirds as the proxy-truth for Spearman off-anchor.

DETECTORS (threat-matched suite, all run on every threat):
  FLDetector / STD-DAGMM   model-free (logs only)
  FLTrust                  val-gradient cosine (loss_fn/pkeys)
  FedDQC                   on-device data quality (client data + base model; matched to noisy)

SCALE (task8): SMOKE_MODEL selects 1B/3B/7B -- 1B/3B = meta-llama/Llama-3.2-{1B,3B}-Instruct,
7B = meta-llama/Llama-2-7b-hf (plan task8; also the FL-LLM literature-standard 7B, so the
Track-D comparability arm reuses this rung); the per-scale memory knobs (batch/val_chunk/
val_maxlen) are env-overridable and change ONLY peak memory, not the values (exact chunk-sum,
fp32 throughout -- 7B needs no bf16 here, the (b)/estimator path is fp32; bf16 is the deferred
(a) retrain oracle, which 7B does not run).

Run from codes/ (env-parameterized per cell; seeds shard across GPUs 0-3 like phase1):
  # cheap tier -- N=5 cross-silo, all 4 threats, all methods, 3 seeds:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. REGIME=silo5 python -u experiments/phase2_matrix.py
  # one threat + one seed (batch across GPUs):
  ... REGIME=silo5 THREAT=noisy SEED=0 python -u experiments/phase2_matrix.py
  # POISON needs the full D2b INSTALL config (lr=2e-3 / batch=8 / EPOCHS=5 / frac=0.8; the attacker's
  # local model must install the backdoor), SEPARATE from the lr=1e-3 valuation threats.  silo5 -> ASR 1.0;
  # device100 -> ASR 0.75 at per_client=300 (240 poisoned > install threshold) + R=60 (converged G):
  ... REGIME=silo5     THREAT=poison LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=0.8 SEED=0 python -u experiments/phase2_matrix.py
  ... REGIME=device100 THREAT=poison LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=0.8 ROUNDS=60 MAX_STEPS=10 SEED=0 python -u experiments/phase2_matrix.py
  # cross-device alpha-sweep cell (cheap methods + detectors only, Flirds proxy-truth):
  ... REGIME=device100 ALPHA=0.1 SEED=0 python -u experiments/phase2_matrix.py
  # cross-device alpha=0.5 ANCHOR (turn the (b) oracle + coalition baselines on):
  ... REGIME=device100 ALPHA=0.5 ORACLE_B=1 COALITION=1 SEED=0 python -u experiments/phase2_matrix.py
  # R4 fidelity cell (gsm50k5; nr defaults to the R4 dose 0.7; one threat/seed per process):
  ... REGIME=gsm50k5 THREAT=noisy SEED=0 python -u experiments/phase2_matrix.py
"""
import os
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from flirds.backends.llm import make_llm_loss
from flirds.baselines.banzhaf import in_run_banzhaf
from flirds.baselines.comfedsv import comfedsv_from_logs
from flirds.baselines.feddqc import feddqc_scores
from flirds.baselines.fedif import fedif_from_logs
from flirds.baselines.fedsv import fedsv_from_logs
from flirds.baselines.fldetector import fldetector_from_logs
from flirds.baselines.fltrust import fltrust_from_logs
from flirds.baselines.gtg import gtg_from_logs
from flirds.baselines.ripple import _flat
from flirds.baselines.shapleyfl import BETA as SFL_BETA, shapleyfl_from_logs
from flirds.baselines.std_dagmm import std_dagmm_from_logs
from flirds.core.flirds_estimator import flirds_values
from flirds.data.corruptors import BACKDOOR_TRIGGER
from flirds.data.llm import build, build_alpaca_iid, build_crossdevice, build_gsm8k_iid, build_val_batches
from flirds.eval.generate import backdoor_asr
from flirds.eval.metrics import cosine_distance, euclidean_distance, max_difference, pearson
from flirds.fl.llm_server import client_optimizer, run_llm_fedavg_logs
from flirds.oracle.in_run_sv import (in_run_shapley, in_run_shapley_perround,
                                     in_run_singletons)
from flirds.repro import seed_everything
from flirds.hf_pin import rev
from flirds.run_logger import RunLogger
from flirds.timing import PhaseTimer

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
SCALE = ("7B" if "Llama-2-7b" in MODEL else
         MODEL.split("-")[-2] if "Llama-3.2-" in MODEL else MODEL.split("/")[-1])  # "1B"/"3B"/"7B"
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ORDER = ["medical", "legal", "finance", "math", "general"]

REGIME = os.environ.get("REGIME", "silo5")
SILO_LIKE = REGIME in ("silo5", "iid5")    # N=5 full-participation: (b)=exact 2^N, Banzhaf on, no ComFedSV.
#   silo5 = 5-domain non-IID stage; iid5 = alpaca IID stage (same N=5 config, only the data builder differs)
#   -> the corruption-axis x non-IID-axis matrix: {iid5, silo5} x {clean, noisy, free-rider, poison}.
GSM_LIKE = REGIME == "gsm50k5"             # R4 stage: partial participation -> per-round (b), ComFedSV partial.
ALPHA = float(os.environ.get("ALPHA", "0.5"))
BD_TARGET = os.environ.get("BD_TARGET", "delicious")       # single-token target (D1/D2 install regime)
POISON_FRAC = float(os.environ.get("POISON_FRAC", "0.5"))  # clean-preserving knob (D1: 0.5-0.8)
POISON_TRAIN = int(os.environ.get("POISON_TRAIN", "1000"))  # poison-threat client train size (D1/D2
#   install needs ~1000; the valuation config train=200 is too small to install -> the poison
#   trajectory overrides to this (silo5; device100 attacker is per_client-sized -- see build_trajectory).
ATTACKER_EPOCHS = float(os.environ.get("EPOCHS", "3"))     # attacker local epochs (D1 install strength)
ATTACKER_LR = float(os.environ.get("ATTACKER_LR", "2e-3"))  # attacker install lr (D1/D2 validated; > FL lr)

# --- C-1~C-5 mitigation experiments (removal_dose; all default to no-op = current behavior) ---
NOISY_RATE = float(os.environ.get("NOISY_RATE",            # Exp B: graded answer_swap dose (1.0 = full swap = current)
                                  "0.7" if GSM_LIKE else "1.0"))   # gsm50k5 default = the R4 dose (nr0.7)
DOSE_MULT = float(os.environ.get("DOSE_MULT", "1.0"))      # Exp B: free-rider-random amplitude multiplier
REMOVAL = os.environ.get("REMOVAL", "0") == "1"            # Exp A2: removal/selection curve (extra retrains; silo only)
REMOVAL_METHODS = [m for m in os.environ.get("REMOVAL_METHODS", "").split(",") if m]  # [] = all val methods

# per-scale memory knobs (task8): the fp32 estimator HVP is O(seq^2) -> bigger models need a
# smaller val_chunk / batch.  chunk-sum is exact, so these change ONLY peak memory, not values.
MODEL_CFG = {"1B": dict(batch=16, val_chunk=10, val_maxlen=384),
             "3B": dict(batch=8, val_chunk=5, val_maxlen=384),
             "7B": dict(batch=4, val_chunk=2, val_maxlen=320)}
MCFG = dict(MODEL_CFG.get(SCALE, MODEL_CFG["1B"]))

# per-regime trajectory config (env-overridable; the corrupt sets place the threat).
SILO = dict(n_clients=5, train=200, val=20, test=40, rounds=10, max_steps=10, lr=1e-3,
            maxlen=768, k_frac=1.0, warmup=2, noisy={0}, freerider={1}, attacker=0)
# per_client=300 (poison-compatible): a cross-device backdoor needs each attacker's local model to
# install (>= ~200 poisoned samples = D1 threshold), so per_client>=300/frac0.8 -> 240 poisoned; at the
# small per_client=40 the local X never installs and the threat is a no-op (verified 2026-06-09: ASR=0 at
# 40, ASR=0.75 at 300).  noisy/free-rider are unaffected by client size, so 300 unifies the regime.
DEVICE = dict(n_clients=100, per_client=300, pool=7000, val=10, test=40, rounds=30, max_steps=5,
              lr=1e-3, maxlen=768, k_frac=0.1, warmup=3,
              noisy={10, 30, 50, 70, 90}, freerider={10, 30, 50, 70, 90}, attacker=0)
# Track H R4 stage spec (runs/track_h/README.md §1.6; matches the Tier A rundir config exactly:
# N=50, K=5/round (k_frac=0.1), R=200, val=200 from the official test split, corrupt = clients 0-19).
GSM = dict(n_clients=50, val=200, test=0, rounds=200, max_steps=10, lr=1e-3,
           maxlen=512, k_frac=0.1, warmup=3,
           noisy=set(range(20)), freerider=set(range(20)), attacker=0)
RCFG = dict(SILO if SILO_LIKE else GSM if GSM_LIKE else DEVICE)   # iid5 shares silo5's N=5 config (only the data builder differs)
for k in ("rounds", "max_steps", "train", "per_client", "pool", "val", "test", "warmup"):
    if os.environ.get(k.upper()):
        RCFG[k] = int(os.environ[k.upper()])
for k in ("batch", "val_chunk", "val_maxlen"):
    if os.environ.get(k.upper()):
        MCFG[k] = int(os.environ[k.upper()])
if os.environ.get("LR"):                                       # the poison threat reproduces D2b at lr=2e-3
    RCFG["lr"] = float(os.environ["LR"])                       # (vs the valuation lr=1e-3 for noisy/free-rider)

_OUT = os.path.join(os.environ.get("TMPDIR") or "/tmp", f"flirds_matrix_{os.getpid()}")  # per-process (no cross-cell collision)
_CODES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../flirds/codes
RUNDIR_ROOT = os.environ.get("RUNDIR_ROOT",
                             os.path.join(os.path.dirname(_CODES), "runs", "phase2_matrix", "rundirs"))


# --------------------------------------------------------------------------- #
# setup + trajectory helpers                                                  #
# --------------------------------------------------------------------------- #
def _load(device):
    """fp32 + eager LoRA model (the (b)/estimator path; all methods share one load)."""
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev(MODEL))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                             attn_implementation="eager",
                                             revision=rev(MODEL)).to(device)
    seed_everything(0)                     # pin LoRA-A init (else entropy-seeded per process -> non-reproducible)
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=TARGET,
                                     lora_dropout=0.0, task_type="CAUSAL_LM"))
    init = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    return tok, m, init, list(init)


def _build_clients(seed, noisy=frozenset(), backdoor=frozenset(), train=None):
    """Regime-appropriate (clients, val, test); answer_swap=noisy, trigger->target=backdoor.
    `train` overrides the per-client train size (the poison threat needs the larger install size)."""
    if REGIME == "silo5":
        return build(RCFG["n_clients"], train or RCFG["train"], RCFG["val"], RCFG["test"], seed=seed,
                     noisy=noisy, backdoor=backdoor, noisy_rate=NOISY_RATE,
                     backdoor_kwargs=dict(target=BD_TARGET, poison_frac=POISON_FRAC))
    if REGIME == "iid5":                          # IID leg: alpaca shards, silo5-matched totals (x n_clients)
        n = RCFG["n_clients"]
        return build_alpaca_iid(n, total_train=(train or RCFG["train"]) * n,
                                n_val=RCFG["val"] * n, n_test=RCFG["test"] * n, seed=seed,
                                noisy=noisy, backdoor=backdoor, noisy_rate=NOISY_RATE,
                                backdoor_kwargs=dict(target=BD_TARGET, poison_frac=POISON_FRAC))
    if GSM_LIKE:                                  # R4 loader (no backdoor seam -- poison out of scope)
        return build_gsm8k_iid(RCFG["n_clients"], n_val=RCFG["val"], n_test=RCFG["test"],
                               seed=seed, noisy=noisy, noisy_rate=NOISY_RATE)
    return build_crossdevice(RCFG["n_clients"], alpha=ALPHA, per_client_train=train or RCFG["per_client"],
                             per_domain_pool=RCFG["pool"], per_domain_val=RCFG["val"],
                             per_domain_test=RCFG["test"], seed=seed, noisy=noisy, backdoor=backdoor,
                             noisy_rate=NOISY_RATE,
                             backdoor_kwargs=dict(target=BD_TARGET, poison_frac=POISON_FRAC))


def _fl(model, tok, clients, init, seed, rounds=None, **kw):
    """One FedAvg pass from init -> logs (regime k_frac, max_steps; passes free-rider kwargs)."""
    model.load_state_dict(init, strict=False)
    return run_llm_fedavg_logs(model, tok, clients, rounds or RCFG["rounds"], RCFG["lr"],
                               RCFG["max_steps"], batch_size=MCFG["batch"], max_length=RCFG["maxlen"],
                               sample_frac=RCFG["k_frac"], seed=seed, **kw)


def _benign_std(logs, free_riders=frozenset()):
    """Mean std of the benign clients' flattened updates (the Lin et al. free-rider tuning target)."""
    keys = sorted(next(iter(logs[0][1].values()))[0].keys())
    return float(np.mean([float(_flat(dm[c][0], keys).std())
                          for _, dm in logs for c in dm if c not in free_riders]))


def _final_global(init, logs, pkeys, device):
    """Deployed global G = init + sum_r (round FedAvg aggregate)."""
    G = {n: init[n].detach().clone().float().to(device) for n in pkeys}
    for _, dm in logs:
        tot = sum(nc for _, nc in dm.values())
        for k in dm:
            d, nc = dm[k]
            for n in pkeys:
                G[n] += (nc / tot) * d[n].float().to(device)
    return G


def _train_delta(model, tok, dataset, G, pkeys, seed, *, epochs=None, steps=None, lr=None):
    """Load G, locally train `dataset` (SGD mom=0), return the CPU delta (logged-delta convention)."""
    lr = lr or RCFG["lr"]
    model.load_state_dict({k: G[k] for k in pkeys}, strict=False)
    kw = dict(num_train_epochs=epochs) if epochs is not None else dict(max_steps=steps)
    cfg = SFTConfig(output_dir=_OUT, per_device_train_batch_size=MCFG["batch"], learning_rate=lr,
                    max_length=RCFG["maxlen"], lr_scheduler_type="constant", warmup_steps=0,
                    completion_only_loss=True, bf16=False, fp16=False, report_to="none",
                    logging_strategy="no", save_strategy="no", seed=seed, **kw)
    SFTTrainer(model=model, args=cfg, train_dataset=dataset, processing_class=tok,
               optimizer_cls_and_kwargs=client_optimizer(lr)).train()   # CLIENT_OPT (default SGD mom=0)
    after = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    return {k: (after[k] - G[k]).detach().cpu() for k in pkeys}


# --------------------------------------------------------------------------- #
# trajectory builders (one per threat)                                        #
# --------------------------------------------------------------------------- #
def build_trajectory(threat, seed, model, tok, init, pkeys, device, exclude=frozenset()):
    """Build the threat-matched frozen trajectory.

    Returns (logs, score_clients, corrupt_set, asr, val): `score_clients` is what FedDQC scores
    (the corrupt-data view), `corrupt_set` the AUROC labels, `asr` the deployed triggered-ASR
    (poison only, else None), `val` the shared §3.4 held-out set (threat-independent).
    `exclude` = client ids dropped from the poison attack-round cohort (Exp A2 poison removal,
    k=1): excluding the attacker skips the scaled inject -> the backdoor never deploys."""
    n = RCFG["n_clients"]
    if threat == "clean":                                          # no corruption: fidelity-only baseline
        clients, val, _ = _build_clients(seed)                     # (AUROC undefined, corrupt=empty set)
        logs = _fl(model, tok, clients, init, seed)
        return logs, clients, set(), None, val

    if threat == "noisy":
        corrupt = set(RCFG["noisy"])
        clients, val, _ = _build_clients(seed, noisy=corrupt)
        logs = _fl(model, tok, clients, init, seed)
        return logs, clients, corrupt, None, val

    if threat in ("freerider_random", "freerider_zero", "freerider_delta"):
        corrupt = set(RCFG["freerider"])
        clients, val, _ = _build_clients(seed)                     # clean data; the UPDATE is fabricated
        if threat == "freerider_zero":
            logs = _fl(model, tok, clients, init, seed, free_riders=corrupt, free_rider_mode="zero")
        elif threat == "freerider_delta":                          # E7: recycle previous-round aggregate (Lin delta-weights)
            logs = _fl(model, tok, clients, init, seed, free_riders=corrupt, free_rider_mode="delta")
        else:                                                      # random @ benign-std (evasion case)
            warm = _fl(model, tok, clients, init, seed, rounds=RCFG["warmup"])   # short CLEAN warmup
            scale = _benign_std(warm) * (3 ** 0.5) * DOSE_MULT     # U(-s,s) std = s/sqrt(3) -> benign std x dose (Exp B)
            logs = _fl(model, tok, clients, init, seed, free_riders=corrupt,
                       free_rider_mode="random", free_rider_scale=scale)
        return logs, clients, corrupt, None, val

    if threat == "poison":                                         # D2b synthesis
        if GSM_LIKE:
            raise ValueError("poison is out of scope for gsm50k5 (threat scope: clean/noisy/freerider_zero)")
        a = RCFG["attacker"]
        corrupt = {a}
        ptrain = POISON_TRAIN if SILO_LIKE else None               # device100 attacker is per_client-sized
        clean, val, test = _build_clients(seed, train=ptrain)
        bd, _, _ = _build_clients(seed, backdoor=corrupt, train=ptrain)   # attacker's trigger->target data
        logs = _fl(model, tok, clean, init, seed)                  # 1. benign FL -> logs + G
        G = _final_global(init, logs, pkeys, device)
        K = round(RCFG["k_frac"] * n)                              # cohort size = full-replacement gamma
        gamma = float(K)
        attack_dm = {}
        if a not in exclude:                                       # Exp A2 removal: dropping the attacker skips the inject
            delta0 = _train_delta(model, tok, bd[a], G, pkeys, seed,   # 2. backdoor X (install lr)
                                  epochs=ATTACKER_EPOCHS, lr=ATTACKER_LR)
            attack_dm[a] = ({k: gamma * delta0[k] for k in pkeys}, len(bd[a]))              # 3. scaled inject
        rng = np.random.default_rng(seed)
        benign_pool = [c for c in range(n) if c != a and c not in exclude]
        benign_ids = (benign_pool if SILO_LIKE
                      else rng.choice(benign_pool, size=K - 1, replace=False).tolist())
        for c in benign_ids:                                       # fresh benign deltas (the rest of the cohort)
            attack_dm[c] = (_train_delta(model, tok, clean[c], G, pkeys, seed, steps=RCFG["max_steps"]),
                            len(clean[c]))
        logs.append(({k: G[k].detach().cpu() for k in pkeys}, attack_dm))
        Gbd = _final_global(init, logs, pkeys, device)             # sanity: backdoor present in deployed G
        model.load_state_dict({k: Gbd[k] for k in pkeys}, strict=False)
        asr_dom = "alpaca" if REGIME == "iid5" else "medical"     # iid5 test is single-domain "alpaca"
        med = [r["prompt"] for r in test if r["domain"] == asr_dom]
        asr, _ = backdoor_asr(model, tok, med, BACKDOOR_TRIGGER, BD_TARGET, device)
        return logs, bd, corrupt, asr, val

    raise ValueError(f"unknown threat {threat!r}")


# --------------------------------------------------------------------------- #
# method computation (gated by regime + ORACLE_B/COALITION)                   #
# --------------------------------------------------------------------------- #
def _timed(fn, device):
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t


def compute_methods(logs, score_clients, model, tok, init, loss_fn, pkeys, lc, device, seed,
                    oracle_b, coalition):
    """Run the active methods on the frozen logs; return [(name, kind, vec, runtime)].

    kind: "val" (good->low phi; Spearman + AUROC) or "det" (high=suspicious; AUROC only).
    Calls are bit-identical to phase1_baseline_compare; the gates drop the infeasible/expensive
    methods.  (b) is exact 2^N at silo5, per-round at device100."""
    n = RCFG["n_clients"]
    out = []
    silo = SILO_LIKE                                 # N=5 full-participation -> exact 2^N (b), Banzhaf, no ComFedSV

    if oracle_b:
        b_fn = in_run_shapley if silo else in_run_shapley_perround
        (phi_b, _), t_b = _timed(lambda: b_fn(logs, n, loss_fn, pkeys, device), device)
        out.append(("(b)oracle", "val", np.asarray(phi_b), t_b))

    (phi_e, _), t_e = _timed(lambda: flirds_values(logs, loss_fn, pkeys, device, second_order=True,
                                                   n_clients=n, loss_chunks=lc), device)
    out.append(("Flirds", "val", np.asarray(phi_e), t_e))
    (phi_1, _), t_1 = _timed(lambda: flirds_values(logs, loss_fn, pkeys, device, second_order=False,
                                                   n_clients=n, loss_chunks=lc), device)
    out.append(("Flirds1st", "val", np.asarray(phi_1), t_1))
    phi_if, t_if = _timed(lambda: fedif_from_logs(logs, n, loss_fn, pkeys, device, loss_chunks=lc), device)
    out.append(("FedIF", "val", -np.asarray(phi_if, dtype=float), t_if))      # influence good->HIGH -> negate

    if coalition:
        phi_g, t_g = _timed(lambda: gtg_from_logs(logs, None, n, None, device, seed=seed,
                            loss_fn=loss_fn, pkeys=pkeys, round_trunc=0.0, eps=0.0), device)
        out.append(("GTG", "val", np.asarray(phi_g), t_g))
        phi_f, t_f = _timed(lambda: fedsv_from_logs(logs, None, n, None, device, seed=seed,
                            loss_fn=loss_fn, pkeys=pkeys, trunc_eps=0.0), device)
        out.append(("FedSV", "val", np.asarray(phi_f), t_f))
        phi_s, t_s = _timed(lambda: shapleyfl_from_logs(logs, None, n, None, device, beta=SFL_BETA,
                            loss_fn=loss_fn, pkeys=pkeys), device)
        out.append(("ShapleyFL", "val", -np.asarray(phi_s, dtype=float), t_s))   # good->high -> negate
        if silo:                                                                 # Banzhaf = 2^N -> silo only
            (phi_z, _), t_z = _timed(lambda: in_run_banzhaf(logs, n, loss_fn, pkeys, device), device)
            out.append(("Banzhaf", "val", np.asarray(phi_z), t_z))

    # ComFedSV: the partial-participation Shapley baseline (device100).  Now ALSO run at silo5
    # (partial=not silo -> partial=False) for full method coverage -- CAVEAT (review C-4): under
    # full participation the low-rank/partial approximation is trivially exact, a DEGENERATE
    # regime, so flag it in the results caption rather than read it as a win.
    phi_c, t_c = _timed(lambda: comfedsv_from_logs(logs, None, n, None, device, seed=seed,
                        loss_fn=loss_fn, pkeys=pkeys, partial=not silo), device)
    out.append(("ComFedSV", "val", -np.asarray(phi_c, dtype=float), t_c))        # loss-decrease util -> negate

    phi_h, t_h = _timed(lambda: in_run_singletons(logs, n, loss_fn, pkeys, device), device)
    out.append(("loss-heur", "val", phi_h, t_h))

    # ---- detectors (AUROC only) ----
    fld, t_fld = _timed(lambda: fldetector_from_logs(logs, n, device="cpu"), device)
    out.append(("FLDetector", "det", np.asarray(fld), t_fld))
    sdg, t_sdg = _timed(lambda: std_dagmm_from_logs(logs, n, seed=seed, device="cpu"), device)
    out.append(("STD-DAGMM", "det", np.asarray(sdg), t_sdg))
    flt, t_flt = _timed(lambda: fltrust_from_logs(logs, n, loss_fn, pkeys, device, loss_chunks=lc), device)
    out.append(("FLTrust", "det", np.asarray(flt), t_flt))
    model.load_state_dict(init, strict=False)                # FedDQC scores with the base model (smoke-matched)
    fdq, t_fdq = _timed(lambda: feddqc_scores(score_clients, model, tok, device, seed=seed), device)
    out.append(("FedDQC", "det", np.asarray(fdq), t_fdq))
    return out


# --------------------------------------------------------------------------- #
# removal / selection curves (Exp A2; silo full-participation only)           #
# --------------------------------------------------------------------------- #
def removal_retrain_curves(methods, score_clients, model, tok, init, loss_fn, pkeys, device, seed, n):
    """Exp A2: worst-first / best-first removal curves by ACTUAL retraining (the GPU cost).

    For every val method, rank clients by phi (suspicion orientation, good->LOW) and retrain
    clean FedAvg on the kept subset, scoring the deployed global's val loss.  U(kept) is cached
    by frozenset so it is computed ONCE and shared across methods/directions (<= 2^N retrains;
    methods that agree collapse to the same chain).  The retrain is clean (no fabricated update),
    so this is a game-INDEPENDENT downstream ruler (C-1/C-4 defence): noisy clients carry
    corrupted DATA -> removing them should lower val loss; free-riders carry clean data ->
    removing them is ~neutral (the honest nuance).
      worst_first -- drop the highest-phi (most-suspicious) client each step (kept n..1)
      best_first  -- drop the lowest-phi  (most-valuable)   client each step
    Returns ({method: {"worst_first": [[k_dropped, val_loss], ...], "best_first": [...]}},
    n_retrains, mean_retrain_s)."""
    cache, times = {}, []
    def util(kept):
        key = frozenset(kept)
        if key not in cache:
            logs_k, dt = _timed(lambda: _fl(model, tok, [score_clients[c] for c in sorted(kept)],
                                            init, seed), device)
            times.append(dt)
            G = _final_global(init, logs_k, pkeys, device)
            model.eval()                                  # _fl leaves train mode + the embed hook
            model.get_input_embeddings()._forward_hooks.clear()
            with torch.no_grad():
                cache[key] = float(loss_fn({k: G[k] for k in pkeys}, {}))
            del logs_k, G
            if device == "cuda":
                torch.cuda.empty_cache()
        return cache[key]
    def curve(order):
        pts, kept = [], list(range(n))
        for k in range(n):                                # kept sizes n, n-1, ..., 1 (skip empty)
            pts.append([k, util(kept)])
            kept.remove(order[k])
        return pts
    sel = REMOVAL_METHODS or [nm for nm, kind, _, _ in methods if kind == "val"]
    out = {}
    for name, kind, vec, _rt in methods:
        if kind != "val" or name not in sel:
            continue
        v = np.asarray(vec, dtype=float)
        out[name] = {"worst_first": curve(list(np.argsort(-v))),   # high phi (worst) dropped first
                     "best_first": curve(list(np.argsort(v)))}     # low phi (best) dropped first
    return out, len(cache), (sum(times) / len(times) if times else 0.0)


def poison_removal_asr(methods, seed, model, tok, init, pkeys, device, baseline_asr):
    """Exp A2 poison (k=1, locked): drop each method's TOP suspect (highest phi) from the attack
    cohort, rebuild the poison trajectory, and report the deployed backdoor ASR.  A method that
    flags the attacker as #1 drops ASR toward the clean baseline; a wrong flag leaves the attack
    intact.  Cached by excluded client (methods agreeing on the top suspect share one rebuild)."""
    sel = REMOVAL_METHODS or [nm for nm, kind, _, _ in methods if kind == "val"]
    cache, by_method = {}, {}
    for name, kind, vec, _rt in methods:
        if kind != "val" or name not in sel:
            continue
        top = int(np.argmax(np.asarray(vec, dtype=float)))         # most-suspicious client
        if top not in cache:
            _, _, _, asr_x, _ = build_trajectory("poison", seed, model, tok, init, pkeys, device,
                                                 exclude={top})
            cache[top] = asr_x
        by_method[name] = {"excluded": top, "asr": cache[top]}
    return {"baseline_asr": baseline_asr, "by_method": by_method}


# --------------------------------------------------------------------------- #
# reporting                                                                   #
# --------------------------------------------------------------------------- #
def _auroc(vec, corrupt, idx):
    lab = [1 if c in corrupt else 0 for c in idx]
    if 0 not in lab or 1 not in lab:
        return float("nan")
    return float(roc_auc_score(lab, [vec[c] for c in idx]))


def _rho(vec, truth, idx):
    return float(spearmanr([vec[c] for c in idx], [truth[c] for c in idx]).correlation)


def report(threat, methods, corrupt, logs, asr):
    """Per-(threat,seed) table: AUROC + Spearman vs truth + runtime per method.  Returns the
    dict {column: {method: value}} for seed aggregation."""
    selected = sorted({k for _, dm in logs for k in dm})          # scoreable clients (silo: all N)
    vecs = {name: vec for name, _, vec, _ in methods}
    truth_method = "(b)oracle" if "(b)oracle" in vecs else "Flirds"   # off-anchor -> Flirds proxy-truth
    truth = vecs[truth_method]
    truth_name = "(b)" if truth_method == "(b)oracle" else "Flirds*"

    seen_corrupt = sorted(corrupt & set(selected))
    print(f"\n[{threat}] selected={len(selected)}/{RCFG['n_clients']} corrupt={sorted(corrupt)} "
          f"seen_corrupt={seen_corrupt}" + (f" deployed-ASR={asr:.2f}" if asr is not None else ""))
    if not seen_corrupt:
        print("  (no corrupt client participated -> AUROC undefined; raise ROUNDS or change SEED)")

    res = {"auroc": {}, "spearman": {}, "pearson": {}, "cos_d": {}, "euc_d": {},
           "max_diff": {}, "runtime": {}}                  # rank + value-level (Pearson + GTG distances) fidelity
    print(f"  {'method':11s} {'AUROC':>7s} {'Spearman/'+truth_name:>14s} {'Pearson':>8s} "
          f"{'runtime':>9s}   corrupt-rank")
    for name, kind, vec, rt in methods:
        au = _auroc(vec, corrupt, selected)
        res["auroc"][name] = au
        res["runtime"][name] = rt
        rho, pear = ("(truth)" if name == truth_method else ""), ""
        if kind == "val" and name != truth_method:
            mv, tv = [vec[c] for c in selected], [truth[c] for c in selected]
            r = _rho(vec, truth, selected)
            res["spearman"][name] = r
            res["pearson"][name] = pearson(mv, tv)         # value-level (affine-invariant); key where rank saturates at +1
            res["cos_d"][name] = cosine_distance(mv, tv)
            res["euc_d"][name] = euclidean_distance(mv, tv)
            res["max_diff"][name] = max_difference(mv, tv)
            rho, pear = f"{r:+.3f}", f"{res['pearson'][name]:+.3f}"
        ranks = ",".join(str(sum(1 for j in selected if vec[j] > vec[c]))   # rank among SELECTED (AUROC set)
                         for c in seen_corrupt) or "-"
        print(f"  {name:11s} {au:7.3f} {rho:>14s} {pear:>8s} {rt:8.1f}s   {ranks}/{len(selected)}")
    return res


# Rundir identity (protocol §1.7).  First block = exactly what the cell NAME encodes, so
# "same name, different identity" is by construction a name-generation bug.  Second block =
# semantic knobs the name does NOT carry (sfl_beta was a source literal, removal/client_opt
# arrived later): listing them makes a change fail at launch instead of silently
# overwriting the canonical run -- which is how beta 0.5 -> 0.3 got in unannounced.
# Re-running a cell whose stored config predates one of these needs RUNDIR_REPLACE=1 once.
IDENTITY = ("scale", "model", "regime", "alpha", "threats", "seeds", "oracle_b",
            "coalition", "noisy_rate", "dose_mult",
            "sfl_beta", "removal", "client_opt")


def _persist(phi_rows, run_metrics, threats, seeds, oracle_b, coalition, timing=None):
    """Save the cell's per-client phi vectors + metrics + config + git/env provenance to a
    RunLogger run-dir (protocol §6) so any re-analysis needs no method re-run.  Best-effort:
    a save failure warns but never loses the printed .log results.  PERSIST=0 disables."""
    if os.environ.get("PERSIST", "1") != "1":
        return
    _abbr = {"freerider_random": "frrand", "freerider_zero": "frzero",
             "freerider_delta": "frdelta"}                               # canonical condition tokens
    _cond = "-".join(_abbr.get(t, t) for t in threats)
    _setting = REGIME if (SILO_LIKE or GSM_LIKE) else f"{REGIME}-a{ALPHA}"   # silo5 / iid5 / gsm50k5 / device100-a0.5
    _anchor = "_anchor" if (not SILO_LIKE and not GSM_LIKE and oracle_b and coalition) else ""
    _dose = ((f"_nr{NOISY_RATE:g}" if NOISY_RATE != 1.0 else "")     # Exp B dose tokens (default: none)
             + (f"_dm{DOSE_MULT:g}" if DOSE_MULT != 1.0 else ""))
    _seedtok = f"_s{seeds[0]}" if os.environ.get("SEED") else ""   # 1-seed/process shard -> distinct dir (else silent overwrite)
    name = os.environ.get("RUN_NAME") or f"{SCALE}_{_setting}_{_cond}{_anchor}{_dose}{_seedtok}"
    rcfg = {k: (sorted(v) if isinstance(v, (set, frozenset)) else v) for k, v in RCFG.items()}
    config = {"scale": SCALE, "model": MODEL, "regime": REGIME, "alpha": ALPHA,
              "threats": threats, "seeds": seeds, "oracle_b": oracle_b, "coalition": coalition,
              "noisy_rate": NOISY_RATE, "dose_mult": DOSE_MULT, "removal": REMOVAL,
              "client_opt": os.environ.get("CLIENT_OPT", "sgd"), "sfl_beta": SFL_BETA,
              "poison": {"bd_target": BD_TARGET, "poison_train": POISON_TRAIN,   # poison-cell knobs (self-describing)
                         "attacker_lr": ATTACKER_LR, "attacker_epochs": ATTACKER_EPOCHS},
              "rcfg": rcfg, "mcfg": MCFG,
              "env": {k: os.environ[k] for k in
                      ("POOL", "LR", "BATCH", "EPOCHS", "POISON_FRAC", "ROUNDS", "MAX_STEPS",
                       "PER_CLIENT", "VAL", "NOISY_RATE", "DOSE_MULT", "REMOVAL", "CLIENT_OPT",
                       "REMOVAL_METHODS") if k in os.environ}}
    try:
        rl = RunLogger(RUNDIR_ROOT, name, config, repo_root=_CODES, identity=IDENTITY)
        try:
            rl.save_phi(phi_rows)                                  # parquet (protocol convention)
        except Exception:                                         # CSV fallback if no parquet engine
            import pandas as pd
            pd.DataFrame(phi_rows).to_csv(rl._p("phi.csv"), index=False)
        rl.save_metrics(run_metrics)
        if timing is not None:
            rl.save_timing(timing)                                # §15.1 per-phase wall + GPU-hours + peak
        print(f"\n[persist] {rl.dir}  ({len(phi_rows)} phi rows)", flush=True)
    except Exception as e:
        print(f"\n[persist] WARNING: run-dir save failed ({e!r}); .log still has results", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pt = PhaseTimer(device, n_gpus=int(os.environ.get("N_GPUS", "1")))   # §15.1 timing.json substrate
    threats = ([os.environ["THREAT"]] if os.environ.get("THREAT")
               else ["clean", "noisy", "freerider_zero"] if GSM_LIKE
               else ["noisy", "freerider_random", "freerider_zero", "poison"])
    seeds = [int(os.environ["SEED"])] if os.environ.get("SEED") else [0, 1, 2]
    oracle_b = os.environ.get("ORACLE_B", "1" if (SILO_LIKE or GSM_LIKE) else "0") == "1"
    coalition = os.environ.get("COALITION", "1" if (SILO_LIKE or GSM_LIKE) else "0") == "1"

    print(f"=== step5 MATRIX | {SCALE} {REGIME}" + (f" alpha={ALPHA}" if not SILO_LIKE else "")
          + f" | R={RCFG['rounds']} K_frac={RCFG['k_frac']} lr={RCFG['lr']} batch={MCFG['batch']} "
          f"val_chunk={MCFG['val_chunk']} | ORACLE_B={oracle_b} COALITION={coalition} | "
          f"threats={threats} seeds={seeds} ===", flush=True)

    tok, model, init, pkeys = _load(device)
    agg = {t: [] for t in threats}                                # per-threat list of per-seed res dicts
    phi_rows, run_metrics = [], {}                                # run-dir persistence (phi vectors + metrics)
    for seed in seeds:
        for threat in threats:
            seed_everything(seed)
            with pt.phase("client-training"):                     # §15.1: the previously-untimed log generation
                logs, score_clients, corrupt, asr, val = build_trajectory(threat, seed, model, tok,
                                                                          init, pkeys, device)
            val_chunks = build_val_batches(val, tok, MCFG["val_maxlen"], device, MCFG["val_chunk"])
            loss_fn, _pk, lc = make_llm_loss(model, val_chunks, device)
            with pt.phase("valuation"):                           # all methods' phi-estimation + oracles (peak = HVP)
                methods = compute_methods(logs, score_clients, model, tok, init, loss_fn, pkeys, lc,
                                          device, seed, oracle_b, coalition)
            print(f"\n----- seed {seed} -----", flush=True)
            res = report(threat, methods, corrupt, logs, asr)
            agg[threat].append(res)
            sel = sorted({k for _, dm in logs for k in dm})       # scoreable clients (silo: all N)
            for nm, kind, vec, _rt in methods:                    # persist every method's raw phi vector
                phi_rows += [{"threat": threat, "seed": seed, "method": nm, "kind": kind,
                              "client": int(c), "phi": float(vec[c])} for c in sel]
            run_metrics[f"{threat}_seed{seed}"] = {
                "auroc": res["auroc"], "spearman": res["spearman"], "pearson": res["pearson"],
                "cos_d": res["cos_d"], "euc_d": res["euc_d"], "max_diff": res["max_diff"],
                "runtime": res["runtime"],
                "corrupt": sorted(int(x) for x in corrupt), "selected": [int(c) for c in sel],
                "asr": asr}
            if REMOVAL and SILO_LIKE:                             # Exp A2: removal/selection curves (extra retrains)
                mk = f"{threat}_seed{seed}"
                if threat == "poison":
                    pr = poison_removal_asr(methods, seed, model, tok, init, pkeys, device, asr)
                    run_metrics[mk]["poison_removal"] = pr
                    print(f"  [removal] poison k=1 baseline-ASR={asr} -> "
                          + ", ".join(f"{m}:excl{d['excluded']}=ASR{d['asr']:.2f}"
                                      for m, d in pr["by_method"].items()), flush=True)
                else:
                    rc, nrt, mrt = removal_retrain_curves(methods, score_clients, model, tok, init,
                                                          loss_fn, pkeys, device, seed, RCFG["n_clients"])
                    run_metrics[mk]["removal_curve"] = rc
                    run_metrics[mk]["removal_orient"] = "val_loss (lower=better); phi good->low"
                    run_metrics[mk]["removal_retrain_s"] = mrt
                    print(f"  [removal] {threat}: {nrt} distinct retrains "
                          f"({mrt:.1f}s/retrain) over {len(rc)} methods x 2 dirs", flush=True)
            del loss_fn
            if device == "cuda":
                torch.cuda.empty_cache()

    if len(seeds) > 1:
        print(f"\n=== aggregate (mean+/-std over {len(seeds)} seeds) ===", flush=True)
        for threat in threats:
            runs = agg[threat]
            names = list(runs[0]["auroc"])                        # method order from the first run
            print(f"\n[{threat}]")
            for name in names:
                au = [r["auroc"][name] for r in runs]
                line = f"  {name:11s} AUROC={np.nanmean(au):.3f}+/-{np.nanstd(au):.3f}"
                if name in runs[0]["spearman"]:
                    sp = [r["spearman"][name] for r in runs]
                    line += f"  Spearman={np.nanmean(sp):+.3f}+/-{np.nanstd(sp):.3f}"
                rt = [r["runtime"][name] for r in runs]
                line += f"  runtime={np.mean(rt):.1f}s"
                print(line)
    _persist(phi_rows, run_metrics, threats, seeds, oracle_b, coalition, timing=pt.to_timing())
    print("\nMATRIX DONE  (matched detector should top AUROC on-threat; Flirds/valuation track the "
          "(b) oracle; off-threat detectors degrade -- the §3.9 separation story).", flush=True)


if __name__ == "__main__":
    main()
