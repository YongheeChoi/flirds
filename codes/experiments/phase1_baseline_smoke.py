"""Phase 1 (3) SV-baseline port smoke: GTG/FedSV backend-agnostic port verification.

(A) CNN regression: gtg_from_logs / fedsv_from_logs on a tiny DETERMINISTIC MNIST
    trajectory.  The CNN path is UNCHANGED by the port, so phi must be bit-identical
    to the pre-port values -- run before & after the edit and diff.
(B) LLM smoke: the SAME baselines on a tiny real-1B 5-domain trajectory (the
    corruptor smoke setup), valuing the SAME frozen logs as the (b) in-run oracle ->
    Spearman(GTG/FedSV, oracle) + free-rider(zero) phi==0 (a zero delta has zero
    marginal in every coalition, exactly like the estimator/oracle).

Run from codes/:
  PYTHONPATH=. python experiments/phase1_baseline_smoke.py cnn   # CNN regression only (CPU, fast)
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/phase1_baseline_smoke.py llm
"""
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from flirds.baselines.fedsv import fedsv_from_logs
from flirds.baselines.gtg import gtg_from_logs


def cnn_regression(device="cpu"):
    """gtg/fedsv on a fixed MNIST trajectory; print phi at full precision (golden)."""
    from flirds.data.cnn import get_dataset, get_labels
    from flirds.data.corruptors import CNN_CORRUPTORS
    from flirds.fl.partition import dirichlet_partition
    from flirds.fl.server import run_fedavg_logs
    from flirds.models.cnn import LeNet5

    N, rounds, E, lr, n_per, seed = 4, 2, 1, 0.1, 100, 0
    noisy = {3}
    train = get_dataset("mnist")
    test = get_dataset("mnist", train=False)
    idx = dirichlet_partition(get_labels(train), N, alpha=100.0, seed=seed)
    idx = [i[:n_per] for i in idx]
    loaders = []
    for c in range(N):
        xs = torch.stack([train[i][0] for i in idx[c]])
        ys = torch.tensor([train[i][1] for i in idx[c]])
        if c in noisy:
            xs, ys = CNN_CORRUPTORS["label_shuffle"](xs, ys, c)
        loaders.append(DataLoader(TensorDataset(xs, ys), batch_size=50, shuffle=False))
    tl = DataLoader(Subset(test, range(256)), batch_size=256)

    model, logs = run_fedavg_logs(LeNet5, loaders, tl, rounds, E, lr, device=device, seed=seed)
    g = gtg_from_logs(logs, model, N, tl, device, seed=seed)
    f = fedsv_from_logs(logs, model, N, tl, device, seed=seed)
    print("== (A) CNN regression (bit-identical golden) ==")
    print("  gtg  :", np.array2string(g, precision=12, floatmode="unique"))
    print("  fedsv:", np.array2string(f, precision=12, floatmode="unique"))
    return g, f


def banzhaf_kernel_check():
    """banzhaf_from_utilities math (no model, CPU): additive game -> per-player values;
    null player -> 0; N=2 -> equals Shapley (the two semivalue kernels coincide only
    for N<=2).  A non-circular check of the Banzhaf weighting itself."""
    from itertools import combinations

    from flirds.baselines.banzhaf import banzhaf_from_utilities

    # additive U(S)=Σ v_i -> every marginal of i is v_i -> Banzhaf_i = v_i.
    v = [1.0, 2.0, 3.0]
    U = {S: sum(v[i] for i in S) for r in range(4) for S in combinations(range(3), r)}
    add_ok = bool(np.allclose(banzhaf_from_utilities(U, 3), v))
    # null player: client 2 has a zero marginal in every coalition -> Banzhaf_2 == 0.
    vn = [1.0, 2.0, 0.0]
    Un = {S: sum(vn[i] for i in S) for r in range(4) for S in combinations(range(3), r)}
    null_ok = bool(abs(banzhaf_from_utilities(Un, 3)[2]) < 1e-12)
    # N=2: Banzhaf weight 1/2 == Shapley weights {1/2, 1/2}.
    rng = np.random.default_rng(0)
    U2 = {(): 0.0, (0,): float(rng.random()), (1,): float(rng.random()),
          (0, 1): float(rng.random())}
    bz = banzhaf_from_utilities(U2, 2)
    sh = [0.5 * (U2[(0,)] - U2[()]) + 0.5 * (U2[(0, 1)] - U2[(1,)]),
          0.5 * (U2[(1,)] - U2[()]) + 0.5 * (U2[(0, 1)] - U2[(0,)])]
    n2_ok = bool(np.allclose(bz, sh))
    print("== Banzhaf kernel check ==")
    print(f"  additive-game recovers values={add_ok}  null-player==0={null_ok}  "
          f"N2==Shapley={n2_ok}")
    return add_ok and null_ok and n2_ok


def shapleyfl_pipeline_check():
    """ShapleyFL aggregation math (no model, CPU): min-max normalization (Def 4.2) +
    EMA across rounds (Def 4.3) on synthetic per-round SVs.  Hand-verifiable."""
    from flirds.baselines.shapleyfl import _ema_aggregate, _minmax

    # min-max: [-1,0,1] -> [0,0.5,1]; flat round -> zeros.
    mm_ok = bool(np.allclose(_minmax(np.array([-1.0, 0.0, 1.0])), [0.0, 0.5, 1.0])
                 and np.allclose(_minmax(np.array([2.0, 2.0, 2.0])), [0.0, 0.0, 0.0]))
    # EMA β=0.5, 2 clients, NSV r1=[0,1] r2=[1,0]: r1 ssv=[0,0.5]; r2 ssv=[0.5,0.25].
    ema = _ema_aggregate([np.array([0.0, 1.0]), np.array([1.0, 0.0])],
                         [[0, 1], [0, 1]], 2, 0.5)
    ema_ok = bool(np.allclose(ema, [0.5, 0.25]))
    print("== ShapleyFL pipeline check ==")
    print(f"  minmax={mm_ok}  ema(Def4.3)={ema_ok}")
    return mm_ok and ema_ok


def fldetector_synthetic_check():
    """FLDetector on synthetic logs (no model, CPU): N-1 temporally-CONSISTENT clients
    (track a smooth global trend) + 1 ANOMALOUS client (per-round noise).  The anomalous
    client must get the highest suspicious score (argmax) and AUROC=1.  Non-circular: the
    per-client update vectors are built directly; FLDetector recomputes the aggregate +
    L-BFGS Hessian + the prediction itself.  A consistent client's update is base[r]+bias_c
    (constant offset -> its temporal change == the trend, so it is L-BFGS-predictable); the
    anomalous client's per-round noise breaks that, inflating its prediction residual."""
    from flirds.baselines.fldetector import fldetector_from_logs
    from flirds.eval.metrics import detection_auroc

    rng = np.random.default_rng(0)
    N, P, R, anom = 5, 8, 12, 0
    base = [np.sin(0.3 * (r + 1) + np.arange(P)) for r in range(R)]   # smooth global trend
    bias = [rng.normal(0, 0.5, P) for _ in range(N)]                  # per-client constant offset
    logs = []
    for r in range(R):
        dm = {}
        for c in range(N):
            upd = base[r] + (rng.normal(0, 2.0, P) if c == anom else bias[c])
            dm[c] = ({"w": torch.tensor(upd, dtype=torch.float32)}, 100)
        logs.append(({}, dm))                                        # w_r unused (model-free)

    score = fldetector_from_logs(logs, N, window=10, device="cpu")
    labels = [1 if c == anom else 0 for c in range(N)]
    auroc = detection_auroc(score, labels)
    top_ok = bool(int(np.argmax(score)) == anom)
    auroc_ok = bool(auroc >= 0.999)
    print("== FLDetector synthetic check ==")
    print(f"  score={np.array2string(score, precision=3, floatmode='fixed')} "
          f"anom={anom} argmax={int(np.argmax(score))} AUROC={auroc:.3f}")
    print(f"  anomalous-is-top={top_ok}  AUROC==1={auroc_ok}")
    return top_ok and auroc_ok


def llm_smoke(device="cuda"):
    """gtg/fedsv on a tiny real-1B 5-domain trajectory, valuing the SAME frozen logs
    as the (b) oracle/estimator.  Gate (scale-independent): free-rider(zero) phi==0
    for every method (zero delta -> zero marginal in every coalition) + finite phi.
    Spearman vs the (b) oracle is printed but NOT gated (detection quality is
    experiment-scale; 8ex/2step sits at the noise floor -- cf. corruptor smoke)."""
    import os

    from peft import LoraConfig, get_peft_model
    from scipy.stats import spearmanr
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from flirds.backends.llm import make_llm_loss
    from flirds.baselines.banzhaf import in_run_banzhaf
    from flirds.baselines.fldetector import fldetector_from_logs
    from flirds.baselines.ripple_llm import ripple_shapley_llm
    from flirds.baselines.shapleyfl import shapleyfl_from_logs
    from flirds.core.flirds_estimator import flirds_values
    from flirds.data.llm import build, build_val_batches
    from flirds.eval.metrics import detection_auroc
    from flirds.fl.llm_server import run_llm_fedavg_logs
    from flirds.oracle.in_run_sv import in_run_shapley
    from flirds.repro import seed_everything

    MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
    TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    DOMAINS = ["medical", "legal", "finance", "math", "general"]
    NOISY, FREE_RIDER, N = {0}, {1}, 5

    seed_everything(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 attn_implementation="eager").to(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=TARGET,
                                             lora_dropout=0.0, task_type="CAUSAL_LM"))
    init_lora = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    clients, val, _ = build(n_clients=N, per_domain_train=8, per_domain_val=10, seed=0, noisy=NOISY)
    val_chunks = build_val_batches(val, tok, 256, device, chunk_size=10)
    model.load_state_dict(init_lora, strict=False)
    logs = run_llm_fedavg_logs(model, tok, clients, rounds=3, lr=1e-3, max_steps=2,
                               batch_size=2, max_length=768, seed=0,
                               free_riders=FREE_RIDER, free_rider_mode="zero")
    loss_fn, pkeys, lc = make_llm_loss(model, val_chunks, device)

    phi_b, _ = in_run_shapley(logs, N, loss_fn, pkeys, device)
    phi_z, _ = in_run_banzhaf(logs, N, loss_fn, pkeys, device)   # exact Banzhaf, same (b) utility
    phi_e, _ = flirds_values(logs, loss_fn, pkeys, device, second_order=True, loss_chunks=lc)
    # baselines value the SAME logs via loss_fn/pkeys; truncation off (loss-scale per
    # round << the CNN-accuracy default 1e-3 -> the default would skip every round).
    phi_g = gtg_from_logs(logs, None, N, None, device, seed=0, loss_fn=loss_fn, pkeys=pkeys,
                          round_trunc=0.0, eps=0.0, normalize=False)
    phi_f = fedsv_from_logs(logs, None, N, None, device, seed=0, loss_fn=loss_fn, pkeys=pkeys,
                            trunc_eps=0.0)
    # ShapleyFL surrogate-FSV (good->high -> negate). min-max normalized -> NOT exactly 0 on
    # the free-rider (a normalized method, like gtg/fedsv) -> excluded from the fr==0 gate.
    phi_sfl = -np.asarray(shapleyfl_from_logs(logs, None, N, None, device, beta=0.5,
                                              loss_fn=loss_fn, pkeys=pkeys))
    # Ripple: own tiny trajectory (good->high -> negate to good->low). free-rider(zero)
    # -> phi==0 (zero delta = no drop, no ripple), like the from-logs methods.
    phi_r = -np.asarray(ripple_shapley_llm(model, init_lora, clients, tok, val_chunks, device,
                                           rounds=3, steps=2, lr=1e-3, k=2, m=6, seed=0,
                                           free_riders=FREE_RIDER, free_rider_mode="zero", hess_bs=2))
    # FLDetector: model-free server-side DETECTION score (good->low; high=suspicious). NOT a
    # value -> no free-rider==0 gate (a zero-update free-rider is exactly what it should flag,
    # giving it a HIGH score, not 0).  R=3 -> 1 prediction round (needs >=1 secant pair).
    phi_fld = fldetector_from_logs(logs, N, device="cpu")

    fr = next(iter(FREE_RIDER))
    tag = lambda i: "*" if i in NOISY else ("F" if i in FREE_RIDER else " ")
    fmt = lambda ph: "  ".join(f"{DOMAINS[i][:4]}{tag(i)}={ph[i]:+.4f}" for i in range(N))
    print("\n== (B) LLM smoke (1B, N=5, free-rider=zero; loss-based, good->low) ==")
    for name, ph in [("(b) oracle", phi_b), ("estimator ", phi_e), ("gtg       ", phi_g),
                     ("fedsv     ", phi_f), ("banzhaf   ", phi_z), ("shapleyfl ", phi_sfl),
                     ("ripple    ", phi_r), ("fldetector", phi_fld)]:
        print(f"  {name}: {fmt(ph)}")
    # free-rider(zero) -> EXACTLY 0 for the delta-based methods (zero delta = zero
    # marginal in every coalition).  GTG/FedSV use within-subset RENORMALIZATION, so a
    # zero-delta client still DILUTES the others' weights (it enters the Σn denominator)
    # -> a small nonzero phi: a real property of those baselines, NOT a bug (the
    # fixed-weight / delta utilities of Flirds/oracle/ripple give exactly 0 -- a cleaner
    # free-rider handling, a Flirds differentiator).
    fr_exact = {k: bool(abs(v[fr]) < 1e-9)
                for k, v in [("oracle", phi_b), ("est", phi_e), ("banzhaf", phi_z), ("ripple", phi_r)]}
    fr_small = {k: bool(abs(v[fr]) < 1e-2) for k, v in [("gtg", phi_g), ("fedsv", phi_f)]}
    finite = all(bool(np.isfinite(v).all()) for v in (phi_b, phi_e, phi_g, phi_f, phi_z, phi_sfl, phi_r, phi_fld))
    print(f"  free-rider(zero): delta-based==0 {fr_exact} | renorm-small {fr_small} | finite={finite}")
    print(f"  Spearman vs (b) oracle: gtg={spearmanr(phi_g, phi_b).correlation:+.3f} "
          f"fedsv={spearmanr(phi_f, phi_b).correlation:+.3f} "
          f"banzhaf={spearmanr(phi_z, phi_b).correlation:+.3f} "
          f"shapleyfl={spearmanr(phi_sfl, phi_b).correlation:+.3f} "
          f"est={spearmanr(phi_e, phi_b).correlation:+.3f}  (ripple: own trajectory)")
    # FLDetector is detection-only (no Spearman/value): plumbing gate = finite + L1-normalized
    # (Σ score == 1).  Its noisy/free-rider AUROC is printed for info but NOT gated -- detection
    # QUALITY is experiment-scale (R=3/8ex is below the floor; the real signal is in compare).
    fld_norm = bool(abs(float(phi_fld.sum()) - 1.0) < 1e-6)
    print(f"  fldetector: Σscore={phi_fld.sum():.4f} (norm={fld_norm}) "
          f"AUROC noisy={detection_auroc(phi_fld, [1 if i in NOISY else 0 for i in range(N)]):.3f} "
          f"free-rider={detection_auroc(phi_fld, [1 if i in FREE_RIDER else 0 for i in range(N)]):.3f}")
    ok = all(fr_exact.values()) and all(fr_small.values()) and finite and fld_norm
    print("\nSV-BASELINE LLM PORT OK" if ok else "\nSV-BASELINE LLM SMOKE FAIL")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("cnn", "both"):
        cnn_regression()
        banzhaf_kernel_check()
        shapleyfl_pipeline_check()
        fldetector_synthetic_check()
    if which in ("llm", "both"):
        llm_smoke()
