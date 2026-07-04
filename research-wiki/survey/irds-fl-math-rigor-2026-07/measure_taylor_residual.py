#!/usr/bin/env python
"""항목 1 실측: per-round Taylor 잔차 — 실제 u_r(S) vs 1차/2차 Taylor 예측 (전 부분집합 S ⊆ P_r).

수학 골격(math_skeleton.md P2·P3·P5)과 코드 감사(estimator_audit.md §1–3)의 실측 파트.
flirds 코드는 임포트만 하고 일절 수정하지 않는다 (codes/ 에서 PYTHONPATH=. 로 실행).

측정하는 게임 (고정가중; (b) oracle in_run_sv 와 동일한 게임 — 감사 §2.1):
  u_r(S)  = ℓ(w_r + Δ_S) − ℓ(w_r),   Δ_S = Σ_{k∈S} a_k,  a_k = p_k^r δ_k,
            p_k^r = n_k / Σ_{j∈P_r} n_j   (분모 = 라운드 전체 P_r, S-비의존)
  û¹(S)   = ⟨g^r, Δ_S⟩ = Σ_{k∈S} b_k,          b_k  = ⟨g^r, a_k⟩          (1차 Taylor)
  û²(S)   = û¹(S) + ½ Σ_{i,j∈S} q_ij,          q_ij = ⟨a_i, H^r a_j⟩      (2차 Taylor)
  ũ(S)    = ℓ(w_r + c_S Δ_S) − ℓ(w_r),         c_S  = Σ_{P_r} n / Σ_S n   (--renorm; P5 대안게임)

라운드당 비용: forward 2^K(+renorm 2^K−1) + grad 1회 + HVP K회 (h_k = H a_k,
torch.func jvp∘grad = flirds_estimator._chunked 재사용; q_ij·‖Δ_S‖ 는 K×K 캐시로 행렬연산만).

φ-수준 3자(+α) 비교 (라운드별 exact Shapley 는 2^K 직접):
  phi_exact   = Σ_r Shapley(u_r)           — 진짜 게임의 라운드별 exact Shapley 합 (P1 분해)
  phi_t2      = Σ_r Shapley(û²_r)          — Taylor-2 게임의 exact Shapley
  phi_closed  = Σ_r [b_k + ½ Σ_j q_kj]     — P2 닫힌형 (û² Shapley 와 이론상 동일; fp만 차이)
  phi_flirds  = flirds_values(logs, ...)   — 본 estimator 직접 호출 (대조; HVP 방향이 ΔW 단일이라
                                             contraction 순서가 달라 bit-일치는 기대 불가, ~fp 오차)
  phi_renorm  = Σ_r Shapley(ũ_r)           — 재정규화 게임 (--renorm; P5 순위 비교)
  (--check_inrun 시 in_run_shapley 전체 2^N 직접 호출과 Σ_r Shapley(u_r) 대조 = P1 수치 확인)

sanity 기대: median|u−û²| < median|u−û¹|; 잔차>0; |u−û¹|/‖Δ_S‖² 및 |u−û²|/‖Δ_S‖³ 유계;
telescoping: u_r(P_r) = ℓ(w_{r+1})−ℓ(w_r) (S⊇P_r 에서 섭동=실현 업데이트; 감사 §2.1).

출력 (--outdir):
  coalitions.csv  (r, S, size, n_S, u_true, u_t1, u_t2, resid1, resid2, norm_dS[, c_S, u_renorm])
  phi.csv         (client, n, phi_exact, phi_t1, phi_t2, phi_closed, phi_flirds2, phi_flirds1[, phi_renorm])
  summary.json    (라운드별 잔차 통계 + 스케일링 + φ 비교 + Spearman + sanity 판정 + timing)

실행 예 (codes/ 에서):
  # gpt2 CPU 스모크:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 PYTHONPATH=. python -u <this file> \
    --model gpt2 --device cpu --rounds 3 --train 40 --val_size 10 --max_steps 2 \
    --batch 4 --maxlen 256 --val_maxlen 128 --val_chunk 5 --renorm --check_inrun --outdir <dir>
  # 1B 본실행 (RUN_1B.md 참조): 기본값 = 2026-06-06 valuation-baseline 무대
  #   (Llama-3.2-1B-Instruct, silo5 N=5, R=10, per-domain train=200, val=100, lr=1e-3,
  #    max_steps=10, batch=16, maxlen=768, val_maxlen=384, val_chunk=10, LoRA r=16 α=32).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from math import factorial

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- flirds 재사용 (수정 금지; import 만) ----------------------------------- #
from flirds.backends.llm import make_llm_loss
from flirds.core.flirds_estimator import _chunked, flirds_values
from flirds.data.llm import build, build_val_batches
from flirds.fl.llm_server import run_llm_fedavg_logs
from flirds.oracle.in_run_sv import _perturbed_params, _round_weight, in_run_shapley

LLAMA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def load_model(model_name, device, lora_r):
    """fp32 + eager LoRA 모델 (phase2_matrix._load 와 동일 규약; eager = HVP forward-AD 필수)."""
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    target = LLAMA_TARGET if "llama" in model_name.lower() else None  # None -> peft 아키텍처 기본 (gpt2: c_attn)
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32,
                                             attn_implementation="eager").to(device)
    m = get_peft_model(m, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, target_modules=target,
                                     lora_dropout=0.0, task_type="CAUSAL_LM"))
    init = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    return tok, m, init, list(init)


def exact_shapley_dict(players, u):
    """|players|-인 게임 u (dict: tuple(sorted ids) -> float, u[()]=0) 의 exact Shapley.

    in_run_sv.in_run_shapley_perround 의 라운드-내부 커널과 동일한 조합가중 r!(K-r-1)!/K!."""
    K = len(players)
    phi = {}
    for k in players:
        others = [c for c in players if c != k]
        acc = 0.0
        for r in range(K):
            w = factorial(r) * factorial(K - r - 1) / factorial(K)
            for S in itertools.combinations(others, r):
                acc += w * (u[tuple(sorted(S + (k,)))] - u[S])
        phi[k] = acc
    return phi


def _dot(x, y, pkeys):
    """Σ_n <x[n], y[n]> (fp32 텐서 내적 -> Python float; estimator 의 float(sum(...)) 규약과 동일)."""
    return float(sum((x[n] * y[n]).sum() for n in pkeys))


def _stats(arr):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return {}
    return {"max": float(a.max()), "median": float(np.median(a)), "mean": float(a.mean())}


def _loglog_slope(norms, resids):
    """log|resid| vs log‖Δ_S‖ 최소제곱 기울기 (잔차 스케일링 차수 추정; resid>0 인 S 만)."""
    x = np.asarray(norms, dtype=float)
    y = np.asarray(resids, dtype=float)
    m = (x > 0) & (y > 0)
    if m.sum() < 3:
        return None
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


# --------------------------------------------------------------------------- #
# per-round measurement                                                        #
# --------------------------------------------------------------------------- #
def measure_round(r, w_r, dm, loss_fn, loss_chunks, pkeys, device, renorm):
    """라운드 r 에서 (coalition rows, per-round phi dicts, round summary) 를 계산."""
    players = sorted(dm.keys())
    K = len(players)
    assert K <= 16, f"round {r}: 2^{K} coalition enumeration too large"
    pr = _round_weight(dm)                                        # p_k^r (S-비의존; oracle 과 동일)
    tot_n = sum(n for _, n in dm.values())
    # estimator 와 동일한 fp32 캐스팅 (flirds_estimator.py L101-103; 로그는 이미 fp32라 no-op)
    params = {n: w_r[n].detach().float().to(device) for n in pkeys}
    buffers = {n: w_r[n].detach().to(device) for n in w_r if n not in pkeys}
    a = {k: {n: pr[k] * dm[k][0][n].float().to(device) for n in pkeys} for k in players}

    with torch.no_grad():
        base = float(loss_fn(params, buffers))

    # ---- g^r 1회 + 클라별 HVP h_k = H^r a_k (jvp∘grad; _chunked 는 g 를 함께 반환) ----
    t0 = time.perf_counter()
    g, h = None, {}
    for k in players:
        gk, hk = _chunked(loss_chunks, buffers, params, a[k], pkeys)   # (∇ℓ, H·a_k) 청크 정확합
        if g is None:
            g = gk
        h[k] = hk
    t_hvp = time.perf_counter() - t0

    # ---- 캐시: b_k = <g, a_k>,  Q_ij = <a_i, H a_j>,  Gram_ij = <a_i, a_j> ----
    b = {k: _dot(g, a[k], pkeys) for k in players}
    Q = np.zeros((K, K))
    Gram = np.zeros((K, K))
    for i, ki in enumerate(players):
        for j, kj in enumerate(players):
            Q[i, j] = _dot(a[ki], h[kj], pkeys)
            Gram[i, j] = _dot(a[ki], a[kj], pkeys)
    q_asym = float(np.abs(Q - Q.T).max())          # true-Hessian 대칭성의 fp 잔차 (진단)

    # ---- 전 부분집합: 실제 u (forward; oracle 의 _perturbed_params 재사용) + Taylor 재구성 ----
    rows = []
    u_true = {(): 0.0}
    u_t2d = {(): 0.0}
    u_ren = {(): 0.0} if renorm else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for size in range(1, K + 1):
            for S in itertools.combinations(players, size):
                idx = [players.index(k) for k in S]
                pert = _perturbed_params(params, dm, S, pr, pkeys)     # w_r + Σ_{k∈S} p_k δ_k
                u = float(loss_fn(pert, buffers)) - base
                del pert
                u1 = sum(b[k] for k in S)                              # û¹(S)
                u2 = u1 + 0.5 * float(Q[np.ix_(idx, idx)].sum())       # û²(S)
                nrm = float(np.sqrt(max(Gram[np.ix_(idx, idx)].sum(), 0.0)))   # ‖Δ_S‖
                n_S = sum(dm[k][1] for k in S)
                row = dict(r=r, S="+".join(map(str, S)), size=size, n_S=n_S,
                           u_true=u, u_t1=u1, u_t2=u2,
                           resid1=abs(u - u1), resid2=abs(u - u2), norm_dS=nrm)
                u_true[S] = u
                u_t2d[S] = u2
                if renorm:                                             # ũ(S) = ℓ(w_r + c_S Δ_S) − base
                    c_S = tot_n / n_S
                    dS = {n: sum(a[k][n] for k in S) for n in pkeys}
                    pert = {n: params[n] + c_S * dS[n] for n in pkeys}
                    row["c_S"] = c_S
                    row["u_renorm"] = float(loss_fn(pert, buffers)) - base
                    u_ren[S] = row["u_renorm"]
                    del pert, dS
                rows.append(row)
    t_fwd = time.perf_counter() - t0

    # ---- 라운드별 φ: exact(u) / exact(û²) / 닫힌형 / exact(ũ) ----
    phi_exact = exact_shapley_dict(players, u_true)
    phi_t2 = exact_shapley_dict(players, u_t2d)
    phi_closed = {k: b[k] + 0.5 * float(Q[i, :].sum()) for i, k in enumerate(players)}  # P2 닫힌형
    phi_t1 = dict(b)                                                   # 가산 게임: Shapley = 자기항
    phi_ren = exact_shapley_dict(players, u_ren) if renorm else None

    ne = [row for row in rows]                                         # 비공집합 S 전부
    summ = dict(
        r=r, n_players=K, base_loss=base, t_hvp_s=round(t_hvp, 2), t_forward_s=round(t_fwd, 2),
        norm_dW=float(np.sqrt(max(Gram.sum(), 0.0))),                  # ‖Δ_{P_r}‖
        # fp32 forward-eval 노이즈 플로어: u = (fp32 loss)−(fp32 loss) 차이므로 ~ulp(base) 아래
        # 잔차는 측정 불능 (δ 극소 스모크에선 2차항이 이 플로어에 묻힘 — sanity 판정에 반영).
        ulp_base=float(np.spacing(np.float32(base))),
        Q_asym_max=q_asym,
        resid1=_stats([w["resid1"] for w in ne]),
        resid2=_stats([w["resid2"] for w in ne]),
        frac_t2_le_t1=float(np.mean([w["resid2"] <= w["resid1"] for w in ne])),
        ratio_r1_over_d2=_stats([w["resid1"] / w["norm_dS"] ** 2 for w in ne if w["norm_dS"] > 0]),
        ratio_r2_over_d3=_stats([w["resid2"] / w["norm_dS"] ** 3 for w in ne if w["norm_dS"] > 0]),
        loglog_slope_r1=_loglog_slope([w["norm_dS"] for w in ne], [w["resid1"] for w in ne]),
        loglog_slope_r2=_loglog_slope([w["norm_dS"] for w in ne], [w["resid2"] for w in ne]),
        u_grand=u_true[tuple(players)],
        max_abs_phi_t2_vs_closed=max(abs(phi_t2[k] - phi_closed[k]) for k in players),
    )
    return rows, dict(exact=phi_exact, t1=phi_t1, t2=phi_t2, closed=phi_closed, renorm=phi_ren), summ


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--n_clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--train", type=int, default=200, help="per-domain(=per-client at N=5) train size")
    ap.add_argument("--val_size", type=int, default=100, help="total val records (per-domain = /5)")
    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=768)
    ap.add_argument("--val_maxlen", type=int, default=384)
    ap.add_argument("--val_chunk", type=int, default=10)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--renorm", action="store_true", help="재정규화 게임 ũ 부가 측정 (P5)")
    ap.add_argument("--check_inrun", action="store_true",
                    help="in_run_shapley 전체 2^N 직접 호출과 Σ_r Shapley(u_r) 대조 (P1; forward 2배)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = args.device
    timing = {}

    # ---- 1) 무대: silo5 데이터 + fp32 LoRA 모델 + FedAvg 궤적 (logs 는 RAM 상주; 감사 §9) ----
    t0 = time.perf_counter()
    tok, model, init, pkeys0 = load_model(args.model, device, args.lora_r)
    clients, val_records, _ = build(args.n_clients, args.train, max(1, args.val_size // 5), 0,
                                    seed=args.seed)
    timing["setup_s"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    model.load_state_dict(init, strict=False)
    logs = run_llm_fedavg_logs(model, tok, clients, args.rounds, args.lr, args.max_steps,
                               batch_size=args.batch, max_length=args.maxlen,
                               sample_frac=1.0, seed=args.seed)
    timing["fl_run_s"] = round(time.perf_counter() - t0, 1)
    print(f"[fl] {len(logs)} rounds logged in {timing['fl_run_s']}s", flush=True)

    # ---- 2) val loss 빌더 (estimator/oracle 이 공유하는 것과 동일 closure) ----
    val_batches = build_val_batches(val_records, tok, args.val_maxlen, device, args.val_chunk)
    loss_fn, pkeys, loss_chunks = make_llm_loss(model, val_batches, device)
    assert set(pkeys) == set(pkeys0), "pkeys mismatch between init and make_llm_loss"

    # ---- 3) 라운드별 실측 ----
    N = args.n_clients
    all_rows, round_summaries = [], []
    acc = {name: np.zeros(N) for name in ("exact", "t1", "t2", "closed", "renorm")}
    t0 = time.perf_counter()
    for r, (w_r, dm) in enumerate(logs):
        rows, phis, summ = measure_round(r, w_r, dm, loss_fn, loss_chunks, pkeys, device,
                                         args.renorm)
        all_rows += rows
        round_summaries.append(summ)
        for name in acc:
            if phis[name] is not None:
                for k, v in phis[name].items():
                    acc[name][k] += v
        print(f"[round {r}] base={summ['base_loss']:.6f} ‖ΔW‖={summ['norm_dW']:.4g} "
              f"med|u-û¹|={summ['resid1']['median']:.3g} med|u-û²|={summ['resid2']['median']:.3g} "
              f"frac(û²≤û¹)={summ['frac_t2_le_t1']:.2f}", flush=True)
    timing["measure_s"] = round(time.perf_counter() - t0, 1)

    # ---- 4) estimator 직접 호출 대조 (flirds_values; 2차 + 1차) ----
    t0 = time.perf_counter()
    phi_fl2, _ = flirds_values(logs, loss_fn, pkeys, device, second_order=True,
                               n_clients=N, loss_chunks=loss_chunks)
    phi_fl1, _ = flirds_values(logs, loss_fn, pkeys, device, second_order=False,
                               n_clients=N, loss_chunks=loss_chunks)
    timing["flirds_values_s"] = round(time.perf_counter() - t0, 1)

    phi_inrun = None
    if args.check_inrun:
        t0 = time.perf_counter()
        phi_inrun, _ = in_run_shapley(logs, N, loss_fn, pkeys, device)
        timing["in_run_shapley_s"] = round(time.perf_counter() - t0, 1)

    # ---- 5) telescoping 확인: u_r(P_r) == base_{r+1} − base_r (r < R−1) ----
    telescoping = [abs((round_summaries[r + 1]["base_loss"] - round_summaries[r]["base_loss"])
                       - round_summaries[r]["u_grand"])
                   for r in range(len(round_summaries) - 1)]

    # ---- 6) 저장 ----
    import pandas as pd
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.outdir, "coalitions.csv"), index=False)
    try:
        df.to_parquet(os.path.join(args.outdir, "coalitions.parquet"))
    except Exception as e:                                             # pyarrow 부재 시 CSV 만
        print(f"[warn] parquet skipped: {e}", flush=True)

    ns = [len(c) for c in clients]
    phi_df = pd.DataFrame({"client": range(N), "n": ns,
                           "phi_exact": acc["exact"], "phi_t1": acc["t1"], "phi_t2": acc["t2"],
                           "phi_closed": acc["closed"], "phi_flirds2": phi_fl2,
                           "phi_flirds1": phi_fl1})
    if args.renorm:
        phi_df["phi_renorm"] = acc["renorm"]
    if phi_inrun is not None:
        phi_df["phi_inrun_2N"] = phi_inrun
    phi_df.to_csv(os.path.join(args.outdir, "phi.csv"), index=False)

    def sp(x, y):
        return float(spearmanr(x, y)[0])

    resid1_all = df["resid1"].to_numpy()
    resid2_all = df["resid2"].to_numpy()
    compare = {
        "max_abs_closed_vs_flirds2": float(np.abs(acc["closed"] - phi_fl2).max()),
        "allclose_closed_vs_flirds2_rtol1e-4": bool(np.allclose(acc["closed"], phi_fl2,
                                                                rtol=1e-4, atol=1e-9)),
        "bit_identical_closed_vs_flirds2": bool(np.array_equal(acc["closed"], phi_fl2)),
        "max_abs_t1_vs_flirds1": float(np.abs(acc["t1"] - phi_fl1).max()),
        "max_abs_t2_vs_closed": float(np.abs(acc["t2"] - acc["closed"]).max()),
        "spearman": {
            "exact_vs_t1": sp(acc["exact"], acc["t1"]),
            "exact_vs_t2": sp(acc["exact"], acc["t2"]),
            "exact_vs_closed": sp(acc["exact"], acc["closed"]),
            "exact_vs_flirds2": sp(acc["exact"], phi_fl2),
        },
    }
    if args.renorm:
        compare["spearman"]["exact_vs_renorm"] = sp(acc["exact"], acc["renorm"])
    if phi_inrun is not None:
        compare["max_abs_perround_vs_inrun2N"] = float(np.abs(acc["exact"] - phi_inrun).max())

    med_r1 = float(np.median(resid1_all))
    med_r2 = float(np.median(resid2_all))
    # fp32 forward-eval 노이즈 플로어(라운드 최대 ulp(base)): 2차 개선분이 이 밑이면 t2-vs-t1 판정 불능
    noise_floor = float(max(s["ulp_base"] for s in round_summaries))
    t2_verdict = ("t2_better" if med_r2 < med_r1 else
                  "inconclusive_noise_floor" if med_r2 <= 4 * noise_floor else
                  "t1_better_CHECK")
    sanity = {
        "resid_positive": bool(resid2_all.max() > 0),
        "t2_median_lt_t1_median": bool(med_r2 < med_r1),
        "t2_vs_t1_verdict": t2_verdict,
        "fp32_eval_noise_floor": noise_floor,
        "t2_le_t1_frac_overall": float(np.mean(resid2_all <= resid1_all)),
        "closed_matches_flirds_values": compare["allclose_closed_vs_flirds2_rtol1e-4"],
        "t2_shapley_matches_closed_form": bool(compare["max_abs_t2_vs_closed"] < 1e-10),
        "telescoping_max_gap": float(max(telescoping)) if telescoping else None,
    }
    sanity["verdict"] = ("PASS" if sanity["resid_positive"]
                         and t2_verdict in ("t2_better", "inconclusive_noise_floor")
                         and sanity["closed_matches_flirds_values"]
                         and sanity["t2_shapley_matches_closed_form"] else "CHECK")

    summary = {
        "config": vars(args),
        "pooled": {
            "resid1": _stats(resid1_all), "resid2": _stats(resid2_all),
            "loglog_slope_r1": _loglog_slope(df["norm_dS"], resid1_all),
            "loglog_slope_r2": _loglog_slope(df["norm_dS"], resid2_all),
        },
        "rounds": round_summaries,
        "phi_full": {k: (acc[k].tolist() if k != "renorm" or args.renorm else None)
                     for k in acc},
        "phi_flirds2": phi_fl2.tolist(), "phi_flirds1": phi_fl1.tolist(),
        "phi_inrun_2N": phi_inrun.tolist() if phi_inrun is not None else None,
        "phi_compare": compare,
        "telescoping_gaps": telescoping,
        "sanity": sanity,
        "timing": timing,
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({"sanity": sanity, "phi_compare": compare, "timing": timing},
                     indent=2, ensure_ascii=False), flush=True)
    print(f"[done] outputs in {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
