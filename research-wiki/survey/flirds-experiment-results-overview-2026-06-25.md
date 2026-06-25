---
type: survey
title: "Flirds 실험 결과 전체 한눈에 보기 (2026-06-25)"
created: 2026-06-25
updated: 2026-06-25
tags: [survey, results, experiments, master, fidelity, detection, cost]
---

# Flirds 실험 결과 전체 한눈에 보기

> **날짜**: 2026-06-25 · **git HEAD**: `e89af94` (branch `main`) · **스코프**: flirds 프로젝트의
> 네 실험 트랙(Foundational `phase1` / LLM standard `track_d` / CNN `track_c` / Robustness
> `phase2_matrix`)에서 *실제로 돌아 디스크에 남은* 모든 셀·baseline·하이퍼파라미터를 표로 정리한다.
> 계획만 잡혀 있고 아직 안 돌린 실험도 행으로 넣되 수치는 빈칸(⬚).
>
> **모든 수치는 아래 파일에서 직접 집계**(기억·CLAUDE.md·메모리 숫자 미사용 — stale 가능). per-seed
> CSV는 method별 mean±std로 집계하고 사람용 요약(RESULTS.md/RESULTS.txt)과 교차검증했다.
>
> **수치 출처 파일**
> - LLM standard: `runs/track_d/fidelity.csv` · `runs/track_d/rundirs/*/{config.yaml,metrics.json}`
> - CNN: `runs/track_c/fidelity.csv` · `runs/track_c/RESULTS.txt` · `runs/track_c/{c1,c2,c1_oracle}/*/{config,metrics}.json`
> - Robustness: `runs/phase2_matrix/RESULTS.md` · `runs/phase2_matrix/analysis/00_overview/master_metrics.csv` · `runs/phase2_matrix/rundirs/*/{config.yaml,meta.json}`
> - Foundational: `runs/phase1/rundirs/*/{config.yaml,metrics.json}`
>
> **수치 재생성 경로** (rundir만으로 재실행 가능, GPU 불필요):
> - `python runs/track_d/make_fidelity.py` → `runs/track_d/fidelity.csv` 재생성 (1B/3B/7B × std20/anchor5)
> - `python runs/phase2_matrix/make_analysis.py` → `runs/phase2_matrix/analysis/*` + `RESULTS.md` 재생성
> - CNN `RESULTS.txt`는 Track C의 결과 정리 스크립트가 rundir에서 재생성.
>
> 자매 문서(중복 X, 링크만): baseline 수치가 각 방법 원 논문과 얼마나 맞는지는
> [[baseline-original-paper-verification-2026-06-22]]. 선행연구 지형 분류는 [[prior-work-taxonomy/README]].

---

## 1. 범례 (legend)

**분류축 (6축, [[prior-work-taxonomy/taxonomy]] 기준)**
- **Federation**: `C` centralized · **`F` federated**(star, server) · `D` decentralized. — 우리 실험은 전부 **F**.
- **Valuation 기반**: `in-run`(단일 궤적, 재학습 X) · `retrain`(조합마다 재학습) · `IF`(gradient/influence) · `recon`(sub-model 재구성=FL-Shapley 계열) · `other`. — Flirds = **in-run (1st+2nd Taylor, true Hessian)**.
- **Model**: CNN/MLP · LLM(LoRA) · kernel · tabular.
- **Unit (평가 단위)**: sample · **client** · class · dataset · node. — 우리 = **client-level**.
- **Purpose (검증 목적)**: Fidelity · Selection→downstream performance · Corrupt-client detection · Fairness·reward · Stability(replication) · Cost·scalability · Aggregation quality.
- **Exact/Approx**: Shapley 계산이 exact(2ᴺ 열거) / approx(MC·Taylor) / both. — estimator=approx, oracle=exact.

**지표 방향**
- Spearman / Kendall = 순위 상관 ↑ (1=완벽 일치) · Pearson = 값-수준 상관 ↑ · cosine_d / euclid_d / max_diff = 거리 ↓ (0=동일)
- AUROC = 탐지 ↑ (1=완벽; 0.5=무작위; <0.5=뒤집힘) · ASR = backdoor 공격 성공률 · runtime = wall-clock(초) ↓

**오라클/기준점**
- **In-run oracle (b)** = 한 학습 궤적에서 exact 2ᴺ 분해 (full-participation→2ᴺ 열거; partial→exact per-round 분해). 1차 fidelity 정답.
- **Retrain oracle (a)** = 조합마다 처음부터 재학습한 exact 2ᴺ Shapley (val-loss utility). 별도 정답(문헌 공백).
- **Flirds proxy reference** = 정확 oracle이 비싼 칸(device100 비-anchor)에서 검증된 Flirds를 기준 대용으로 씀 → 그 칸의 Spearman은 *vs Flirds*임에 주의.
- **Flirds (1st-order only) = Flirds-1st** (2차 Hessian 항 끔).

**마커**: ● 실측(파일에 수치 있음) · ◐ 기준점만/부분(예: 1-seed) · ○ 설계상 제외 · ⬚ 미실행(계획·빈칸) · – 해당없음 · **(미기록)** = 파일에 칸은 있으나 값 없음.

---

## 2. 마스터 한눈에 표 (실험당 1행)

> 참여형태 표기: `full`=매 라운드 전원, `k/N`=라운드당 k명 참여, `K%`=비율. Federation은 전부 **F**.
> "valuation 기반" 열은 *주체 방법(Flirds=in-run)* 기준이며, 같은 셀에서 비교한 baseline은 recon/IF/other를 함께 포함(섹션의 baseline-set 노트 참조).

| # | 실험 (rundir 코드) | Model · N · 참여 | Unit | valuation 기반 (+baseline) | 검증 목적 | Exact/Approx | oracle / truth | seeds | status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | LLM standard · 1B · std20 (`track_d/1B_std20`) | Llama-3.2-1B · N=20 · 2/20 | client | in-run (+recon/IF) | Fidelity·Perf·Conv·Cost | approx vs exact(b) | (b) exact per-round | 3 | ● |
| 2 | LLM standard · 1B · anchor (`track_d/1B_anchor5`) | Llama-3.2-1B · N=5 · full | client | in-run (+recon/IF/Banzhaf) | Fidelity·Perf·Conv·Cost | approx vs exact(a)&(b) | **(a) 2⁵ retrain + (b) 2⁵ in-run** | 3 | ● |
| 3 | LLM standard · 3B · std20 (`track_d/3B_std20`) | Llama-3.2-3B · N=20 · 2/20 | client | in-run (+recon/IF) | Fidelity·Perf·Conv·Cost | approx vs exact(b) | (b) exact per-round | 3 | ● |
| 4 | LLM standard · 3B · anchor (`track_d/3B_anchor5`) | Llama-3.2-3B · N=5 · full | client | in-run (+recon/IF/Banzhaf) | Fidelity·Perf·Conv·Cost | approx vs exact(b) | (b) 2⁵ in-run · **(a) ⬚** | 3 | ● |
| 5 | LLM standard · 7B · std20 (`track_d/7B_std20`) | Llama-2-7B · N=20 · 2/20 | client | in-run (+recon/IF) | Fidelity·Perf·Conv·Cost | approx vs exact(b) | (b) exact per-round | 3 | ● |
| 6 | LLM standard · 7B · anchor (`track_d/7B_anchor5`) | Llama-2-7B · N=5 · full | client | in-run (+recon/IF/Banzhaf) | Fidelity·Perf·Conv·Cost | approx vs exact(b) | (b) 2⁵ in-run · **(a) ⬚** | 3 | ● |
| 7 | CNN · cross-silo N=10 (`track_c/c1`+`c1_oracle`) | 소형 CNN · N=10 · full | client | in-run (+recon/IF/Ripple/Banzhaf) | **Fidelity·Stability**·Cost | approx vs exact(a)&(b) | **(a) 2¹⁰ retrain + (b) 2¹⁰ in-run** | 3 | ● |
| 8 | CNN · cross-device N=100 (`track_c/c2`) | 소형 CNN · N=100 · 10/100 | client | in-run intervention arms (+recon/sfedavg) | **Perf·Detection·Conv** | – (개입 arm) | corrupt 마스크 | 3 | ● |
| 9 | Robustness · 1B cross-silo N=5 (`phase2_matrix/1B_silo5_*`) | Llama-3.2-1B · N=5 · full | client | in-run (+recon/IF/Banzhaf/탐지기4) | **Fidelity·Detection**·Cost | approx vs exact(b) | (b) 2⁵ in-run | 3 | ● |
| 10 | Robustness · 1B cross-device N=100 α-sweep (`.../1B_device100-a{0,0.01,0.1,5.0}_*`) | Llama-3.2-1B · N=100 · 10/100 | client | in-run (+IF/ComFedSV/탐지기4) | **Detection**·Fidelity | approx vs Flirds-proxy | **Flirds proxy reference** | 3 | ● |
| 11 | Robustness · 1B cross-device N=100 α=0.5 **Anchor cell** (`.../1B_device100-a0.5_*_anchor`) | Llama-3.2-1B · N=100 · 10/100 | client | in-run (+recon/IF/ComFedSV/탐지기4) | **Fidelity·Detection**·Cost | approx vs exact(b)-perround | **(b) per-round** | 3 | ● |
| 12 | Robustness · 1B cross-device N=100 poison (`.../1B_device100-a{0,0.5}_poison`) | Llama-3.2-1B · N=100 · 10/100 | client | in-run (+IF/ComFedSV/탐지기4) | **Detection**·Fidelity | approx vs Flirds-proxy | Flirds proxy | 3 | ● |
| 13 | Robustness · 3B cross-silo N=5 (`phase2_matrix/3B_silo5_*`) | Llama-3.2-3B · N=5 · full | client | in-run (+IF/탐지기4) | Fidelity·Detection | approx vs exact(b) | (b) 2⁵ in-run | **1** | ◐ |
| 14 | Foundational · 1B 첫 clean run (`phase1/...full-lr*`) | Llama-3.2-1B · N=5 · full | client | in-run | Detection·Perf(selection) | approx | 주입 라벨(oracle 없음) | 3×2lr | ● (부록) |
| 15 | Foundational · 1B LR sweep (`phase1/...sweep-lr*`) | Llama-3.2-1B · N=5 · full | client | in-run | Detection·Perf(selection) | approx vs exact(b) | (b) 2⁵ in-run | 1×4lr | ● (부록) |
| — | **이하 계획·미실행 (수치 ⬚)** | | | | | | | | |
| P1 | LLM N=10 (a)/(b) oracle | LLM · N=10 | client | retrain+in-run | Fidelity(고-power) | exact | (a)/(b) 2¹⁰ | ⬚ | ⬚ deferred(비용) |
| P2 | LLM 3B anchor **(a) retrain oracle** | Llama-3.2-3B · N=5 | client | retrain | Fidelity(dual oracle) | exact | (a) 2⁵ retrain | ⬚ | ⬚ |
| P3 | LLM 7B anchor **(a) retrain oracle** | Llama-2-7B · N=5 | client | retrain | Fidelity(dual oracle) | exact | (a) 2⁵ retrain | ⬚ | ⬚ |
| P4 | Robustness · 7B (silo5/device100) | Llama-2-7B | client | in-run+탐지기 | Detection·Fidelity | approx vs (b) | (b) | ⬚ | ⬚ |
| P5 | Robustness · 3B 3-seed 완성 | Llama-3.2-3B · N=5 | client | in-run+탐지기 | Detection·Fidelity | approx vs (b) | (b) 2⁵ | ⬚ (현 1 seed) | ⬚ |
| P6 | Fairness·reward 전용 실험 | – | client | – | Fairness·reward | – | – | ⬚ | ⬚ 미설계 |

---

# 3. 검증 목적별 결과 (핵심 질문 위계 순)

> 위계(루트 CLAUDE.md): **1차 = Fidelity**(기여도 추정 정확도) → 2차 = ① Selection→performance / Aggregation → ② Convergence → ③ Detection → 비용. 아래 섹션 순서가 이 위계다.

---

## 3.1 Fidelity (1차 핵심) — 기여도 추정이 정답 oracle 순위를 얼마나 재현하나

### 3.1.1 LLM standard fidelity (`track_d`)

**(a) 세팅**
- model: Llama-3.2-1B-Instruct / Llama-3.2-3B-Instruct / **Llama-2-7B-hf** (7B = FL-LLM 문헌 표준 rung)
- 두 스테이지: **Standard stage `std20`** = N=20, 라운드당 2명 참여, R=200, local 10 steps · **Precision stage `anchor5`** = N=5 전원 참여, R=30, local 10 steps
- batch 16(1B/3B) / 4(7B); lr=1e-3, optimizer **plain SGD momentum=0**; **LoRA r=16, α=32** (target = q/k/v/o/gate/up/down proj); dataset = alpaca-gpt4 20k IID (clean, 오염 없음); seq len 512; val=200 / test=1000; fp32
- oracle: **(b) in-run** — std20=exact per-round 분해(2/round), anchor5=exact 2⁵ 전수; **(a) retrain 2⁵ = 1B anchor5에서만** (3B/7B는 ⬚)
- seeds=3 (각 스테이지·스케일)

**(b) 결과 — vs (b) in-run oracle, method별 3-seed mean±std**

#### std20 스테이지 (N=20, 2/round) — 순위·값 상관

| method | 1B Spearman | 1B Kendall | 1B Pearson | 3B Spearman | 3B Kendall | 3B Pearson | 7B Spearman | 7B Kendall | 7B Pearson |
|---|---|---|---|---|---|---|---|---|---|
| **Flirds** | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 0.999±.001 | 0.996±.005 | 1.000±.000 |
| **Flirds-1st** | 0.999±.001 | 0.996±.005 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 0.998±.001 | 0.986±.010 | 1.000±.000 |
| loss-heur | 1.000±.000 | 1.000±.000 | 1.000±.000 | 0.999±.001 | 0.996±.005 | 1.000±.000 | 0.999±.001 | 0.996±.005 | 1.000±.000 |
| GTG | 0.975±.018 | 0.916±.043 | 0.995±.001 | 0.988±.005 | 0.944±.022 | 0.996±.000 | 0.977±.017 | 0.916±.045 | 0.989±.006 |
| FedSV | 0.910±.073 | 0.786±.117 | 0.959±.013 | 0.952±.018 | 0.853±.026 | 0.972±.004 | 0.968±.010 | 0.881±.030 | 0.976±.006 |
| FedIF | 0.157±.303 | 0.111±.199 | 0.229±.222 | 0.211±.184 | 0.139±.115 | 0.262±.137 | 0.480±.101 | 0.323±.061 | 0.508±.054 |
| ShapleyFL | 0.194±.351 | 0.133±.244 | 0.245±.283 | 0.227±.143 | 0.161±.092 | 0.246±.143 | 0.406±.081 | 0.274±.054 | 0.431±.026 |
| ComFedSV | 0.093±.146 | 0.060±.108 | 0.095±.193 | -0.129±.066 | -0.105±.034 | -0.093±.038 | 0.039±.171 | 0.039±.110 | 0.048±.115 |

#### std20 스테이지 — 거리 (↓)

| method | 1B cosine_d | 1B euclid_d | 1B max_diff | 3B cosine_d | 3B euclid_d | 3B max_diff | 7B cosine_d | 7B euclid_d | 7B max_diff |
|---|---|---|---|---|---|---|---|---|---|
| **Flirds** | .0000 | .0000 | .0000 | .0000 | .0000 | .0000 | .0000 | .0001 | .0000 |
| **Flirds-1st** | .0000 | .0006 | .0002 | .0000 | .0004 | .0002 | .0000 | .0002 | .0001 |
| loss-heur | .0000 | .0001 | .0001 | .0000 | .0001 | .0000 | .0000 | .0000 | .0000 |
| GTG | .0007 | .0010 | .0005 | .0005 | .0009 | .0005 | .0015 | .0011 | .0005 |
| FedSV | .0057 | .0028 | .0017 | .0034 | .0024 | .0013 | .0055 | .0023 | .0012 |
| FedIF | .1911 | 2.636 | .9760 | .1768 | 2.598 | .9837 | .1141 | 2.667 | .9870 |
| ShapleyFL | .2087 | 2.737 | .9905 | .2063 | 2.729 | .9943 | .1570 | 2.733 | .9946 |
| ComFedSV | .8447 | .0262 | .0098 | 1.009 | .0292 | .0103 | .9264 | .0213 | .0077 |

#### anchor5 스테이지 (N=5, full) — 순위·값 상관

| method | 1B Spearman | 1B Kendall | 1B Pearson | 3B Spearman | 3B Kendall | 3B Pearson | 7B Spearman | 7B Kendall | 7B Pearson |
|---|---|---|---|---|---|---|---|---|---|
| **Flirds** | 1.000±.000 | 1.000±.000 | 1.000±.000 | 0.967±.047 | 0.933±.094 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 |
| **Flirds-1st** | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 |
| loss-heur | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 |
| Banzhaf | 1.000±.000 | 1.000±.000 | 1.000±.000 | 0.967±.047 | 0.933±.094 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 |
| GTG | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 0.999±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 |
| FedSV | 0.700±.163 | 0.600±.163 | 0.824±.116 | 0.700±.000 | 0.600±.000 | 0.882±.056 | 0.933±.047 | 0.867±.094 | 0.961±.024 |
| ShapleyFL | 0.700±.283 | 0.600±.283 | 0.764±.258 | 0.100±.000 | 0.000±.000 | 0.353±.415 | 0.833±.125 | 0.733±.189 | 0.903±.067 |
| ComFedSV | 0.500±.432 | 0.467±.340 | 0.563±.356 | 0.600±.294 | 0.533±.249 | 0.475±.272 | 0.600±.216 | 0.467±.189 | 0.588±.256 |
| FedIF | 0.067±.531 | 0.067±.411 | -0.068±.626 | 0.067±.492 | 0.000±.432 | 0.335±.475 | 0.200±.616 | 0.200±.490 | 0.368±.509 |
| **(a)oracle**¹ | 0.933±.047 | 0.867±.094 | 0.933±.054 | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |

¹ **(a) retrain 2⁵ oracle vs (b) in-run oracle** = 듀얼 오라클 일치도 (1B anchor5만 실행). Spearman 0.933±.047 → 두 정답 정의가 거의 동률(완전 일치 아님은 N=5 coarse + retrain noise). 3B/7B (a) = ⬚.

> anchor5 거리는 생략하지 않음 — 핵심만: Flirds/Flirds-1st/loss-heur/Banzhaf cosine_d≈0(<.0001), GTG≈.003–.011, FedSV≈.005–.013; FedIF/ShapleyFL euclid_d≈1.2–1.5(부호 불안정). 전체 6-metric은 `make_fidelity.py` 재실행 시 `fidelity.csv`.

**출처**: `runs/track_d/fidelity.csv` (재생성: `python runs/track_d/make_fidelity.py`) · 셀별 원시 = `runs/track_d/rundirs/{1B,3B,7B}_{std20,anchor5}_seed{0,1,2}/metrics.json`

**(c) baseline-set 노트**
- **포함(9종 + (a)/(b) oracle)**: Flirds, Flirds-1st, loss-heur, GTG, FedSV, ComFedSV, ShapleyFL(β=0.3, Def 4.3), FedIF, Banzhaf(anchor5만). truth = (b) in-run oracle; anchor5 1B는 (a) retrain oracle도.
- **제외**: **Banzhaf** = exact 2ᴺ → std20(N=20)에선 비실행(2²⁰), anchor5(N=5)만 ─ *적용규칙: exact 2ᴺ은 N≤10*. **(a) retrain oracle** = 비용(2⁵×R 재학습) → 1B anchor5만, 3B/7B ⬚. **Ripple** = sample-level 이미지 전용 → LLM 트랙 설계상 제외 ─ *적용규칙: 적용성*.

---

### 3.1.2 CNN fidelity (`track_c` C1, cross-silo N=10) — 싼 고-power 듀얼 오라클

**(a) 세팅**
- 소형 CNN (**mnist=LeNet5 / cifar10=FedSVCNN**, 전체 모델 학습 — LoRA 아님), **N=10 전원 참여**, R=10, local epochs=5, lr=0.01, batch=64, SGD mom=0; val=2000 / test=8000
- datasets: **mnist, cifar10**; 5 시나리오 = `iid` · `label_skew` · `quantity_skew` · `label_flip` · `feature_noise` (GTG-Shapley 5-시나리오 무대 이식). 10 scenario × 3 seed = 30 셀; (a) oracle도 30 셀.
- oracle: **(a) exact 2¹⁰ retrain** (`oracle.exact_sv.subset_utility_valloss`) + **(b) exact 2¹⁰ in-run** (`oracle.in_run_sv.in_run_shapley`); 둘 다 val-loss 게임. Oracle-a 효율 gap ≤1e-15 (모든 셀).

**(b) 결과 — 10 scenario × 3 seed pool 한 method별 mean±std** (Spearman/Pearson; Kendall·거리는 fidelity.csv)

| method | vs (b) Spearman | vs (b) Pearson | vs (a) Spearman | vs (a) Pearson |
|---|---|---|---|---|
| **Flirds** | **0.919±.134** | **0.934±.128** | 0.352±.462 | 0.354±.461 |
| **Flirds-1st** | 0.832±.194 | 0.853±.159 | 0.408±.435 | 0.412±.421 |
| loss-heur | 0.860±.154 | 0.885±.134 | 0.425±.429 | 0.423±.408 |
| Banzhaf | 0.989±.019 | 0.998±.004 | 0.355±.441 | 0.357±.459 |
| GTG | 0.569±.343 | 0.612±.317 | 0.374±.412 | 0.332±.456 |
| FedSV | 0.401±.410 | 0.410±.406 | 0.284±.479 | 0.215±.466 |
| ComFedSV | 0.348±.377 | 0.328±.396 | 0.338±.398 | 0.309±.431 |
| ShapleyFL | 0.391±.385 | 0.392±.425 | 0.453±.380 | 0.443±.410 |
| FedIF | 0.491±.391 | 0.506±.427 | 0.380±.393 | 0.368±.431 |
| Ripple | 0.373±.444 | 0.404±.437 | 0.213±.462 | 0.158±.470 |

> **주의(caveat)**: 위 pool 평균은 **`iid` 셀 포함**(오염·skew 신호가 없어 fidelity가 의미상 낮음) → 깎인 값.
> `iid` 제외 시(8 scenario×3seed=24): Flirds vs (b) **0.928±.136**, Flirds-1st 0.875±.172, loss-heur 0.884±.153, Banzhaf 0.991±.020, GTG 0.626±.315.

**시나리오별 vs (b) Spearman** (3-seed 평균; 신호 강한 칸이 보이게)

| dataset/scenario | Flirds | Flirds-1st | GTG | FedSV | ComFedSV | Banzhaf | ShapleyFL | FedIF | loss-heur | Ripple |
|---|---|---|---|---|---|---|---|---|---|---|
| cifar10 / feature_noise | 1.00 | 0.89 | 0.59 | 0.40 | 0.19 | 1.00 | 0.20 | 0.62 | 0.90 | -0.01 |
| cifar10 / iid | 0.95 | 0.54 | 0.21 | 0.22 | 0.12 | 0.98 | 0.18 | 0.45 | 0.69 | 0.31 |
| cifar10 / label_flip | 1.00 | 0.95 | 0.64 | 0.54 | 0.31 | 1.00 | 0.37 | 0.74 | 0.95 | 0.32 |
| cifar10 / label_skew | 0.98 | 0.92 | 0.49 | 0.53 | 0.31 | 1.00 | 0.29 | 0.68 | 0.88 | 0.26 |
| cifar10 / quantity_skew | 0.99 | 0.96 | 0.78 | 0.56 | 0.67 | 1.00 | 0.44 | -0.20 | 0.98 | 0.68 |
| mnist / feature_noise | 0.79 | 0.70 | 0.41 | 0.13 | 0.21 | 0.95 | 0.48 | 0.57 | 0.78 | 0.00 |
| mnist / iid | 0.81 | 0.78 | 0.47 | 0.04 | 0.10 | 0.98 | 0.47 | 0.73 | 0.84 | 0.18 |
| mnist / label_flip | 1.00 | 0.99 | 0.99 | 0.97 | 0.95 | 1.00 | 0.98 | 0.98 | 0.99 | 0.97 |
| mnist / label_skew | 0.71 | 0.61 | 0.33 | -0.01 | 0.14 | 0.98 | -0.02 | 0.41 | 0.63 | 0.06 |
| mnist / quantity_skew | 0.96 | 0.98 | 0.78 | 0.63 | 0.49 | 1.00 | 0.52 | -0.07 | 0.96 | 0.96 |

**출처**: `runs/track_c/fidelity.csv` (열 `spearman_b/pearson_b/spearman_a/pearson_a`) · `runs/track_c/RESULTS.txt` (C1 (a)-oracle 절). 코드 = `codes/experiments/track_c1.py`.

**(c) baseline-set 노트**
- **포함(9종 + Ripple + (a)/(b))**: Flirds, Flirds-1st, loss-heur, GTG, FedSV, ComFedSV, ShapleyFL, FedIF, Banzhaf, **Ripple**(sample-level 자체 게임). truth = (a) 2¹⁰ retrain + (b) 2¹⁰ in-run 듀얼.
- **제외**: 탐지기(FLDetector/FLTrust/STD-DAGMM/FedDQC) = C1엔 오염축이 없음(시나리오는 skew/flip/noise이지 update-level 위협 아님) → *적용규칙: 탐지기는 오염축 있는 실험만* (탐지는 Track C2/Robustness). Banzhaf·(a)/(b) exact 가능 = N=10 ≤ 10이라 *exact 2ᴺ 규칙 충족*.

---

### 3.1.3 Fidelity under corruption (`phase2_matrix`) — 오염 주입 무대에서의 fidelity

> Robustness 무대의 Spearman(vs (b) 또는 Flirds proxy)은 §3.4 Detection 표에 AUROC와 함께 통합 수록(같은 셀). 요지만:
> - **silo5 (N=5, (b) 2⁵)**: clean·noisy·free-rider 위협서 Flirds/Flirds-1st/loss-heur Spearman **1.000**, FedSV 0.93~1.0, GTG 1.0, FedIF 0.90~0.93. **poison 위협**서 near-additive 동률 붕괴: FedSV **0.367±.262**, GTG 0.867, Flirds-1st **0.000**(회피), Flirds 0.967.
> - **device100 anchor (N=100, (b) per-round)**: Flirds/Flirds-1st/loss-heur **1.000**, GTG 0.78~0.84, FedSV 0.75~0.81, ShapleyFL 0.58~0.69, FedIF 0.72~0.83, ComFedSV ≈ 0(low-rank 가정 위배).
> 전체 수치 → §3.4.

---

### 3.1.4 Stability (replication) — `track_c` C1 cross-seed 안정성

> 위계의 5개 주요 섹션엔 없는 목적(Stability·replication)이지만, C1 fidelity 런에서 같이 측정된 결과이므로 여기 둔다.
> **rho_xseed** = 한 method φ 랭킹의 seed 간 상관(↑=재현성). **topJ/botJ** = 상위/하위 20% 클라셋 Jaccard.

| method | rho_xseed (10 scenario pool) | topJ | botJ |
|---|---|---|---|
| (b)oracle (자체) | 0.518±.453 | 0.522±.395 | 0.555±.453 |
| **Flirds** | **0.547±.394** | 0.544±.324 | 0.500±.419 |
| Flirds-1st | 0.510±.461 | 0.611±.395 | 0.455±.386 |
| Banzhaf | 0.508±.463 | 0.511±.407 | 0.555±.453 |
| Ripple | 0.514±.388 | 0.533±.333 | 0.489±.359 |
| loss-heur | 0.474±.448 | 0.500±.287 | 0.467±.384 |
| GTG | 0.311±.441 | 0.467±.358 | 0.378±.345 |
| FedSV | 0.289±.385 | 0.356±.351 | 0.345±.292 |
| FedIF | 0.243±.413 | 0.322±.296 | 0.244±.293 |
| ComFedSV | 0.198±.383 | 0.300±.268 | 0.289±.366 |
| ShapleyFL | 0.124±.431 | 0.200±.276 | 0.344±.331 |

> 읽기: (b) oracle 자체의 cross-seed 안정성이 0.518(CNN은 seed별 기여가 실제로 갈림) → **Flirds(0.547)는 oracle의 내재 안정성을 그대로 추종**, recon MC baseline(GTG/FedSV/ComFedSV/ShapleyFL)은 추가 분산으로 0.12~0.31로 떨어짐.

**출처**: `runs/track_c/RESULTS.txt` (C1 stability 절, 10 scenario pool).

---

## 3.2 Selection → downstream performance / Aggregation quality (2차 ①)

### 3.2.1 LLM standard intervention arms (`track_d`) — clean-IID do-no-harm parity

**(a) 세팅**: §3.1.1과 동일 궤적. arms = 같은 vanilla 로그에서 파생한 온라인 개입 6종.
- `base` = 학습 전 베이스 모델 · `vanilla` = 표준 FedAvg · `flirds_w` = 곱셈 가중 w∝n·s (EMA β=0.5) · `flirds_sel` = softmax 선택 (cohort가 진부분집합인 std20만) · `shapleyfl_w` = 교체 가중 (β=0.3) · `fedif_w` = 교체 가중 (β=0.7=1-γ)
- 평가: **MMLU full-test(14,042) 0-shot** + 같은분포 **Alpaca-test(1k) ROUGE-L**. clean-IID 기대 = parity(do-no-harm); 차이는 finding.

**(b) 결과 — MMLU / ROUGE-L, 3-seed mean±std**

| stage·scale | arm | MMLU | ROUGE-L | stage·scale | arm | MMLU | ROUGE-L |
|---|---|---|---|---|---|---|---|
| **1B std20** | base | 0.4822±.0000 | 0.2168±.0019 | **1B anchor5** | base | 0.4822±.0000 | 0.2168±.0019 |
| | vanilla | 0.4742±.0001 | 0.2841±.0051 | | vanilla | 0.4801±.0003 | 0.2725±.0032 |
| | flirds_w | 0.4745±.0003 | 0.2848±.0050 | | flirds_w | 0.4802±.0007 | 0.2741±.0025 |
| | flirds_sel | 0.4739±.0005 | 0.2838±.0041 | | shapleyfl_w | 0.4802±.0007 | 0.2741±.0026 |
| | shapleyfl_w | 0.4742±.0005 | 0.2845±.0050 | | fedif_w | 0.4797±.0008 | 0.2713±.0037 |
| | fedif_w | 0.4741±.0003 | 0.2847±.0046 | | | | |
| **3B std20** | base | 0.6230±.0000 | 0.2219±.0015 | **3B anchor5** | base | 0.6230±.0000 | 0.2219±.0015 |
| | vanilla | 0.6147±.0006 | 0.3017±.0024 | | vanilla | 0.6215±.0001 | 0.2749±.0035 |
| | flirds_w | 0.6137±.0006 | 0.3015±.0018 | | flirds_w | 0.6214±.0002 | 0.2755±.0042 |
| | flirds_sel | 0.6139±.0014 | 0.3029±.0039 | | shapleyfl_w | 0.6214±.0002 | 0.2755±.0042 |
| | shapleyfl_w | 0.6136±.0005 | 0.3016±.0018 | | fedif_w | 0.6213±.0002 | 0.2730±.0035 |
| | fedif_w | 0.6139±.0007 | 0.3022±.0024 | | | | |
| **7B std20** | base | 0.4175±.0000 | 0.1496±.0024 | **7B anchor5** | base | (미기록)¹ | (미기록)¹ |
| | vanilla | 0.4038±.0024 | 0.2778±.0026 | | vanilla | (미기록)¹ | (미기록)¹ |
| | flirds_w | 0.4026±.0028 | 0.2780±.0027 | | | | |
| | flirds_sel | 0.4025±.0022 | 0.2790±.0044 | | | | |
| | shapleyfl_w | 0.4027±.0027 | 0.2787±.0028 | | | | |
| | fedif_w | 0.4030±.0023 | 0.2763±.0033 | | | | |

¹ 7B anchor5 metrics.json은 fidelity·runtime만 담고 arm(MMLU/ROUGE) 블록은 비어 있음 → **(미기록)**. (7B std20는 arm 있음.)

> 읽기(do-no-harm): 모든 개입 arm의 MMLU·ROUGE가 vanilla와 ±0.001~0.003 이내 = clean-IID에서 기여도-가중이 성능을 **해치지도 크게 올리지도 않음**(기대대로 parity). ROUGE는 학습으로 base 대비 크게 상승(예: 3B std20 0.222→0.302); MMLU는 SFT로 소폭 하락(외부 벤치, 분포 밖).

**출처**: `runs/track_d/rundirs/*/metrics.json` (`arms.{arm}.{mmlu,rouge_l}`).

**(c) baseline-set 노트**: 포함 arm = base/vanilla/flirds_w/shapleyfl_w/fedif_w (+ std20만 flirds_sel). **flirds_sel 제외@anchor5** = 전원 참여라 선택이 무의미(degenerate) ─ *적용규칙: 참여형태*. arm은 valuation 비교가 아니라 *개입 효과* 측정.

---

### 3.2.2 CNN cross-device intervention (`track_c` C2) — 최종 정확도

**(a) 세팅**
- 소형 CNN, **N=100, 라운드당 10% 참여**, R=120, local epochs=5, lr=0.01, batch=64, SGD mom=0, val=2000/test=8000, target acc=0.6(cifar) / dataset별; 3 seeds.
- datasets: cifar10, fmnist; partitions: `iid` · `dir1`(Dirichlet α=1, label+size skew) · `shard`(McMahan 2-shard); threats: `clean` · `label_flip` · `free_rider` · `grad_noise` (+ 강도 변형); 총 **30 셀**.
- arms(8): vanilla · **flirds_mult** · flirds_repl(dir1만) · flirds_add(dir1만) · **flirds_select** · shapleyfl(β=0.5) · fedif(β=0.7) · sfedavg(S-FedAvg).

**(b) 결과 — 최종 test 정확도, threat별 그룹 mean±std** (셀 pool; 셀별은 `RESULTS.txt`)

| arm | clean (6셀) | free_rider (6셀) | grad_noise (8셀) | label_flip (10셀) |
|---|---|---|---|---|
| vanilla | 0.686±.127 | 0.646±.146 | 0.499±.241 | 0.583±.184 |
| **flirds_mult** | 0.698±.122 | 0.662±.144 | 0.609±.187 | 0.626±.170 |
| flirds_repl (dir1) | 0.734±.096 | 0.704±.111 | 0.621±.185 | 0.652±.149 |
| flirds_add (dir1) | 0.733±.094 | 0.702±.110 | 0.604±.195 | 0.635±.161 |
| flirds_select | 0.679±.150 | 0.656±.148 | 0.548±.231 | 0.618±.172 |
| shapleyfl | 0.702±.126 | 0.645±.136 | 0.645±.183 | 0.622±.168 |
| fedif | 0.685±.127 | 0.654±.154 | 0.624±.178 | 0.623±.169 |
| sfedavg | 0.695±.128 | 0.655±.139 | 0.510±.252 | 0.598±.186 |

> **그룹 평균임을 명시**: 위는 partition(iid/dir1/shard)·강도·dataset(cifar/fmnist)을 한 threat 내에서 pool한 값이라 std가 크다(예: grad_noise는 str0.05↔strmain 혼합). 셀별 30칸 acc는 `runs/track_c/RESULTS.txt` C2 절.
> 읽기: 오염 위협(grad_noise/label_flip)에서 기여도-가중 arm이 vanilla 대비 정확도를 회복(예 grad_noise vanilla 0.499 → flirds_mult 0.609 / shapleyfl 0.645). clean에선 parity~소폭↑.

**출처**: `runs/track_c/c2/*/metrics.json` (`arms.{arm}.final_acc`). 코드 = `codes/experiments/track_c2.py`.

**(c) baseline-set 노트**: 포함 8 arm. **flirds_repl/flirds_add 제외@iid·shard** = size-skew(dir1)에서만 MULT와 갈리므로 dir1 전용 ─ *적용규칙: 참여형태/적용성*. valuation fidelity baseline(GTG/FedSV exact 등)은 C2엔 없음(C2는 개입-성능 무대; fidelity는 C1).

---

## 3.3 Convergence (2차 ②) — 수렴 속도

### 3.3.1 LLM standard (`track_d`) — final val-loss + rounds-to-target

**(a) 세팅**: §3.2.1 arms와 동일 로그. rounds-to-target = vanilla 최종 val-loss에 도달한 첫 라운드.

**(b) 결과 — final val-loss / rounds-to-target, 3-seed mean±std**

| stage·scale | arm | final val-loss | rounds-to-target | stage·scale | arm | final val-loss | rounds-to-target |
|---|---|---|---|---|---|---|---|
| **1B std20** (R=200) | vanilla | 1.2653±.0216 | 200.0±0.0 | **1B anchor5** (R=30) | vanilla | 1.2977±.0209 | 30.0±0.0 |
| | flirds_w | 1.2653±.0215 | 199.0±1.0 | | flirds_w | 1.2964±.0209 | 29.7±0.5 |
| | flirds_sel | 1.2654±.0217 | 199.0±0.0 | | shapleyfl_w | 1.2964±.0209 | 29.7±0.5 |
| | shapleyfl_w | 1.2653±.0215 | 199.0±1.0 | | fedif_w | 1.2976±.0204 | 30.0±0.0 |
| | fedif_w | 1.2652±.0215 | 198.0±0.0 | | | | |
| **3B std20** | vanilla | 1.1483±.0270 | 198.3±2.4 | **3B anchor5** | vanilla | 1.1970±.0272 | 30.0±0.0 |
| | flirds_w | 1.1479±.0271 | 192.3±3.3 | | flirds_w | 1.1961±.0272 | 30.0±0.0 |
| | flirds_sel | 1.1481±.0273 | 193.5±0.5 | | shapleyfl_w | 1.1960±.0272 | 30.0±0.0 |
| | shapleyfl_w | 1.1479±.0271 | 192.3±3.3 | | fedif_w | 1.1970±.0269 | 30.0±0.0 |
| | fedif_w | 1.1478±.0271 | 191.0±3.6 | | | | |
| **7B std20** | vanilla | 1.0357±.0244 | 184.7±18.3 | **7B anchor5** | — | (미기록)¹ | (미기록)¹ |
| | flirds_w | 1.0348±.0245 | 153.0±18.8 | | | | |
| | flirds_sel | 1.0351±.0243 | 153.0±23.6 | | | | |
| | shapleyfl_w | 1.0347±.0245 | 151.3±19.9 | | | | |
| | fedif_w | 1.0348±.0246 | 158.0±12.7 | | | | |

¹ 7B anchor5 arm 블록 비어 있음 → **(미기록)**.

> 읽기: 수렴은 clean-IID라 arm 간 거의 동률. 두드러진 칸 = **7B std20**: 개입 arm이 vanilla 184.7 라운드 대비 ~151~158 라운드로 target 도달(약 14~18% 빠름); 3B std20도 flirds/shapleyfl_w ~192 vs vanilla 198. 1B·anchor5는 차이 미미.

**출처**: `runs/track_d/rundirs/*/metrics.json` (`arms.{arm}.{final_val_loss,rounds_to_target,val_curve}`).

### 3.3.2 CNN cross-device (`track_c` C2) — rounds-to-target

| arm | clean | free_rider | grad_noise | label_flip |
|---|---|---|---|---|
| vanilla | 35.7±27.3 | 41.2±39.1 | 27.2±35.0 | 19.3±12.1 |
| flirds_mult | 34.1±27.7 | 38.8±34.8 | 13.7±13.6 | 31.3±30.9 |
| flirds_repl (dir1) | 41.5±34.9 | 7.7±0.5 | 6.3±0.5 | 10.9±4.0 |
| flirds_add (dir1) | 39.5±32.8 | 34.2±44.9 | 6.8±0.7 | 11.8±4.0 |
| flirds_select | 35.3±26.9 | 50.4±43.2 | 21.7±24.4 | 16.1±7.7 |
| shapleyfl | 35.6±28.1 | 17.4±11.7 | 18.7±26.1 | 33.1±35.0 |
| fedif | 36.0±28.7 | 44.7±35.8 | 23.9±24.5 | 30.6±30.4 |
| sfedavg | 35.3±28.3 | 42.4±38.0 | 21.9±24.5 | 19.3±10.6 |

> 그룹 평균(셀 pool); rounds-to-target은 target 미달 셀에서 분산이 큼. 셀별은 `RESULTS.txt`.
> 읽기: grad_noise에서 기여도-가중이 target 도달을 크게 앞당김(vanilla 27.2 → flirds_mult 13.7 / flirds_repl 6.3).

**출처**: `runs/track_c/c2/*/metrics.json` (`arms.{arm}.rounds_to_target`).

---

## 3.4 Corrupt-client detection (2차 ③) — 오염 클라 탐지 AUROC

> 위계상 **마지막**(기여도≠탐지). valuation φ를 탐지 스코어(corrupt=high-φ)로 쓴 AUROC와 전용 탐지기 AUROC를 같은 셀에서 함께 본다. Fidelity Spearman(vs (b)/Flirds-proxy)도 같은 표에 병기.

### 3.4.1 Robustness cross-silo N=5 (`phase2_matrix/1B_silo5_*`)

**(a) 세팅**: Llama-3.2-1B, **N=5 전원**, R=10, local 10 steps, batch 16, lr=1e-3 (poison=2e-3), maxlen 768, train=200/val=20/test=40, warmup=2; 위협별 1명 오염(noisy=client0, free-rider=client1, poison=client0); **(b)=exact 2⁵**; 3 seeds. **poison** = D2b model-replacement backdoor(lr=2e-3, batch=8, epochs=5, frac=0.8), deployed ASR≈1.00.

**(b) 결과 — AUROC(corrupt=high-φ) + Spearman vs (b) + Pearson + runtime, 3-seed mean±std**

| method | noisy AUROC | noisy Sp | frrand AUROC | frrand Sp | frzero AUROC | frzero Sp | **poison AUROC** | **poison Sp** | runtime |
|---|---|---|---|---|---|---|---|---|---|
| Flirds | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.917±.118** | 0.967±.047 | ~107s |
| Flirds-1st | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.000** | **0.000** | ~35s |
| loss-heur | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ~170s |
| FedIF | 1.000 | 0.933±.05 | 1.000 | 0.900±.08 | 1.000 | 0.933±.05 | 1.000 | 0.967±.05 | ~37s |
| GTG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.867±.12 | ~540s |
| FedSV | 1.000 | 1.000 | 1.000 | 0.933±.05 | 1.000 | 1.000 | 1.000 | **0.367±.26** | ~535s |
| ShapleyFL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ~530s |
| Banzhaf | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ~535s |
| (b)oracle | 1.000 | (truth) | 1.000 | (truth) | 1.000 | (truth) | 1.000 | (truth) | ~530s |
| FLDetector | 0.750 | – | 1.000 | – | 0.750 | – | 1.000 | – | ~30s |
| STD-DAGMM | 0.417±.31 | – | 1.000 | – | 0.250±.20 | – | 0.750±.20 | – | ~120s |
| FLTrust | 1.000 | – | 1.000 | – | 1.000 | – | 1.000 | – | ~37s |
| FedDQC | 0.917±.12 | – | 0.750 | – | 0.750 | – | 1.000 | – | ~22s |

> **읽기**: noisy·free-rider 위협에선 valuation·전용탐지기 거의 전부 AUROC 1.0 + Spearman 1.0(near-additive). **poison(clean-보존 backdoor)이 분리점**: 공격자가 clean val-loss를 낮춰 φ가 "기여 높음"으로 나옴 → **Flirds-1st AUROC 0.000 / Spearman 0.000 = 완전 회피**(2차항 있는 Flirds는 0.917로 일부 잡음). loss-heur·(b) oracle·Banzhaf·FedIF·GTG·FLDetector·FLTrust는 1.0으로 잡음. FedSV Spearman이 poison서 0.367로 추락(per-round MC 분산 + near-additive 붕괴 첫 사례).

**출처**: `runs/phase2_matrix/analysis/00_overview/master_metrics.csv` + `RESULTS.md` (교차검증 일치). 코드 = `codes/experiments/phase2_matrix.py`.

**(c) baseline-set 노트**: 포함 13 = valuation 8(Flirds/Flirds-1st/loss-heur/FedIF/GTG/FedSV/ShapleyFL/Banzhaf) + (b)oracle + 탐지기 4(FLDetector/STD-DAGMM/FLTrust/FedDQC). 모든 method가 모든 위협서 동작("category-together"). ComFedSV 제외(=partial-participation 전용, silo full엔 부적합). Ripple 제외(이미지 전용).

### 3.4.2 Robustness cross-device N=100 α-sweep (`phase2_matrix/1B_device100-a*`)

**(a) 세팅**: Llama-3.2-1B, **N=100, 라운드당 10명(10%)**, R=30, local 5 steps, batch 16, lr=1e-3, per_client=300, Dirichlet α∈{0, 0.01, 0.1, **0.5=anchor**, 5.0}, val=10/test=40, warmup=3; 오염 클라 5명(idx 10/30/50/70/90); 3 seeds. **α=0.5 = Anchor cell**: (b) per-round 오라클 + coalition baseline(GTG/FedSV/ShapleyFL) 켬. 그 외 α: cheap method + Flirds proxy reference.

**(b1) 결과 — noisy 위협, detection AUROC** (α별 3-seed mean±std)

| method | α=0.0 | α=0.01 | α=0.1 | α=0.5 (anchor) | α=5.0 |
|---|---|---|---|---|---|
| (b)oracle | – | – | – | 0.604±.041 | – |
| Flirds | 0.774±.058 | 0.575±.055 | 0.605±.056 | 0.604±.041 | 0.596±.039 |
| Flirds-1st | 0.772±.058 | 0.575±.057 | 0.606±.055 | 0.605±.042 | 0.597±.038 |
| loss-heur | 0.772±.058 | 0.574±.056 | 0.607±.056 | 0.605±.042 | 0.597±.038 |
| FedIF | 0.973±.017 | 0.568±.106 | 0.693±.126 | 0.830±.085 | 0.973±.022 |
| GTG | – | – | – | 0.734±.112 | – |
| FedSV | – | – | – | 0.708±.142 | – |
| ShapleyFL | – | – | – | 0.762±.095 | – |
| ComFedSV | 0.442±.115 | 0.419±.054 | 0.432±.032 | 0.371±.028 | 0.396±.002 |
| FLDetector | 0.535±.048 | 0.482±.085 | 0.525±.070 | 0.539±.055 | 0.532±.058 |
| STD-DAGMM | 0.856±.037 | 0.652±.190 | 0.659±.147 | 0.671±.142 | 0.760±.040 |
| FLTrust | 1.000 | 0.602±.096 | 0.720±.136 | 0.854±.090 | 0.994±.008 |
| **FedDQC** | 0.960±.057 | 1.000 | 1.000 | 1.000 | 1.000 |

> noisy@device100: **FedDQC(데이터-품질 전용)가 1.0으로 최강**, valuation φ는 0.57~0.77(비-IID서 clean 클라가 상위 → 침식). FedIF/FLTrust는 α 높을수록 회복.

**(b2) 결과 — free-rider(random / zero) detection AUROC** (대표 α; 전체 α는 master_metrics)

| method | frrand α=0.0 | frrand α=0.5 | frzero α=0.0 | frzero α=0.5 |
|---|---|---|---|---|
| Flirds / Flirds-1st / loss-heur | 1.000 | 1.000 | 1.000 | 1.000 |
| FedIF | 0.983±.009 | 0.981±.003 | 0.989±.004 | 0.987 |
| ComFedSV | 0.449±.130 | 0.383±.045 | 0.441±.122 | 0.367±.028 |
| FLDetector | 0.606±.030 | 0.617±.039 | 0.529±.024 | 0.540±.061 |
| STD-DAGMM | 0.960±.029 | 0.588±.205 | 0.870±.094 | 0.963±.036 |
| FLTrust | 1.000 | 1.000 | 1.000 | 1.000 |
| FedDQC | 0.140±.014 | 0.573±.113 | 0.140±.014 | 0.573±.113 |

> free-rider@device100: **gradient 쓰는 method(Flirds/Flirds-1st/loss-heur/FLTrust=1.0)가 깔끔**, model-free STD-DAGMM은 0.59~0.96 가변, FedDQC는 off-threat(free-rider는 데이터-품질 아님)이라 0.14~0.57.

**(b3) 결과 — Spearman vs truth** (α=0.5→vs (b)perround; 그 외→vs Flirds proxy reference)

| method | noisy α=0.5 | frrand α=0.5 | frzero α=0.5 | (off-anchor Flirds-1st·loss-heur) |
|---|---|---|---|---|
| Flirds | 1.000 | 1.000 | 1.000 | (proxy 기준) |
| Flirds-1st | 1.000 | 1.000 | 1.000 | 모든 α 1.000 |
| loss-heur | 1.000 | 1.000 | 1.000 | 모든 α 1.000 |
| FedIF | 0.721±.027 | 0.827±.022 | 0.824±.017 | α별 0.62~0.83 |
| GTG | 0.784±.021 | 0.817±.022 | 0.843±.026 | (anchor만) |
| FedSV | 0.752±.020 | 0.795±.020 | 0.814±.018 | (anchor만) |
| ShapleyFL | 0.582±.075 | 0.685±.054 | 0.681±.049 | (anchor만) |
| ComFedSV | -0.023±.127 | -0.051±.153 | -0.051±.142 | 모든 α ≈0 |

**출처**: `runs/phase2_matrix/RESULTS.md` + `master_metrics.csv` (둘 일치). anchor cell runtime: (b)perround ≈25,000s, GTG ≈16,000–18,000s, FedSV ≈4,970s, ShapleyFL ≈24,900s, Flirds ≈157s, Flirds-1st ≈53s.

**(c) baseline-set 노트**:
- **off-anchor(α≠0.5)** 포함 9 = Flirds/Flirds-1st/loss-heur/FedIF/ComFedSV + 탐지기 4. **제외**: GTG/FedSV/ShapleyFL/(b)oracle/Banzhaf ─ *적용규칙: MC Shapley/exact = 대규모서 비용 게이팅 → anchor만*. ComFedSV는 partial-participation Shapley baseline으로 포함 ─ *적용규칙: 참여형태(partial→ComFedSV)*.
- **anchor(α=0.5)** 포함 13 = 위 + GTG/FedSV/ShapleyFL/(b)perround 켬.

### 3.4.3 Robustness cross-device N=100 poison (`phase2_matrix/1B_device100-a{0,0.5}_poison`)

**(a) 세팅**: §3.4.2 + poison(D2b, lr=2e-3, batch=8, R=60, max_steps=10, frac=0.8); α=0.0(ASR≈1.00) / α=0.5(ASR≈0.50); truth=Flirds proxy; 3 seeds.

**(b) 결과**

| method | α=0.0 AUROC | α=0.0 Sp | α=0.5 AUROC | α=0.5 Sp |
|---|---|---|---|---|
| Flirds | 1.000 | (proxy truth) | 1.000 | (proxy truth) |
| Flirds-1st | 1.000 | 0.997±.002 | 0.670±.467 | 0.980±.028 |
| loss-heur | 1.000 | 0.997±.002 | 1.000 | 0.999 |
| FedIF | 0.542±.258 | 0.620±.204 | 0.458±.284 | 0.439±.071 |
| ComFedSV | 0.778±.314 | 0.104±.054 | 0.727±.386 | 0.025±.098 |
| FLDetector | 0.987±.019 | – | 0.983±.024 | – |
| STD-DAGMM | 1.000 | – | 0.983±.024 | – |
| FLTrust | 0.650±.180 | – | 0.498±.281 | – |
| FedDQC | 1.000 | – | 1.000 | – |

> device100 poison은 silo5만큼 강하게 설치 안 됨(cross-device 희석; α=0.5 ASR 0.50). 여기선 Flirds(2차) AUROC 1.0(α=0)으로 회피 안 됨 = 설정 의존(silo5 3B와 대비). caveat: tiny val=10.

**출처**: `master_metrics.csv` (03_device100_poison).

### 3.4.4 Robustness cross-silo N=5 · 3B (`phase2_matrix/3B_silo5_*`) — **1 seed (◐)**

**(a) 세팅**: Llama-3.2-3B, N=5 full, R=10, batch 8; poison lr=2e-3/frac=0.8; **seeds=[0]만**(3-seed ⬚=계획 P5); (b)=exact 2⁵.

**(b) 결과 (1 seed)**

| method | noisy AUROC | noisy Sp | frrand AUROC | frzero AUROC | **poison AUROC** | **poison Sp** | runtime(noisy) |
|---|---|---|---|---|---|---|---|
| Flirds | 1.000 | 1.000 | 1.000 | 1.000 | **0.000** | **0.000** | ~251s |
| Flirds-1st | 1.000 | 1.000 | 1.000 | 1.000 | **0.000** | **0.000** | ~82s |
| loss-heur | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ~384s |
| FedIF | 1.000 | 0.600 | 1.000 | 1.000 | 1.000 | 0.600 | ~82s |
| (b)oracle | 1.000 | (truth) | 1.000 | 1.000 | 1.000 | (truth) | ~1244s |
| FLDetector | 1.000 | – | 1.000 | 1.000 | 1.000 | – | ~146–382s |
| STD-DAGMM | 0.250 | – | 1.000 | 0.000 | 0.750 | – | ~206–745s |
| FLTrust | 1.000 | – | 1.000 | 1.000 | 1.000 | – | ~83–91s |
| FedDQC | 1.000 | – | 0.750 | 0.750 | 1.000 | – | ~46–50s |

> **3B poison**: Flirds·Flirds-1st 둘 다 AUROC 0.000 / Spearman 0.000 = **clean-보존 backdoor에 완전 회피**(silo5 1B는 Flirds 2차가 0.917로 일부 버텼으나 3B는 둘 다 0). loss-heur·(b)·FedIF·FLDetector·FLTrust·FedDQC = 1.0으로 잡음.

**출처**: `master_metrics.csv` (05_scale_3b).

**(c) baseline-set 노트 (3.4.3–3.4.4)**: poison/3B 셀 포함 9 = Flirds/Flirds-1st/loss-heur/FedIF(+ComFedSV@device) + 탐지기 4 + (b)@silo. GTG/FedSV/ShapleyFL/Banzhaf 제외 = poison 셀은 coalition off(비용/설계) ─ *MC Shapley·exact 게이팅*.

---

## 3.5 Cost · scalability — wall-clock

### 3.5.1 LLM standard runtime (`track_d`) — method별, 3-seed mean±std (초)

| method | 1B std20 | 1B anchor5 | 3B std20 | 3B anchor5 | 7B std20 | 7B anchor5 |
|---|---|---|---|---|---|---|
| **Flirds-1st** | 1531±37 | 231±5 | 3630±76 | 547±10 | 6485±87 | 975±13 |
| **Flirds** | 4697±112 | 707±16 | 11147±228 | 1674±30 | 20180±250 | 3027±36 |
| FedIF | 1534±37 | 232±5 | 3638±75 | 549±10 | 6495±88 | 978±13 |
| loss-heur | 2913±72 | 1093±26 | 6909±150 | 2585±48 | 12299±167 | 4613±62 |
| ComFedSV | 2330±22 | 2557±215 | 5526±34 | 6043±525 | 9839±110 | 10792±1044 |
| GTG | 3647±90 | 3552±82 | 8647±189 | 8393±153 | 15393±206 | 14972±193 |
| FedSV | 3646±90 | 3536±86 | 8647±189 | 8356±161 | 15393±208 | 14907±270 |
| ShapleyFL | 2917±72 | 3513±83 | 6916±150 | 8303±155 | 12312±167 | 14812±199 |
| Banzhaf | – (N=20) | 3527±83 | – | 8329±155 | – | 14844±198 |
| **(b)oracle** | 2917±72 | 3528±83 | 6916±151 | 8329±156 | 12310±165 | 14839±196 |
| **(a)oracle** | – | **30817±244** | – | ⬚ | – | ⬚ |

> 읽기: **Flirds-1st = 최저가**(1 HVP/round); Flirds(2차)는 그 ~3배지만 여전히 (b) oracle·coalition baseline(GTG/FedSV/Banzhaf/ShapleyFL)의 **~1/5~1/7**. (a) retrain oracle은 (b)의 **~9배**(1B anchor5 30,817s vs 3528s). 스케일에 따라 모든 method가 ~선형 증가.

**출처**: `runs/track_d/rundirs/*/metrics.json` (`runtime`).

### 3.5.2 다른 트랙 runtime 요약
- **Robustness** (§3.4 표에 병기): N=5 silo5 — Flirds-1st ~35s / Flirds ~107s / (b)·coalition ~530s / 탐지기 22~136s. N=100 anchor — Flirds-1st ~53s / Flirds ~157s vs (b)perround **~25,000s** / GTG ~16–18k s / ShapleyFL ~24.9k s / FedSV ~4970s.
- **CNN** C1: 셀별 wall-clock은 `runs/track_c/c1/*/metrics.json`에 method별 기록(여기 표엔 미전사 — `RESULTS.txt`는 fidelity/stability 중심). **(미기록)**: 본 문서에 CNN wall-clock 테이블 미수록(파일엔 존재).

---

# 4. 부록

## 4.1 Foundational validation (`phase1`) — 첫 clean run + LR sweep

**세팅**: Llama-3.2-1B, **N=5 full**, K=3, 오염 주입(noisy=client0 answer-swap / free-rider=client1 zero-update), per_domain train=12000·val=200·test=2000(full) / 500·100·200(sweep), local 10 steps. **full** = R=50, lr∈{1e-3, 3e-3} × 3 seed, oracle_b off. **sweep** = R=20, lr∈{1e-4, 3e-4, 1e-3, 3e-3} × 1 seed, oracle_b on. 메트릭: AUROC(noisy/free-rider) + selection(K=3 keep) + arms(full/flirds_topk/random_k val-loss).

**결과 — AUROC + selection**

| group | noisy AUROC | free-rider AUROC | flirds_keep (seed별) | random_keep |
|---|---|---|---|---|
| full lr1e-3 (3 seed) | 0.750±.000 | 1.000±.000 | [3,2,4] 매 seed (=clean) | seed별 가변 |
| full lr3e-3 (3 seed) | 1.000±.000 | 0.750±.000 | [2,3,4] 매 seed (=clean) | seed별 가변 |
| sweep lr1e-4 (1 seed) | 0.750 | 1.000 | [3,2,4] | [2,3,4] |
| sweep lr3e-4 (1 seed) | 0.750 | 1.000 | [3,2,4] | [2,3,4] |
| sweep lr1e-3 (1 seed) | 0.750 | 1.000 | [3,2,4] | [2,3,4] |
| sweep lr3e-3 (1 seed) | 0.750 | 1.000 | [3,2,4] | [2,3,4] |

> **lr 의존 반전**: full lr1e-3은 noisy 0.75/FR 1.0, lr3e-3은 noisy 1.0/FR 0.75 (AUROC가 lr에 의존). **selection**: flirds_keep이 매 seed에서 정확히 clean 클라 3개(client 0=noisy·1=free-rider 항상 드롭) → 안정적 분리.

**결과 — selection arms (final val-loss, 3-seed mean±std)**

| group | full(전원) | flirds_topk | random_k |
|---|---|---|---|
| full lr1e-3 | 2.4064±.0234 | **2.3978±.0226** | 2.4111±.0133 |
| full lr3e-3 | 2.3931±.0223 | **2.3926±.0219** | 2.4055±.0100 |

> flirds_topk val-loss ≤ random_k (양 lr) 그리고 ≤ full(오염 드롭이 도움) → "random은 hard bar"를 넘김.

**출처**: `runs/phase1/rundirs/*/metrics.json` (`auroc_noisy`, `auroc_freerider`, `selection`, `arms`).

## 4.2 Caveats (주의)

1. **3B robustness = 1 seed** (`phase2_matrix/3B_silo5_*` seeds=[0]). 3-seed 미완(계획 P5). 3B robustness 수치는 단일 seed.
2. **(a) retrain oracle = 1B anchor5만**(track_d). 3B/7B anchor5는 fidelity·runtime만 있고 (a) 없음(⬚, 계획 P2/P3). *프로젝트 노트엔 별도 task6에서 3B (a)-valloss≈0.900 언급이 있으나 track_d rundir엔 없음 → 파일-only 원칙상 본 표엔 미수록.*
3. **device100 비-anchor truth = Flirds proxy reference** (정확 (b)가 칸당 ~25,000s라 α=0.5만 실측). 그 칸의 Spearman은 *vs Flirds*이지 vs exact oracle 아님 → 1.000은 "Flirds-1st·loss-heur가 Flirds와 동일 순위"의 뜻.
4. **CNN fidelity pool 평균은 `iid` 포함 → 깎임**. iid 셀은 오염·skew 신호가 없어 fidelity가 의미상 낮다(§3.1.2에 iid 제외 값 병기).
5. **CNN C2 / track_c 그룹 테이블은 partition·강도·dataset을 threat 내에서 pool** → std 큼. 셀별 30칸은 `RESULTS.txt`.
6. **7B anchor5 arm(MMLU/ROUGE/val-loss) = (미기록)**(metrics.json arm 블록 비어 있음). fidelity·runtime은 있음.
7. **tiny val** caveat: Robustness silo5 val=20 / device100 val=10 — 작은 검증셋이라 AUROC가 coarse(특히 noisy φ-as-detector).
8. **poison ASR**은 deployed-model 기준(silo5≈1.00, device100 α0≈1.00/α0.5≈0.50, 3B≈1.00).

## 4.3 상호 링크
- 선행연구 6축 분류 + 마스터 표: [[prior-work-taxonomy/README]] · [[prior-work-taxonomy/taxonomy]]
- E1–E7 검증실험 카탈로그(CNN/LLM 트랙 분리): [[prior-work-taxonomy/validation-experiments]]
- metric·benchmark·ground-truth 출처: [[prior-work-taxonomy/metrics-and-benchmarks]]
- baseline 수치 ↔ 원 논문 대조: [[baseline-original-paper-verification-2026-06-22]]

## 4.4 각주 — E1–E7 ↔ 영어 목적명 매핑
> 본문은 영어 목적명만 사용. taxonomy의 E# 코드와의 대응: **E1**=Fidelity · **E2**=Selection→downstream performance · **E3**=Corrupt-client detection · **E4**=Fairness·reward · **E5**=Stability(replication) · **E6**=Cost·scalability · **E7**=Aggregation quality.

---

# 5. 유지보수 / 갱신

**미실행(⬚) 행을 채우는 법**
- 그 실험을 돌린 뒤 해당 rundir가 생기면, 아래 재집계로 수치가 채워진다. 마스터 표(§2)의 `status`를 ⬚→●로, 본문 표의 ⬚ 칸을 mean±std로 교체.
- 예: 7B anchor5 (a) retrain oracle(P3) → `ORACLE_A=1 REGIME=anchor5 SMOKE_MODEL=7B` 로 track_d 실행 → `make_fidelity.py` 재실행 → §3.1.1 anchor5 (a)oracle 7B 칸 채움.

**수치 갱신 (rundir만으로, GPU 불필요)**
- LLM standard: `python runs/track_d/make_fidelity.py` → `fidelity.csv` 재생성(1B/3B/7B × std20/anchor5 자동 포함). 그 후 §3.1.1 표 재집계.
- Robustness: `python runs/phase2_matrix/make_analysis.py` → `analysis/00_overview/master_metrics.csv` + `RESULTS.md` 재생성. 그 후 §3.4/§3.5 표 재집계.
- CNN: Track C 결과 스크립트로 `RESULTS.txt` 재생성(C1 fidelity/stability/(a)-oracle + C2 arms).

**새 실험 완료 시**: §2에 행 1개 추가(축 분류 + status ●) → 해당 검증목적 섹션(§3.x)에 세팅 블록 + 결과 테이블 + baseline-set 노트 추가. 구조는 위 섹션과 동일하게.

---

## 6. 커버리지 자가점검

| 트랙 | 디스크 rundir/셀 수 | 본 문서 수록 | 비고 |
|---|---|---|---|
| LLM standard (`track_d`) | 18 rundir (3 scale × 2 stage × 3 seed) = 6 셀 | 6 셀 전부 (fidelity+arms+conv+runtime) | 7B anchor5 arm = (미기록) |
| CNN (`track_c`) | C1 30 + C1_oracle 30 + C2 90 = 150 rundir (10+10+30 셀 × 3 seed) | C1 10 시나리오 (fidelity/stability/(a)) + C2 30 셀(threat 4그룹 pool) | C2는 그룹 평균(셀별=RESULTS.txt); CNN wall-clock 미전사 |
| Robustness (`phase2_matrix`) | 25 셀 (master_metrics 25, RESULTS.md 25) | 25 셀 전부 (AUROC+Sp+Pe+cos+runtime) | – |
| Foundational (`phase1`) | 12 rundir (full 6 + sweep 4 + mini/smoke 2) | full 6 + sweep 4 = 10 (부록) | mini/smoke 2 = 진단용, 미수록(명시) |
| **계획·미실행** | – | P1–P6 (6행, 수치 ⬚) | – |

---

## 7. 작성 메모 — 확신 없는/누락 의심 칸 (보고)

> 아래는 작성자가 *파일 근거가 약하거나 해석 주의가 필요하다고 판단한 칸*. 모두 위 본문에 마커로 표기했고, 여기 한 번 더 모은다.

1. **3B (a) retrain oracle 수치** — CLAUDE.md/노트엔 "3B (a)-valloss vs (b)≈0.900"이 있으나 **track_d 3B_anchor5 rundir엔 (a) 행이 없음**(파일 확인). 파일-only 원칙상 ⬚로 두고 노트로만 언급(§4.2-2). 별도 `phase2_llm_a_oracle.py` 산출물이 어딘가 있다면 추가 가능 — 미확인.
2. **7B anchor5 arm(MMLU/ROUGE/conv)** — metrics.json에 arm 블록이 비어 **(미기록)**. fidelity·runtime은 정상. 7B std20는 arm 정상 → anchor5만 arm 미수집(의도/누락 불명).
3. **CNN wall-clock** — `track_c/c1/*/metrics.json`에 method별 wall-clock이 있다고 코드 docstring이 명시하나, 본 문서엔 테이블로 전사 안 함(§3.5.2에 (미기록) 표기). 필요 시 c1 metrics에서 추출 가능.
4. **CNN C2 그룹 평균** — 30셀을 threat 4그룹으로 pool해 std가 큼(partition/강도/dataset 혼합). 셀별 정밀치가 필요하면 `RESULTS.txt` 직접 참조. 본 문서는 "그룹 평균" 명시.
5. **Robustness free-rider α 전체 그리드** — §3.4.2 (b2)는 대표 α(0.0/0.5)만 표로; α∈{0.01,0.1,5.0}의 free-rider 칸은 `master_metrics.csv`에 있음(noisy는 전체 α 수록). 지면상 free-rider는 대표값만 — 누락 아님, 압축.
6. **near-additive 동률의 "1.000" 과밀** — silo5·anchor의 clean/noisy/free-rider서 다수 method가 정확히 1.000인 것은 N이 작고(5) 게임이 near-additive라 exact와 동률이기 때문(파일 그대로). poison에서 이 동률이 깨지는 게 핵심 신호.
