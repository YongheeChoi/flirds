---
type: survey
title: "Flirds 실험 결과 전체 한눈에 보기"
created: 2026-06-25
updated: 2026-07-22
tags: [survey, results, experiments, master, fidelity, detection, cost]
---

# Flirds 실험 결과 전체 한눈에 보기

> **기준일**: 2026-07-22 · **git HEAD**: `f5d40a7`(2026-07-21) · **스코프**: flirds 프로젝트의
> 실험 트랙 전체(Foundational `phase1` / LLM standard `track_d` / CNN `track_c` / Robustness
> `phase2_matrix`(+B축 드라이버 `matrix_cxni`) / Signal-size `probe_signal` / Removal-dose `removal_dose` /
> φ-게이팅 `track_g` / 점수원 경쟁 `track_h` / 계측 `measured_2026-07` / β-재실행 `rerun_beta03`)에서
> *실제로 돌아 디스크에 남은* 모든 셀·baseline·하이퍼파라미터를 표로 정리한다.
> 계획만 잡혀 있고 아직 안 돌린 실험도 행으로 넣되 수치는 빈칸(⬚).
> 이 문서는 **최신 상태만** 유지한다 — 갱신 이력 블록을 상단에 쌓지 않으며, 변경 내역은 git 히스토리가 담당.
>
> **문서 구조(2026-07-22 재구조화)**: 실험을 실행-세트명이 아니라 **검증 목적** 기준으로
> 3층 분류한다 — **§3 Main**(논문 본문 후보; 핵심 질문 위계 순 = 1차 fidelity → 2차 selection→성능
> → 탐지) / **§4 Ablation** / **§5 기타 분석 모음**(종합판정·감사·해석). 옛 세트명↔새 위치 매핑 = §2.1.
> 종합 판정(위계별 승·패)은 **§5.1**.
>
> **모든 수치는 아래 파일에서 직접 집계**(기억·CLAUDE.md·메모리 숫자 미사용 — stale 가능). per-seed
> CSV는 method별 mean±std(ddof=0; §3.1.3 E4·E5만 ddof=1)로 집계하고 사람용 요약(RESULTS.txt 등)과 교차검증했다.
> 잔여 작업(GPU 대기·문서 반영 대기)의 정본 목록 = 루트 `REMAINING.md`(2026-07-21 개정).
>
> **수치 출처 파일**
> - LLM standard: `runs/track_d/fidelity.csv` · `runs/track_d/rundirs/*/{config.yaml,metrics.json}` · Fed-LOO 스위트 `runs/track_d/rundirs_e4_fedloo/*` · N=10 oracle `runs/track_d/rundirs_e5_n10/*` · target-stability `runs/track_d/target_stability.csv`(파생)
> - CNN: `runs/track_c/fidelity.csv`(파생) · `runs/track_c/RESULTS.txt` · `runs/track_c/{c1,c2,c1_oracle}/*/{config,metrics}.json`
> - Robustness: `runs/phase2_matrix/analysis/00_overview/master_metrics.csv`(파생; `make_analysis.py` 재생성) · `runs/phase2_matrix/rundirs/*/{config.yaml,meta.json}` — ⚠ 1B_silo5 오염 4셀은 **β0.3 재실행판(ce0b454, 2026-07-20)이 canonical**(재실행 전 값은 git 이력) · frdelta `runs/phase2_matrix/rundirs_2026-07/1B_silo5_frdelta/*`
> - Foundational: `runs/phase1/rundirs/*/{config.yaml,metrics.json}`
> - Signal-size probe(§4.2–4.3): LLM A축 `runs/probe_signal/rundirs/1B_*` · `runs/probe_signal/noise_probe/*` · CNN A축 `runs/probe_signal/cnn_c{1,2}/pc*` · B축 매트릭스(§3.1.5) `runs/phase2_matrix/rundirs/1B_{iid5,silo5}_*`(드라이버=`runs/matrix_cxni/`) — 배경 [[flirds-signal-size-diagnosis]]
> - Removal-dose(§4.4–4.6): `runs/removal_dose/rundirs*/` (LLM A2·B·A1·D=AdamW 3-seed + CNN A3 `rundirs_cnn/`)
> - 계측(§3.4.1·§6.3): `runs/measured_2026-07/{taylor,e3_cost_smoke,timing_device100,microbench,acct,loss_heur_acct,tf32_ab}/`
> - φ-부호 감사(§5.2): `runs/track_g/audit/` (전 rundir 파생 재분석) · φ-게이팅 Phase B(§3.2.3–4): `runs/track_g/rundirs/`(LLM 218; ⚠ `rundirs_llm/` 폴더는 존재하지 않음) + `rundirs_cnn/`(36) + `rundirs_cnn_v3/`(12) · 정본 `runs/track_g/analysis/{llm_summary.csv,cnn_summary.csv}`
> - 점수원 경쟁(§3.2.6·§4.8): `runs/track_h/rundirs_cnn/`(204=경쟁 96+P5 108) · `rundirs_llm/`(12) · `rundirs_cnn_scale/`(21) · `rundirs_cnn_dyn/`(9) · 정본 `runs/track_h/analysis/*.csv` + `scale/analysis/` + `dyn/analysis/`
> - β-재실행(§4.7): `runs/rerun_beta03/figures/{beta_provenance.csv,beta_contrast_3b.csv}` · 재개법 `runs/rerun_beta03/RESUME_AFTER_MIGRATION.md`
>
> **수치 재생성 경로** (rundir만으로 재실행 가능, GPU 불필요):
> - `python runs/track_d/make_fidelity.py` → `runs/track_d/fidelity.csv` (1B/3B/7B × std20/anchor5)
> - `python runs/phase2_matrix/make_analysis.py` → `runs/phase2_matrix/analysis/*` (`RESULTS.md`는 별도 `make_report.py`; 둘 다 gitignored 파생 산출물. ⚠ iid5·silo5_clean·frdelta 셀은 미포함 → rundir 직접 집계)
> - CNN `fidelity.csv` = `codes/slurm/scripts/merge_oracle_a.py`; `RESULTS.txt`는 Track C 결과 스크립트가 재생성
> - track_g/track_h/probe/removal/β: 각 폴더 `make_analysis.py`/`make_figures.py` (rundir-only)
> - Figure: 실험별 `python runs/<exp>/make_figures.py` → `runs/<exp>/figures/`(+`MANIFEST.md`) — 본 문서는 그림을 임베드하지 않는다(구 임베드 사본 폴더 `overview-figures-2026-07/`·`removal-dose-2026-07/`는 정리 커밋 e3b40e5로 삭제됨)
>
> 자매 문서(중복 X, 링크만): baseline 수치가 각 방법 원 논문과 얼마나 맞는지는
> [[baseline-original-paper-verification]]. 선행연구 지형 분류는 [[prior-work-taxonomy/README]].
> 수학 검증(P1–P8)은 [[irds-fl-math-rigor-2026-07/irds-fl-math-rigor]].
> **논문 실험 배치안(본문/ablation/appendix) = [[paper-experiment-placement-plan]]** (2026-07-22 신규).

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
- **표 헤더의 ↑/↓** = 그 열 지표가 **클수록 좋음(↑) / 작을수록 좋음(↓)**. Spearman·Kendall·Pearson·AUROC·MMLU·ROUGE-L·정확도·rho/Jaccard = ↑ · 거리(cosine_d·euclid_d·max_diff)·val-loss·runtime = ↓ · ASR·flirds_keep 등은 방법 우열이 아니라 설정/산출이라 방향 표기 없음. 열이 α·threat·scale·arm 이름인 표는 캡션에 지표 화살표를 단다.

**오라클/기준점**
- **In-run oracle (b)** = 한 학습 궤적에서 exact 2ᴺ 분해 (full-participation→2ᴺ 열거; partial→exact per-round 분해). 1차 fidelity 정답.
- **Retrain oracle (a)** = 조합마다 처음부터 재학습한 exact 2ᴺ Shapley (val-loss utility). 별도 정답(문헌 공백).
- **Flirds proxy reference** = 정확 oracle이 비싼 칸(device100 비-anchor)에서 검증된 Flirds를 기준 대용으로 씀 → 그 칸의 Spearman은 *vs Flirds*임에 주의.
- **Flirds (1st-order only) = Flirds-1st** (2차 Hessian 항 끔).

**마커**: ● 실측(파일에 수치 있음) · ◐ 기준점만/부분(예: 1-seed) · ○ 설계상 제외 · ⬚ 미실행(계획·빈칸) · – 해당없음 · **(미기록)** = 파일에 칸은 있으나 값 없음.

---

## 2. 마스터 한눈에 표 (실험당 1행)

> 참여형태 표기: `full`=매 라운드 전원, `k/N`=라운드당 k명 참여, `K%`=비율. Federation은 전부 **F**.
> "valuation 기반" 열은 *주체 방법(Flirds=in-run)* 기준이며, 같은 셀에서 비교한 baseline은 recon/IF를 함께 포함(섹션의 baseline-set 노트 참조).
> **§ 열 = 본 문서에서 그 실험의 결과가 실리는 섹션**(재구조화 후 위치; 매핑 상세 = §2.1).

| #     | 실험 (rundir 코드)                                                                              | §                | Model · N · 참여                      | Unit   | valuation 기반 (+baseline)                  | 검증 목적                                   | Exact/Approx                | oracle / truth                       | seeds        | status                  |
| ----- | ------------------------------------------------------------------------------------------- | ---------------- | ----------------------------------- | ------ | ----------------------------------------- | --------------------------------------- | --------------------------- | ------------------------------------ | ------------ | ----------------------- |
| 1     | LLM standard · 1B · std20 (`track_d/1B_std20`)                                              | §3.1.1·§3.2.1·§3.4.2 | Llama-3.2-1B · N=20 · 2/20          | client | in-run (+recon/IF)                        | Fidelity·Perf·Cost                      | approx vs exact(b)          | (b) exact per-round                  | 3            | ●                       |
| 2     | LLM standard · 1B · anchor (`track_d/1B_anchor5`)                                           | §3.1.1·§3.2.1·§3.4.2 | Llama-3.2-1B · N=5 · full           | client | in-run (+recon/IF)                        | Fidelity·Perf·Cost                      | approx vs exact(a)&(b)      | **(a) 2⁵ retrain + (b) 2⁵ in-run**   | 3            | ●                       |
| 3     | LLM standard · 3B · std20 (`track_d/3B_std20`)                                              | §3.1.1·§3.2.1·§3.4.2 | Llama-3.2-3B · N=20 · 2/20          | client | in-run (+recon/IF)                        | Fidelity·Perf·Cost                      | approx vs exact(b)          | (b) exact per-round                  | 3            | ●                       |
| 4     | LLM standard · 3B · anchor (`track_d/3B_anchor5`)                                           | §3.1.1·§3.2.1·§3.4.2 | Llama-3.2-3B · N=5 · full           | client | in-run (+recon/IF)                        | Fidelity·Perf·Cost                      | approx vs exact(b)          | (b) 2⁵ in-run · **(a) ⬚**            | 3            | ●                       |
| 5     | LLM standard · 7B · std20 (`track_d/7B_std20`)                                              | §3.1.1·§3.2.1·§3.4.2 | Llama-2-7B · N=20 · 2/20            | client | in-run (+recon/IF)                        | Fidelity·Perf·Cost                      | approx vs exact(b)          | (b) exact per-round                  | 3            | ●                       |
| 6     | LLM standard · 7B · anchor (`track_d/7B_anchor5`)                                           | §3.1.1·§3.2.1·§3.4.2 | Llama-2-7B · N=5 · full             | client | in-run (+recon/IF)                        | Fidelity·Perf·Cost                      | approx vs exact(b)          | (b) 2⁵ in-run · **(a) ⬚**            | 3            | ●                       |
| 7     | CNN · cross-silo N=10 (`track_c/c1`+`c1_oracle`)                                            | §3.1.2·§3.4.3    | 소형 CNN · N=10 · full                | client | in-run (+recon/IF)                        | **Fidelity·Stability**·Cost             | approx vs exact(a)&(b)      | **(a) 2¹⁰ retrain + (b) 2¹⁰ in-run** | 3            | ●                       |
| 8     | CNN · cross-device N=100 (`track_c/c2`)                                                     | §3.2.2·§3.3.5    | 소형 CNN · N=100 · 10/100             | client | in-run intervention arms (+recon/sfedavg) | **Perf·Detection**                      | – (개입 arm)                  | corrupt 마스크                          | 3            | ●                       |
| 9     | Robustness · 1B cross-silo N=5 (`phase2_matrix/1B_silo5_*`)                                 | §3.3.1(·§3.1.4)  | Llama-3.2-1B · N=5 · full           | client | in-run (+recon/IF/탐지기4)                   | **Fidelity·Detection**·Cost             | approx vs exact(b)          | (b) 2⁵ in-run                        | 3            | ● (오염 4셀=β0.3 재실행판 ce0b454) |
| 10    | Robustness · 1B cross-device N=100 α-sweep (`.../1B_device100-a{0,0.01,0.1,5.0}_*`)         | §3.3.2           | Llama-3.2-1B · N=100 · 10/100       | client | in-run (+IF/ComFedSV/탐지기4)                | **Detection**·Fidelity                  | approx vs Flirds-proxy      | **Flirds proxy reference**           | 3            | ●                       |
| 11    | Robustness · 1B cross-device N=100 α=0.5 **Anchor cell** (`.../1B_device100-a0.5_*_anchor`) | §3.3.2           | Llama-3.2-1B · N=100 · 10/100       | client | in-run (+recon/IF/ComFedSV/탐지기4)          | **Fidelity·Detection**·Cost             | approx vs exact(b)-perround | **(b) per-round**                    | 3            | ●                       |
| 12    | Robustness · 1B cross-device N=100 poison (`.../1B_device100-a{0,0.5}_poison`)              | §3.3.2           | Llama-3.2-1B · N=100 · 10/100       | client | in-run (+IF/ComFedSV/탐지기4)                | **Detection**·Fidelity                  | approx vs Flirds-proxy      | Flirds proxy                         | 3            | ●                       |
| 13    | Robustness · 3B cross-silo N=5 (`phase2_matrix/3B_silo5_*`)                                 | §3.3.3           | Llama-3.2-3B · N=5 · full           | client | in-run (+IF/탐지기4)                         | Fidelity·Detection                      | approx vs exact(b)          | (b) 2⁵ in-run                        | **1**        | ◐                       |
| 14    | Foundational · 1B 첫 clean run (`phase1/...full-lr*`)                                        | §6.1·§3.2.5      | Llama-3.2-1B · N=5 · full           | client | in-run                                    | Detection·Perf(selection)               | approx                      | 주입 라벨(oracle 없음)                     | 3×2lr        | ●                       |
| 15    | Foundational · 1B LR sweep (`phase1/...sweep-lr*`)                                          | §6.3(검증-전용)      | Llama-3.2-1B · N=5 · full           | client | in-run                                    | Detection·Perf(selection)               | approx vs exact(b)          | (b) 2⁵ in-run                        | 1×4lr        | ●                       |
| 16    | Signal-size probe C1 (`probe_signal/cnn_c1`)                                                | §4.3             | 소형 CNN · N=10 · 2·5·10/10           | client | in-run (+recon/IF)                        | **Fidelity** (폭·참여 sweep)               | approx vs exact(b)          | (b) 2¹⁰/per-round in-run             | 3            | ●                       |
| 17    | Signal-size probe C2 (`probe_signal/cnn_c2`)                                                | §4.3             | 소형 CNN · N=100 · 5·10/100           | client | in-run intervention arms (+sfedavg)       | **Perf·Detection** (폭·참여 sweep)         | – (개입 arm)                  | corrupt 마스크                          | 3            | ●                       |
| 18    | Signal-size probe LLM A축 (`probe_signal/rundirs`+`noise_probe`)                             | §4.2             | Llama-3.2-1B · N=5·full / N=50·5/50 | client | in-run (+recon/IF)                        | **Fidelity** (rank·참여·lr·steps·noise)   | approx vs exact(b)          | (b) 2⁵ / per-round                   | 1–3 (셀별)     | ● (lr격자 st10·std50k5 r16·noise=3-seed, 나머지 seed0) |
| 19    | Signal-size B축 매트릭스 (`phase2_matrix/1B_{iid5,silo5}_*`; 드라이버 `matrix_cxni`)                | §3.1.5           | Llama-3.2-1B · N=5 · full           | client | in-run (+탐지기4)                            | **Fidelity·Detection** (오염×비IID)        | approx vs exact(b)          | (b) 2⁵ in-run                        | 3            | ●                       |
| 20    | Fed-LOO 스위트 E4 (`track_d/rundirs_e4_fedloo`)                                               | §3.1.3           | Llama-3.2-1B · N=20 2/20 + N=5 full | client | in-run + **Fed-LOO**(in-run LOO)          | **Fidelity·Cost** (LOO 공백 메움)          | approx vs exact(b)          | (b) per-round / 2⁵                   | 3            | ●                       |
| 21    | N=10 exact oracle E5 (`track_d/rundirs_e5_n10`)                                             | §3.1.3           | Llama-3.2-1B · N=10 · full          | client | in-run (경량 4종)                            | **Fidelity·Cost** (2¹⁰ 실측)             | approx vs exact(b)          | **(b) exact 2¹⁰**                    | 1 (seed0)    | ◐                       |
| 22    | Robustness frdelta E7 (`phase2_matrix/rundirs_2026-07/1B_silo5_frdelta`)                    | §3.3.4           | Llama-3.2-1B · N=5 · full           | client | in-run (+Fed-LOO/탐지기4)                    | **Detection** (delta-FR 한계)            | approx vs exact(b)          | (b) 2⁵ in-run                        | 3            | ●                       |
| 23    | Taylor 물리잔차 E2 (`measured_2026-07/taylor`)                                                  | §5.5(보류)         | Llama-3.2-1B · N=5 · full           | client | – (계측)                                   | **검증(명제 P3)**                          | –                           | exact ΔL per-coalition               | 3            | ●                       |
| 24    | φ-부호 감사 Stage 0 (`track_g/audit`, derived)                                                 | §5.2             | 기존 전 rundir 309개                    | client | – (재분석)                                   | **Track G 예측 확정/수정**                   | –                           | rundir φ 부호                          | –            | ●                       |
| 25    | φ 부호-게이팅 Phase B (`track_g/rundirs`+`rundirs_cnn(+_v3)`)                                  | §3.2.3–4         | 1B N=5 full (silo5·iid5) + 소형 CNN N=100 | client | in-run 게이트 arms (sign/z/V2w/V3)          | **Perf(온라인 배제)·게이트 P/R**              | – (개입 arm)                  | oracle_excl 상한 + corrupt 마스크        | 3 (일부 seed0) | ● (V2w 불승격)             |
| 26    | Track H 확증 런 3종 — P5 신뢰-게이트(108런)·Scale 완전참여(21런)·Dyn 재추첨(9런) (`track_h/rundirs_cnn*`)   | §4.8             | 소형 CNN · N=100 · 10/100 및 100/100   | client | in-run 게이트 arms (P1/P5h/P5s)             | **Perf(정책 확증; 사전등록 예측 대조)**          | – (개입 arm)                  | oracle_excl/random_excl + corrupt 마스크 | 3            | ●                       |
| —     | **이하 계획 행 (⬚=미실행; 완료된 행은 status ●/◐·§ 참조)**                                               |                  |                                     |        |                                           |                                         |                             |                                      |              |                         |
| P1    | LLM N=10 (a)/(b) oracle                                                                     | §3.1.3           | LLM · N=10                          | client | retrain+in-run                            | Fidelity(고-power)                       | exact                       | (a)/(b) 2¹⁰                          | (b)=1        | ◐ (b) seed0 완료=행21; (a) 2¹⁰·seeds1-2 ⬚(장기 대기, 루트 REMAINING §1.4) |
| P2    | LLM 3B anchor **(a) retrain oracle**                                                        | §3.1.1           | Llama-3.2-3B · N=5                  | client | retrain                                   | Fidelity(dual oracle)                   | exact                       | (a) 2⁵ retrain                       | ⬚            | ⬚                       |
| P3    | LLM 7B anchor **(a) retrain oracle**                                                        | §3.1.1           | Llama-2-7B · N=5                    | client | retrain                                   | Fidelity(dual oracle)                   | exact                       | (a) 2⁵ retrain                       | ⬚            | ⬚                       |
| P4    | Robustness · 7B (silo5/device100)                                                           | –                | Llama-2-7B                          | client | in-run+탐지기                                | Detection·Fidelity                      | approx vs (b)               | (b)                                  | ⬚            | ⬚                       |
| P5    | Robustness · 3B 3-seed 완성                                                                   | §3.3.3           | Llama-3.2-3B · N=5                  | client | in-run+탐지기                                | Detection·Fidelity                      | approx vs (b)               | (b) 2⁵                               | ⬚ (현 1 seed) | ⬚ (β0.3 재실행 잔여 18셀에 3B silo5 4셀 포함 — 루트 REMAINING §1.2) |
| P6    | Fairness·reward 전용 실험                                                                       | –                | –                                   | client | –                                         | Fairness·reward                         | –                           | –                                    | ⬚            | ⬚ 미설계                   |
| P7    | Removal/selection curve (`removal_dose/*_removal_*`+`rundirs_cnn`)                          | §4.4             | 1B silo5 N=5 + track_d anchor5 + CNN N=10(A3) | client | in-run+(a)retrain                         | **Fidelity(게임-무관 downstream; C-1/C-4)** | – (removal 재학습)             | (a) 재학습 val-loss                     | 3            | ●                       |
| P8    | Dose-response (`removal_dose/*_dose_*`)                                                     | §4.5             | 1B silo5 N=5                        | client | in-run(+탐지기)                              | **Fidelity(φ vs 오염강도; C-3)**            | approx vs (b)               | (b) 2⁵                               | 3            | ●                       |
| P9    | AdamW-fidelity (`removal_dose/*_adamw`)                                                     | §4.6             | 1B anchor5 N=5                      | client | in-run                                    | **Fidelity(external-validity; C-5)**    | approx vs (a)&(b)           | (a)/(b) 2⁵                           | 3            | ●                       |
| P10   | **Track H 점수원 경쟁** (`track_h/rundirs_cnn`+`rundirs_llm`)                                   | §3.2.6           | CNN N=100 dir1 + 1B silo5 noisy (+std50k5 seed0) | client | in-run **전 점수원 8종** × 정책 4(sign±가중/mult/z) × 시점 2(online/retrain) | **Perf(경쟁: 어느 φ 정의가 학습을 잘 만드나)**   | – (개입 arm)                  | oracle_excl 상한 + corrupt 마스크        | 3 (Tier1·2) / R2 seed0 | ◐ Tier 1 CNN 96런+Tier 2 LLM 12런 완주(FAIL 0); Tier 3 std50k5 12런은 07-21 REMAINING 개정판 큐 미등재(std50k5 서술=seed0 파일럿 동결·LLM 경쟁 무대는 R4가 대체) |
| P11   | **Track H R4 — gsm50k5 accuracy 무대** (`REGIME=gsm50k5`)                                     | (미착지)            | Llama-3.2-1B · N=50 · 5/50 · GSM8K  | client | in-run 게이트 arms (P1·T2, 이후 P5-leg)       | **Perf(LLM 경쟁; exact-match 심판)**       | – (개입 arm)                  | oracle_excl + corrupt 마스크           | Tier A=seed0 | ◐ **Tier A seed0 서버 실행 중**(2026-07-20 23:29~; rundir 미착지 — 수치 없음. 루트 REMAINING §1.1; 스펙 `runs/track_h/README.md` §1.6, 예측 H-8~11) |
| Exp C | (b) target self-stability (derived)                                                         | §5.4(보류)         | 기존 track_d/phase2 rundir            | client | – (재분석)                                   | **Stability(C-2)**                      | –                           | (b) φ xseed                          | 3            | ● (재실행 0, 로컬 완료)        |

> **"검증 목적" 용어 정의** (각 본문 섹션이 정본; E1–E7 매핑은 §5.7):
> - **Fidelity** = 추정 φ가 정답 oracle의 *기여도 순위/값*을 얼마나 재현하나 (vs (a)/(b); Spearman·Kendall·Pearson·거리). **1차 핵심.** → §3.1
> - **Perf**(= Selection→downstream performance) = 측정한 φ로 클라를 *선택/가중*해 학습했을 때 다운스트림 성능(MMLU·ROUGE·정확도)이 오르나/유지되나. → §3.2
> - **Aggregation**(= Aggregation quality) = φ-가중 *집계*가 만든 글로벌 모델의 품질(특히 오염 하 CNN 정확도). Perf와 같은 표에서 측정. → §3.2
> - **Conv**(= Convergence) = 목표 손실까지 *수렴 속도*. **(overview 스코프 제외, 2026-07-22 — clean-IID에서 arm 간 사실상 동률·7B std20 차이도 seed-std와 중첩. 데이터는 rundir 존속, 포인터 = §5.6.)**
> - **Detection**(= Corrupt-client detection) = φ(또는 탐지기)로 오염 클라를 *분리*하는 AUROC. 위계상 마지막(기여도≠탐지). → §3.3
> - **Cost**(= Cost·scalability) = 방법별 *wall-clock* 비용. → §3.4
> - **Stability**(= replication) = φ 순위의 *seed 간 재현성*(rho_xseed·Jaccard); oracle 자체 안정성 대비. → §3.1.2
> - **Fairness·reward** = 공정한 보상 분배 관점(공리·ECDF 등) — 본 프로젝트는 *전용 실험 미설계*(P6, ⬚).

### 2.1 옛 실행 세트명/코드 ↔ 새 분류 위치 매핑 (추적성)

> 2026-07-22 재구조화 전 문서(git 이력)와 실행-세트 코드로 결과를 찾던 독자를 위한 대응표.
> 실행 세트명·rundir 코드는 각 섹션의 "(c) 출처" 블록에만 유지한다.

| 옛 세트명 / 실행 코드 | 내용 | 새 위치 |
|---|---|---|
| `track_d` 본체 (구 §3.1.1·§3.2.1·§3.3.1·§3.5.1) | LLM 표준 무대 fidelity·개입·runtime | §3.1.1 · §3.2.1 · §3.4.2 (수렴은 스코프 제외 §5.6) |
| `track_d` E4 (`rundirs_e4_fedloo`, 구 §3.1.6) | Fed-LOO 경량 스위트 | §3.1.3 |
| `track_d` E5 (`rundirs_e5_n10`, 구 §3.1.6) | N=10 exact 2¹⁰ | §3.1.3 (비용 확장 §3.4.3) |
| Exp C (파생 `target_stability.csv`, 구 §3.1.5) | (b) target self-stability | §5.4 (보류: appendix 후보) |
| E2 (`measured_2026-07/taylor`, 구 §3.1.7) | Taylor 물리잔차(P3) | §5.5 (보류: appendix 후보) |
| E7 (`1B_silo5_frdelta`, 구 §3.4.5) | delta-재활용 free-rider | §3.3.4 |
| `track_c` C1+`c1_oracle` (구 §3.1.2·§3.1.4·§3.5.2) | CNN 듀얼 오라클 fidelity·안정성·runtime | §3.1.2 · §3.4.3 |
| `track_c` C2 (구 §3.2.2·§3.3.2) | CNN cross-device 개입 | §3.2.2 (+탐지 §3.3.5) |
| `phase2_matrix` 25셀 (구 §3.1.3·§3.4.1–4) | 오염 무대 fidelity+탐지 | §3.1.4(요지) · §3.3.1–3(표) |
| `matrix_cxni` → `phase2_matrix/rundirs/1B_{iid5,silo5}_*` (구 §3.6.4) | B축 오염×비IID 매트릭스 | §3.1.5 |
| `probe_signal` LLM A축 (`rundirs`+`noise_probe`, 구 §3.6.1) | rank·참여·lr·steps·noise lever | §4.2 |
| `probe_signal` CNN (`pc1_*`/`pc2_*`, 구 §3.6.2–3) | 폭×참여 probe | §4.3 |
| `removal_dose` A2·A3 (구 §3.7.1–2·§3.7.5) | removal-curve(게임-무관) + poison 한계 | §4.4 |
| `removal_dose` B (구 §3.7.3) | dose-response | §4.5 |
| `removal_dose` A1·D (구 §3.7.4) | (a)oracle 브리지·AdamW | §4.6 |
| `rerun_beta03` (구 §4.2 caveat 9) | β 통일 재실행·provenance | §4.7 (+캠페인 상태 = §6.2 caveat 9) |
| `track_g` Stage 0 (`audit/`, 구 §3.8) | φ-부호 감사 | §5.2 |
| `track_g` Phase B (구 §3.2.3–4) | φ 부호-게이팅 + V3 | §3.2.3–4 (번호 유지) |
| `track_h` Tier 1·2·R2 (구 §3.2.6) | 점수원 경쟁 본판 | §3.2.6 (번호 유지) |
| `track_h` P5·Scale·Dyn (구 §3.2.6 하위 블록) | 확증 런 3종 | §4.8 |
| `track_h` R4 (`gsm50k5`) | LLM accuracy 경쟁 무대 | §2 P11 (실행 중 — 결과 미착지) |
| `phase1` full (구 §4.1·§3.2.5) | foundational 탐지·top-k selection | §6.1 · §3.2.5 |
| `phase1` LR sweep (구 §4.1) | lr 민감도 | §6.3 (검증-전용 축약) |
| `measured_2026-07` op-count·microbench (구 §3.5.3 ①) | 하드웨어-독립 비용 축 | §3.4.1 |
| `measured_2026-07` timing·e3·tf32_ab·acct (구 §3.5.3 ②③) | 계측 세부 | §6.3 (검증-전용 축약) |
| 구 §2.5·§3.2.7 종합판정 | 위계별 승·패 분석 | §5.1 |
| 구 §3.3 Convergence | 수렴 속도 | **삭제** — 데이터 포인터 = §5.6 |

---

# 3. Main — 논문 본문 후보 실험 (핵심 질문 위계 순)

> 위계(루트 CLAUDE.md): **1차 = Fidelity**(기여도 추정 정확도) → 2차 = ① Selection→performance / Aggregation → ③ Detection → 비용. 아래 섹션 순서가 이 위계다(수렴 축은 overview 스코프 제외 — §5.6).
> 모든 실험 섹션 = **(a) 세팅 / (b) 결과 / (c) 출처·baseline-set 노트** 3블록.

## 3.1 Fidelity (1차 핵심) — 기여도 추정이 정답 oracle 순위를 얼마나 재현하나

### 3.1.1 LLM 표준 무대 fidelity + 듀얼 오라클 (`track_d`)

**(a) 세팅**
- model: Llama-3.2-1B-Instruct / Llama-3.2-3B-Instruct / **Llama-2-7B-hf** (7B = FL-LLM 문헌 표준 rung)
- 두 스테이지: **Standard stage `std20`** = N=20, 라운드당 2명 참여, R=200, local 10 steps · **Precision stage `anchor5`** = N=5 전원 참여, R=30, local 10 steps
- batch 16(1B/3B) / 4(7B); lr=1e-3, optimizer **plain SGD momentum=0**; **LoRA r=16, α=32** (target = q/k/v/o/gate/up/down proj); dataset = alpaca-gpt4 20k IID (clean, 오염 없음); seq len 512; val=200 / test=1000; fp32
- oracle: **(b) in-run** — std20=exact per-round 분해(2/round), anchor5=exact 2⁵ 전수; **(a) retrain 2⁵ = 1B anchor5에서만** (3B/7B는 ⬚)
- seeds=3 (각 스테이지·스케일)

**(b) 결과 — vs (b) in-run oracle, method별 3-seed mean±std**

#### std20 스테이지 (N=20, 2/round) — 순위·값 상관

| method | 1B Spearman ↑ | 1B Kendall ↑ | 1B Pearson ↑ | 3B Spearman ↑ | 3B Kendall ↑ | 3B Pearson ↑ | 7B Spearman ↑ | 7B Kendall ↑ | 7B Pearson ↑ |
|---|---|---|---|---|---|---|---|---|---|
| **Flirds** | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 0.999±.001 | 0.996±.005 | 1.000±.000 |
| **Flirds-1st** | 0.999±.001 | 0.996±.005 | 1.000±.000 | 0.997±.002 | 0.982±.013 | 1.000±.000 | 0.998±.001 | 0.986±.010 | 1.000±.000 |
| loss-heur | 1.000±.000 | 1.000±.000 | 1.000±.000 | 0.999±.001 | 0.996±.005 | 1.000±.000 | 0.999±.001 | 0.996±.005 | 1.000±.000 |
| GTG | 0.975±.018 | 0.916±.043 | 0.995±.001 | 0.990±.005 | 0.947±.026 | 0.997±.000 | 0.977±.017 | 0.916±.045 | 0.989±.006 |
| FedSV | 0.910±.073 | 0.786±.117 | 0.959±.013 | 0.966±.006 | 0.870±.018 | 0.973±.004 | 0.968±.010 | 0.881±.030 | 0.976±.006 |
| FedIF | 0.157±.303 | 0.111±.199 | 0.229±.222 | 0.203±.194 | 0.132±.122 | 0.264±.151 | 0.480±.101 | 0.323±.061 | 0.508±.054 |
| ShapleyFL | 0.194±.351 | 0.133±.244 | 0.245±.283 | 0.211±.158 | 0.147±.101 | 0.178±.163 | 0.406±.081 | 0.274±.054 | 0.431±.026 |
| ComFedSV | 0.093±.146 | 0.060±.108 | 0.095±.193 | -0.137±.065 | -0.098±.030 | -0.104±.040 | 0.039±.171 | 0.039±.110 | 0.048±.115 |

#### std20 스테이지 — 거리 (↓)

| method | 1B cosine_d ↓ | 1B euclid_d ↓ | 1B max_diff ↓ | 3B cosine_d ↓ | 3B euclid_d ↓ | 3B max_diff ↓ | 7B cosine_d ↓ | 7B euclid_d ↓ | 7B max_diff ↓ |
|---|---|---|---|---|---|---|---|---|---|
| **Flirds** | .0000 | .0000 | .0000 | .0000 | .0000 | .0000 | .0000 | .0001 | .0000 |
| **Flirds-1st** | .0000 | .0006 | .0002 | .0000 | .0004 | .0002 | .0000 | .0002 | .0001 |
| loss-heur | .0000 | .0001 | .0001 | .0000 | .0001 | .0000 | .0000 | .0000 | .0000 |
| GTG | .0007 | .0010 | .0005 | .0005 | .0009 | .0005 | .0015 | .0011 | .0005 |
| FedSV | .0057 | .0028 | .0017 | .0033 | .0024 | .0013 | .0055 | .0023 | .0012 |
| FedIF | .1911 | 2.636 | .9760 | .1781 | 2.605 | .9837 | .1141 | 2.667 | .9870 |
| ShapleyFL | .2087 | 2.737 | .9905 | .2471 | 2.863 | .9954 | .1570 | 2.733 | .9946 |
| ComFedSV | .8447 | .0262 | .0098 | 1.026 | .0292 | .0104 | .9264 | .0213 | .0077 |

#### anchor5 스테이지 (N=5, full) — 순위·값 상관

| method | 1B Spearman ↑ | 1B Kendall ↑ | 1B Pearson ↑ | 3B Spearman ↑ | 3B Kendall ↑ | 3B Pearson ↑ | 7B Spearman ↑ | 7B Kendall ↑ | 7B Pearson ↑ |
|---|---|---|---|---|---|---|---|---|---|
| **Flirds** | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 |
| **Flirds-1st** | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 |
| loss-heur | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 |
| GTG | 1.000±.000 | 1.000±.000 | 1.000±.000 | 0.967±.047 | 0.933±.094 | 0.999±.000 | 1.000±.000 | 1.000±.000 | 1.000±.000 |
| FedSV | 0.700±.163 | 0.600±.163 | 0.824±.116 | 0.667±.205 | 0.600±.163 | 0.881±.054 | 0.933±.047 | 0.867±.094 | 0.961±.024 |
| ShapleyFL | 0.700±.283 | 0.600±.283 | 0.764±.258 | 0.167±.094 | 0.067±.094 | 0.280±.496 | 0.833±.125 | 0.733±.189 | 0.903±.067 |
| ComFedSV | 0.500±.432 | 0.467±.340 | 0.563±.356 | 0.600±.327 | 0.533±.340 | 0.451±.298 | 0.600±.216 | 0.467±.189 | 0.588±.256 |
| FedIF | 0.067±.531 | 0.067±.411 | -0.068±.626 | 0.200±.374 | 0.133±.340 | 0.356±.466 | 0.200±.616 | 0.200±.490 | 0.368±.509 |
| **(a)oracle**¹ | 0.933±.047 | 0.867±.094 | 0.933±.054 | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |

¹ **(a) retrain 2⁵ oracle vs (b) in-run oracle** = 듀얼 오라클 일치도 (1B anchor5만 실행). Spearman 0.933±.047 → 두 정답 정의가 거의 동률(완전 일치 아님은 N=5 coarse + retrain noise). 3B/7B (a) = ⬚.

#### anchor5 스테이지 — **모든 방법 vs (a) retrain oracle** (1B anchor5, 3-seed mean±std)

> `make_fidelity.py`는 truth=(b)만 산출해 위 표(본표)는 *vs (b)*다. 아래는 동일 `1B_anchor5/phi.parquet`의 (a) φ를 truth로 잡아 **각 방법을 (a) retrain oracle과 직접 비교**한 값(이 문서에서 추가 산출; CNN §3.1.2는 원래 vs (a)/(b) 둘 다 있음). 3B/7B·std20은 (a)가 없어 ⬚.

| method | Spearman vs (a) ↑ | Kendall vs (a) ↑ | Pearson vs (a) ↑ | max_diff vs (a) ↓ | (참고) Spearman vs (b) ↑ |
|---|---|---|---|---|---|
| Flirds | 0.933±.047 | 0.867±.094 | 0.933±.055 | .001 | 1.000 |
| Flirds-1st | 0.933±.047 | 0.867±.094 | 0.929±.060 | .001 | 1.000 |
| loss-heur | 0.933±.047 | 0.867±.094 | 0.931±.057 | .001 | 1.000 |
| GTG | 0.933±.047 | 0.867±.094 | 0.937±.052 | .002 | 1.000 |
| FedSV | 0.733±.170 | 0.600±.163 | 0.685±.249 | .003 | 0.700 |
| ShapleyFL | 0.767±.330 | 0.733±.377 | 0.916±.084 | .983 | 0.700 |
| ComFedSV | 0.467±.450 | 0.467±.411 | 0.598±.280 | .014 | 0.500 |
| FedIF | 0.167±.613 | 0.200±.490 | 0.048±.585 | .984 | 0.067 |

> 읽기: Flirds·Flirds-1st·loss-heur·GTG가 모두 vs (a) **0.933**으로 동률인 이유 = 이들이 (b)와 거의 완전 일치(vs (b)≈1.000)하므로, 이들의 vs (a) 점수가 곧 **(b)-vs-(a) 일치도(0.933)** 와 같아진다(천장 효과). FedSV/ShapleyFL은 vs (a)가 vs (b)보다 약간 높음(0.733/0.767 > 0.700) = (a)·(b) 두 정답 사이에서 어느 쪽과도 부분적 일치. **출처**: `runs/track_d/rundirs/1B_anchor5_seed{0,1,2}/phi.parquet` (재현: phi.parquet의 method 피벗에서 truth=`(a)oracle`으로 spearmanr/kendalltau/corrcoef).

> anchor5 거리는 생략하지 않음 — 핵심만: Flirds/Flirds-1st/loss-heur cosine_d≈0(<.0001), GTG≈.003–.011, FedSV≈.005–.013; FedIF/ShapleyFL euclid_d≈1.2–1.5(부호 불안정). 전체 6-metric은 `make_fidelity.py` 재실행 시 `fidelity.csv`.

**(c) 출처·baseline-set 노트**
- **출처**: `runs/track_d/fidelity.csv` (재생성: `python runs/track_d/make_fidelity.py`) · 셀별 원시 = `runs/track_d/rundirs/{1B,3B,7B}_{std20,anchor5}_seed{0,1,2}/metrics.json`
- **포함(8종 + (a)/(b) oracle)**: Flirds, Flirds-1st, loss-heur, GTG, FedSV, ComFedSV, ShapleyFL(β=0.3, Def 4.3), FedIF. truth = (b) in-run oracle; anchor5 1B는 (a) retrain oracle도.
- **제외**: **(a) retrain oracle** = 비용(2⁵×R 재학습) → 1B anchor5만, 3B/7B ⬚. **Fed-LOO** = 본 스위트엔 없음 → **별도 경량 재실행으로 실측(§3.1.3 E4, 1B 양 스테이지 3-seed)**. (전역 비교군 제외 2종 = §6.2 caveat 12.)

---

### 3.1.2 CNN 듀얼 오라클 fidelity + cross-seed 안정성 (`track_c` C1, cross-silo N=10)

**(a) 세팅**
- 소형 CNN (**mnist=LeNet5 / cifar10=FedSVCNN**, 전체 모델 학습 — LoRA 아님), **N=10 전원 참여**, R=10, local epochs=5, lr=0.01, batch=64, SGD mom=0; val=2000 / test=8000
- datasets: **mnist, cifar10**; 5 시나리오 = `iid` · `label_skew` · `quantity_skew` · `label_flip` · `feature_noise` (GTG-Shapley 5-시나리오 무대 이식). 10 scenario × 3 seed = 30 셀; (a) oracle도 30 셀.
- oracle: **(a) exact 2¹⁰ retrain** (`oracle.exact_sv.subset_utility_valloss`) + **(b) exact 2¹⁰ in-run** (`oracle.in_run_sv.in_run_shapley`); 둘 다 val-loss 게임. Oracle-a 효율 gap ≤1e-15 (모든 셀).

**(b1) 결과 — 10 scenario × 3 seed pool 한 method별 mean±std** (Spearman/Pearson; Kendall·거리는 아래)

| method | vs (b) Spearman ↑ | vs (b) Pearson ↑ | vs (a) Spearman ↑ | vs (a) Pearson ↑ |
|---|---|---|---|---|
| **Flirds** | **0.919±.134** | **0.934±.128** | 0.352±.462 | 0.354±.461 |
| **Flirds-1st** | 0.832±.194 | 0.853±.159 | 0.408±.435 | 0.412±.421 |
| loss-heur | 0.860±.154 | 0.885±.134 | 0.425±.429 | 0.423±.408 |
| GTG | 0.569±.343 | 0.612±.317 | 0.374±.412 | 0.332±.456 |
| FedSV | 0.401±.410 | 0.410±.406 | 0.284±.479 | 0.215±.466 |
| ComFedSV | 0.348±.377 | 0.328±.396 | 0.338±.398 | 0.309±.431 |
| ShapleyFL | 0.391±.385 | 0.392±.425 | 0.453±.380 | 0.443±.410 |
| FedIF | 0.491±.391 | 0.506±.427 | 0.380±.393 | 0.368±.431 |

> **주의(caveat)**: 위 pool 평균은 **`iid` 셀 포함**(오염·skew 신호가 없어 fidelity가 의미상 낮음) → 깎인 값.
> `iid` 제외 시(8 scenario×3seed=24): Flirds vs (b) **0.928±.136**, Flirds-1st 0.875±.172, loss-heur 0.884±.153, GTG 0.626±.315 — **비교군 내 Flirds 1위**.

**시나리오별 vs (b) Spearman ↑** (3-seed 평균; 신호 강한 칸이 보이게; 값 클수록 충실)

| dataset/scenario | Flirds | Flirds-1st | GTG | FedSV | ComFedSV | ShapleyFL | FedIF | loss-heur |
|---|---|---|---|---|---|---|---|---|
| cifar10 / feature_noise | 1.00 | 0.89 | 0.59 | 0.40 | 0.19 | 0.20 | 0.62 | 0.90 |
| cifar10 / iid | 0.95 | 0.54 | 0.21 | 0.22 | 0.12 | 0.18 | 0.45 | 0.69 |
| cifar10 / label_flip | 1.00 | 0.95 | 0.64 | 0.54 | 0.31 | 0.37 | 0.74 | 0.95 |
| cifar10 / label_skew | 0.98 | 0.92 | 0.49 | 0.53 | 0.31 | 0.29 | 0.68 | 0.88 |
| cifar10 / quantity_skew | 0.99 | 0.96 | 0.78 | 0.56 | 0.67 | 0.44 | -0.20 | 0.98 |
| mnist / feature_noise | 0.79 | 0.70 | 0.41 | 0.13 | 0.21 | 0.48 | 0.57 | 0.78 |
| mnist / iid | 0.81 | 0.78 | 0.47 | 0.04 | 0.10 | 0.47 | 0.73 | 0.84 |
| mnist / label_flip | 1.00 | 0.99 | 0.99 | 0.97 | 0.95 | 0.98 | 0.98 | 0.99 |
| mnist / label_skew | 0.71 | 0.61 | 0.33 | -0.01 | 0.14 | -0.02 | 0.41 | 0.63 |
| mnist / quantity_skew | 0.96 | 0.98 | 0.78 | 0.63 | 0.49 | 0.52 | -0.07 | 0.96 |

**데이터셋별 평균** (위 시나리오 표를 dataset로 묶음; 각 5 scenario × 3 seed = 15)

| dataset | 기준 | Flirds | Flirds-1st | GTG | FedSV | ComFedSV | ShapleyFL | FedIF | loss-heur |
|---|---|---|---|---|---|---|---|---|---|
| cifar10 | vs (b) Sp ↑ | 0.98 | 0.85 | 0.54 | 0.45 | 0.32 | 0.30 | 0.46 | 0.88 |
| mnist | vs (b) Sp ↑ | 0.85 | 0.81 | 0.60 | 0.35 | 0.38 | 0.49 | 0.52 | 0.84 |
| cifar10 | vs (a) Sp ↑ | 0.26 | 0.29 | 0.37 | 0.30 | 0.40 | 0.30 | 0.20 | 0.34 |
| mnist | vs (a) Sp ↑ | 0.44 | 0.53 | 0.38 | 0.27 | 0.27 | 0.60 | 0.56 | 0.51 |

> mnist는 cifar10보다 vs (b) fidelity가 약간 낮고(신호 작음) vs (a) 일치는 높다. 시나리오별 개별 값은 바로 위 시나리오 표 + `fidelity.csv`.

**vs (b) Kendall + 거리 pool** (10 scenario × 3 seed; c1 rundir의 φ에서 파생 계산 — metrics.json 저장 키는 `phi`·`runtime`(+ladder 셀 `auroc`·`spearman_vs_rate`)이고 Kendall·거리 3종은 저장 φ에서 재계산):

| method | Kendall_b ↑ | cosine_d ↓ | euclid_d ↓ | max_diff ↓ |
|---|---|---|---|---|
| Flirds | 0.849±.192 | 0.001±.003 | 0.133±.114 | 0.054±.048 |
| Flirds-1st | 0.733±.223 | 0.009±.020 | 0.207±.186 | 0.085±.068 |
| loss-heur | 0.757±.187 | 0.009±.018 | 0.216±.164 | 0.090±.060 |
| GTG | 0.470±.302 | 0.074±.078 | 0.175±.124 | 0.117±.110 |
| FedSV | 0.324±.344 | 0.225±.185 | 0.547±.418 | 0.346±.346 |
| ComFedSV | 0.270±.318 | 0.329±.221 | 0.756±.507 | 0.490±.370 |
| ShapleyFL | 0.307±.327 | 0.057±.030 | 1.456±.305 | 0.697±.088 |
| FedIF | 0.399±.333 | 0.078±.071 | 1.092±.192 | 0.601±.097 |

**(b2) Stability (replication) — cross-seed 안정성** (같은 C1 런에서 측정; **rho_xseed** = 한 method φ 랭킹의 seed 간 상관 ↑, **topJ/botJ** = 상위/하위 20% 클라셋 Jaccard)

| method | rho_xseed ↑ (10 scenario pool) | topJ ↑ | botJ ↑ |
|---|---|---|---|
| (b)oracle (자체) | 0.518±.453 | 0.522±.395 | 0.555±.453 |
| **Flirds** | **0.547±.394** | 0.544±.324 | 0.500±.419 |
| Flirds-1st | 0.510±.461 | 0.611±.395 | 0.455±.386 |
| loss-heur | 0.474±.448 | 0.500±.287 | 0.467±.384 |
| GTG | 0.311±.441 | 0.467±.358 | 0.378±.345 |
| FedSV | 0.289±.385 | 0.356±.351 | 0.345±.292 |
| FedIF | 0.243±.413 | 0.322±.296 | 0.244±.293 |
| ComFedSV | 0.198±.383 | 0.300±.268 | 0.289±.366 |
| ShapleyFL | 0.124±.431 | 0.200±.276 | 0.344±.331 |

> 읽기: (b) oracle 자체의 cross-seed 안정성이 0.518(CNN은 seed별 기여가 실제로 갈림) → **Flirds(0.547)는 oracle의 내재 안정성을 그대로 추종**, recon MC baseline(GTG/FedSV/ComFedSV/ShapleyFL)은 추가 분산으로 0.12~0.31로 떨어짐. (LLM 무대의 *(b) target 자체* 안정성 = §5.4.)

**(c) 출처·baseline-set 노트**
- **출처**: `runs/track_c/fidelity.csv` (열 `spearman_b/pearson_b/spearman_a/pearson_a`; 재생성 = `codes/slurm/scripts/merge_oracle_a.py`) · `runs/track_c/RESULTS.txt` (C1 stability 절 — (b2)는 그 인쇄값 기준 풀링) · 셀별 원시 = `runs/track_c/c1/*/metrics.json`. 코드 = `codes/experiments/track_c1.py`.
- **포함(8종 + (a)/(b))**: Flirds, Flirds-1st, loss-heur, GTG, FedSV, ComFedSV, ShapleyFL, FedIF. truth = (a) 2¹⁰ retrain + (b) 2¹⁰ in-run 듀얼.
- **제외**: 탐지기(FLDetector/FLTrust/STD-DAGMM/FedDQC) = C1엔 update-level 위협 축이 없음(시나리오는 skew/flip/noise) → *적용규칙: 탐지기는 오염축 있는 실험만* (탐지는 §3.2.2 C2·§3.3). (a)/(b) exact 가능 = N=10 ≤ 10이라 *exact 2ᴺ 규칙 충족*.

> **확장**: 이 C1을 **모델 폭 w×참여 k로 sweep**한 신호크기 probe = **§4.3** (폭·참여는 fidelity·φ 크기·신호 실재성을 안 키움). C1 ladder 셀의 **φ-as-detector AUROC** = §3.3.5.

---

### 3.1.3 확장: Fed-LOO 스위트 + N=10 exact 2¹⁰ oracle (`track_d` E4·E5)

**(a) 세팅** — §3.1.1과 동일 무대(1B, LoRA r16/α32, plain SGD, lr 1e-3, 10 steps, val 200)의 **경량 재실행**(method 4종만 = Fed-LOO·Flirds·Flirds-1st·loss-heur; coalition baseline off = `METHODS` 필터). truth = (b) in-run; (a) retrain off.
- **E4 Fed-LOO** (`runs/track_d/rundirs_e4_fedloo/`, 커밋 155324b): std20(N=20, 2/20, R=200) · anchor5(N=5 full, R=30) × **3-seed**. **Fed-LOO = in-run leave-one-out**(`in_run_loo`: 라운드별 U(P_r)−U(P_r∖{i}) 누적) — deferred-rigor 인벤토리의 "Federated-LOO 수치 부재" 공백을 메움(**loss-heur는 singleton U({i})라 LOO가 아님** — 코드 확인).
- **E5 N=10 oracle** (`runs/track_d/rundirs_e5_n10/1B_anchor10_seed0/`, 커밋 3895396): **N=10 전원 참여**, R=30, **(b) = exact 2¹⁰**(1024 coalition 완전열거). 계획 P1의 (b)쪽 절반 — **seed0만**(셀당 ~33h → Yonghee 결정 2026-07-18로 seed0 한정; (a) 2¹⁰ retrain·seeds 1-2는 ⬚ = 루트 REMAINING §1.4 장기 대기).

**(b1) fidelity vs (b) — Spearman ↑** (E4 = 3-seed mean±std / E5 = seed0)

| method | E4 std20 Sp ↑ | E4 anchor5 Sp ↑ | E5 N=10 Sp ↑ | E5 N=10 Pearson ↑ |
|---|---|---|---|---|
| **Fed-LOO** | **1.000±.000** | **1.000±.000** | **1.000** | 0.99993 |
| Flirds | 1.000±.000 | 1.000±.000 | 1.000 | 0.999999 |
| Flirds-1st | 0.998±.003 | 1.000±.000 | 1.000 | 0.99991 |
| loss-heur | 0.999±.001 | 1.000±.000 | 1.000 | 0.99994 |

> 읽기: **same-game 방법 전원이 N=10 exact 2¹⁰에서도 순위 완전일치** — near-additive 동률이 N=5를 넘어 N=10으로 확장(P6 "N=5 coarse 축퇴" 비판의 규모 축 반쪽 해소; (a)축은 여전히 ⬚). **Fed-LOO도 +1.000** = near-additive 무대에선 semivalue(Shapley↔LOO) 구별이 없다는 §3.1.1 구조의 재확인. 방법 간 격차는 Pearson 잔차 수준에서만(Flirds 1−Pe ≈ 9e-7로 최소, Flirds-1st ≈ 9e-5). E4 std20 Kendall도 Fed-LOO/Flirds 1.000(Flirds-1st seed1 0.979·loss-heur seed1-2 0.989만 소폭).

**(b2) runtime** (초 ↓; E4 = 3-seed mean±std / E5 = seed0)

| method | E4 anchor5 ↓ | E4 std20 ↓ | E5 N=10 (2¹⁰) ↓ |
|---|---|---|---|
| **Flirds-1st** | 232±8 | 1535±46 | 240 |
| **Flirds** | 716±29 | 4703±133 | **733** |
| loss-heur | 657±19 | 2199±60 | 1240 |
| Fed-LOO | 773±23 | 2925±88 | 1373 |
| **(b)oracle** | 3568±160 | 2943±104 | **117,649 (=32.7h)** |

> 읽기: **(b) 지수 비용의 실측 확장** — full 참여 N=5(2⁵) 3568s → **N=10(2¹⁰) 117,649s**(33×)인데 Flirds는 733s 그대로(1 HVP/round 고정) → **160×**(§3.4 비용 모델·device100 anchor ~159×와 정합; E5 timing.json = valuation 121,234s의 97.0%가 (b) 단독, 셀 총 34.7 GPU-h). **Fed-LOO 비용 = O(Σ_r|P_r|)** → 라운드 많은 std20에선 (b) per-round와 동급(2925≈2943s), anchor5에선 1/4.6. **loss-heur 행 = C6 측정버그 fix 이후 정본**(anchor5 657s = pre-fix 1093s의 1/1.66 · std20 2199s = pre-fix 2913s의 1/1.32 — 이론 예측 fwd 2|P_r|→1+|P_r| 비율과 정확 일치; §6.2 caveat 11).

**(c) 출처·baseline-set 노트**: 포함 4종 + (b). GTG/FedSV/ComFedSV/ShapleyFL/FedIF **제외 = 경량 재실행 설계**(coalition류 fidelity는 §3.1.1 본 스위트가 담당). ⚠ E5 rundir의 config `regime` 필드가 코드 분기용 `anchor5` 라벨로 남아 있으나 실제 = N=10(`n_clients: 10`)·exact 2¹⁰ 적용 확인. E4 std20_seed2와 E5의 git_sha가 타 셀과 다름(전 셀 git_dirty=true) — 엄밀 재현 시 유의. 이 표만 std=ddof1(각 파일과 정확 부합).

---

### 3.1.4 오염 무대 fidelity 요지 (`phase2_matrix`) — 표는 §3.3에 통합

> Robustness 무대의 Spearman(vs (b) 또는 Flirds proxy)은 **§3.3 Detection 표에 AUROC와 함께 통합 수록**(같은 셀에서 측정). 요지만:
> - **silo5 (N=5, (b) 2⁵; β0.3 재실행판 ce0b454)**: clean·noisy·free-rider 위협서 Flirds/Flirds-1st/loss-heur/Fed-LOO Spearman **1.000**, FedSV 0.93~1.0, GTG 1.0, FedIF 0.90~0.93. **poison 위협**서 near-additive 동률 붕괴: FedSV **0.367±.262**, Flirds-1st **0.000**(회피), **Flirds 0.600±.283**(seed-혼재 .4/.4/1.0; 재실행 전 rundir는 0.967 — git 이력), GTG 0.867, ComFedSV 0.733.
> - **device100 anchor (N=100, (b) per-round)**: Flirds/Flirds-1st/loss-heur **1.000**, GTG 0.78~0.84, FedSV 0.75~0.81, ShapleyFL 0.58~0.69, FedIF 0.72~0.83, ComFedSV ≈ 0(low-rank 가정 위배).
> 전체 수치 → §3.3.1–3.3.3.

---

### 3.1.5 신호 실재성 — 오염축×비IID 2×2 매트릭스 (B축; `phase2_matrix/1B_{iid5,silo5}_*`, 1B 3-seed)

**(a) 세팅**: §3.3.1 silo5와 동일 무대(N=5 full, R=10, 10 steps, batch 16, lr=1e-3[poison 2e-3/batch8/epochs5/frac0.8], maxlen 768, train200/val20/test40, warmup2, (b)=exact 2⁵, 3 seed), 단 **비IID축과 오염축을 2×2로 분리**: 비IID축 = `iid5`(build_alpaca_iid 균질) ↔ `silo5`(5-domain 비IID); 오염축 = clean ↔ {noisy·free-rider·poison}. 신규 = iid5 5셀 + silo5_clean(silo5 오염 3셀은 §3.3.1 재사용 — **β0.3 재실행판**). 목적 = "silo5의 신호가 오염 때문인가 도메인 이질성 때문인가"의 분리. 드라이버 = `runs/matrix_cxni/`(rundir는 `phase2_matrix/rundirs/`에 착지).

**(b1) 1차 fidelity — (b)oracle 자기순위 cross-seed ρ ↑** (신호 실재성; 1=완전재현·≈0=추첨노이즈; 3-seed 쌍별 Spearman 평균)

| 무대 \ 오염 | clean | noisy | free-rider(zero) | poison |
|---|---|---|---|---|
| **IID** (iid5) | 0.13 | 0.60 | 0.70 | 0.73 |
| **non-IID** (silo5) | **0.87** | 0.93 | 0.93 | 1.000 |

> **headline (결정타)**: **non-IID clean ρ 0.87** — 오염이 0인데 **도메인 분리만으로** oracle 자기순위가 재현된다 → silo5의 높은 fidelity가 오염이 아니라 **도메인 이질성** 때문임을 확정(진단 §1.4 caveat 해소). 대비 **IID clean ρ 0.13**(≈0, 신호 거의 없음 = §3.1.2 (b2)·진단 §1.4 재현). 두 축이 각각 독립적으로 신호를 만든다(오염 하나만 있어도 IID 0.60~0.73; 둘 다면 0.93~1.00). → **A축(rank·참여·lr·steps §4.2, 폭 §4.3)이 신호를 못 만든 것과 정확히 대비: 신호는 B축(클라 간 실제 차이)이 만든다.** ⚠ silo5 frzero 칸은 β0.3 재실행(ce0b454) 후 0.93(재실행 전 rundir는 1.000 — seed2에서 근소차 순위 1쌍 뒤집힘; tracked `matrix_cxni/figures/crossseed_rho.csv`는 재실행 전 산출이라 1.000으로 남아 있음 — rundir 재계산이 정본).

**(b2) 2차 탐지 AUROC — IID vs non-IID 배경 대조** (오염 클라 탐지 ↑, 3-seed mean; non-IID 열 = §3.3.1과 같은 셀)

| 무대 + 오염 | Flirds | FedDQC | FLTrust | STD-DAGMM |
|---|---|---|---|---|
| IID + noisy | 1.00 | **1.00** | 1.00 | 0.17 |
| non-IID + noisy | 1.00 | 0.92 | 1.00 | 0.25 |
| IID + free-rider(zero) | 1.00 | 0.58 | 1.00 | 0.00 |
| non-IID + free-rider(zero) | 1.00 | 0.75 | 1.00 | 0.00 |
| IID + poison | **0.00** | 1.00 | 1.00 | 0.67 |
| non-IID + poison | **0.50** | 1.00 | 1.00 | 0.83 |

> - **Flirds·FLTrust는 배경 무관 noisy/free-rider 탐지 1.00**(gradient 기반 강건).
> - **FedDQC(data-quality)는 IID 균질 배경서 noisy 탐지가 더 깨끗**(1.00 vs non-IID 0.92) — clean 클라가 다 비슷해 오염 클라가 뚜렷; non-IID는 clean 도메인도 튀어 대비↓. "균질 배경이 오염 탐지를 돕나"의 답 = data-quality 축에선 그렇다.
> - **poison(clean-preserving backdoor)은 IID서 Flirds 완전 회피(0.00) vs non-IID 0.50**(seed-혼재 .25/.25/1.0; β0.3 재실행 정본 — 재실행 전 rundir는 0.92) — 균질 배경일수록 backdoor가 clean val-loss에 덜 드러나 더 잘 숨고, 비-IID에서도 방어는 간헐적(§3.3.1·§3.3.3 Flirds-회피 경계와 정합). **⚠ [07-19 정정 유지] IID선 Taylor 추정만이 아니라 val-loss 게임 자체가 회피됨**: `1B_iid5_poison`서 **(b)oracle·loss-heur 도 AUROC 0.00**(3-seed 전부; non-IID silo5선 둘 다 1.00으로 잡음 — §3.3.1). 잡는 건 FedDQC·FLTrust·FLDetector·FedIF·GTG·FedSV·ShapleyFL(1.00)·STD-DAGMM(0.67) — coalition-**MC** 계열은 근사 분산 덕에 잡는 반면 exact 게임((b))과 그 직독(loss-heur)은 회피됨.
> - STD-DAGMM(model-free)은 전반 약하고 배경 무관 저조(silo5 값은 β0.3 재실행판: noisy 0.25·frzero 0.00·poison 0.83).

**(c) 출처·baseline-set 노트**
- **출처**: `runs/phase2_matrix/rundirs/1B_{iid5,silo5}_{clean,noisy,frrand,frzero,poison}/` — cross-seed ρ = `phi.parquet`의 (b)oracle를 seed로 피벗한 쌍별 Spearman, AUROC = `metrics.json`의 per-seed `auroc` 평균. **⚠ `make_analysis.py`(06-19 생성)엔 iid5/silo5_clean 미포함** → 이 표는 rundir 직접 집계(master_metrics.csv에 없음). **정식 재생성 = `runs/matrix_cxni/make_figures.py`**(B축 10셀 전담; tracked figures는 β0.3 재실행 전 산출 — 재실행 후 재생성 필요).
- valuation φ(Flirds 등) + 탐지기 4종. iid5 poison = 별도 install config(lr2e-3/batch8/epochs5/frac0.8). silo5 오염 3셀 재사용(§3.3.1).

> **A/B축 종합 판정**(신호크기 진단의 결론) = **§5.3**.

---

## 3.2 Selection → downstream performance / Aggregation quality (2차 ①)

> **이 절의 목적** = 측정한 기여도 φ로 학습 자체를 더 잘 만들 수 있음을 보여 **계산된 기여도가 실효적으로 유의미함**을 증명한다 — 기여도 기반 개입(soft 가중 · softmax 선택 · sign-게이팅 · 부분집합 재학습)이 오염 무대에선 성능을 회복시키고, clean 무대에선 해치지 않아야(do-no-harm) 한다. 사후 제거-재학습(removal) 증거 = §4.4, 게이트 정책 확증 런(P5·Scale·Dyn) = §4.8, 종합 판정·최고 세팅 = **§5.1**.
>
> **4축 커버리지 매트릭스** — ① LLM/CNN ② IID/non-IID ③ 계산 시점(**retrain** = 전체 학습 후 φ로 부분집합 선택 → init부터 재학습 / **per-update** = 라운드마다 누적 φ로 온라인 결정) ④ 선택 규칙(**top-k** / **양수-only** = cum φ>0 게이트):

| 정책 (③×④) | LLM · IID | LLM · non-IID | CNN · IID | CNN · non-IID |
|---|---|---|---|---|
| retrain × top-k | — | ● phase1 K=3/5 (§3.2.5) | — | — |
| retrain × 양수-only (V3) | ● iid5 (§3.2.3) | ● silo5 (§3.2.3) | ◐ 오염 시나리오만(파티션-skew V3 없음) (§3.2.4) | ● dir1 T2 **점수원 8종**×{plain, 크기가중} (§3.2.6) |
| per-update × soft top-k (softmax 선택) | ● std20 `flirds_sel` (§3.2.1) | — | ● C2 `flirds_select` iid (§3.2.2) | ● C2 dir1·shard (§3.2.2) |
| per-update × 양수-only (sign-게이트) | ● iid5 · ◐ std50k5 seed0 (§3.2.3·§3.2.6) | ● silo5 (§3.2.3; noisy 셀 **점수원 8종** §3.2.6) | ● iid (§3.2.4) | ● dir1 (§3.2.4; **점수원 8종** §3.2.6) |

> 매트릭스 주: **하드 top-k per-update arm은 전 무대 설계상 부재** — per-update 선택은 softmax 샘플링(확률적 soft top-k) 아니면 parameter-free sign-게이트뿐이며, top-k 상한은 `oracle_excl`이 대역(`runs/track_g/README.md` §7). soft **가중** arm(flirds_w/mult/repl/add 등 — 선택이 아니라 집계 가중)은 §3.2.1–2에 함께 수록.
>
> **왜 baseline-set이 fidelity(§3.1, 8~9종)보다 작은가** — ① **비용 구조가 다름**: fidelity는 방법 전부가 *같은 학습 궤적 하나*를 사후 채점(값싼 후처리)하지만, 개입 arm은 가중/선택이 궤적 자체를 바꾸므로 **arm 1개 = FL 학습 전체 1회** → 비용이 arm 수 × 셀 × seed에 선형(track_g LLM만 이미 218 rundir). ② **공정성(자기 논문 방식) 원칙**: 개입 레시피가 원 논문에 정의된 방법만 arm화 — ShapleyFL(교체 가중)·FedIF(교체 가중)·S-FedAvg(선택). ③ **Track G는 baseline 동물원이 아니라 통제 설계**: 같은 V2 게이트 정책에 *점수원만* 교체(flirds / loss-heur / (b)oracle=정책 천장 / ShapleyFL=붕괴-무대 대조) + oracle_excl·random_excl 상·하한 — GTG/FedSV 게이트는 noisy 발화가 coalition-renorm 값-오차의 부산물이라 당시 의도 제외(`track_g/README.md`, 감사 권고2).
>
> **[07-19 Yonghee 판정 — 위 공백 자체가 채워야 할 결함]**: fidelity는 우리가 정의한 게임((b))의 자기-일치라 타 정의 방법의 심판이 될 수 없음 — "다른 baseline들이 계산한 기여도로 **똑같은 실험**을 했을 때 우리가 더 잘한다"를 보여야 기여도 정의의 실효 우열이 증명됨(다운스트림 = 게임-무관 중립 심판). sign-게이트 경쟁도 **불공정이 아님**: 0의 의미론(zero-semantics)은 기여도 품질의 일부이므로, 0을 다르게 정의한 방법의 오발화는 그 방법의 실측 감점(②·③의 제외 근거를 경쟁 실험에선 승격). → **Track H 점수원 경쟁**(스펙·예측표 = §2 P10, `runs/track_h/README.md`; 점수원 8종 × 정책 4종[sign±크기가중·mult·z] × 시점 2종[online/retrain] × 차이-무대) — **Tier 1(CNN)+Tier 2(LLM) 실행 완료 = §3.2.6**; LLM 경쟁 무대의 본판은 **R4 gsm50k5(accuracy 심판)가 대체·실행 중**(§2 P11). 이로써 온라인 사분면 공백(track_g 점수원 3종뿐)은 8종으로 채워졌고, retrain 사분면도 T2(관찰자→재학습)가 8종 커버 — removal(§4.4, 전 방법·사후 제거)과 상보.

### 3.2.1 LLM 표준 개입 arm (`track_d`) — clean-IID do-no-harm parity

**(a) 세팅**: §3.1.1과 동일 궤적. arms = 같은 vanilla 로그에서 파생한 온라인 개입 6종.
- `base` = 학습 전 베이스 모델 · `vanilla` = 표준 FedAvg · `flirds_w` = 곱셈 가중 w∝n·s (EMA β=0.5) · `flirds_sel` = softmax 선택 (cohort가 진부분집합인 std20만) · `shapleyfl_w` = 교체 가중 (β=0.3) · `fedif_w` = 교체 가중 (β=0.7=1-γ)
- 평가: **MMLU full-test(14,042) 0-shot** + 같은분포 **Alpaca-test(1k) ROUGE-L**. clean-IID 기대 = parity(do-no-harm); 차이는 finding.

**(b) 결과 — MMLU / ROUGE-L, 3-seed mean±std**

| stage·scale  | arm         | MMLU ↑       | ROUGE-L ↑    | stage·scale    | arm         | MMLU ↑       | ROUGE-L ↑    |
| ------------ | ----------- | ------------ | ------------ | -------------- | ----------- | ------------ | ------------ |
| **1B std20** | base        | 0.4822±.0000 | 0.2168±.0019 | **1B anchor5** | base        | 0.4822±.0000 | 0.2168±.0019 |
|              | vanilla     | 0.4742±.0001 | 0.2841±.0051 |                | vanilla     | 0.4801±.0003 | 0.2725±.0032 |
|              | flirds_w    | 0.4745±.0003 | 0.2848±.0050 |                | flirds_w    | 0.4802±.0007 | 0.2741±.0025 |
|              | flirds_sel  | 0.4739±.0005 | 0.2838±.0041 |                | shapleyfl_w | 0.4802±.0007 | 0.2741±.0026 |
|              | shapleyfl_w | 0.4742±.0005 | 0.2845±.0050 |                | fedif_w     | 0.4797±.0008 | 0.2713±.0037 |
|              | fedif_w     | 0.4741±.0003 | 0.2847±.0046 |                |             |              |              |
| **3B std20** | base        | 0.6230±.0000 | 0.2219±.0015 | **3B anchor5** | base        | 0.6230±.0000 | 0.2219±.0015 |
|              | vanilla     | 0.6149±.0003 | 0.3017±.0025 |                | vanilla     | 0.6218±.0001 | 0.2753±.0028 |
|              | flirds_w    | 0.6145±.0006 | 0.3016±.0019 |                | flirds_w    | 0.6215±.0002 | 0.2758±.0037 |
|              | flirds_sel  | 0.6142±.0015 | 0.3022±.0039 |                | shapleyfl_w | 0.6214±.0001 | 0.2757±.0038 |
|              | shapleyfl_w | 0.6143±.0006 | 0.3019±.0017 |                | fedif_w     | 0.6214±.0004 | 0.2734±.0030 |
|              | fedif_w     | 0.6143±.0007 | 0.3025±.0024 |                |             |              |              |
| **7B std20** | base        | 0.4175±.0000 | 0.1496±.0024 | **7B anchor5** | base        | 0.4175±.0000 | 0.1496±.0024 |
|              | vanilla     | 0.4038±.0024 | 0.2778±.0026 |                | vanilla     | 0.4206±.0012 | 0.1651±.0016 |
|              | flirds_w    | 0.4026±.0028 | 0.2780±.0027 |                | flirds_w    | 0.4210±.0014 | 0.1680±.0023 |
|              | flirds_sel  | 0.4025±.0022 | 0.2790±.0044 |                | shapleyfl_w | 0.4210±.0014 | 0.1680±.0023 |
|              | shapleyfl_w | 0.4027±.0027 | 0.2787±.0028 |                | fedif_w     | 0.4204±.0008 | 0.1656±.0011 |
|              | fedif_w     | 0.4030±.0023 | 0.2763±.0033 |                |             |              |              |

> 7B anchor5 arm(MMLU/ROUGE)은 **2026-06-26 추가**(arm-only 재실행; fidelity·runtime은 phi.parquet 불변이라 §3.1.1·§3.4.2와 동일). anchor5는 전원 참여 → flirds_sel 없음. 읽기: anchor5(R=30 경량학습)는 std20과 달리 vanilla MMLU가 base보다 살짝 ↑(0.4175→0.4206); ROUGE는 base 0.1496→vanilla 0.1651로 std20(0.2778)보다 작게 상승(학습량 적음). flirds_w/shapleyfl_w가 vanilla 대비 ROUGE 미세 ↑(0.1651→0.1680).

> 읽기(do-no-harm): 모든 개입 arm의 MMLU·ROUGE가 vanilla와 ±0.001~0.003 이내 = clean-IID에서 기여도-가중이 성능을 **해치지도 크게 올리지도 않음**(기대대로 parity). ROUGE는 학습으로 base 대비 크게 상승(예: 3B std20 0.222→0.302); MMLU는 SFT로 소폭 하락(외부 벤치, 분포 밖).

**개입 arm의 가중 메커니즘** (코드 `flirds/fl/intervene.py`; 각 baseline은 *자기 논문 방식*을 씀):
온라인 점수기 `OnlineScorer`가 라운드별 raw 기여도를 EMA로 누적(`s ← β·s + (1−β)·raw`), 누적 s로 다음 라운드 FedAvg 가중을 바꾼다. 가중 규칙 4종:
- **multiplicative** `w_i ∝ n_i · s_i` — FedAvg의 데이터-크기 가중에 기여도를 **곱함**. **Flirds 기본**(`flirds_w` / CNN `flirds_mult`; Yonghee 규칙).
- **replacement** `w_i ∝ s_i` — n-가중을 기여도로 **대체**. **FedIF·ShapleyFL 논문 관행** (`fedif_w` β=0.7=1−γ / `shapleyfl_w` β=0.3; 두 논문 모두 per-round min-max→EMA→대체).
- **additive** `w_i = λ·s_i/Σs + (1−λ)·n_i/Σn`, λ=0.5 — 기여도와 n-가중을 **혼합** (CNN `flirds_add`).
- **selection** `softmax(s/T)`로 k명 **선택** 샘플링(비복원). **S-FedAvg 관행** (`flirds_sel` / CNN `flirds_select`; cohort가 진부분집합일 때만 = std20·N=100).
> 주의: n_i가 모두 같으면 multiplicative==replacement (크기-skew에서만 갈림 — 그래서 IID std20/anchor에선 flirds_w·shapleyfl_w의 *가중식*은 같고 점수원·β만 다름). raw 점수원: **Flirds**=estimator, **FedIF**=per-round 1차 influence, **ShapleyFL**=per-round exact Shapley, **S-FedAvg**=자체 MC-relevance. 즉 각 arm은 *논문 방식+자기 점수+자기 β* 조합이라 공정 비교.

**(c) 출처·baseline-set 노트**
- **출처**: `runs/track_d/rundirs/*/metrics.json` (`arms.{arm}.{mmlu,rouge_l}`).
- 포함 arm = base/vanilla/flirds_w/shapleyfl_w/fedif_w (+ std20만 flirds_sel). **flirds_sel 제외@anchor5** = 전원 참여라 선택이 무의미(degenerate) ─ *적용규칙: 참여형태*. arm은 valuation 비교가 아니라 *개입 효과* 측정. **개별 결과**: 위 MMLU/ROUGE 표는 (scale × stage) 6 셀로 *개별* 수록(pool 아님); CNN C2(§3.2.2)만 threat 4그룹 pool이고 셀별 30칸은 `RESULTS.txt`.

---

### 3.2.2 CNN cross-device 개입 (`track_c` C2) — 최종 정확도

**(a) 세팅**
- 소형 CNN, **N=100, 라운드당 10% 참여**, R=120, local epochs=5, lr=0.01, batch=64, SGD mom=0, val=2000/test=8000, target acc=0.6(cifar) / dataset별; 3 seeds.
- datasets: cifar10, fmnist; partitions: `iid` · `dir1`(Dirichlet α=1, label+size skew) · `shard`(McMahan 2-shard); threats: `clean` · `label_flip` · `free_rider` · `grad_noise` (+ 강도 변형); 총 **30 셀**.
- arms(8): vanilla · **flirds_mult** · flirds_repl(dir1만) · flirds_add(dir1만) · **flirds_select** · shapleyfl(β=0.3) · fedif(β=0.7) · sfedavg(S-FedAvg).

**(b) 결과 — 최종 test 정확도 ↑, threat별 그룹 mean±std** (열=threat 그룹, 값=정확도 ↑; 셀 pool, 셀별은 `RESULTS.txt`)

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

**(c) 출처·baseline-set 노트**
- **출처**: `runs/track_c/c2/*/metrics.json` (`arms.{arm}.final_acc`). 코드 = `codes/experiments/track_c2.py`.
- 포함 8 arm. **flirds_repl/flirds_add 제외@iid·shard** = size-skew(dir1)에서만 MULT와 갈리므로 dir1 전용 ─ *적용규칙: 참여형태/적용성*. valuation fidelity baseline(GTG/FedSV exact 등)은 C2엔 없음(C2는 개입-성능 무대; fidelity는 C1). arm별 탐지 AUROC = §3.3.5.

> **확장**: 이 C2를 **폭 w×참여 f로 sweep**한 신호크기 probe = **§4.3** (clean은 폭·참여 무관 parity, label-flip 개입 이득 ~0.09도 폭 무관).

---

### 3.2.3 LLM φ-게이팅 + V3 재학습 (Track G Phase B; `track_g/rundirs`, LLM 218 rundir)

> 전제가 된 Stage 0 부호 감사 = §5.2. ⚠ **경로 정정**: LLM rundir는 `runs/track_g/rundirs/`(218개)이며 별도 `rundirs_llm/` 폴더는 존재하지 않는다.

**(a) 세팅**: silo5 {clean, noisy(nr1.0), frrand, frzero}×3-seed + iid5 {clean, frzero}×3-seed + silo5 noisy-nr0.75 seed0 + std50k5-mixed(총 8런 = seed0 5-arm 커버 + flirds s1 + shapleyfl 3-seed; **상태 = seed0 파일럿 동결**[1-seed caveat 필수, 루트 REMAINING §2-5] — 3-seed 확장 중단, LLM 참여축 성능 주장은 R4가 대체) — 전 셀 arms + per-round `phi_rounds.parquet`(프로젝트 최초 per-round φ 영속). arm = sign-게이트 2종(V1/V2; V2=burn-in·probation 포함; **per-update × 양수-only**), z-게이트(cohort-상대), lossheur/oracleb 게이트 대조, soft 가중 `flirds_w`, `oracle_excl`/`random_excl` 상·하한, **V3**(vanilla 완주 후 게이트 판정 kept로 init부터 재학습 = **retrain × 양수-only**: sign/z/random). recovery = (vanilla−arm)/(vanilla−oracle_excl).

**(b) 결과 — final val-loss ↓, arm×셀, 3-seed mean±std** (llm_summary.csv 재집계; silo5 noisy-nr0.75는 seed0뿐이라 생략, std50k5-mixed는 파일럿이라 생략 — §3.2.6 R2)

| arm | iid5 clean | iid5 frzero | silo5 clean | silo5 noisy nr1.0 | silo5 frrand | silo5 frzero |
|---|---|---|---|---|---|---|
| vanilla | 1.3147±.0556 | 1.3199±.0557 | 2.3322±.0474 | 2.3340±.0471 | 2.3364±.0459 | 2.3362±.0454 |
| oracle_excl (상한) | – | 1.3126±.0554 | – | 2.3323±.0472 | 2.3328±.0459 | 2.3325±.0453 |
| random_excl (통제) | – | 1.3184±.0503 | – | 2.3331±.0464 | 2.3349±.0475 | 2.3347±.0470 |
| **flirds_gate_v2** | 1.3147±.0556 | **1.3126±.0554** | 2.3322±.0474 | 2.3340±.0471 | 2.3347±.0465 | **2.3325±.0453** |
| flirds_gate_v1 | 1.3147±.0556 | 1.3134±.0554 | 2.3322±.0474 | 2.3340±.0471 | 2.3347±.0465 | 2.3329±.0453 |
| flirds_zgate_v2 | 1.3144±.0559 | 1.3125±.0555 | 2.3322±.0474 | 2.3340±.0471 | 2.3339±.0454 | 2.3336±.0449 |
| flirds_w (soft 가중) | 1.3141±.0556 | 1.3133±.0554 | 2.3273±.0477 | 2.3294±.0471 | 2.3314±.0460 | 2.3310±.0454 |
| lossheur_gate_v2 | 1.3147±.0556 | 1.3126±.0554 | 2.3322±.0474 | 2.3340±.0471 | 2.3341±.0466 | 2.3325±.0453 |
| oracleb_gate_v2 (silo5만) | – | – | 2.3322±.0474 | 2.3340±.0471 | 2.3348±.0465 | 2.3325±.0453 |
| v3_sign (retrain) | 1.3147±.0556 | 1.3126±.0554 | 2.3322±.0474 | 2.3340±.0471 | 2.3351±.0475 | 2.3325±.0453 |
| v3_z (retrain) | 1.3144±.0558 | 1.3126±.0554 | 2.3322±.0474 | 2.3340±.0471 | 2.3339±.0454 | 2.3336±.0449 |
| v3_random (retrain 통제) | 1.3147±.0556 | 1.3191±.0580 | 2.3322±.0474 | 2.3340±.0471 | 2.3364±.0459 | 2.3363±.0481 |

> 읽기 주의: ±std는 **seed 간 손실-수준 분산**(무대 자체 변동, ~.05)이라 arm 효과(~.004)보다 큼 — arm 효과는 seed-쌍 대응 delta/recovery로 읽는다(아래 표). clean 두 열에서 게이트 arm이 vanilla와 소수 4자리 동일 = 게이트 무발화(do-no-harm)의 표 형태.

**recovery = (vanilla−arm)/(vanilla−oracle_excl) ↑, 오염 셀만, 3-seed mean**

| arm | iid5 frzero | silo5 frzero | silo5 frrand | silo5 noisy nr1.0 |
|---|---|---|---|---|
| **flirds_gate_v2** | **1.000** | **1.000** | 0.462 | 0.000(침묵) |
| flirds_gate_v1 | 0.900 | 0.898 | 0.462 | 0.000 |
| flirds_zgate_v2 | 1.018 | 0.667 | 0.667 | 0.000 |
| flirds_w (soft) | 0.903 | 1.421¹ | 1.418¹ | 2.708¹ |
| lossheur_gate_v2 | 1.000 | 1.000 | 0.663 | 0.000 |
| oracleb_gate_v2 | – | 1.000 | 0.429 | 0.000 |
| v3_sign (retrain) | 1.000 | 1.000 | 0.333 | 0.000 |
| v3_z (retrain) | 1.000 | 0.667 | 0.667 | 0.000 |
| v3_random (retrain 통제) | 0.161 | 0.018 | −0.001 | 0.000 |
| random_excl (통제) | 0.185 | 0.390 | 0.385 | 0.511 |

> ¹ recovery>1 = 분모(vanilla−oracle_excl 갭, +0.0015~0.0037)가 작아 soft 가중의 절대 이득(+0.0045~0.0052)이 상한을 넘어 보이는 것(분모 작음 주의 — 아래 noisy 불릿).

- **frzero (핵심 성공)**: sign-게이트 V2·lossheur/oracleb 게이트·v3_sign 전부 **recovery 1.000 정확**(silo5 3/3·iid5 3/3 seed; oracle_excl 최종손실과 소수 4자리 동일) — bit-exact φ=0(§5.2 판정 2)이 **온라인 자동배제로 그대로 이어짐**. per-round 게이트 정밀도/재현율 **1.0/1.0, 오배제 0쌍**. V1은 0.90(제외 시점 semantics 차), z-게이트는 1.000–1.037(iid5; oracle_excl보다 소폭 우위 셀 존재). 예외 2건 = silo5 seed2의 z-게이트·v3_z 미발화(0.0; z-임계 미달).
- **clean (무해성)**: sign-게이트 발화 0(오배제 0쌍, iid5·silo5 전 seed), 게이트·V3 arm 기준 max Δ최종손실 = **0.00056**(soft 가중 flirds_w 제외) — do-no-harm 성립. 유일 예외 = z-게이트가 iid5 clean에서 오배제 2–12쌍(precision 0; 손실 영향은 +0.00003~0.00056 미미) → cohort-상대 게이트의 구조적 오발화 리스크.
- **noisy (예측 적중 = 작동영역 없음)**: 전 게이트 침묵(발화 0; nr0.75·nr1.0 동일) — §5.2 예측 수정 ①(nr≤1에서 누적 φ 0-교차 없음, 외삽 nr≈3.4) 그대로. oracle_excl 갭 자체가 +0.0015~0.0020으로 작고, soft 가중 flirds_w만 +0.0045~0.0047(recovery 2.2–3.2 — 분모 작음 주의). **noisy 대응은 게이트가 아니라 탐지(AUROC 1.0)+selection의 몫**(위계 정합).
- **frrand (코인플립 예측 적중)**: 누적부호가 0 근방이라 sign-게이트 회수 0.30–0.70 seed-의존(recall 0.29–0.86), z-게이트 2/3 seed 1.00(seed2 recall 0.14) — §5.2 예측 수정 ② 정합. flirds_w는 3 seed 전부 +0.0048~0.0052.
- **비용**: silo5 게이팅 4셀 = seed0 기준 셀당 1.73–2.29 GPU-h(합 7.77; **3-seed 전체 25.2 GPU-h** + nr0.75 12런 2.3). std50k5-mixed(누적): seed0 gate_v2 4.55 / vanilla 4.39 / oracle_excl 3.34 / random_excl 3.14 + flirds s1 4.59 · shapleyfl s0 9.94/s1 9.65/s2 9.83 GPU-h(shapleyfl 3-seed 완비·flirds 2-seed; vanilla 앵커 s1–s2 미완이라 recovery는 seed0만 — 셀 결과·경쟁 판정 = **§3.2.6 R2**).

**(c) 출처·baseline-set 노트**
- **출처(정본)**: `runs/track_g/analysis/{README.md,llm_summary.csv}`(rundir-only 재생성; 스팟 대조 3셀 rundir 일치 확인) · rundir = `runs/track_g/rundirs/`.
- arm은 valuation 방법 비교가 아니라 **같은 V2 게이트 정책에 점수원을 교체한 통제 실험** — flirds / **loss-heur**(경쟁 점수원) / **(b)oracle**(정책 천장, silo5만) / **ShapleyFL**(fidelity-붕괴 무대 대조, std50k5만·파일럿) + oracle_excl·random_excl 상·하한 + v3_random(재학습 통제). **제외**: GTG/FedSV/ComFedSV 게이트 — 개입 정책이 원 논문에 없고, GTG/FedSV는 frzero exact-0이 아니어서(coalition-renorm, §5.2 판정 2) sign-게이트 발화가 값-오차 부산물이 됨 ─ *적용규칙: 점수원 통제 설계*. 이 제외는 *게이트 정책 실효성* 질문용이고, *점수원 경쟁* 질문에선 **§3.2.6(Track H)이 renorm 4종 게이트를 포함해 실측**(오발화 = 그 방법의 실측 감점, Yonghee 07-19). arm당 FL 학습 전체 1회 = 비용 선형(§3.2 서두 ①).

### 3.2.4 CNN 게이트 그리드 + V2w 승격 판정 (Track G Phase B; `track_g/rundirs_cnn` 36 + `rundirs_cnn_v3` 12 = 48/48셀)

**(a) 세팅**: cifar10 {iid, dir1}×{clean, label-flip@{0.15,0.35,0.70}, grad-noise, free-rider}×3-seed c2-게이트 36셀(**per-update × 양수-only**) + c1 V3 12셀(**retrain × 양수-only**; mnist·cifar10 × {label_flip, feature_noise} × 3-seed, `rundirs_cnn_v3/` 별도 영속) — 4-GPU 샤딩, 실패 0.

**(b) 결과 — dAcc(= arm − vanilla final test acc) ↑, arm×위협, 3-seed mean** (cnn_summary.csv 재집계; vanilla 행만 절대 acc)

**iid 파티션**

| arm | clean | free-rider | grad-noise | lf0.15 | lf0.35 | lf0.70 |
|---|---|---|---|---|---|---|
| vanilla (절대 acc) | .6488 | .6083 | .2564 | .6277 | .5923 | .5171 |
| oracle_excl (상한) | – | +.0273 | +.3793 | +.0033 | +.0387 | +.1138 |
| random_excl (통제) | – | −.0097 | +.0081 | −.0259 | −.0200 | −.0148 |
| **flirds_gate_v2** | −.0060 | +.0225 | **+.3580** | −.0049 | +.0102 | +.0796 |
| flirds_gate_v1 | −.0073 | +.0126 | +.3232 | −.0115 | +.0242 | +.0710 |
| flirds_zgate_v2 | −.0007 | .0000 | +.0628 | −.0024 | +.0055 | +.0244 |
| flirds_gatew_v2 (V2w) | −.0075 | +.0234 | +.3621 | −.0140 | +.0069 | +.0860 |
| flirds_gatew_v1 (ablation) | −.0148 | +.0166 | +.3208 | −.0075 | +.0189 | +.0892 |
| flirds_mult (soft 가중) | −.0021 | +.0176 | +.2765 | −.0034 | +.0239 | +.0905 |

**dir1 파티션 (Dirichlet α=1, non-IID)**

| arm | clean | free-rider | grad-noise | lf0.15 | lf0.35 | lf0.70 |
|---|---|---|---|---|---|---|
| vanilla (절대 acc) | .6389 | .5879 | .2436 | .6172 | .5849 | .5247 |
| oracle_excl (상한) | – | +.0324 | +.3767 | +.0064 | +.0387 | +.0990 |
| random_excl (통제) | – | −.0041 | +.0154 | −.0249 | −.0250 | −.0228 |
| **flirds_gate_v2** | −.0074 | +.0269 | **+.3232** | −.0182 | −.0126 | +.0465 |
| flirds_gate_v1 | −.0032 | +.0227 | +.3025 | −.0012 | +.0120 | +.0573 |
| flirds_zgate_v2 | −.0048 | −.0039 | +.0982 | −.0024 | +.0032 | +.0092 |
| flirds_gatew_v2 (V2w) | −.0201 | +.0177 | +.3438 | −.0240 | −.0028 | +.0563 |
| flirds_gatew_v1 (ablation) | −.0191 | +.0234 | +.2744 | −.0146 | +.0068 | +.0233 |
| flirds_mult (soft 가중) | +.0037 | +.0100 | +.1927 | −.0001 | +.0154 | +.0623 |

> 읽기: **grad-noise = 게이트 최대 성공**(V2 +.32~.36 = oracle 상한 +.377~.379의 회수 0.86–0.94; vanilla 절대 acc .24→.58~.61) · **free-rider 부분 회수** 0.81–0.84 · **label-flip dose-의존**(0.15는 개입 여지 ~0, 0.70서 회수 0.44–0.90) · **clean 열 = CNN 오발화**(V2 −.006~−.007, V2w −.008~−.020; LLM §3.2.3의 무발화와 대비 — V2w 불승격 사유).

- **V2w 승격 판정 = DO NOT PROMOTE** (spec §5-2 자동판정): clean parity 6셀 중 5셀 위반(dAcc −0.010~−0.030, 기준 |d|<0.006) + corrupt 6그룹 중 2그룹 V2w<V2(dir1 FR −0.0092, iid lf −0.0020) → **CNN-only 정직 보고, LLM ARMS에 V2w 미추가·백필 없음**.
- **V3(c1, 12셀)**: sign-kept 집합이 Flirds와 (b)oracle 판정 **전 셀 동일**(feature-noise·label-flip 대부분 kept=전원 — 누적 양수; z-kept가 일부 클라 제외 시 소폭 개선 사례 존재).

**(c) 출처·baseline-set 노트**
- **출처(정본)**: `runs/track_g/analysis/cnn_summary.csv` · rundir = `runs/track_g/{rundirs_cnn,rundirs_cnn_v3}/`.
- CNN 게이트 그리드도 §3.2.3과 같은 **점수원-통제 설계**(Flirds 점수 하나에 게이트 정책 5종 + oracle/random 상·하한) — valuation baseline 게이트는 동일 사유로 제외(§3.2 서두 ②③). soft 개입 arm 8종 전 비교(shapleyfl/fedif/sfedavg 포함)는 **§3.2.2 C2가 담당**(같은 무대) — 이 그리드는 C2 위에 게이트 arm만 추가한 것. cifar10만(fmnist·shard 파티션 없음 — C2엔 있음). **점수원 경쟁 확장 = §3.2.6(Track H)**; 게이트 정책 확증 런(P5·Scale·Dyn) = **§4.8**.

### 3.2.5 Foundational top-k selection→재학습 (`phase1` full runs) — retrain × top-k 유일 칸

**(a) 세팅**: §6.1의 full run(1B silo5 N=5 non-IID, noisy+free-rider 주입, lr∈{1e-3,3e-3}×3-seed)과 동일 궤적. 전체 학습 후 φ 계산 → **하위-φ 2명 드롭, K=3 keep으로 init부터 재학습**. arm = full(전원) / flirds_topk / random_k. 본 문서 유일의 **retrain×top-k** 칸(§3.2 매트릭스).

**(b) 결과 — final val-loss ↓, 3-seed mean±std**

| group | full(전원) | flirds_topk | random_k |
|---|---|---|---|
| full lr1e-3 | 2.4064±.0234 | **2.3978±.0226** | 2.4111±.0133 |
| full lr3e-3 | 2.3931±.0223 | **2.3926±.0219** | 2.4055±.0100 |

> flirds_topk val-loss ≤ random_k(양 lr) 그리고 ≤ full(오염 드롭이 도움) → "random은 hard bar"를 넘김. flirds_keep은 매 seed 정확히 clean 3명(client 0=noisy·1=free-rider 항상 드롭; keep 상세·AUROC = §6.1). 개선 폭 자체는 작음(Δ~0.005–0.013 val-loss).

**(c) 출처·baseline-set 노트**: phase1은 프로젝트 최초 foundational run이라 arm이 **3종뿐**(full=vanilla 상당 / flirds_topk / random_k 통제) — 다른 valuation 방법의 top-k arm은 설계에 없었음(당시 baseline 미구현 단계). retrain×top-k 칸을 다른 점수원으로 채우려면 신규 실험 필요(§3.2 매트릭스의 공백). **출처**: `runs/phase1/rundirs/*/metrics.json` (`arms`, `selection`).

### 3.2.6 Track H 점수원 경쟁 — 같은 개입 정책에서 어느 기여도 정의가 학습을 잘 만드나 (`track_h/rundirs_cnn` 96런 + `rundirs_llm` 12런)

> §3.2 서두 [07-19 Yonghee 판정]의 실행 — §3.2.3–4(Track G)가 *점수원-통제*였다면 여기는 **점수원-경쟁**: 같은 정책·같은 무대·같은 seed에 **점수원만 교체**. 스펙·예측표(H-1~7 사전 등록) = `runs/track_h/README.md`. **Tier 1(CNN)+Tier 2(LLM) 완주(실패 0)**; std50k5 Tier 3(12런)은 2026-07-21 루트 REMAINING 개정판 실행 큐에 미등재 — R2 서술은 seed0 파일럿 동결(1-seed caveat), **LLM 경쟁 무대는 R4 gsm50k5(§2 P11, 실행 중)가 대체**. 정책 확증 후속 런(P5·Scale·Dyn) = **§4.8**.

**(a) 세팅**: 점수원 8종 = Flirds / Flirds-1st / loss-heur / GTG / FedSV / ComFedSV / ShapleyFL(β0.3 un-normalized raw) / FedIF — 전부 contribution orientation(도움=양수) 통일, per-round in-run 채점(`fl/score_providers.py`; ComFedSV는 per-round 대용치[균등평균 submodel + loss-감소 효용, 논문 Eq.6] — Yonghee 승인 caveat). 정책 4종 = **P1** sign-게이트(cum>0 참여·n-가중, V2) / **P2** sign+크기가중(w∝n·max(cum,0)) / **P3** soft 곱셈가중(EMA min-max — 부호 파괴) / **P4** z-게이트(cohort-상대) × 시점 2종 = **T1** online / **T2** retrain(**관찰자 런** 1회에 8 점수원 동시 부착[vanilla와 비트동일 궤적] → 최종 누적 부호로 kept 결정 → init부터 재학습; kept-set 동일 시 재학습 공유, kept=전원이면 vanilla와 동일 처리). 무대 = **R1** CNN cifar10 dir1 {clean, grad-noise, free-rider, label-flip@0.70}×3-seed(Flirds arm·통제 = §3.2.4 재사용) / **R3** silo5 noisy nr1.0×3-seed(renorm 4종만 신규; Flirds·loss-heur·(b) = §3.2.3 재사용) / **R2** std50k5 mixed(seed0 동결). **판정 = 학습 성능만**(탐지 AUROC류 없음): 각 셀에서 vanilla(바닥)~oracle_excl(천장) 사이 절대 성능(CNN acc / LLM val-loss)으로 직접 비교. 분석 정본 = `runs/track_h/analysis/`(CSV엔 정규화 지표[셀 간 합산용]도 있으나 본 절 표는 **절대값만** — Yonghee 2026-07-20 지시).

> ⚠ **집계 노트(07-20 로컬, make_analysis 수정·재생성)**: 서버 커밋(`df3e8e9`) 시점 분석 스크립트가 ① track_g CNN label-flip 셀 config에 dose 키가 없어 track_h lf@0.7 arm이 vanilla/oracle 앵커와 join 실패(점수원별 집계 셀-집합 불일치), ② T2 kept=전원 스킵(`equals_vanilla`)을 결측으로 탈락시켜 재학습 셀이 증발 — 둘 다 수정해 **전 점수원 dir1 공통 셀**로 재집계. 커밋 메시지의 순위 문구("lossheur > flirds", "fedif=flirds1st T2 최고")는 정정 전 산물이며, 정정 후 = 아래 절대값 표.

**(b1) 결과 — CNN dir1 절대 test 정확도**(3-seed mean; vanilla=바닥·oracle_excl=천장 사이 어디에 앉나로 직접 비교). 각 표 = 한 정책·한 시점, 행 = vanilla + 상/하한 통제 + 8 점수원, 열 = 위협. **읽는 법**: 오염 열에서 vanilla보다 높으면 그 기여도로 개입이 성능을 올린 것, vanilla보다 낮으면(오배제 등) 되레 해친 것. clean 열은 vanilla가 천장이라 낮을수록 오발화.

> 표 공통 축: **오염-평균** 열 = free-rider·grad-noise·label-flip 3위협 평균(위협 축 접기); 하단 **계열평균** 2행 = exact-0(Flirds·Flirds-1st·loss-heur·FedIF) / renorm(GTG·FedSV·ComFedSV·ShapleyFL) 4종 평균(점수원 축 접기). ⚠ 위협을 가로질러 평균하면 vanilla 기준선이 다른 셀(clean .64↔grad-noise .24)을 섞으므로 오염-평균이 높다는 건 *가장 어려운 셀을 회복한 쪽*에 유리하게 가중됨 — 순위 요약일 뿐 셀별 값을 대체하지 않음.

**P1 sign-게이트 · online**(cum>0 참여, 매 라운드)

| arm              | clean     | free-rider | grad-noise | label-flip@0.70 | **오염-평균** |
| ---------------- | --------- | ---------- | ---------- | --------------- | --------- |
| vanilla (바닥)     | .6389     | .5879      | .2436      | .5247           | .4521     |
| oracle_excl (천장) | –         | .6203      | .6203      | .6236           | .6214     |
| random_excl (통제) | –         | .5838      | .2590      | .5018           | .4482     |
| **flirds**       | .6315     | .6148      | .5668      | .5712           | .5843     |
| flirds1st        | .6384     | **.6216**  | .2479      | .5717           | .4804     |
| lossheur         | .6264     | .6114      | .5981      | .5670           | **.5922** |
| fedif            | **.6386** | .6143      | .2479      | **.5728**       | .4783     |
| gtg              | .6051     | .3915      | .5972      | .5479           | .5122     |
| fedsv            | .5982     | .3966      | .5972      | .5164           | .5034     |
| comfedsv         | .5963     | .3918      | .5871      | .5152           | .4981     |
| shapleyfl        | .6045     | .4020      | **.6115**  | .5278           | .5138     |

**P1 sign-게이트 · retrain**(관찰자 최종 누적으로 kept 결정 → init부터 재학습)

| arm              | clean     | free-rider | grad-noise | label-flip@0.70 | **오염-평균** |
| ---------------- | --------- | ---------- | ---------- | --------------- | --------- |
| vanilla (바닥)     | .6389     | .5879      | .2436      | .5247           | .4521     |
| oracle_excl (천장) | –         | .6203      | .6203      | .6236           | .6214     |
| **flirds**       | .6277     | .6063      | .6065      | .6192           | **.6107** |
| flirds1st        | .6386     | **.6252**  | .2436      | **.6236**       | .4975     |
| lossheur         | .6293     | .6125      | .4518      | .6205           | .5616     |
| fedif            | **.6417** | **.6252**  | .2436      | .6217           | .4968     |
| gtg              | .6265     | .5158      | **.6203**  | .5991           | .5784     |
| fedsv            | .6166     | .5140      | **.6203**  | .5904           | .5749     |
| comfedsv         | .6232     | .5200      | **.6203**  | .5921           | .5775     |
| shapleyfl        | .6223     | .5113      | **.6203**  | .6028           | .5781     |

**P2 sign+크기가중 · online / retrain**(cum>0 배제 + 양수는 크기 가중; = Yonghee "기여 정도 가중 참가") — 좌=online, 우=retrain, 각 끝열 = 오염-평균

| arm              | cln·on | FR·on | GN·on | LF·on | **오염평균·on** | cln·re | FR·re | GN·re | LF·re | **오염평균·re** |
| ---------------- | ------ | ----- | ----- | ----- | ----------- | ------ | ----- | ----- | ----- | ----------- |
| vanilla (바닥)     | .6389  | .5879 | .2436 | .5247 | .4521       | 〃      | 〃     | 〃     | 〃     | .4521       |
| oracle_excl (천장) | –      | .6203 | .6203 | .6236 | .6214       | –      | 〃     | 〃     | 〃     | .6214       |
| **flirds**       | .6188  | .6056 | .5874 | .5810 | **.5913**   | .6123  | .5838 | .5904 | .6135 | .5959       |
| flirds1st        | .6378  | .6172 | .1868 | .5615 | .4552       | .6328  | .6160 | .1824 | .6160 | .4715       |
| lossheur         | .6274  | .6143 | .5998 | .5626 | .5922       | .6215  | .5935 | .5882 | .6118 | .5978       |
| fedif            | .6375  | .6178 | .6043 | .5811 | **.6011**   | .6338  | .6157 | .6114 | .6206 | **.6159**   |
| gtg              | .6076  | .3978 | .6113 | .5471 | .5187       | .6104  | .4428 | .6171 | .5927 | .5509       |
| fedsv            | .6040  | .3772 | .6057 | .5204 | .5011       | .6106  | .4240 | .6168 | .5932 | .5447       |
| comfedsv         | .5899  | .3791 | .6059 | .5118 | .4989       | .6077  | .4406 | .6160 | .5962 | .5509       |
| shapleyfl        | .5995  | .3657 | .6117 | .5303 | .5026       | .6128  | .4141 | .6172 | .6038 | .5450       |

**P3 soft-mult / P4 z-게이트 · online**(rank-기반 통제 — P3는 게이트 없이 순위로 soft 가중[부호 파괴], P4는 상대 z<−1.5 배제) — 좌=P3, 우=P4, 각 끝열 = 오염-평균

| arm              | cln·P3 | FR·P3 | GN·P3 | LF·P3 | **오염평균·P3** | cln·P4 | FR·P4 | GN·P4 | LF·P4 | **오염평균·P4** |
| ---------------- | ------ | ----- | ----- | ----- | ----------- | ------ | ----- | ----- | ----- | ----------- |
| vanilla (바닥)     | .6389  | .5879 | .2436 | .5247 | .4521       | 〃      | 〃     | 〃     | 〃     | .4521       |
| oracle_excl (천장) | –      | .6203 | .6203 | .6236 | .6214       | –      | 〃     | 〃     | 〃     | .6214       |
| **flirds**       | .6425  | .5979 | .4364 | .5870 | .5404       | .6341  | .5840 | .3419 | .5339 | .4866       |
| flirds1st        | .6415  | .6156 | .2039 | .5586 | .4594       | .6362  | .5931 | .2569 | .5399 | .4633       |
| lossheur         | .6408  | .6062 | .4147 | .5693 | .5301       | .6432  | .5816 | .3242 | .5246 | .4768       |
| fedif            | .6411  | .6169 | .5234 | .5761 | **.5721**   | .6385  | .5945 | .2458 | .5254 | .4552       |
| gtg              | .6405  | .5626 | .5222 | .5702 | .5517       | .6400  | .5823 | .2797 | .5239 | .4620       |
| fedsv            | .6401  | .5687 | .4254 | .5620 | .5187       | .6436  | .5768 | .2739 | .5195 | .4567       |
| comfedsv         | .6385  | .5717 | .4091 | .5610 | .5140       | .6412  | .5818 | .2860 | .5261 | .4647       |
| shapleyfl        | .6412  | .5645 | .5103 | .5727 | .5492       | .6383  | .5839 | .2690 | .5121 | .4550       |

**읽기(절대 정확도로 직접; 축 평균 포함):**
- **renorm 4종의 free-rider 파국**(P1/P2 online): vanilla .588인데 gtg/fedsv/comfedsv/shapleyfl은 **.37~.40으로 vanilla보다도 한참 낮음** — 게이트가 free-rider(아무것도 안 보낸 클라)는 남기고 멀쩡한 클라를 내쫓았기 때문. 반면 estimator 4종(flirds .615·flirds1st .622·lossheur .611·fedif .614)은 천장 .620에 근접. 원인 진단(`observer_zero_semantics.csv`): renorm은 free-rider에 raw≤0을 **한 번도 안 주면서**(coalition-renorm이 zero-delta에도 몫을 배분) clean 참가자의 63–71%에 raw≤0을 줌. retrain·P4에선 파국이 완화(.51~.58) — 온라인 복리 악화가 없어서.
- **grad-noise = 2차항 판별 셀**: vanilla .244(오염 심함)·천장 .620. **flirds만 estimator 중 회복**(.567 online / .607 retrain), 반면 **flirds1st·fedif는 .248/.244로 vanilla와 사실상 동일 = noise 클라를 아예 못 걸러 전원 유지**(1차 정보만으론 안 보임). loss-heur는 online .598로 잡으나 retrain .452로 떨어짐. renorm은 GN에선 정상(.59~.62; 상대-순위만으로 충분). §5.1·§4.3 "2차항(HVP) 존재 이유"가 성능으로 재현.
- **label-flip@0.70 = 재학습 우위**: 전 estimator가 online(.57대) → retrain(.62대)로 상승(vanilla .525 대비). P2가 P1보다 소폭 나은 경향(크기가중이 kept 내 저기여 클라를 눌러줌).
- **clean(오발화)**: vanilla .6389가 천장. flirds1st .638·fedif .639는 발화 0(천장 유지)이나 **flirds .632·loss-heur .626은 소폭 하락**(clean 일부 오배제 — LLM 무발화 §3.2.3과 대비), renorm은 .60~.605로 가장 큰 오발화. P3/P4(부호 파괴·상대 게이트)는 전원 clean .638~.644로 안전하나 GN 회복도 약함(.34~.52).
- **계열평균 = 트레이드오프의 요약**(각 표 하단 2행): exact-0은 **free-rider 압승**(P1on .6155 vs renorm .3955), renorm은 **grad-noise 압승**(P1on .5983 vs exact-0 .4152 — 단 exact-0 평균은 flirds1st·fedif의 1차 실명 .248이 끌어내린 것, **Flirds 단독 GN .567~.607**). clean·label-flip은 근소. 정책 효과: exact-0은 **P2(크기가중) 오염평균 최고**(.5599 on / .5703 re), renorm은 **P1-retrain에서만 exact-0 상회**(.5772 vs .5416 — 온라인 free-rider 복리붕괴가 재학습엔 없고 grad-noise 강점만 남음); P4(z-게이트)는 양 계열 최저.
- **전 정책·시점 총평**(6개 오염-평균 열을 다시 평균한 점수원 순위): **Flirds .568 1위** > loss-heur .558 > FedIF .537 > renorm 4종 .517~.529 > **Flirds-1st .471 최하** — Flirds-1st가 estimator인데 꼴찌인 건 grad-noise 실명이 평균을 깎아서(2차항 유무가 계열 내에서도 가름). 앵커 = vanilla .452 / oracle_excl .621 / random_excl .448.

**(b2) 결과 — LLM silo5 noisy nr1.0 (R3)** — LLM 무대는 **accuracy 지표가 없고 final val-loss만**(오염 게이팅 rundir엔 MMLU/ROUGE 미측정 — §3.2.1 참조). renorm 4종 gate_v2 ×3-seed 신규(12런). 낮을수록 좋음; ±std는 seed 간 무대 분산 ~.046이라 arm 차이(4번째 소수)를 가림 — 같은 seed 내 대소로 읽음.

| arm | final val-loss ↓ (3-seed mean±std) | 게이트 P / R (오배제쌍) |
|---|---|---|
| vanilla (바닥) | 2.3340±.0471 | 발화 0 |
| oracle_excl (천장) | 2.3323±.0472 | – |
| random_excl (통제) | 2.3331±.0464 | – |
| gtg_gate_v2 | 2.3310±.0460 | 1.0 / 0.86~1.0 (0) |
| fedsv·comfedsv_gate_v2 (동일값) | **2.3308±.0462** | 0.875~1.0 / 0.86~1.0 (1쌍, s0·s2) |
| shapleyfl_gate_v2 | 2.3309±.0461 | 1.0 / 1.0 (0) |
| flirds·lossheur·oracleb_gate_v2 (§3.2.3 재사용) | 2.3340±.0471 | 발화 0(침묵) |
| flirds_w (soft 가중 참조, §3.2.3) | 2.3294±.0471 | – |

- **여기선 renorm이 이긴다(역전)**: renorm 4종 val-loss 2.3308~2.3310으로 **vanilla·oracle_excl보다도 낮음**. flirds/loss-heur는 게이트가 **침묵**(val-loss=vanilla 2.3340) — noisy 클라의 누적 φ가 계속 양수라 배제 대상이 없음(§5.2 판정 3: 0-교차가 nr≈3.4 외삽 = 도달불가). renorm은 자기 값-오차 덕에 noisy가 0 아래로 내려가 게이트 발화 → noisy 조기 배제로 4 점수원이 사실상 같은 궤적 수렴(FedSV≡ComFedSV 전 seed·GTG≡ShapleyFL 2/3 seed 소수 6자리 동일). 단 절대 갭이 val-loss 0.003 수준으로 작고, FedSV/ComFedSV는 예측대로 clean 오배제 1쌍 동반(2/3 seed).
- **CNN free-rider(renorm 파국) ↔ LLM noisy(renorm 이득) = zero-semantics의 위협-의존 트레이드오프**: 절대-0(estimator)은 zero-delta 위협(free-rider)에 정확·noisy엔 침묵; 상대-0(renorm)은 free-rider에 파국·noisy엔 (값-오차 부산물이지만) 유효. renorm의 LLM clean 셀은 미실행(Tier 2 스코프 밖) — LLM 오발화 리스크는 CNN clean(.60~.605 vs vanilla .639)으로만 실측.

**(b3) 결과 — std50k5 mixed (R2) seed0 파일럿**(**동결** — 1-seed caveat 필수; LLM이라 val-loss ↓):

| arm (seed0)        | final val-loss ↓ | 게이트 P / R   |
| ------------------ | ---------------- | ----------- |
| vanilla (바닥)       | 1.2887           | –           |
| oracle_excl (천장)   | 1.2864           | –           |
| random_excl (통제)   | 1.2884           | –           |
| **flirds_gate_v2** | **1.2860**       | .928 / .571 |
| shapleyfl_gate_v2  | 1.2863           | .901 / .858 |

50명 중 5명만 참여하는 부분참여 무대. flirds·shapleyfl 게이트 둘 다 vanilla·oracle보다 낮음(둘 다 val-loss로는 안 붕괴). **H-3("ShapleyFL은 random 수준으로 붕괴") 예측과 어긋나는 방향** — 이 무대는 fidelity(순위 재현)가 음수로 붕괴하는 곳인데(§4.2) 다운스트림 성능은 안 붕괴(un-normalized raw의 부호가 corrupt를 구분). 단 1셀·seed0 동결 — LLM 경쟁의 확정 판정은 R4(accuracy 심판)가 담당.

**예측표 H-1~7 대조** (MISS 포함 그대로 — 스펙 §3-3):

| # | 판정 | 근거 |
|---|---|---|
| H-1 (R1 순서 ≈ fidelity 순서) | 부분 적중 | 계열-수준(estimator 상위·renorm 하위)은 적중; 단 GN 셀 자체는 renorm도 .59~.62로 정상 — 순서가 갈리는 곳은 free-rider·label-flip |
| H-2 (R1 clean 무발화) | **MISS(정직 보고)** | Flirds .632·loss-heur .626 = vanilla .639보다 하락(parity 위반; CNN cum 0-교차 노이즈, LLM 무발화와 대비). renorm .60~.605; Flirds-1st .638·FedIF .639만 OK |
| H-3 (R2 ShapleyFL ≤ random) | 어긋남(예비·동결) | seed0: shapleyfl 1.2863 < random 1.2884(둘 다 vanilla보다 좋음) — 확정 판정은 R4로 이관 |
| H-4 (R3 renorm 발화) | **적중** | 발화·이득 실측(val-loss 2.3308~2.3310 < vanilla 2.3340); clean-오배제 동반은 FedSV/ComFedSV만(2/3 seed 1쌍) |
| H-5 (P2>P1 오염·clean 악화) | 점수원 의존 | FedIF·loss-heur는 P2가 P1보다 오염서 소폭↑; Flirds는 P1≈P2에 clean만 악화; renorm은 P2도 free-rider 붕괴 |
| H-6 (FR 동률·점진 T1 우위) | **MISS 다수** | label-flip은 retrain 압도(.62대 vs online .57대); free-rider는 flirds online>retrain·renorm은 retrain이 덜 나쁨(온라인 복리 악화 없음) |
| H-7 (fidelity가 다운스트림 예측) | 계열-수준만 | exact-0 vs renorm 계열 구분은 예측대로; 계열 내 1위는 정책·위협 의존(fidelity 순위와 불일치) + R2 ShapleyFL 반례(예비) = "fidelity ≠ 다운스트림"의 1급 사례 후보 |

**판정(Tier 1+2)**: ① **0-의미론이 게이트 실효성을 가른다** — exact-0 계열(Flirds·Flirds-1st·loss-heur·FedIF)만 free-rider 무대 생존(acc .61~.62 ≈ 천장), renorm 4종은 clean 오배제+free-rider 방치로 붕괴(acc .37~.40 < vanilla .59; CNN 3-seed). ② 그 반대급부 = **위협-의존 트레이드오프**(LLM noisy선 renorm 발화가 이득·절대-0은 침묵) — "0을 어떻게 정의하느냐"가 *어느 위협에서 개입이 작동하는지*를 결정. ③ 계열 내 1위는 정책·시점 의존 — 정확한 클레임은 "**Flirds = 전 정책·전 시점 상위권 + grad-noise를 잡는 유일한 estimator(2차항: flirds GN .567~.607 vs flirds1st/fedif .244~.248)**"이지 단독 1위가 아님(clean·label-flip 등 개별 칸 최고는 FedIF·loss-heur도 차지). ④ clean 오발화 없이 완주는 FedIF·Flirds-1st뿐 — Flirds·loss-heur의 CNN clean 소폭 하락은 §3.2.4 V2w 불승격과 같은 뿌리.

**(c) 출처·baseline-set 노트**
- **출처**: `runs/track_h/rundirs_cnn/`(경쟁 96런)·`rundirs_llm/`(12런) + §3.2.3–4 재사용(track_g) · `runs/track_h/analysis/{competition_score,cnn_competition,llm_competition,observer_zero_semantics}.csv`(**07-20 집계 정정판** — 수정 내역 3건은 `make_analysis.py` docstring; R2 표의 정본 = `llm_competition.csv`). 코드 = `codes/flirds/fl/score_providers.py` + `experiments/track_c2.py`(관찰자·T2)·`track_g.py`(provider 확장) + `tests/test_track_h.py`.
- baseline-set = 점수원 8종 전부(경쟁 설계라 renorm 게이트 포함 — §3.2 서두 판정). 통제 = vanilla / oracle_excl / random_excl.

---

## 3.3 Corrupt-client detection (2차 ③) — 오염 클라 탐지 AUROC와 정직한 한계

> 위계상 **마지막**(기여도≠탐지). valuation φ를 탐지 스코어(corrupt=high-φ)로 쓴 AUROC와 전용 탐지기 AUROC를 같은 셀에서 함께 본다. Fidelity Spearman(vs (b)/Flirds-proxy)도 같은 표에 병기(§3.1.4의 전체 표가 여기).

### 3.3.1 Robustness cross-silo N=5 (`phase2_matrix/1B_silo5_*`) — **β0.3 재실행판(ce0b454) 정본**

**(a) 세팅**: Llama-3.2-1B, **N=5 전원**, R=10, local 10 steps, batch 16, lr=1e-3 (poison=2e-3), maxlen 768, train=200/val=20/test=40, warmup=2; 위협별 1명 오염(noisy=client0, free-rider=client1, poison=client0); **(b)=exact 2⁵**; 3 seeds. **poison** = D2b model-replacement backdoor(lr=2e-3, batch=8, epochs=5, frac=0.8), deployed ASR≈1.00. ⚠ 오염 4셀 rundir는 **2026-07-20 β0.3 재실행판(ce0b454)이 canonical**(스위트에 Fed-LOO·ComFedSV 추가, timing.json 신설; 재실행 전 값은 git 이력 — 아래 poison 각주).

**(b) 결과 — AUROC(corrupt=high-φ) + Spearman vs (b) + runtime, 3-seed mean±std**

| method | noisy AUROC ↑ | noisy Sp ↑ | frrand AUROC ↑ | frrand Sp ↑ | frzero AUROC ↑ | frzero Sp ↑ | **poison AUROC ↑** | **poison Sp ↑** | runtime ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Flirds | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.500±.354** | **0.600±.283** | ~107s |
| Flirds-1st | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.000** | **0.000** | ~35s |
| loss-heur | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ~99s |
| Fed-LOO | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ~118–129s |
| FedIF | 1.000 | 0.933±.05 | 1.000 | 0.900±.08 | 1.000 | 0.900±.082 | 1.000 | 0.967±.05 | ~37s |
| GTG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.867±.12 | ~540s |
| FedSV | 1.000 | 1.000 | 1.000 | 0.967±.047 | 1.000 | 0.933±.047 | 1.000 | **0.367±.26** | ~535s |
| ShapleyFL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ~530s |
| ComFedSV | 1.000 | 0.833±.170 | 1.000 | 0.867±.125 | 1.000 | 0.867±.125 | 1.000 | 0.733±.189 | ~387–429s |
| (b)oracle | 1.000 | (truth) | 1.000 | (truth) | 1.000 | (truth) | 1.000 | (truth) | ~530s |
| FLDetector | 0.750 | – | 1.000 | – | 0.750 | – | 1.000 | – | ~40–91s |
| STD-DAGMM | 0.250±.204 | – | 1.000 | – | 0.000±.000 | – | 0.833±.118 | – | ~180–390s |
| FLTrust | 1.000 | – | 1.000 | – | 1.000 | – | 1.000 | – | ~37s |
| FedDQC | 0.917±.12 | – | 0.750 | – | 0.750 | – | 1.000 | – | ~22s |

> **읽기**: noisy·free-rider 위협에선 valuation 거의 전부 AUROC 1.0 + Spearman 0.87~1.0(near-additive). **poison(clean-보존 backdoor)이 분리점**: 공격자가 clean val-loss를 낮춰 φ가 "기여 높음"으로 나옴 → **Flirds-1st AUROC 0.000 / Spearman 0.000 = 완전 회피**; **2차항 있는 Flirds도 0.500±.354(per-seed .25/.25/1.0) = seed-혼재 간헐 방어**. loss-heur·(b) oracle·Fed-LOO·FedIF·GTG·FLDetector·FLTrust는 1.0으로 잡음. FedSV Spearman이 poison서 0.367로 추락(per-round MC 분산 + near-additive 붕괴 첫 사례).
> **poison 각주(정본 이력)**: β0.3 재실행 전 rundir(2026-06 캠페인)는 Flirds poison AUROC **0.917±.118**·Sp 0.967±.047 — 재실행(같은 무대·seed, EMA β만 통일+스위트 확장)으로 0.500/0.600이 됨. 즉 "2차항의 poison 방어"는 **run-인스턴스 의존**(§4.5 dose 전이대의 seed-불안정과 정합; 재현성 정정 H1[§6.2 caveat 13]의 절대값 비재현과도 무관하지 않음). 3B에선 Flirds도 0.000(§3.3.3).
> **runtime**: loss-heur ~99s = C6 측정버그 post-fix 실측(96.6/100.1/99.9s; §6.2 caveat 11). FLDetector·STD-DAGMM는 재실행판에서 위협·seed별 분산 큼(각 40–91s / 184–385s). free-rider 변형 **frdelta**(직전 글로벌 delta 재활용)는 별도 셀 → **§3.3.4**.

**(c) 출처·baseline-set 노트**
- **출처**: `runs/phase2_matrix/analysis/00_overview/master_metrics.csv`(gitignored 파생 — `make_analysis.py` 재생성) ← `runs/phase2_matrix/rundirs/1B_silo5_*`. 코드 = `codes/experiments/phase2_matrix.py`.
- 포함 14 = valuation 9(Flirds/Flirds-1st/loss-heur/Fed-LOO/FedIF/GTG/FedSV/ShapleyFL/ComFedSV) + (b)oracle + 탐지기 4(FLDetector/STD-DAGMM/FLTrust/FedDQC). 모든 method가 모든 위협서 동작("category-together"). Fed-LOO·ComFedSV는 β0.3 재실행 스위트에서 추가(재실행 전엔 ComFedSV = partial-participation 전용 사유로 제외였음 — silo full에서도 실측되며 Sp 0.73~0.87 하위권).

### 3.3.2 Robustness cross-device N=100 α-sweep + anchor + poison (`phase2_matrix/1B_device100-a*`)

**(a) 세팅**: Llama-3.2-1B, **N=100, 라운드당 10명(10%)**, R=30, local 5 steps, batch 16, lr=1e-3, per_client=300, Dirichlet α∈{0, 0.01, 0.1, **0.5=anchor**, 5.0}, val=10/test=40, warmup=3; 오염 클라 5명(idx 10/30/50/70/90); 3 seeds. **α=0.5 = Anchor cell**: (b) per-round 오라클 + coalition baseline(GTG/FedSV/ShapleyFL) 켬. 그 외 α: cheap method + Flirds proxy reference. poison 셀 = §3.3.2(b4).

**(b1) noisy 위협, detection AUROC ↑** (열=α; 3-seed mean±std)

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
> **정본 확정(재검증)**: (b)oracle noisy anchor AUROC **0.604±.041** = 3-seed 정본(per-seed 0.660/0.563/0.589, master_metrics 직접 확인; Flirds도 seed별 동일값 = Spearman 1.0의 귀결). 일부 문서에 돌던 **0.660은 seed0 단독값** — 3-seed 평균 아님.

**(b2) free-rider(random / zero) detection AUROC ↑** (대표 α; 전체 α는 master_metrics)

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

**(b3) Spearman vs truth ↑** — 여기서 **`truth` = 그 셀에서 Spearman을 잰 비교 기준(정답)**이고, 셀마다 다르다:
> - **α=0.5 (anchor) 3열**: `truth` = **(b) per-round exact oracle** (그 칸은 (b)를 실제로 돌린 anchor cell) → Flirds=+1.000은 *진짜 exact oracle 대비* 일치.
> - **맨 오른쪽 (off-anchor) 열**: `truth` = **Flirds proxy reference** (α∈{0,0.01,0.1,5.0}는 정확 (b)가 칸당 ~25,000s라 미실행 → 검증된 Flirds를 기준 대용) → Flirds-1st·loss-heur의 +1.000은 *"Flirds와 동일 순위"* 를 뜻하지, vs exact oracle이 아니다.

| method | noisy α=0.5<br>(truth=(b)) | frrand α=0.5<br>(truth=(b)) | frzero α=0.5<br>(truth=(b)) | off-anchor α<br>(truth=Flirds proxy) |
|---|---|---|---|---|
| Flirds | 1.000 | 1.000 | 1.000 | (proxy 기준) |
| Flirds-1st | 1.000 | 1.000 | 1.000 | 모든 α 1.000 |
| loss-heur | 1.000 | 1.000 | 1.000 | 모든 α 1.000 |
| FedIF | 0.721±.027 | 0.827±.022 | 0.824±.017 | α별 0.62~0.83 |
| GTG | 0.784±.021 | 0.817±.022 | 0.843±.026 | (anchor만) |
| FedSV | 0.752±.020 | 0.795±.020 | 0.814±.018 | (anchor만) |
| ShapleyFL | 0.582±.075 | 0.685±.054 | 0.681±.049 | (anchor만) |
| ComFedSV | -0.023±.127 | -0.051±.153 | -0.051±.142 | 모든 α ≈0 |

**(b4) cross-device poison** (`1B_device100-a{0,0.5}_poison`; §3.3.2(a) + poison(D2b, lr=2e-3, batch=8, R=60, max_steps=10, frac=0.8); α=0.0(ASR≈1.00) / α=0.5(ASR≈0.50); truth=Flirds proxy; 3 seeds)

| method | α=0.0 AUROC ↑ | α=0.0 Sp ↑ | α=0.5 AUROC ↑ | α=0.5 Sp ↑ |
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

> device100 poison은 silo5만큼 강하게 설치 안 됨(cross-device 희석; α=0.5 ASR 0.50). 여기선 Flirds(2차) AUROC 1.0(α=0)으로 회피 안 됨 = 설정 의존(silo5·3B와 대비). caveat: tiny val=10.

**(c) 출처·baseline-set 노트**
- **출처**: `master_metrics.csv` (02_device100_sweep · 03_device100_poison). anchor cell runtime: (b)perround ≈25,000s, GTG ≈16,700–19,800s(seed별 16,667/17,947/19,832; mean ≈18,100), FedSV ≈4,970s, ShapleyFL ≈24,900s, Flirds ≈157s, Flirds-1st ≈53s.
- **off-anchor(α≠0.5)** 포함 9 = Flirds/Flirds-1st/loss-heur/FedIF/ComFedSV + 탐지기 4. **제외**: GTG/FedSV/ShapleyFL/(b)oracle ─ *적용규칙: MC Shapley/exact = 대규모서 비용 게이팅 → anchor만*. ComFedSV는 partial-participation Shapley baseline으로 포함 ─ *적용규칙: 참여형태(partial→ComFedSV)*.
- **anchor(α=0.5)** 포함 13 = 위 + GTG/FedSV/ShapleyFL/(b)perround 켬. poison 셀 포함 9 = Flirds/Flirds-1st/loss-heur/FedIF/ComFedSV + 탐지기 4(coalition off — 비용/설계).

### 3.3.3 Robustness cross-silo N=5 · 3B (`phase2_matrix/3B_silo5_*`) — **1 seed (◐)**

**(a) 세팅**: Llama-3.2-3B, N=5 full, R=10, batch 8; poison lr=2e-3/frac=0.8; **seeds=[0]만**(3-seed ⬚=계획 P5; β0.3 재실행 잔여 18셀에 3B silo5 4셀 포함 — 루트 REMAINING §1.2); (b)=exact 2⁵.

**(b) 결과 (1 seed)**

| method | noisy AUROC ↑ | noisy Sp ↑ | frrand AUROC ↑ | frzero AUROC ↑ | **poison AUROC ↑** | **poison Sp ↑** | runtime(noisy) ↓ |
|---|---|---|---|---|---|---|---|
| Flirds | 1.000 | 1.000 | 1.000 | 1.000 | **0.000** | **0.000** | ~251s |
| Flirds-1st | 1.000 | 1.000 | 1.000 | 1.000 | **0.000** | **0.000** | ~82s |
| loss-heur | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ~384s |
| FedIF | 1.000 | 0.600 | 1.000 | 1.000 | 1.000 | 0.600 | ~82s |
| (b)oracle | 1.000 | (truth) | 1.000 | 1.000 | 1.000 | (truth) | ~1244s |
| FLDetector | 1.000 | – | 1.000 | 1.000 | 1.000 | – | ~133–382s |
| STD-DAGMM | 0.250 | – | 1.000 | 0.000 | 0.750 | – | ~206–745s |
| FLTrust | 1.000 | – | 1.000 | 1.000 | 1.000 | – | ~83–91s |
| FedDQC | 1.000 | – | 0.750 | 0.750 | 1.000 | – | ~46–50s |

> **3B poison**: Flirds·Flirds-1st 둘 다 AUROC 0.000 / Spearman 0.000 = **clean-보존 backdoor에 완전 회피**(1B silo5의 Flirds 2차는 seed-혼재 0.500으로 간헐 방어했으나 3B는 둘 다 0). loss-heur·(b)·FedIF·FLDetector·FLTrust·FedDQC = 1.0으로 잡음.

**(c) 출처·baseline-set 노트**: `master_metrics.csv` (05_scale_3b). 포함 9 = valuation 4 + (b) + 탐지기 4(coalition off).

### 3.3.4 frdelta — delta-재활용 free-rider (E7, `rundirs_2026-07/1B_silo5_frdelta`)

**(a) 세팅**: §3.3.1 silo5 무대 동일(N=5 full, R=10, (b)=exact 2⁵, 3-seed), 위협만 교체 — **freerider_delta**: client 1이 로컬 학습 없이 **직전 라운드의 실현된 글로벌 집계 delta(Δw = w^r − w^{r−1})를 그대로 재제출**(Lin et al. delta-weights 공격; round 0은 Δw=0 폴백; 데이터는 clean, update만 조작). zero/random과 달리 재활용 delta는 val gradient와 **실제 정렬**이 있어 1차항 ⟨−∇ℓ_val, Δw⟩ ≠ 0 → "free-rider φ = exact 0" 성질(§3.3.1 frzero)이 구조적으로 깨지는 스트레스 셀. method 스위트 = valuation 계열(Fed-LOO 포함) + 탐지기 4.

**(b) 결과 — AUROC ↑ · Spearman vs (b) ↑** (3-seed mean±std; per-seed는 이산 {0,.25,.5,.75,1})

| method | AUROC ↑ | Sp vs (b) ↑ | | method | AUROC ↑ | Sp vs (b) ↑ |
|---|---|---|---|---|---|---|
| **(b)oracle** | **0.333±.118** (.25/.25/.50) | (truth) | | ComFedSV | 0.667±.236 | 0.600±.356 |
| **Flirds** | **0.333±.118** (.25/.25/.50) | **1.000±.000** | | FedIF | 0.000 | 0.733±.094 |
| Flirds-1st | 0.333±.118 | 1.000±.000 | | FLDetector | 0.000 | – |
| loss-heur | 0.333±.118 | 1.000±.000 | | FLTrust | 0.000 | – |
| Fed-LOO | 0.333±.118 | 1.000±.000 | | **STD-DAGMM** | **1.000±.000** | – |
| GTG | 0.333±.118 | 1.000±.000 | | FedDQC | 0.750±.000 | – |
| FedSV / ShapleyFL | 0.333 / 0.250 | 0.933 / 0.967 | | | | |

> **읽기 — "φ=0 exact는 zero/random 한정"의 실측 확정**: Flirds는 Spearman +1.000(전 seed)으로 oracle을 완벽 추종하는데 AUROC가 낮다(0.25/0.25/0.50). 그런데 **(b)oracle 자신의 AUROC가 seed별로 정확히 같다** — 직전 글로벌 delta는 val-loss를 실제로 낮추는 방향이라 **val-loss 게임의 정직한 답이 "이 클라는 기여함"**((b) 저장-φ 전 클라 음수 −0.0017~−0.0052 = 기여 방향 전원 양수, free-rider가 중간 순위에 섞임; Flirds φ와 소수 6자리 일치 cos_d≈6e-9[seed0]). 즉 낮은 AUROC는 추정 실패가 아니라 **게임 자체의 성질**(루트 위계 2차-③ "기여도≠탐지"의 정확한 사례; poison C-8과 같은 구조가 free-rider축에도 존재). 탐지 자체는 update-패턴 탐지기의 몫 — **STD-DAGMM 1.0**(복제 delta의 이상 패턴 탐지); 반면 FedIF·FLDetector·FLTrust는 **0.0**(free-rider를 최고 가치로 오정렬 = 방향 정렬 기반 방법들이 완전히 속음).

**(c) 출처·baseline-set 노트**: §3.3.1과 동일 규약 + Fed-LOO 추가(E-세션 스위트). 단일 rundir에 3-seed 동거(`freerider_delta_seed{0,1,2}` 키; `make_analysis.py` 미반영 → rundir 직접 집계). **출처**: `runs/phase2_matrix/rundirs_2026-07/1B_silo5_frdelta/{config.yaml,metrics.json,phi.parquet}` (커밋 5ed9b9e); 위협 구현 = `codes/flirds/data/corruptors.py` `free_rider(mode="delta")`.

### 3.3.5 CNN 탐지 AUROC (`track_c`; §3.1.2·§3.2.2와 같은 셀)

**(a) 세팅**: §3.1.2 C1과 동일 런. φ-as-detector AUROC(corrupt=high-φ)는 **per-client 오염 ladder가 있는 4개 시나리오 그룹만 정의**(label_flip·feature_noise × mnist·cifar10; iid/label_skew/quantity_skew 셀엔 auroc 키 없음 — 오염 ladder 부재).

**(b) 결과 — AUROC ↑, 3-seed mean** (±std는 파일 재계산으로; 대표 분산 예 = GTG cifar10/label_flip ±.319, (b) cifar10/feature_noise ±.059)

| dataset/scenario | (b)oracle | Flirds | Flirds-1st | GTG | FedSV | ComFedSV | ShapleyFL | FedIF | loss-heur |
|---|---|---|---|---|---|---|---|---|---|
| cifar10 / label_flip | 1.000 | 1.000 | 1.000 | .625 | .542 | .625 | .500 | .854 | .979 |
| cifar10 / feature_noise | .833 | .833 | .812 | .604 | .604 | .708 | .646 | .792 | .812 |
| mnist / label_flip | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | .979 | 1.000 | 1.000 | 1.000 |
| mnist / feature_noise | .375 | .542 | .500 | .438 | .667 | .521 | .292 | .458 | .521 |

> 읽기: Flirds·Flirds-1st가 (b)와 같은 수준(label_flip 1.0, cifar10 feature_noise 0.81~0.83) — φ-as-detector도 oracle-동률. mnist/feature_noise는 **(b) 자체가 0.375** = mild σ 무대라 게임 수준에서 신호가 약함(방법 실패 아님; §4.4 A3의 mnist feature_noise ≈중립과 정합). 같은 셀 metrics.json엔 `spearman_vs_rate`(φ vs 오염강도)도 저장(예 cifar10/label_flip: (b) .968·Flirds .960).
> **C2 arm-level 탐지 AUROC**(개입 arm의 φ-as-detector·N=100)는 셀별 metrics.json(`arms.<arm>.auroc`)에 존재하며 본 문서엔 표로 미수록 — 예: cifar10_dir1_free_rider에서 fedif .996 / flirds_mult .385 / shapleyfl .000. probe 확장판의 AUROC는 §4.3.

**(c) 출처·baseline-set 노트**: `runs/track_c/c1/*/metrics.json` (`methods.<name>.auroc`; ladder 4그룹 12셀). 포함 = §3.1.2와 동일 8종 + (b).

---

## 3.4 Cost · scalability — wall-clock과 연산수

### 3.4.1 op-count 축 (`measured_2026-07/op_counts.py`) — 하드웨어·정밀도 독립 비용 모델

**(a) 세팅**: 방법별 라운드-누적 연산수(해석적) × microbench per-op 실측(fp32·val100·B200: **forward 1.60s · HVP 10.36s → HVP/fwd = 6.47**)의 곱이 측정 wall-clock을 재현하는지 검증(fp32/재구현 caveat 우회; 논문 tab:opcount).

**(b) 결과 — 지배 연산 수** (fwd 환산 제외):

| regime | Flirds | Flirds-1st | loss-heur | Fed-LOO | (b) |
|---|---|---|---|---|---|
| silo (N5·R10) | 10 HVP | 10 grad | 60 fwd | 70 fwd | 320 fwd |
| anchor (N5·R30) | 30 HVP | 30 grad | 180 fwd | 210 fwd | 960 fwd |
| device (N100·R30·K10) | 30 HVP | 30 grad | 330 fwd | 360 fwd | **30,720 fwd** |

> 교차검증: silo Flirds 10 HVP×10.36 = **104s**(실측 ~107) · (b) 320 fwd×1.60 = **512s**(~530) · loss-heur 60 fwd = **96s**(post-fix 실측 96.6~100.1s와 일치; pre-fix 170s가 버그였음을 op-count가 독립 확인 §6.2 caveat 11) · device (b) 30,720 fwd → 실측 **24,975s** vs Flirds 157s ≈ **159×**(§3.3.2 정합).

**(c) 출처**: 산출 = stdout(재실행 `python runs/measured_2026-07/op_counts.py`; microbench 입력 = `runs/measured_2026-07/microbench/summary.json`). 학습시간 위상분리·E3 CNN cost 스모크 등 계측 세부 = §6.3.

### 3.4.2 LLM 표준 runtime (`track_d`) — method별, 3-seed mean±std (초 ↓)

| method | 1B std20 | 1B anchor5 | 3B std20 | 3B anchor5 | 7B std20 | 7B anchor5 |
|---|---|---|---|---|---|---|
| **Flirds-1st** | 1531±37 | 231±5 | 3633±79 | 548±12 | 6485±87 | 975±13 |
| **Flirds** | 4697±112 | 707±16 | 11163±238 | 1679±37 | 20180±250 | 3027±36 |
| FedIF | 1534±37 | 232±5 | 3638±77 | 550±12 | 6495±88 | 978±13 |
| loss-heur | 2913±72 | 1093±26 | 6912±156 | 2591±60 | 12299±167 | 4613±62 |
| ComFedSV | 2330±22 | 2557±215 | 5531±38 | 6057±509 | 9839±110 | 10792±1044 |
| GTG | 3647±90 | 3552±82 | 8653±197 | 8415±192 | 15393±206 | 14972±193 |
| FedSV | 3646±90 | 3536±86 | 8654±199 | 8378±199 | 15393±208 | 14907±270 |
| ShapleyFL | 2917±72 | 3513±83 | 6921±159 | 8324±193 | 12312±167 | 14812±199 |
| **(b)oracle** | 2917±72 | 3528±83 | 6923±154 | 8350±192 | 12310±165 | 14839±196 |
| **(a)oracle** | – | **30817±244** | – | ⬚ | – | ⬚ |

> 읽기 (비용 모델 — 중요): **Flirds-1st = 항상 최저가**(1 val-gradient/round, Hessian 없음). **Flirds(2차)의 비용은 라운드당 cohort 크기와 무관**(1 HVP/round 고정)인데 **(b) oracle 비용은 라운드당 cohort에 지수적**(2^k coalition-eval/round). 그래서 우열이 무대마다 갈린다:
> - **cohort가 크면 Flirds(2차)가 (b)를 크게 이김**: anchor5(전원 N=5 → 2⁵/round) 707s vs (b) 3528s(≈1/5); device100 anchor(K=10 → 2¹⁰/round) 157s vs (b) **25,000s**(≈1/159).
> - **cohort가 작으면 (b)가 더 쌈 → std20에서 Flirds(2차)가 (b)보다 오래 걸림**: std20은 **라운드당 2명만 참여(2²=4 coalition-eval/round)라 (b) per-round가 이미 저렴** → 1B std20 (b) 2917s **<** Flirds(2차) 4697s (1 HVP[forward+backward]가 4 forward-pass보다 비싸서). 단 Flirds-1st 1531s는 이 레짐에서도 최저.
> (a) retrain oracle은 (b)의 **~9배**(1B anchor5 30,817s vs 3528s). **요지: Flirds(2차)의 비용 우위는 "라운드당 참여가 많아 exact 2^k가 비싼" 무대(anchor5 full·device100)에서만 나오고, std20처럼 cohort가 작아 (b)가 싼 곳에선 Flirds-1st만 우위다.**
> ⚠ **loss-heur 행 = C6 측정버그 pre-fix 값**(singleton fwd 2\|P_r\|→1+\|P_r\| 수정; §6.2 caveat 11): 1B post-fix 정본 = anchor5 **657±19s**·std20 **2199±60s**(E4 rundir §3.1.3) — 표의 1093/2913s는 각 1.66×/1.32× 과대(이론 비율과 정확 일치; 3B/7B 열도 동일 구조 과대, 재측정 대기). full 참여 지수 비용의 N=10 실측 연장 = **§3.1.3 E5**: (b) exact 2¹⁰ **117,649s(32.7h)** vs Flirds 733s = **160×**.

**출처**: `runs/track_d/rundirs/*/metrics.json` (`runtime`).

### 3.4.3 오염 무대·CNN runtime 요약

- **Robustness** (§3.3 표에 병기): N=5 silo5 — Flirds-1st ~35s / Flirds ~107s / loss-heur ~99s(post-fix) / Fed-LOO ~118–129s / (b)·coalition ~530s / 탐지기 22~390s. N=100 anchor — Flirds-1st ~53s / Flirds ~157s vs (b)perround **~25,000s** / GTG ~16.7–19.8k s(mean ~18.1k) / ShapleyFL ~24.9k s / FedSV ~4970s.
- **CNN** C1 (cross-silo N=10, mode=full, 3-seed mean±std, 초 ↓; c1 metrics.json `methods.<name>.runtime`):

**MNIST**

| method | iid | feature-noise | label-flip | label-skew | quantity-skew |
|---|---|---|---|---|---|
| **Flirds-1st** | 0.08±0.00 | 0.08±0.00 | 0.08±0.00 | 0.08±0.00 | 0.08±0.00 |
| **Flirds** | 0.73±0.03 | 0.58±0.05 | 0.59±0.03 | 0.61±0.06 | 0.55±0.03 |
| FedIF | 0.18±0.00 | 0.16±0.01 | 0.17±0.01 | 0.17±0.01 | 0.16±0.01 |
| loss-heur | 0.24±0.00 | 0.24±0.00 | 0.24±0.00 | 0.24±0.00 | 0.24±0.00 |
| ComFedSV | 5.13±0.18 | 5.17±0.04 | 5.13±0.18 | 5.12±0.14 | 5.10±0.15 |
| GTG | 18.11±0.11 | 19.77±0.48 | 16.29±1.51 | 25.01±0.90 | 17.26±0.95 |
| FedSV | 6.23±0.07 | 6.29±0.11 | 6.29±0.14 | 6.22±0.11 | 6.21±0.04 |
| ShapleyFL | 30.65±0.26 | 30.80±0.43 | 30.58±0.64 | 30.47±0.56 | 30.31±0.31 |
| **(b)oracle** | 31.09±0.36 | 31.33±0.47 | 31.04±0.62 | 30.89±0.55 | 30.91±0.48 |
| _traj_time(FL학습)_ | 93.88±1.32 | 92.85±1.02 | 91.81±1.16 | 84.57±1.56 | 85.36±1.38 |

**CIFAR-10**

| method | iid | feature-noise | label-flip | label-skew | quantity-skew |
|---|---|---|---|---|---|
| **Flirds-1st** | 0.34±0.01 | 0.37±0.04 | 0.38±0.05 | 0.34±0.01 | 0.34±0.01 |
| **Flirds** | 1.20±0.08 | 1.62±0.49 | 1.44±0.34 | 1.27±0.17 | 5.78±6.24 |
| FedIF | 0.44±0.01 | 0.48±0.02 | 0.50±0.07 | 0.54±0.15 | 0.49±0.06 |
| loss-heur | 1.27±0.01 | 1.30±0.03 | 1.27±0.00 | 1.28±0.01 | 1.30±0.02 |
| ComFedSV | 18.45±0.36 | 19.18±1.10 | 19.07±0.82 | 18.83±0.19 | 19.04±0.51 |
| GTG | 88.98±2.02 | 87.84±5.34 | 88.25±3.10 | 102.46±0.58 | 75.50±0.96 |
| FedSV | 22.29±0.27 | 23.10±0.73 | 23.19±0.91 | 23.87±1.76 | 22.81±0.25 |
| ShapleyFL | 110.83±1.28 | 115.58±4.74 | 112.25±1.82 | 113.03±1.14 | 115.42±3.39 |
| **(b)oracle** | 110.48±0.93 | 119.41±4.78 | 114.54±3.85 | 112.87±1.16 | 168.69±71.77 |
| _traj_time(FL학습)_ | 80.21±0.45 | 87.28±5.38 | 86.70±7.24 | 83.12±2.05 | 91.23±3.36 |

> 읽기: 순서는 LLM 트랙과 동일 — **Flirds-1st 최저**(0.08/0.34s) < Flirds(0.6/~2s) ≪ exact 2¹⁰급 (b)oracle≈ShapleyFL(~31s MNIST / ~113s CIFAR), GTG(truncated MC, ~19/~89s)·FedSV(~6/~23s)는 그 사이. Flirds·Flirds-1st는 `traj_time`(FL 학습 자체, ~80–94s)보다도 2~3 자릿수 싸다 = "기여도 추정이 학습보다 훨씬 저렴". *(a) 2¹⁰ retrain oracle은 fidelity 비교셋이 아니라 별도 `runs/track_c/c1_oracle/*_aonly_*`에 있어 이 표엔 미포함(t_a ≈ 32,725~42,448s/셀).*

**출처**: `runs/track_c/c1/*/metrics.json` (`methods.<name>.runtime`, `traj_time`). FL 학습시간 위상분리 실측(timing.json)·E3 CNN cost 스모크 = **§6.3 검증-전용 기록**.

---

# 4. Ablation 실험

> 방법의 구성요소(2차항·β)·무대 lever(rank·lr·steps·폭·참여)·프로토콜 변형(removal·dose·AdamW·게이트 정책 변형)이 결론을 바꾸는지 검증하는 실험들. 포맷은 Main과 동일한 3블록.

## 4.1 2차항(HVP)의 기여 — Flirds-1st vs Flirds (근거 종합; 전용 rundir 없음)

> 별도 실험 세트가 아니라 **전 트랙에 걸쳐 Flirds-1st(1차만)와 Flirds(1차+2차)를 병렬 실측**한 결과의 종합. 2차항이 값을 하는 곳:
> - **부분참여·짧은 지평 fidelity**: CNN k=0.2에서 Flirds 0.891 vs **Flirds-1st 0.305**(§4.3); LLM std50k5는 클라당 참여 ~20회라 1차도 1.0 유지 = "참여 분수"가 아니라 **클라당 참여 횟수**가 1차항 정확도의 조건(§6.2 caveat 10).
> - **grad-noise 개입**: Track H에서 Flirds GN acc .567~.607 vs **Flirds-1st/FedIF .244~.248 = vanilla 수준 실명**(§3.2.6) — 1차 정보만으론 noise 클라가 안 보임.
> - **물리 근사**: 2차 잔차가 1차의 ~1/3(2.7–3.4× 개선; §5.5).
> - **한계**: poison(clean-보존)은 2차항 방어도 seed-혼재(1B 0.500)·스케일 취약(3B 0.000; §3.3.1·§3.3.3). 비용은 Flirds-1st가 전 무대 최저가(§3.4.2) — 2차항의 가치는 위 세 무대에서만 비용을 정당화.

## 4.2 A축 lever probe — LLM (rank·참여·lr·steps·noise; `probe_signal`)

> 배경·가설: [[flirds-signal-size-diagnosis]]. 질문 = "IID-clean에서 fidelity·개입 효과가 약한 것이 **학습 강도(A축: 모델 용량·라운드당 참여 수·lr·steps)** 부족 때문인가, 아니면 **클라 간 진짜 차이(B축: 오염·비IID)** 부재 때문인가." B축(§3.1.5)과 분리해 A축 lever만 바꾼다. **판정 요약 = §5.3** (A축 lever는 어느 것도 cross-seed 실재 신호를 못 만든다 — 신호는 B축이 만든다).

**(a) 세팅**: Llama-3.2-1B, plain SGD mom=0, maxlen 512, val=200, batch 16, LoRA α=2r, fp32. truth=(b) in-run oracle; 방법 스위트 = §3.1.1과 동형. seed: 파일럿 seed0 → **lr격자(st10 열)·std50k5(r16)·noise(r16) 3-seed 확정**(std50k5 seeds 1-2는 Flirds·Flirds-1st만 채점하는 경량 스위트); rank probe·st20/30·r32/64는 seed0 유지. lever 3무대 + noise:
- **rank probe** (anchor5): N=5 전원, R=30, 10 steps, lr=1e-3, **LoRA r∈{16,32,64}**(r16 = `track_d/1B_anchor5_seed0` 재사용) = 용량 lever @ full 참여(A축 순수).
- **참여 probe** (std50k5): **N=50 중 5/round**, R=200, 10 steps, lr=1e-3, r∈{16,32,64}, (b)=exact per-round 2⁵ = 저참여 축(std20의 2/20보다 낮은 5/50).
- **lr·steps 격자** (anchor5): N=5 full, R=30, r16, **lr∈{1e-3,2e-3,3e-3}×steps∈{10,20,30}** 3×3(lr1e-3·st10 = 재사용) = 학습 강도 lever.
- **noise probe** (anchor5 r16/r64): trained 모델 고정 + val chunk(20) bootstrap(2000) → φ의 측정노이즈 하한.

**(b1) rank probe — 용량은 φ 크기·fidelity 어느 것도 안 바꿈** (anchor5 full, seed0; vs (b) Spearman ↑)

| rank | (b)φ range | Flirds | Flirds-1st | loss-heur | GTG | FedSV |
|---|---|---|---|---|---|---|
| r16 (재사용) | 0.00119 | 1.000 | 1.000 | 1.000 | 1.000 | 0.700 |
| r32 | 0.00102 | 1.000 | 1.000 | 1.000 | 1.000 | 0.700 |
| r64 | 0.00106 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 |

> rank 4×(16→64)로도 (b)φ range는 0.001 근처 평평(오히려 미세↓), Flirds/Flirds-1st/loss-heur/GTG 전부 vs (b) 1.000 유지. **용량은 IID-clean 신호를 안 키우고 fidelity도 안 흔든다**(HVP가 크기 2r LoRA 공간에서 2차항 비중이 바뀌어도 무영향). full 참여라 near-additive → 대부분 방법이 붕괴-동률(§3.1.1 anchor5와 같은 구조); FedSV 등 부분참여-MC 계열만 seed0 노이즈로 흔들림(r64 FedSV 0.5).

**(b2) 참여 probe — 부분참여가 방법 구별을 만든다 (Flirds 우위)** (std50k5, N=50 5/round, seed0; vs (b) Spearman ↑)

| method | r16 | r32 | r64 |
|---|---|---|---|
| **Flirds** | **1.000** | **1.000** | **1.000** |
| Flirds-1st | 1.000 | 1.000 | 0.999 |
| loss-heur | 1.000 | 1.000 | 0.999 |
| GTG | 0.983 | 0.983 | 0.981 |
| FedSV | 0.910 | 0.899 | 0.909 |
| FedIF | -0.040 | -0.076 | -0.052 |
| ShapleyFL | -0.064 | -0.093 | -0.078 |
| ComFedSV | -0.109 | -0.125 | -0.081 |

> (b)φ range 0.00402 / 0.00440 / 0.00493 (r16/32/64 = rank 거의 무영향, ×1.2). **money finding**: anchor(full 참여, b1 표)에선 방법이 전부 1.000으로 붕괴-동률이나 **5/50 저참여에선 방법이 갈린다** — Flirds/Flirds-1st/loss-heur가 vs (b) 1.000 유지, GTG 0.98·FedSV 0.91, 반면 **uniform-subset/1차-influence 계열(ComFedSV·ShapleyFL·FedIF)은 음수로 붕괴**(−0.04~−0.13 = oracle 반대 순위). near-additive 무대에서 방법을 가르는 축은 rank가 아니라 **부분참여**이고, 그 축에서 Flirds의 per-round HVP가 (b)를 정확 재현 = **Flirds 우위**(CNN C1 §4.3의 "부분참여가 방법 구별을 만듦"과 정합; 단 여기선 클라당 참여 ~20회[R200]라 Flirds-1st도 1.0 유지 = §6.2 caveat 10 "참여 횟수" 조건). seed0 스위트엔 **Fed-LOO도 실측(+0.9998)** — §3.1.3 E4와 정합.
>
> **[3-seed 확정]** r16 seeds 1-2 추가(경량 채점 = Flirds·Flirds-1st만): **Flirds +1.000±.000**(0.9999/1.0000/0.9999)·Flirds-1st 0.999±.000 — **저참여 우위가 seed 재현**. 같은 셀 **(b)oracle 자기순위 cross-seed ρ = −0.09 / +0.13 / +0.15(쌍별; 평균 +0.06 ≈ 0)** = N=50 고검정력에서도 IID-clean 실재 신호 부재를 재확인(§5.4 Exp C·§3.1.5 B축과 정합; N=5 이산-Spearman 저검정력 caveat를 해소하는 보강 증거).

**(b3) lr·steps 격자 — lr의 φ-분리 확대는 seed0 한정, cross-seed 실재 신호는 없음 (fidelity 불변)** ((b)φ range = max−min; steps 축은 seed0, st10 열은 3-seed)

seed0 격자 (steps 축 포함):

| lr \ steps | 10 | 20 | 30 |
|---|---|---|---|
| 1e-3 | 0.00119 | 0.00178 | 0.00106 |
| 2e-3 | 0.00241 | 0.00262 | 0.00227 |
| 3e-3 | 0.00330 | 0.00316 | 0.00326 |

**st10 열 3-seed 확장** — (b)φ range per-seed + (b)oracle 자기순위 cross-seed ρ:

| lr (st10) | range s0 | range s1 | range s2 | xseed ρ (0-1 / 0-2 / 1-2) | 평균 ρ |
|---|---|---|---|---|---|
| 1e-3 | 0.00119 | 0.00147 | 0.00321 | +0.00 / −0.90 / −0.20 | **−0.37** |
| 2e-3 | 0.00241 | 0.00113 | 0.00400 | +0.50 / −0.70 / −0.40 | **−0.20** |
| 3e-3 | 0.00330 | 0.00089 | 0.00340 | +0.10 / −0.70 / −0.10 | **−0.23** |

> **Flirds vs (b) = 전 lr×steps×seed 칸 +1.000**(3-seed 확정 — per-round Δ가 커져도 1-HVP가 (b)를 정확 재현 = **Taylor tradeoff 없음**). **① cross-seed 실재성**: lr를 키워도 (b) 자기순위 xseed ρ ≈ 0(평균 −0.37~−0.20, lr에 따른 개선 추세 없음) → **"lr로 커진 φ는 cross-seed 실재 신호가 아니다" 예측 적중·확정**(N=5 이산-Spearman 저검정력 caveat는 std50k5 N=50의 ρ≈+0.06이 보강 — (b2)). **② 크기 효과 정밀화(seed0 서술 정정)**: 종전 "lr→φ range ~3×"는 **seed0 한정 관측** — seed1은 lr↑에 range 오히려 감소(0.00147→0.00089), seed2는 lr 무관하게 큼(0.0032~0.0040) = **클라 간 분리(range·std)는 seed 분산이 lr 효과를 상회**. 전 seed에서 재현되는 lr 효과는 공통 학습-shift(mean\|φ\| 1.5~1.6e-2 → 1.9~2.1e-2, ~1.3×)뿐이며 이는 분리 지표가 아님. → **A축 결론 강화: lr를 포함한 어느 lever도 클라 간 실재 신호를 못 만든다.** intervention(flirds_w Δval-loss·SNR = 원가설 2차)은 arm metrics로 별도 분석 대기(A축 유일 잔여; 루트 REMAINING §1.4).

**(b4) noise probe — φ 신호가 val 측정노이즈 수준(하한)** (anchor5, chunk 20×bootstrap 2000; r16 = 3-seed)

| | r16 s0 | r16 s1 | r16 s2 | r64 s0 |
|---|---|---|---|---|
| φ spread (클라 간) | 0.00098 | 0.00152 | 0.00311 | 0.00106 |
| φ bootstrap SE (max) | 0.00086 | 0.00111 | 0.00133 | 0.00095 |
| spread / max-SE | 1.15 | 1.37 | 2.34 | 1.11 |
| boot 자기순위 ρ | 0.93 | 0.96 | 0.99 | 0.92 |
| half-split ρ | 0.90 | 0.90 | 1.000 | 1.000 |
| est vs (b) Spearman | 1.000 | 1.000 | 1.000 | 1.000 |

> φ 클라 간 spread가 val bootstrap SE의 **1.1~2.3×**(seed마다 spread 자체가 ~3× 변동 = (b3)의 분리 seed-분산과 정합) = **신호가 측정노이즈 수준~약간 위**. within-seed 순위는 재표본·val 반분할서 ρ 0.9+ 유지(3-seed 일관; 극단 쌍은 SE 위로 분리). **§3.1.2 (b2)·진단 §1.4 cross-seed 불안정과 층위가 다르다**: 같은 seed 안 val 노이즈엔 순위 강건(0.9+)이나, seed를 넘으면 데이터 파티션이 바뀌어 순위 붕괴(≈0, (b3)) — 후자가 B축 신호 부재의 진단. 이 층위 구분이 3-seed로 확정됨.

**(c) 출처·baseline-set 노트**
- **출처**: `runs/probe_signal/rundirs/1B_{anchor5_r*,std50k5_r*,anchor5_lr*_st*}_seed*/` + `runs/probe_signal/noise_probe/noise_1B_r{16,64}_seed*/`({metrics.json,phi.parquet}; seeds 1-2 커밋 = 454db39); r16 anchor·lr1e-3·st10 기준점 = `track_d/rundirs/1B_anchor5_seed0`(seeds 1-2는 probe 신규 rundir). cross-seed ρ = phi.parquet의 (b)oracle φ를 클라×seed 피벗한 쌍별 Spearman(§5.4와 동일 규약). φ range·cross-seed는 phi.parquet 직접 재계산(**정식 재생성 = `runs/probe_signal/make_figures.py`**; 크기 지표=클라 간 분리 — 본문 표=range(max−min), figure=std, 둘 다 lr서 ≈3×로 정합. mean|φ|는 공통 학습-shift가 지배해 ~1.4×에 그치므로 크기 지표로 부적합).
- 방법 스위트 = §3.1.1과 동형(일부 probe rundir[std50k5 전부·lr격자·anchor r64]엔 Fed-LOO도 실측 존재 — 표 미수록). truth=(b) only(ORACLE_A=0).

## 4.3 A축 lever probe — CNN (width×참여; `probe_signal/cnn_c1`·`cnn_c2`)

### 4.3.1 C1 fidelity probe (N=10 R=10, width×참여 sweep)

**(a) 세팅**: §3.1.2 C1과 동형(cifar10 · FedSVCNN · R=10 · epochs=5 · lr=0.01 · batch=64 · SGD mom=0 · val=2000/test=8000), 단 **폭 w∈{0.5,1,2,4}** × **라운드당 참여 k∈{0.2,0.5,1.0}**(=2/5/10명) × 시나리오 {`iid`, `label-flip`(오염 대조군)} × 3 seed. (b) oracle = full 참여면 2¹⁰ 열거·partial이면 exact per-round 분해. 66 신규 + (w=1,k=1.0) 6셀은 §3.1.2 track_c 재사용 = **72셀**. (a) oracle 제외.

**(b1) 신호 실재성: (b)oracle 자기순위 cross-seed ρ ↑** (재현성=오염·용량과 무관한 "진짜 신호"; 1=완전재현·0=추첨노이즈; 3-seed 쌍별 Spearman 평균)

| 시나리오 · 참여 k | w=0.5 | w=1 | w=2 | w=4 |
|---|---|---|---|---|
| iid · k=0.2 | 0.022 | 0.515 | -0.188 | 0.083 |
| iid · k=0.5 | -0.022 | -0.285 | -0.131 | -0.228 |
| **iid · k=1.0 (full)** | 0.034 | -0.042 | 0.038 | 0.123 |
| label-flip · k=0.2 | -0.147 | 0.160 | -0.051 | 0.038 |
| label-flip · k=0.5 | 0.038 | 0.095 | 0.111 | -0.042 |
| **label-flip · k=1.0 (full)** | **0.976** | **0.968** | **0.859** | **0.923** |

> **headline**: **폭을 0.5→4로 8× 키워도 iid의 ρ는 0 근처에서 안 움직인다**(full 참여 행 0.034/−0.042/0.038/0.123). 반면 **오염(label-flip)이면 full 참여에서 ρ≈0.9로 살아나고 이 역시 폭 무관**(0.976~0.859). → **신호 실재성은 A축(용량)이 아니라 B축(오염)이 만든다.** 진단문서 §1.4의 CNN 대조(iid −0.042 vs label_flip 0.968)를 재현·확장(그 값 = 여기 w=1,k=1.0 칸 = `track_c` RESULTS.txt와 교차검증 일치).
> **참여의 별도 역할**: label-flip이라도 **partial 참여(k<1.0)면 ρ가 0으로 붕괴**(0.16/0.10/−0.04). N=10·R=10에선 클라당 참여가 k=0.2→~2회뿐이라 φ per-round 분해가 "누가 그 라운드에 뽑혔나" 추첨에 지배됨(진단 §1.3d). 즉 참여↓는 신호를 *만드는* 게 아니라 φ 랭킹을 *흐린다*.

**(b2) φ 절대 크기: (b)oracle φ range = max−min, full 참여 k=1.0** (클수록 클라 간 분리 큼)

| 시나리오 | w=0.5 | w=1 | w=2 | w=4 |
|---|---|---|---|---|
| iid | 0.047±.008 | 0.029±.007 | 0.058±.010 | 0.064±.014 |
| label-flip | 0.118±.009 | 0.111±.013 | 0.135±.010 | 0.130±.024 |

> φ 크기도 폭에 평평(iid ~0.05, label-flip ~0.12)하고 **label-flip이 iid의 2–4×**. 용량이 아니라 오염이 φ 분리를 만든다. (partial 참여의 φ range는 0.3~0.6로 크지만 그건 참여 추첨 분산=노이즈이며, 위 ρ 붕괴가 이를 확증.)

**(b3) method fidelity vs (b) Spearman ↑** (label-flip, 폭×seed pool=12, 참여 k별)

| method | k=0.2 (2/10) | k=0.5 (5/10) | k=1.0 (full) |
|---|---|---|---|
| **Flirds** | 0.891±.147 | 0.979±.018 | **0.993±.008** |
| Flirds-1st | 0.305±.434 | 0.765±.085 | 0.940±.039 |
| loss-heur | 0.862±.065 | 0.857±.083 | 0.943±.031 |
| GTG | 0.800±.129 | 0.718±.239 | 0.497±.344 |
| FedSV | 0.641±.216 | 0.571±.224 | 0.302±.252 |
| ComFedSV | 0.040±.294 | -0.030±.303 | 0.238±.238 |
| ShapleyFL | 0.222±.202 | 0.029±.342 | 0.182±.293 |
| FedIF | 0.199±.279 | 0.469±.199 | 0.829±.140 |

> 읽기: **Flirds(2차)는 참여·폭·시나리오 전반 0.9+**(전 72셀 pool: Flirds **0.953±.080**). **Flirds-1st는 참여↓에서 붕괴**(k=0.2→0.305, full→0.940) — **2차 Hessian 항이 partial 참여에서 값을 한다**(Flirds 2차는 k=0.2에서도 0.904 유지). GTG/FedSV는 오히려 full 참여에서 하락(큰 2¹⁰ 게임을 근사 못 함). **폭 효과는 어느 method에도 없음**(용량은 fidelity를 안 바꿈).
> **caveat**: 이 partial-참여 Flirds-1st 붕괴는 CNN R=10(클라당 참여 ~2회)의 짧은 지평 탓이 크다 — LLM std20(R=200, 클라당 ~20회 참여)에선 2/20 partial이어도 Flirds-1st 0.999(진단 §1.3). "참여 분수"가 아니라 **클라당 참여 횟수**가 1차항 정확도의 조건.

### 4.3.2 C2 intervention probe (N=100 R=120, width×참여 sweep)

**(a) 세팅**: §3.2.2 C2와 동형(cifar10 iid · N=100 · R=120 · epochs=5 · lr=0.01 · batch=64 · SGD mom=0 · target acc=0.6), 단 **폭 w∈{0.5,1,2,4}@참여 f=0.1** + **참여 f∈{0.05,0.1}@w=1** × 위협 {`clean`, `label-flip`} × 3 seed. arms 6종(vanilla·flirds_mult·flirds_select·shapleyfl·fedif·sfedavg). 24 신규 + (w=1,f=0.1) 6셀 track_c 재사용 = **30셀**. **(w=1,f=0.2)는 구조적 제외**(shapleyfl arm이 라운드별 exact 2²⁰ Shapley → 계산 불가) → 참여 sweep은 {0.05,0.1}까지.

**(b) 결과 — 개입 arm Δacc(vs vanilla, paired) ↑ · 탐지 AUROC ↑** (clean=do-no-harm parity 기대; label-flip=개입 효과 기대)

| 위협 · w · f | vanilla acc ↑ | flirds_mult Δ | shapleyfl Δ | fedif Δ | sfedavg Δ | fl_mult AUROC ↑ |
|---|---|---|---|---|---|---|
| clean · w0.5 · f0.1 | 0.622±.007 | 0.000±.004 | 0.000±.005 | 0.001±.003 | 0.002±.005 | – |
| clean · w1 · f0.1 | 0.648±.001 | -0.002±.002 | -0.001±.002 | -0.001±.001 | -0.002±.006 | – |
| clean · w2 · f0.1 | 0.658±.005 | -0.002±.001 | -0.002±.002 | -0.001±.002 | -0.001±.002 | – |
| clean · w4 · f0.1 | 0.673±.002 | -0.002±.002 | -0.002±.003 | -0.001±.001 | -0.002±.004 | – |
| clean · w1 · f0.05 | 0.640±.004 | -0.001±.004 | -0.002±.000 | -0.000±.001 | 0.005±.002 | – |
| label-flip · w0.5 · f0.1 | 0.491±.023 | 0.087±.018 | 0.077±.011 | 0.071±.016 | -0.019±.028 | 0.986 |
| label-flip · w1 · f0.1 | 0.510±.023 | 0.092±.018 | 0.083±.013 | 0.075±.013 | -0.008±.032 | 0.981 |
| label-flip · w2 · f0.1 | 0.520±.024 | 0.092±.020 | 0.084±.016 | 0.069±.014 | 0.013±.032 | 0.966 |
| label-flip · w4 · f0.1 | 0.536±.019 | 0.089±.010 | 0.085±.012 | 0.068±.008 | 0.010±.029 | 0.955 |
| label-flip · w1 · f0.05 | 0.528±.038 | 0.063±.034 | 0.069±.025 | 0.044±.018 | -0.015±.042 | 0.934 |

> 읽기: **clean = 전 폭·참여에서 parity**(|Δacc|<0.006, seed 노이즈 이내). 폭↑은 raw acc(0.622→0.673)를 올리지만 **개입 이득은 안 만든다**(신호 없음). **label-flip = 개입이 크게 이득**(flirds_mult Δ≈0.09, shapleyfl 0.08, fedif 0.07), 그리고 **이득도 폭 무관**(0.087~0.092). vanilla는 모든 label-flip 칸에서 target 0.6 미달인데 flirds_mult/shapleyfl는 w≥1에서 도달(flirds_mult 도달 라운드 82→69→55 = 폭↑일수록 빠름). 탐지 AUROC 0.93~0.99(참여↑서 소폭↑). **예외 = sfedavg(softmax 선택)**: AUROC 높아도(0.99~1.0) Δacc≈0/음수 = 탐지≠좋은 개입.
> → **개입 효과 크기는 A축(폭·참여)이 아니라 B축(오염)이 지배**. 폭·참여는 동작점(raw acc·속도)만 옮기고 parity↔이득 이분법과 갭(~0.09)은 안 바꾼다. §3.2.2(track_c 오염 무대)와 정합.

**(c) 출처·baseline-set 노트 (4.3 공통)**
- **출처**: `runs/probe_signal/cnn_c1/pc1_*/metrics.json` + `runs/probe_signal/cnn_c2/pc2_*/metrics.json` + 재사용 = `runs/track_c/c1/cifar10_{iid,label-flip}_seed*` · `runs/track_c/c2/cifar10_iid_{clean,label-flip}_strmain_seed*`(§3.2.2와 값 일치 교차검증). **정식 재생성 = `runs/probe_signal/make_figures.py`**(rundir만으로 재현).
- C1 probe 포함 = §3.1.2와 동일 8종 + (b); truth = (b) only((a) retrain은 probe 스코프 밖 — ORACLE_A=0). C2 probe 포함 6 arm(§3.2.2의 flirds_repl/flirds_add는 dir1 size-skew 전용 → iid probe엔 없음). detector 별도 없음(개입 arm의 φ-as-detector AUROC만; truth = corrupt 마스크).

## 4.4 Removal-curve — 게임-무관 인과 검증 (`removal_dose` A2·A3 + poison 한계)

> 서버 5-GPU 풀스윕 79셀 완주(2026-07-17, 실패 0) + CNN A3 18셀. removal = **기여도 순서대로 클라를 실제 제거·재학습**해 순위의 인과적 타당성을 game-정의와 무관한 공통 자(ruler)로 확인(리뷰 C-1/C-4 대응). removal_retrain_s ≈ 317–323s/재학습(silo5; CNN은 mnist ~18s·cifar10 ~40–98s).

### 4.4.1 Exp A2 — LLM silo5 removal-curve

**(a) 세팅**: silo5 무대(§3.3.1) × {noisy, frrand, frzero} × 3-seed. 각 방법의 φ 순위로 worst-first/best-first 제거 → init부터 재학습 → val-loss 변화.

**(b) 결과** (3-seed 평균):

| threat | Flirds ρ(vs b) | worst-first Δval_loss | best-first Δ | 판정 |
|---|---|---|---|---|
| noisy | +1.00 | **+0.0076** | −0.0084 | worst-first 제거가 val_loss 내림 = 순위 인과적 ✅ |
| frrand | +1.00 | +0.0071 | −0.0015 | ✅ |
| frzero | +1.00 | +0.0067 | −0.0016 | ✅ |

- **removal-curve 는 순위(ranking)에만 의존** — 재학습은 제거 순서만 보므로. 곡선 엄밀-일치(9/9 셀, rundir 재검증): **Flirds=Flirds-1st=(b)=loss-heur=Fed-LOO 5종**. GTG **8/9**·ShapleyFL **7/9**·FedSV **6/9**·ComFedSV **3/9**는 일부 seed에서 clean 클라 간 중간 순서만 이탈(곡선 차 ≤0.002; 해당 seed ρ 0.90). **유일 질적 낙오 = FedIF**(frzero worst-first Δ+0.0038, 얕음 = 순위 오류 ρ+0.90 이 곡선에 드러남). → Flirds 가 (b)·coalition 과 동일한 인과-removal 품질을 **5× 싸게**.

### 4.4.2 Exp A2 poison — 정직한 한계 (C-3/C-8)

clean-preserving 백도어(baseline ASR=1.0). **각 방법의 worst 클라 1개 제거 후 ASR** (3-seed):

| 결과 | 방법 |
|---|---|
| **ASR→0.0 (무력화)** ✅ | (b)oracle · GTG · FedSV · ShapleyFL · FedIF · ComFedSV · loss-heur · Fed-LOO |
| **ASR 1.0 유지 (실패)** ❌ | **Flirds**(seed별 1.0/1.0/0.0) · **Flirds1st**(1.0) |

→ clean-val-loss 를 안 낮추는 백도어를, clean-val-loss 기반 **Taylor-gradient(Flirds/Flirds1st)가 "좋은 기여자"로 오판** → 제거해도 공격 안 막힘. **정확 (b)·coalition 계열은 잡음** → 실패는 게임이 아니라 **1차 Taylor 추정 쪽**(C-8/R4 실측 확증). fidelity ρ가 높아도 **top-1 제거는 틀림** = removal 실험이 Spearman 이 가리는 걸 드러냄(루트 CLAUDE.md 2차-③ 각주와 정합).

### 4.4.3 Exp A3 — CNN removal-curve + **accuracy 축** (mnist+cifar10 18셀)

**(a) 세팅**: track_c1 `C1_REMOVAL` 게이트(코드 커밋 `1693531`; A2 패턴 이식, 기본 off = 기존 산출 비트동일 검증) — {mnist, cifar10} × {label_flip, feature_noise, iid} × 3-seed, 전 방법, worst/best-first **실제 재학습**(frozenset 캐시 방법·방향 공유), **val_loss + test acc(8k disjoint) 동시 기록** — 기존 Caveat("accuracy 없음")를 CNN 스테이지에서 해소. (수치 규약: distinct = 방법들을 3-seed worst-first 곡선 시그니처[4자리]로 그룹핑한 수; acc 분리 = 전 곡선점 k=0..9 의 worst−best gap 평균.)

**(b) 결과**:

| dataset · scenario | Flirds ρ(vs b) | 최저 방법 ρ | distinct worst-first 곡선 | acc 분리(worst−best, Flirds) |
|---|---|---|---|---|
| mnist · label_flip | **+1.00** | ComFedSV +0.95 | 9/11 | **+0.0035** ✅ 인과적 |
| mnist · feature_noise | +0.77 | **FedSV +0.13** | **11/11** | +0.0002 (≈중립) |
| mnist · iid (통제군) | +0.84 | **FedSV +0.01** | **11/11** | +0.0006 (≈0) ✅ 기대 정합 |
| cifar10 · label_flip | **+1.00** | ShapleyFL +0.26 | 10/11 | **+0.0445** ✅ 인과적 (mnist 의 ~13×) |
| cifar10 · feature_noise | **+1.00** | **ShapleyFL +0.07** | 10/11 | **+0.0385** (mnist 와 달리 강한 신호) |
| cifar10 · iid (통제군) | +0.97 | ShapleyFL +0.14 | **11/11** | **−0.0033** (소폭 음수 — 아래 불릿) |

- **LLM silo5(§4.4.1)와 정반대 무대**: silo5 는 순위합의(ρ≈1)로 곡선이 축퇴(강한 방법 전부 bit-동일)했지만, CNN 은 fidelity 스프레드가 커서(mnist FedSV +0.01~0.96·cifar10 ShapleyFL +0.07~0.26 최저) **removal 척도가 방법을 실제로 변별** — "removal-curve 는 순위에만 의존"(§4.4.1)의 대우를 실측으로 확인. C-4(게임-무관 공통 자) 방어가 CNN 스테이지로 일반화.
- **acc 축 위계(mnist)**: label_flip(진짜 나쁜 데이터)만 worst/best 분리 뚜렷(+0.0035; (b)oracle +0.0034 와 동급), feature_noise(mild σ) ≈ 중립, iid = 순수 데이터량 손실 — 완만한 ladder(오염 ≤20%)+클라당 6k 샘플이라 절대 acc 는 어느 방향이든 서서히 하락하고 **신호는 분리(gap)에 있음**.
- **cifar10 9셀**: label_flip·feature_noise 서 Flirds ρ **+1.00** + acc 분리 **+0.039~0.045**(mnist 의 ~13×; (b)oracle 동급 +0.038~0.045, **순위 낮은 ShapleyFL 은 분리 ≈0** = 순위→분리 인과 재확인). **feature_noise 가 mnist(≈중립)와 달리 강한 신호** = 같은 σ 라도 어려운 과제에선 오염 실효가 커짐. **iid 통제군 acc 분리 −0.0033**(mnist ≈0 과 달리 소폭 음수; **(b)oracle 도 −0.0027** = Flirds 실패가 아니라 무대 특성 — 어려운 과제에서 데이터량 손실이 지배해 worst-first 도 acc 를 깎음).
- 흥미: **FedIF 역전** — LLM frzero 서 유일 낙오(§4.4.1)였으나 CNN mnist 에선 분리 최고(+0.0042; cifar10 label_flip 최고는 Fed-LOO +0.055). 척도의 스테이지-의존성 자체가 관찰 결과.
- 잔여: label/quantity_skew·pixel-backdoor ASR 은 옵션(README §Exp A3, Yonghee 결정 대기).

**(c) 출처·baseline-set 노트 (4.4 공통)**: `runs/removal_dose/rundirs/`(A2·poison) · `runs/removal_dose/rundirs_cnn/`(A3). LLM removal 지표는 val_loss 뿐(생성 LM) → accuracy 축은 A3 CNN 에서 실측. distinct 곡선 카운트는 문서 편의상 11-method 표기 유지(현 비교군 기준으로도 카운트 재현 확인). C-# = 2026-07 내부 리뷰 항목 코드(원 문서 `review-claude`는 정리 커밋으로 삭제 — git 이력).

## 4.5 Dose-response (`removal_dose` B; 3-seed 확정) — φ 탐지 문턱 vs 오염강도

**(a) 세팅**: silo5 무대 × 위협 강도 ladder {noisy nr, free-rider 배율 dm, poison frac pf} × 3-seed (63셀).

**(b) 결과 — Flirds 탐지 AUROC vs 오염량**:
- **noisy**: 0.75±.00(rate≤0.1) → **1.00±.00(≥0.25)** — 문턱 nr0.25, 3-seed 무분산. (nr0 대조군 = 0.83±0.12 — N=5 coarse-AUROC 의 무신호 기준선이 0.5 가 아님을 보여주는 계측 참조점)
- **free-rider**: 전 배율(dm0.25–4.0) **1.00±.00** — 크기 무관, 3-seed 확정
- **poison**: pf≤0.2 **0.00**(완전 회피) → **pf0.3–0.7 = 0.33–0.42(±0.42–0.47) (seed-혼재 전이대)** → pf0.8 0.75±0.20 → pf≥0.9 **1.00**. ⚠ seed0 단독으로 서술했던 "pf≤0.7=0.0 절벽"을 **정정** — 3-seed 에선 절벽이 아니라 **넓은 seed-불안정 전이대**(§3.3.1 poison 0.500±.354 의 seed-혼재와 같은 구조)

**(c) 출처**: `runs/removal_dose/rundirs/1B_silo5_*_dose_*/metrics.json`.

## 4.6 AdamW 브리지 — external validity (`removal_dose` A1·D; 3-seed 확정)

**(a) 세팅**: anchor5 무대에서 optimizer만 AdamW(상수 lr — "브리지 설정"; 논문 5e-5 cosine 갭은 deviation caveat)로 교체 + (a)/(b) 듀얼 오라클. A1 = SGD anchor5 removal 대조(3-seed).

**(b) 결과**:

| 셀 | Flirds ρ(vs b) | (a) ρ(vs b) | 비고 |
|---|---|---|---|
| anchor5 removal ×3 seed (A1) | **+1.00** (전 seed) | +0.90 / +1.00 / +0.90 | FedIF·ComFedSV IID-clean 서 불안정(seed1 음수) |
| AdamW ×3 seed (D) | +0.90 / +0.50 / +0.90 = **+0.77±.19** | −0.10 / −0.90 / −0.60 = **−0.53±.33 (전 seed 음수)** | AdamW 브리지서 (a)↔(b) 두 oracle 게임이 **상반 순위** = 타깃 자체 seed-불안정 심화(§5.4 Exp C 정합) — Flirds 실패 아님 |

> AdamW 3-seed 참고(전 방법 vs (b), `rundirs_trackd/1B_anchor5_adamw_seed*`): loss-heur +0.967±.047·Fed-LOO +0.933±.094·GTG +0.900±.082 > Flirds +0.767±.189·Flirds-1st +0.700 > ShapleyFL +0.267·FedIF +0.233. **in-run 계열 전반이 (b)를 추종하는데 (a) retrain만 음의 상관** — SGD(§3.1.1 (a)vs(b) +0.933)와 대비되는 optimizer-의존 게임 괴리로, "AdamW에선 in-run 궤적 게임과 retrain 게임이 다른 것을 잰다"는 external-validity 한계(C-5)의 정량화. Flirds 저하(+1.00→+0.77)는 그 다음 순서의 관측.

**(c) 출처**: `runs/removal_dose/rundirs_trackd/1B_anchor5_{removal,adamw}_seed{0,1,2}/`.

## 4.7 β sweep·provenance (`rerun_beta03`) — ShapleyFL EMA β=0.3 통일 재실행

**(a) 세팅**: ShapleyFL surrogate FSV의 cross-round EMA를 논문값 β=0.3으로 통일하는 전 트랙 재실행 캠페인(2026-07-03~; 캠페인 상태 = §6.2 caveat 9). ablation 질문 = "β0.5→0.3이 결과를 바꾸나".

**(b) 결과**:
- **3B 전후 대조** (`figures/beta_contrast_3b.csv`, 6셀×8방법): β0.5(git `b1b95d0~1`) ↔ β0.3 재실행의 ShapleyFL 순위변화 ρ 0.90–1.00 = **타 방법과 동일 분포(재실행 노이즈 플로어 수준)** — β 자체의 효과는 식별 불가 수준.
- **provenance 실측** (`figures/beta_provenance.csv`, 294 rundir meta.json git_sha ancestry 스캔, 07-16 시점): track_c CNN 120셀 = **β0.5-era 코드 산출물**(라벨만 β0.3 — 'CNN은 β-불변' 주장의 canon 미확보 실측), track_d = 3B 6+7B_anchor5 3만 β0.3-era.
- **1B_silo5 오염 4셀 = β0.3 재실행 완주·착지**(ce0b454, 07-20) — §3.3.1이 그 정본. 부수 관찰: 재실행에서 Flirds poison AUROC 0.917→0.500(§3.3.1 각주) = β 효과라기보다 **run-인스턴스 재추첨 효과**(adapter init unseeded — §6.2 caveat 13 H1)와 구분 불가.

**(c) 출처**: `runs/rerun_beta03/figures/{beta_contrast_3b.csv,beta_provenance.csv}` · 재개법 = `runs/rerun_beta03/RESUME_AFTER_MIGRATION.md`(주의: 구 overview 파일명 참조 stale) · 잔여 셀 = 루트 `REMAINING.md` §1.2–1.3.

## 4.8 Track H 확증 런 3종 — P5 신뢰-게이트 / Scale 완전참여 / Dyn 재추첨 (`track_h`)

> §3.2.6 경쟁 본판에서 파생된 **사전등록(preregistered) 정책 확증 런들** — 예측표를 먼저 등록하고 MISS를 그대로 보고. GPU-h 수치는 서버 로그 기준(track_h CNN rundir엔 timing.json 없음).

### 4.8.1 P5 신뢰-기반 sign 정책(hard/soft) — `rundirs_cnn` +108런

**(a) 세팅**: **스펙·공정성·예측(HP-1~6) 정본 = `runs/track_h/p5/RUN_P5.md`** (z=1.645 보편상수·자기 스트림만·셀별 튜닝 금지; 예측은 07-20 오프라인 리플레이 유래 사전 등록). P1(strict sign)이 경계선 분산을 확신-유해와 똑같이 과금한다는 진단에서 출발 — **P5-hard** `cgate`(UCB: cum+1.645·σ̂√n≤0일 때만 배제) / **P5-soft** `pweight`(w∝n·Φ(t) 확률 가중)를 8 점수원 공통 적용. T1 online 96런(8소스×4위협×3seed×2arm) + T2 retrain 12런(관찰자 obsp5 → 최종 통계로 `t2_csign_*`/`t2_pw_*`·dedupe·size-matched random 통제). **실행 각주**: 최초 제출분에서 `flirds_cgate/pweight`가 레거시 분기에 삼켜져 vanilla로 새던 dispatch 실버그 발견(`fa5fc6e` 수정 — 스모크 AUROC=1.000 가드 신설), **t1 flirds 12런만 수정판 재실행**(동명 멱등 덮어쓰기·meta `git_sha` 전수 검증; 타 소스·T2는 diff상 비트동일로 무영향). GPU-h 실측 ≈ **48**(t1 96×~12분 + flirds 재실행 12 + t2 12×~2.2h + pre-flight; 스펙 추정 36–45 소폭 초과 = t2 실측 2.2h > 추정 1–1.5h).

**(b) 결과 — P5-hard(cgate) · online / retrain** (절대 test acc, 3-seed mean; 형식 = §3.2.6 P1/P2 표와 동일)

| arm              | cln·on    | FR·on     | GN·on | LF·on     | **오염평균·on** | cln·re | FR·re     | GN·re     | LF·re | **오염평균·re** |
| ---------------- | --------- | --------- | ----- | --------- | ----------- | ------ | --------- | --------- | ----- | ----------- |
| vanilla (바닥)     | .6389     | .5879     | .2436 | .5247     | .4521       | 〃      | 〃         | 〃         | 〃     | .4521       |
| oracle_excl (천장) | –         | .6203     | .6203 | .6236     | .6214       | –      | 〃         | 〃         | 〃     | .6214       |
| **flirds**       | .6375     | .6070     | .5169 | **.5928** | **.5722**   | .6333  | .6197     | **.6215** | .6210 | **.6207**   |
| flirds1st        | .6390     | **.6195** | .2422 | .5599     | .4739       | .6389  | .6195     | .2436     | .6009 | .4880       |
| lossheur         | .6367     | .6138     | .3959 | .5835     | .5311       | .6370  | **.6210** | .3529     | .6206 | .5315       |
| fedif            | **.6392** | **.6195** | .2422 | .5556     | .4724       | .6389  | .6195     | .2436     | .6032 | .4888       |
| gtg              | .6351     | .5450     | .5169 | .5559     | .5393       | .6363  | .5486     | .6195     | .5850 | .5844       |
| fedsv            | .6300     | .5310     | .5079 | .5541     | .5310       | .6384  | .5588     | .5655     | .5657 | .5634       |
| comfedsv         | .6299     | .5400     | .4952 | .5529     | .5294       | .6388  | .5554     | .5265     | .5582 | .5467       |
| shapleyfl        | .6355     | .5164     | .5301 | .5600     | .5355       | .6340  | .5417     | .6195     | .5865 | .5825       |

**P5-soft(pweight) · online / retrain**

| arm              | cln·on    | FR·on     | GN·on     | LF·on     | **오염평균·on** | cln·re    | FR·re     | GN·re     | LF·re     | **오염평균·re** |
| ---------------- | --------- | --------- | --------- | --------- | ----------- | --------- | --------- | --------- | --------- | ----------- |
| vanilla (바닥)     | .6389     | .5879     | .2436     | .5247     | .4521       | 〃         | 〃         | 〃         | 〃         | .4521       |
| oracle_excl (천장) | –         | .6203     | .6203     | .6236     | .6214       | –         | 〃         | 〃         | 〃         | .6214       |
| **flirds**       | .6370     | **.6132** | .5416     | **.5965** | **.5838**   | .6350     | .6108     | .5989     | **.6188** | **.6095**   |
| flirds1st        | .6387     | .6100     | .2446     | .5680     | .4742       | .6387     | .6201     | .2478     | .6106     | .4928       |
| lossheur         | **.6403** | .6128     | .5373     | .5771     | .5757       | .6371     | .6159     | .4618     | .6103     | .5627       |
| fedif            | .6392     | .6120     | .2423     | .5693     | .4745       | **.6398** | **.6204** | .2393     | .6063     | .4887       |
| gtg              | .6330     | .5315     | .5456     | .5733     | .5501       | .6302     | .5178     | **.6134** | .5860     | .5724       |
| fedsv            | .6330     | .5387     | .5439     | .5722     | .5516       | .6302     | .5210     | .6085     | .5865     | .5720       |
| comfedsv         | .6334     | .5397     | .5420     | .5745     | .5521       | .6310     | .5252     | .6072     | .5884     | .5736       |
| shapleyfl        | .6324     | .5285     | **.5458** | .5770     | .5504       | .6275     | .4992     | .6038     | .5907     | .5646       |

**읽기(P1 대비):**
- **clean 오발화가 정책으로 대부분 회수**: P1 clean .596~.638 → P5h .630~.639 / P5s .632~.640. renorm 4종도 .630~.636(P1 .596~.605)으로 회복 — HP-1의 설계 의도(경계선 분산 보호)가 전 소스에서 작동. flirds .6315→.6375(h)/.6370(s), vanilla(.6389)와 −.0014 차이까지 근접.
- **renorm free-rider "붕괴"의 원인 분리**: FR 검출은 여전히 0(cgate/pweight AUROC .00~.02 — 확신-양수 편향은 z 무관, HP-4 검출 예측 적중)인데 acc는 P1 .39~.40 → P5 .50~.55로 완화. 즉 P1 파국의 주범은 *미검출*이 아니라 *clean 오배제의 온라인 복리*였음이 정책 대조로 실증됨(그래도 vanilla .5879 아래 = 여전히 열세).
- **flirds GN의 hard 트레이드오프**: online cgate .5169 < P1 .5668(UCB가 증거 쌓일 때까지 오염 유입 허용 — 조기배제 이득 상실, HP-2의 MISS 부분) ↔ **retrain csign .6215 = 사전등록 참조치(≈.61) 적중·천장 동급**. 그 결과 **flirds P5h retrain 오염-평균 .6207 ≈ oracle_excl .6214** — 전 위협 .6197~.6215 + clean .6333으로, Track H 전 정책·전 시점 통틀어 estimator 최고 세팅(P1 retrain .6107 대비 +.0100).
- **hard vs soft(사전 미등록 — 관찰만)**: online 오염-평균은 soft가 **8/8 소스 우세**(약신호를 Φ(t)로 부분 반영; lossheur GN .5373 vs hard .3959가 대표 — HP-3 적중), retrain은 혼재(flirds·gtg·shapleyfl hard↑ / lossheur·fedsv·comfedsv soft↑). 오염-평균 1위는 online 양 정책 모두 flirds(.5722/.5838 — 종합 예측 적중).

**예측표 HP-1~6 대조**(MISS 그대로 — RUN_P5.md §4·§7):

| # | 판정 | 근거 |
|---|---|---|
| HP-1 (flirds cgate clean 상승) | **적중** | .6375 > P1 .6315(vanilla −.0014); UCB 배제 ~10명(clean obsp5 csign kept=90; P1 31명 대비) |
| HP-2 (flirds cgate corrupt 유지) | 부분 | LF ↑(.5928>.5712)·FR ≈(.6070 vs .6148)·GN T2 .6215(참조 ≈.61 적중)이나 **GN online .5169 < P1 .5668 = MISS**(UCB 증거-축적 지연 비용) |
| HP-3 (lossheur cgate GN 하락·pweight 중간 회복) | **적중** | cgate .3959(P1 .5981)·pweight .5373(중간) — marginal 신호(t≈−1.27)의 정책별 반응 예측 그대로 |
| HP-4 (renorm FR 붕괴 유지 + clean 오배제 감소·잔존) | 부분 | 미검출 유지 적중(AUROC .00~.02 전 정책)·clean 회복+잔존 적중(.630~.636, vanilla −.003~−.009); **acc "붕괴 유지"는 MISS**(.39→.50~.55 완화 — 붕괴 주범이 오배제였음) |
| HP-5 (flirds1st·fedif GN 실명 ≈ vanilla) | **적중** | 전 정책 .2393~.2478 ≈ vanilla .2436. 각주: fedif는 GN AUROC 1.0으로 *순위*는 가르나 cum이 양수라 UCB 임계 미달 = 배제 0 — 실명 메커니즘이 리플레이 예상(t=+3.1 확신-오판)과 소스별로 다름(flirds1st는 AUROC .49 = 순위조차 실명) |
| HP-6 (pweight clean ≈ vanilla ±band; soft가 lossheur GN 유리) | 대체로 적중 | clean 8소스 중 7개 \|Δ\|≤.006(shapleyfl .0065로 경계 초과); soft>hard lossheur GN 적중 |
| 종합 (오염-평균 flirds cgate 1위) | **적중** | cgate .5722 1위(2위 gtg .5393)·pweight도 .5838 1위 — 전 위협에서 vanilla 위로 생존하는 유일 소스(renorm=FR 열세 지속, 1차 estimator=GN 실명) |

**(c) 출처(P5)**: `runs/track_h/rundirs_cnn/*_{<src>p5,obsp5}_*`(108런; flirds 12런 meta=`fa5fc6e`) · `runs/track_h/analysis/`(P5h/P5s policy 편입 재생성) · 코드 = `fl/intervene.py`(SignAccumulator.stats·conf-gate·prob-weight) + `experiments/track_c2.py`(`_TH_POLICIES` += cgate/pweight·`C2_T2_P5`·dispatch 순서 수정) + `tests/test_p5.py`(10) — R4(gsm50k5) 동일-정책 leg는 Tier A 종료 후 실행(루트 REMAINING §1.1-P5).

### 4.8.2 Scale 완전참여 100/100 — `rundirs_cnn_scale` 12런 + 앵커 9런

**(a) 세팅**: **정본 = `runs/track_h/scale/RUN_SCALE.md`**. Flirds 비용 주장(valuation이 cohort 크기 k에 선형)의 무대: R1과 동일하되 **frac 0.1→1.0**(매 라운드 100/100). k=100에선 coalition 계열이 라운드당 O(2^k)~O(k²) eval이라 **baseline 개입 arm이 존재할 수 없음**(exact 2^100 등) → 비교 대상 = **vanilla 학습만**(observer=vanilla 비트동일; T2/audit 제외 = Yonghee 07-21 결정). arm 4종 = observer / P1 gate_v2 / P5h cgate / P5s pweight(전부 flirds 점수원; `C2_OBS_SRCS=flirds`). 게이트 하이퍼 = R1/P5 동일(z=1.645). **[07-21 후속 결정(Yonghee): oracle_excl·random_excl 앵커 9셀 추가** — "남은 전원이 매 라운드 참여"라 완전참여 정신과 무충돌 판단, §8의 앵커-제외 결정 번복. RUN_SCALE.md는 사후수정 금지 조항대로 무수정; 앵커는 오염 3위협×3seed 신규 rundir `*_anch_*`(clean은 oracle_excl≡vanilla라 R1 관례대로 생략)].

**(b) 결과 — 절대 test acc** (3-seed mean±sd; vanilla 바닥 ~ oracle_excl 천장):

| arm | clean | label-flip@0.70 | free-rider | grad-noise | **오염-평균** |
|---|---|---|---|---|---|
| observer(=vanilla, 바닥) | **.6527**±.003 | .5550±.022 | .6077±.0035 | .5497±.005 | .5708 |
| oracle_excl (천장) | – | .6301±.004 | .6339±.0025 | .6339±.0025 | .6326 |
| random_excl (통제) | – | .5216±.0335 | .5953±.009 | .5136±.0115 | .5435 |
| flirds P1(sign) | .6440±.005 | .5862±.010 | .6223±.007 | .6102±.001 | .6062 |
| flirds P5h(cgate) | .6485±.007 | .6008±.007 | .6216±.003 | .6037±.003 | .6087 |
| flirds P5s(pweight) | .6462±.004 | **.6220**±.002 | **.6268**±.001 | **.6107**±.004 | **.6198** |

**읽기:**
- **완전참여는 vanilla 자체를 강하게 만든다**(GN .5497 vs frac0.1 .2436 — 오염 grad가 100클라 평균에 희석) — 그럼에도 세 정책 모두 오염 3셀 전부 vanilla 상회(+1.4~+6.7pt), 오염-평균 **P5s .6198 > P5h .6087 > P1 .6062 > vanilla .5708**. n_obs≈R 증거-풍부 레짐에서 **soft(Φ-가중)가 1위** — 오염 3셀 모두 세 정책 중 최고(HS-4 방향).
- **앵커 대비 회수율**(recovery = (arm−vanilla)/(oracle−vanilla), 3-seed mean; 07-21 앵커 추가로 산출 가능): **P5s +0.78**(lf +0.89 / fr +0.73 / gn +0.72) > P5h +0.59 > P1 +0.56. random_excl은 전 위협 **−0.43~−0.47**(동수 랜덤 제외는 vanilla보다 해로움) — 게이트 이득이 "그냥 40명 뺀 효과"가 아님의 통제 실증. FR 천장(.6339)과 게이트(.622~.627)의 잔갭 ~0.7–1.2pt는 burn-in 10라운드의 FR 유입 비용.
- **clean 서열도 P5 설계 의도대로**: P1 −.0087(parity band .006 밖 = 오발화) < P5h −.0042(band 내 회복) < vanilla. P5s는 −.0065로 band를 0.0005 초과(경계).
- **게이트 행동**(`scale_gate_behavior.csv`): FR corrupt 참여율 cgate .004 ≈ P1 .005(exact-0 즉시 배제 유지 — HS-3의 FR 수렴 적중); GN·LF는 cgate가 P1보다 관대(corrupt 참여 .28/.13 vs .09/.04)한데도 acc 동급± — UCB가 배제를 늦추는 비용이 완전참여(오염 희석)에선 거의 무해. pweight는 전원 참여 + corrupt 상대가중 .00~.18로 Φ(t) 양극화 그대로.

**HS-1~5 대조**(정본 §4; MISS 그대로):

| # | 판정 | 근거 |
|---|---|---|
| HS-1 (오염 3셀 세 정책 모두 vanilla 수 pt 상회; frzero 빠른 회복) | 대체로 적중 | GN +5.4~+6.1pt·LF +3.1~+6.7pt 적중; FR은 +1.4~+1.9pt로 "수 pt"엔 경계(완전참여 vanilla가 이미 .6077로 강함), corrupt 즉시배제(참여 .004)는 적중 |
| HS-2 (P1 clean 하회 위험·P5h parity 회복) | **적중** | P1 −.0087(band 밖) vs P5h −.0042(band 내) |
| HS-3 (오염 셀 P5h→P1 수렴: t 조기 포화·같은 배제 집합) | 부분 | FR은 배제 집합 동일(.004≈.005)·acc 동급 적중; GN·LF는 cgate가 더 관대(참여 .28/.13 vs .09/.04)해 집합 불일치 — 단 acc는 동급±(−.65~+1.5pt) |
| HS-4 (P5s clean parity + 오염 회복) | 대체로 적중 | 오염 3셀 전부 세 정책 중 1위(오염-평균 .6198) 적중; clean은 −.0065로 band 0.0005 초과(경계 MISS) |
| HS-5 (셀당 wall-clock ≈ R1의 ~10배 이내) | **적중** | 실측 56~77분/셀(4-arm) = R1 2-arm 10–16분의 ~4–6배; 전체 12셀 ≈ **12.8 GPU-h**(사전 추정 60–90의 1/5 — 파일럿 게이트로 확정 후 잔여 제출) |

**(c) 출처(Scale)**: `runs/track_h/rundirs_cnn_scale/`(12런 + 앵커 9런 `*_anch_*`; observer `phi_rounds.parquet` = flirds 전 라운드×전 클라) · `runs/track_h/scale/analysis/{scale_acc,scale_gate_behavior}.csv`(앵커 편입 재생성) · 코드 = `experiments/track_c2.py`(`C2_OBS_SRCS`·frac=1.0 coalition 가드, `fa5fc6e`) + `runs/track_h/scale/sbatch_scale_anchors.sh`.

### 4.8.3 Dyn 매 라운드 오염 재추첨 — `rundirs_cnn_dyn` 9런

**(a) 세팅**: **정본 = `runs/track_h/dyn/RUN_DYN.md`** (Yonghee 지시: 동학=매 라운드 재추첨·무대=R1 frac0.1; arm = P1·P5s 2정책 + vanilla/per-round oracle_excl/per-round random_excl). 오염이 **클라 속성이 아니라 라운드 속성**이 되는 극한 — 전 클라 확률 동질화로 클라-수준 신호가 구조적으로 소멸하는 null-무대에서, 정책들이 해를 안 끼치는지(do-no-harm) 시험. 오염 3위협×3seed=9셀(clean 없음), 구현=`C2_DYN=1`(`make_roundwise_mask`; 정적 경로 비트동일, 테스트 6+회귀 32 green), **~4.6 GPU-h**. ⚠ 9셀 전부 8598cea(repro 강화) **이전** 코드로 완주 — 기존 CNN canon과 동일 수치 체계(meta: 8셀 `e8be385`·1셀 `9ce7fa9`, 전 셀 git_dirty=true[=dyn diff 그 자체]).

**(b) 결과 — 절대 test acc** (3-seed mean±sd; 클라-수준 AUROC는 원리상 정의 불가 — 성능만 판정):

| arm | label-flip@0.70 | free-rider | grad-noise | **오염-평균** |
|---|---|---|---|---|
| vanilla | .5676±.005 | .5992±.009 | .2547±.003 | .4738 |
| oracle_excl (per-round 천장) | **.6456**±.005 | **.6456**±.005 | **.6456**±.005 | .6456 |
| random_excl (per-round 통제) | .5706±.014 | .5998±.003 | .2583±.020 | .4762 |
| flirds P1(sign) | .5179±.026 | **.6253**±.004 | .1771±.068 | .4401 |
| flirds P5s(pweight) | .5682±.006 | .5980±.000 | .1902±.027 | .4521 |

**읽기:**
- **무대 self-check**: per-round oracle이 3위협에서 **동일값 .6456** — 같은 (seed,r) 마스크 스케줄을 완전 제외하면 위협이 아예 발현되지 않아 세 셀이 같은 clean 궤적이 됨(설계 정합 증거). 정적 oracle(.62대)보다 높은 건 제외가 회전이라 100클라 데이터를 전부 커버하기 때문. per-round random도 데이터 무손실 회전이라 **정확히 vanilla parity**(+.001~+.004; 정적 random_excl의 −0.45 recovery와 대조).
- **P5s do-no-harm은 2/3 무대에서 성립, gn에서 붕괴**: lf +.0006·fr −.0013(band 내 — Φ 공통인자 약분 설계 그대로) vs **gn −.0645 band 밖**(.1902, seed 분산 ±.027). 메커니즘: 전 클라가 공통 강음수 드리프트(라운드의 40%가 σ=0.1 노이즈)라 t가 일제히 큰 음수 → Φ(t)가 전원 극소값 → **재정규화가 극소값들 간 꼬리 차이를 증폭**(Φ 1e-9 vs 1e-12 = 가중 1000×)해 사실상 소수 클라 랜덤 집중 학습이 됨. "모두가 나빠 보이는" 레짐에서 w∝Φ(t)의 신규 실패 모드 — floor/온도 정규화 등 보정 후보의 근거.
- **P1은 lf·gn에서 예측대로 해악, fr에선 예상 밖 이득**: lf −.0497·gn −.0776(gn은 평균 배제 94/100명 — 대량 배제로 cohort 붕괴) vs **fr +.0261**(zero-delta는 raw exact-0이라 cum을 안 움직여 게이트가 정적-clean식 선별로 작동 + n-가중 평균에서 zero-delta 희석 제거가 이득; 정적 fr P1 .6223과 거의 같은 값).
- **DP-4 적중(핵심 실증)**: P1 배제 집합의 "지금-오염" 적중률 **.405 ≈ 우연 .40**(전 위협 동일) — 클라-누적 통계는 라운드-수준 오염 정체를 원리상 추적 못 함의 직접 측정.

**예측표 DP-1~4 대조**(MISS 그대로 — RUN_DYN.md §4):

| # | 판정 | 근거 |
|---|---|---|
| DP-1 (P5s 전 셀 parity) | 부분 | lf·fr band 내 적중; **gn −.0645 MISS**(Φ 꼬리-증폭 — 위 메커니즘) |
| DP-2 (P1 gn·lf 하회, fr 소폭 하락) | 부분 | gn −.0776(대량배제)·lf −.0497 적중; **fr은 +.0261 상승으로 방향 MISS**(exact-0이 cum 불변 → 게이트가 유효 선별로 잔존) |
| DP-3 (oracle 전 셀 상회·random ≤ vanilla) | 적중 | oracle +.046~+.391; random은 등호로 성립(parity — 회전 제외는 데이터 무손실이라 "감소" 근거문은 부정확했음) |
| DP-4 (P1 적중률 ≈ 우연 40%) | **적중** | .405/.405/.405 (`dyn_dp4.csv`) |

**(c) 출처(Dyn)**: `runs/track_h/rundirs_cnn_dyn/`(9런; P1·P5s `phi_rounds.parquet` 포함) · `runs/track_h/dyn/analysis/{dyn_acc,dyn_dp4}.csv` · 코드 = `fl/intervene.py`(`make_roundwise_mask`) + `experiments/track_c2.py`(`C2_DYN` 배선) + `tests/test_dyn.py`(6).

---

# 5. 기타 분석 모음 — 종합 판정·감사·해석 (결과 표가 아닌 콘텐츠)

## 5.1 종합 판정 — 위계별 Flirds 승·패 분석

> §2 전 실험을 핵심 질문 위계(§3 순서)로 관통해 **"Flirds 관점에서 어디서 잘 나왔고 어디서 안 나왔나"** 만 추린 요약. 수치·출처의 정본은 각 본문 섹션(여기 숫자는 전부 재인용). 2차-② 수렴 축은 overview 스코프 제외(§5.6).

**총평**: Flirds는 **1차(fidelity)에서 측정 가능한 전 무대 oracle-동률 최상위**이고, 비용 우위는 **라운드당 참여(cohort)가 큰 무대에서만** 성립하며, 실질 약점은 두 곳 — **clean-preserving poison(추정기 결함)** 과 **IID-clean 무대(잴 신호 자체 부재 = 무대 결함)**.

**1차 Fidelity — 승**
- LLM 전 스케일·양 스테이지 Spearman **1.000±.000** vs (b)(§3.1.1); vs (a)도 0.933 = (b)↔(a) 일치도와 동률(천장 효과).
- CNN 비-additive 무대 vs (b) **0.919**(probe 72셀 pool 0.953) = **비교군 내 1위**(§3.1.2).
- **부분참여 스트레스에서 유일 생존급**: std50k5(5/50) **+1.000**(3-seed) vs ComFedSV/ShapleyFL/FedIF 음수 붕괴(§4.2); CNN k=0.2에서 Flirds 0.891 vs **Flirds-1st 0.305** = **2차(HVP) 항의 존재 이유**(§4.3).
- 강건성·확장: A축 lever(rank·lr·steps) 전반 1.000 유지 = Taylor tradeoff 없음(§4.2) · N=10 exact 2¹⁰·Fed-LOO 확장도 동률(§3.1.3) · Taylor 물리잔차 2차<1차 ~3×(§5.5).

**1차 Fidelity — 주의·약세**
- **IID-clean의 +1.000은 무정보**: 매칭 대상 (b) 자체가 seed-불안정(1B anchor5 xseed ρ −0.37, §5.4)이고 near-additive 축퇴로 loss-heur·Fed-LOO까지 전부 동률(§3.1.3) — 신호는 B축(비IID·오염·부분참여)이 만든다(§3.1.5: non-IID clean ρ 0.87 vs IID 0.13).
- AdamW 브리지 +0.77(SGD 1.000 대비 저하; 단 (a)↔(b) 자체가 −0.53 = 타깃 게임 괴리 → external-validity 한계, §4.6). CNN vs (a) 0.35는 전 방법 공통(두 게임 괴리, §3.1.2).
- **정직한 긴장점**: near-additive 무대에선 **loss-heur(singleton)가 fidelity 동률 + 더 저렴**(anchor5 657s vs 716s, §3.1.3) — Flirds 고유 가치는 비-additive fidelity·부분참여 생존·대규모 cohort 비용에서 성립.

**2차-① 성능/집계 — 오염 무대에서만 승**
- 승: CNN C2 grad_noise acc 0.499→0.609(dir1 repl 0.621)·label_flip 0.583→0.626(§3.2.2); removal에서 순위→acc 분리 인과 확인(cifar10 +0.045 = (b) 동급, 순위 낮은 방법은 분리 ≈0; §4.4.3); clean은 기대대로 do-no-harm parity(§3.2.1).
- 약세: clean-IID 이득은 원리적 부재(효과 < 표본 SE); grad_noise 일부 칸 shapleyfl(0.645) 우위; **removal poison에서 Flirds top-1 제거 실패(ASR 1.0 유지)** vs coalition·loss-heur 전부 무력화(§4.4.2) — ρ가 높아도 top-1이 틀리는 사례.
- **승(Track G)**: 온라인 부호-게이팅 — **frzero 자동배제 recovery 1.000**(LLM silo5·iid5 3-seed, 오배제 0 = φ=0 공리의 배포형 활용)·clean 게이트 무발화(max|Δ|=0.00056)·CNN grad-noise 회수 0.86–0.94(§3.2.3–4).
- **약세(Track G)**: noisy는 sign-게이트 작동영역 없음(누적 φ 양수, 예측 적중 — 회수는 탐지+selection 몫)·frrand는 부호 코인플립으로 seed-의존·**CNN clean에서 게이트 오발화 → V2w 승격 불가**(§3.2.4).
- **승(Track H 점수원 경쟁)**: 같은 게이트 정책에 점수원 8종을 넣은 경쟁(§3.2.6)에서 **exact-0 계열(Flirds·Flirds-1st·loss-heur·FedIF)만 CNN 오염 무대 생존**(free-rider acc .61~.62 ≈ 천장 .620) vs coalition-renorm 4종(GTG/FedSV/ComFedSV/ShapleyFL)은 **free-rider서 파국**(acc .37~.40 < vanilla .59; FR은 못 잡고 clean만 오배제)·clean도 .60~.605 오발화 — zero-semantics 결함이 성능 감점으로 실측. grad-noise는 **1차 estimator(Flirds-1st/FedIF) 실명**(acc .244~.248 = vanilla vs Flirds .567~.607) = 2차항 존재 이유의 다운스트림 재현.
- **약세·트레이드오프(Track H)**: 개별 칸 최고는 점수원마다 갈림(clean·label-flip은 FedIF·loss-heur도 상위) — Flirds는 전 정책·전 시점 상위권이나 단독 1위 아님; **LLM silo5-noisy는 역전** — renorm 게이트가 (renorm-오차) 0-교차 덕에 발화해 val-loss 2.3308(<vanilla 2.3340), Flirds τ-게이트는 침묵(=vanilla, 절대-0 의미론의 위협-의존 트레이드오프); std50k5 seed0는 ShapleyFL 게이트도 vanilla보다 나음(fidelity 붕괴 무대인데 다운스트림은 안 붕괴 — H-3 반례 예비, 확정은 R4)(§3.2.6).

**2차-③ 탐지 — 세 층위로 갈림**
- **승**: noisy·free-rider(zero/random) AUROC **1.0**(IID/비-IID 배경 무관, §3.3.1·§3.1.5) + frzero **bit-exact φ=0**(§5.2 감사 확증); dose 문턱 noisy nr0.25↑·FR 전 배율 1.0(§4.5).
- **패 = 추정기 결함(진짜 약점)**: clean-preserving poison — Flirds-1st 0.000(1B)·**3B는 Flirds(2차)도 0.000**(§3.3.3); **1B의 2차항 방어도 seed-혼재**(β0.3 재실행 정본 0.500±.354, per-seed .25/.25/1.0; 재실행 전 run은 0.917 — §3.3.1 각주); dose pf≤0.2 완전 회피 + pf0.3–0.7 seed-불안정 전이대(§4.5); removal ASR 미차단(§4.4.2).
- **패지만 게임 공통(Flirds 책임 아님)**: IID poison은 (b)·loss-heur도 0.00(§3.1.5); **frdelta는 (b)oracle과 seed별 완전 동일 0.33**(§3.3.4) — 루트 위계 "기여도≠탐지"의 실측 사례, 회수는 update-패턴 탐지기 몫(STD-DAGMM 1.0).
- 전용 탐지기 열세: device100 noisy 0.57~0.77 vs FedDQC 1.0(§3.3.2) — 비IID 배경에서 φ-as-detector 침식.

**보조축 — 비용·안정성**
- 비용: cohort 큰 무대 **5~160×**(anchor5 1/5 · device100 1/159 · N=10 2¹⁰ 1/160; §3.4.2·§3.1.3); Flirds-1st는 전 무대 최저가; 학습 자체보다 2~3자릿수 저렴(CNN §3.4.3). **std20(2/round)은 역전**((b) 2943s < Flirds 4703s) = 우위는 조건부 — op-count 축이 이 구조를 하드웨어-독립으로 방어(§3.4.1).
- 안정성: Flirds xseed 0.547 ≈ (b) 자체 0.518(추가 분산 0) vs recon-MC 계열 0.12~0.31(§3.1.2 (b2)).

**판정 매트릭스**

| 조건 | 판정 | 근거 |
|---|---|---|
| 비IID / 오염 / 부분참여 무대 fidelity | **승** (유일 생존급) | §4.2 std50k5 +1.000 · §3.1.2 CNN 0.92~0.95 · §3.3.1 silo5 1.000 |
| 라운드당 참여 큰 무대 비용 | **승** (5~160×) | §3.4.2 anchor·device100 · §3.1.3 N=10 |
| noisy·zero/random FR 탐지 | **승** (1.0 · exact-0) | §3.3.1 · §4.5 · §5.2 |
| 오염 무대 개입(성능) | **승(소폭)** | §3.2.2 +0.11 · §4.4.3 분리 인과 |
| 점수원 경쟁(같은 정책·Track H) | **계열-수준 승** (exact-0 계열 생존 acc≈천장·renorm free-rider 붕괴 .37~.40; 계열 내 1위는 정책·위협 의존 + LLM noisy 역전) | §3.2.6 |
| IID-clean 무대 | **무정보** (전원 동률·타깃 불안정) | §5.4 ρ −0.37 · §3.1.5 B축 |
| clean-preserving poison | **패** (추정기 약점) | §3.3.3 3B 0.000 · §3.3.1 1B seed-혼재 0.500 · §4.4.2 ASR 유지 |
| frdelta · IID-poison | 패지만 **게임 공통** | §3.3.4 · §3.1.5 ((b)oracle 동일 실패) |
| 저cohort(std20) 비용 · AdamW | **약세** | §3.4.2 역전 · §4.6 +0.77 |

**최고 세팅 표** (2차-① 관점 — 어떤 세팅이 가장 잘 됐나):

| 관점 | 최고 세팅 | 수치 |
|---|---|---|
| **최대 절대 이득** | **CNN cifar10 grad-noise × sign-게이트 V2** (per-update 양수-only; iid·dir1) | dAcc **+0.323~+0.358** = oracle_excl(+0.377~0.379)의 회수 0.86–0.94 (§3.2.4). soft 가중도 동일 위협서 최대: vanilla 0.499→flirds_mult 0.609·shapleyfl 0.645 (§3.2.2) |
| **가장 깨끗한 완전 회수** | **LLM frzero × `flirds_gate_v2`** (silo5·iid5, 3-seed) | recovery **1.000 정확**(오배제 0쌍, oracle_excl과 최종손실 소수 4자리 동일; §3.2.3) — φ=0 null-player 공리가 그대로 배포 성능으로; parameter-free(τ=0, k 없음). retrain 대응물 `v3_sign`도 1.000 = **계산 시점 축(③)과 무관하게 성립** |
| **점수원 경쟁 (같은 정책)** | **Track H CNN dir1** | exact-0 계열이 free-rider acc .61~.62(≈천장) vs renorm 4종 .37~.40(<vanilla .59) = zero-semantics 실측 감점; grad-noise는 Flirds(2차)만 .567~.607 vs Flirds-1st/FedIF .244~.248(1차 실명). 역전 무대 = LLM noisy(renorm val-loss 2.3308 < vanilla 2.3340 vs 절대-0 침묵) (§3.2.6) |
| **estimator 최고 확증 세팅** | **flirds × P5-hard retrain**(csign) | 오염-평균 **.6207 ≈ oracle_excl .6214**(전 위협 .6197~.6215 + clean .6333; §4.8.1) — P1 retrain .6107 대비 +.0100 |
| clean 무해성 (do-no-harm) | LLM 전 무대 | 게이트 무발화, 게이트·V3 arm max Δ최종손실 0.00056 (§3.2.3) + track_d MMLU/ROUGE parity (§3.2.1). **예외 = CNN clean 게이트 오발화**(V2w DO NOT PROMOTE §3.2.4; 경쟁에서도 Flirds .632·loss-heur .626 < vanilla .639 vs FedIF/Flirds-1st .638~.639 통과, renorm .60~.605 §3.2.6; P5 정책이 대부분 회수 §4.8.1) |
| retrain×top-k 유일 칸 | phase1 silo5 K=3/5 | flirds_topk 2.3978 < random_k 2.4111 (§3.2.5) — 방향 일관·폭은 작음 |

**서사 한 줄**: 신호가 존재하는 곳(비IID·오염·부분참여)에서는 exact oracle 동률의 fidelity를 5~160× 싸게 달성하고 그 순위는 removal로 인과 검증된다; 한계는 clean-preserving backdoor(1차 Taylor의 구조적 사각지대 — 2차항 방어도 간헐적)와 IID-clean 무대(잴 신호 자체가 없음 — 전 semivalue 공통)다.

**보조 증거(링크)**: 사후 removal-재학습 — worst-first 제거가 val-loss↓(§4.4.1)·cifar10 acc 분리 +0.039~0.045(§4.4.3) = 순위→성능 인과의 게임-무관 확인.

## 5.2 φ 부호 감사 (Track G Stage 0) — 게이팅 전제 확정

> Track G(φ 부호-게이팅 실효성; 스펙 `runs/track_g/README.md`) 두 단계 중 **Stage 0**. Phase B 게이팅 본실험 = §3.2.3–4. **Stage 0** = 재실행 0, 기존 **309개 rundir 전수**의 φ 부호 감사(`phi_sign_audit.py` → `audit/SIGN_AUDIT.md` + `sign_table.csv` 73,288행; 부호 규약 = contribution orientation, 도움=양수)로 예측표 확정·수정(커밋 6623fdf). 판정 2·3의 성능 귀결은 Track H 경쟁이 실측(§3.2.6).

감사 판정 4건:
1. **clean 오배제-0 전제 성립**: canonical clean 전 셀에서 전 method·전 클라 **누적 φ 양수**(예 1B anchor5 (b)oracle 30/30 min +0.0135; 예외 = ComFedSV 간헐 1클라 음수, ShapleyFL 항등 0) → **τ=0 게이트가 clean에서 아무도 제외 안 함** 확정.
2. **frzero exact-0**: (b)oracle·Flirds·Flirds-1st·loss-heur·FedIF·Fed-LOO가 free-rider(zero)에 **bit-exact 0.0**(전건 100%) — strict `cum>0` 규칙이 정확 포착. coalition-renorm 계열(ComFedSV·FedSV·GTG)은 exact-0 아님(작은 음수).
3. **[예측 수정 ①] noisy엔 sign-게이트 작동영역 없음**: silo5 noisy dose ladder(nr 0→1)에서 Flirds φ 전 구간 양수(+0.00242→+0.00181), 0-교차 선형외삽 **nr≈3.44 = 도달불가**((b) 3.44·Flirds-1st 3.42·loss-heur 3.41·Fed-LOO 4.45 동일 구조) → noisy 회수는 sign-게이트가 아니라 z-게이트/V2w 몫. GTG(~0.76)·FedSV(~0.65)만 nr∈(0,1]서 교차 = **coalition-renorm 값오차의 부산물**(진짜 게임값의 0-교차 아님).
4. **[예측 수정 ②] frrand 누적부호 = seed-코인플립**: silo5 Flirds 3+/8−(mean −4.5e-07)·iid5 3+/0−(+7.4e-07) — 누적값이 0 근방이라 frrand 제외 여부는 seed 의존(예측표의 silo5/iid5 부호 분포 정확 일치).

CNN label-flip 게이트 dose 3점 = **{0.15, 0.35, 0.70}** 확정(전 val-method의 per-client 오염율 0-교차 span ~0.13–0.55의 아래/내부/초과 각 1점).

**출처**: `runs/track_g/audit/{SIGN_AUDIT.md,sign_table.csv}` · 예측표 = `runs/track_g/README.md` §2.1(감사 수정판).

## 5.3 신호크기 진단 종합 판정 (A축 §4.2–4.3 · B축 §3.1.5)

신호크기 진단([[flirds-signal-size-diagnosis]])의 결론 — **Yonghee 원가설("val-loss 변화량이 작아 fidelity 저하")은 반쪽만 맞다.**
- **A축(신호 크기 lever)**: LLM(rank·참여·lr·steps §4.2)·CNN(폭·참여 §4.3) 어느 것도 IID-clean에서 **cross-seed 실재 신호를 못 만든다**(핵심 축 3-seed 확정 — lr를 키워도 xseed ρ≈0). **lr의 φ-크기 효과도 3-seed에선 공통 shift(~1.3×)만 남고 클라 간 분리는 seed 분산에 묻힌다**(seed0 "~3×"는 비재현; §4.2 (b3)). 그 외 lever(rank·참여·폭)는 φ 크기조차 거의 안 바꾼다. fidelity(Flirds vs (b))는 A축 전반 1.000 유지 = **Taylor tradeoff 없음**(HVP가 rank·lr↑에 강건). 참여는 별도 역할 — 짧은 지평에서 φ 랭킹을 흐리고 **방법 구별을 만든다**(LLM std50k5·CNN partial 모두 Flirds 우위, uniform-subset 계열 붕괴; 1차항은 클라당 참여 횟수가 적으면 붕괴, 2차항이 방어 §4.1).
- **B축(신호 실재성)**: **클라 간 실제 차이가 신호를 만든다**. **non-IID clean cross-seed ρ 0.87**(오염 0, 도메인 분리만)이 결정타 — IID clean 0.13과 대비(N=50 저참여 IID도 +0.06 ≈ 0, §4.2 (b2)). 오염축·비IID축이 각각 독립적으로 fidelity·탐지 신호를 만들며, 탐지는 배경 이질성에 따라 갈린다(FedDQC는 IID서 유리, poison-회피는 IID서 심화).
- **결론**: 어느 lever도 cross-seed 실재 신호를 못 만들고, 신호는 B축(비IID·오염)이 만든다. A축 잔여 = lr·steps intervention 분석(원가설 2차; 데이터는 기존 rundir에 있음, 재실행 불필요 — 루트 REMAINING §1.4)뿐.

## 5.4 (b) target self-stability (Exp C) — 매칭 대상 자체의 재현성 **[보류: 논문 appendix 후보 — Yonghee 확인]**

> **질문**: fidelity 헤드라인 +1.000 은 *(b) oracle 과의 일치*다. 그 **매칭 대상 (b) 자신이 seed 간 재현되나?** (리뷰 C-2 정면 대응; 재실행 0 — 기존 rundir 의 (b) per-client φ 를 seed 로 피벗해 pairwise Spearman 을 산출·정본화. `runs/track_d/make_target_stability.py`.)

**(b) oracle cross-seed Spearman ↑** (셀당 3-seed pairwise 평균; ↑=재현성. IID-clean 저·비-IID 고 = 신호 실재성)

| 무대 | cell | mean xseed ρ ↑ | pairs (0-1 / 0-2 / 1-2) |
|---|---|---|---|
| track_d **IID-clean** | 1B_anchor5 | **−0.367** | +0.00 / −0.90 / −0.20 |
| track_d IID-clean | 1B_std20 | −0.114 | −0.04 / −0.11 / −0.19 |
| track_d IID-clean | 3B_anchor5 | +0.033 | −0.10 / +0.30 / −0.10 |
| track_d IID-clean | 3B_std20 | −0.243 | −0.09 / −0.19 / −0.45 |
| track_d IID-clean | 7B_anchor5 | +0.733 | +0.80 / +0.50 / +0.90 |
| track_d IID-clean | 7B_std20 | +0.164 | −0.04 / +0.31 / +0.22 |
| phase2 IID (iid5) | 1B_iid5_clean | +0.133 | −0.30 / +0.60 / +0.10 |
| phase2 **비-IID** (silo5) | 1B_silo5_clean | **+0.867** | +1.00 / +0.80 / +0.80 |
| phase2 비-IID (silo5) | 1B_silo5_noisy | +0.933 | +1.00 / +0.90 / +0.90 |
| phase2 비-IID (silo5) | 1B_silo5_frzero | +0.933 | +1.00 / +0.90 / +0.90 |
| phase2 비-IID (silo5) | 1B_silo5_poison | +1.000 | +1.00 / +1.00 / +1.00 |
| phase2 device100 anchor | 1B_device100-a0.5_noisy_anchor | −0.042 | +0.09 / −0.08 / −0.14 (N≈94, per-round (b)) |

> **판정**: **IID-clean 무대의 (b) target 은 seed-불안정**(track_d 1B −0.37~−0.11 = 리뷰 노트값 정확 재현·정본화) → 그 위의 per-seed **+1.000 fidelity 는 *불안정한 GT* 를 좇는 것**(C-2). **비-IID(silo5)선 (b) 가 안정**(+0.87~1.00) → 거기의 +1.000 은 의미 있음. §3.1.5 B축과 정합 — Exp C 는 그 **(b)-target 버전을 전 스케일·전 무대로 정본화**. **7B 는 IID서도 +0.733** → 스케일이 클수록 (b) 안정성↑(추가 조사감). ⚠ silo5 frzero·noisy 행은 β0.3 재실행판(ce0b454) 기준 +0.933(frzero 재실행 전 rundir는 +1.000).
> **프로토콜 격상**(리뷰 §4/§5.1): `make_fidelity.py` 가 fidelity 표 아래 이 xseed ρ 열을 함께 출력 → *fidelity 는 항상 target 안정성과 병기*. 3B_silo5 는 seed0 뿐(1-seed → nan, 표 제외).

**출처**: `runs/track_d/target_stability.csv` · `runs/phase2_matrix/target_stability.csv` (gitignore=파생; 재생성: `python runs/track_d/make_target_stability.py [rundirs_root] [out.csv]`). 배경 = [[flirds-signal-size-diagnosis]] §3.5.

## 5.5 Taylor 물리잔차 실측 (E2, `measured_2026-07/taylor`) — 명제 P3 검증 **[보류: 논문 appendix 후보 — Yonghee 확인]**

**(a) 세팅**: Llama-3.2-1B-Instruct(LoRA r16), N=5·R=10, train 200/val 100, 10 steps, lr 1e-3, batch 16, maxlen 768, **3-seed**. 수학검증 명제 **P3**(Taylor 잔차의 물리 크기)의 1B 실측 — gpt2 스모크에선 잔차가 노이즈 바닥이라 1B 대기였던 것(`measure_taylor_residual.py`).

**(b) 결과** (per-coalition-step 잔차, pooled median; seed0/1/2):
- **1차 잔차** ≈ 1.75e-6 / 1.55e-6 / 1.28e-6 · **2차 잔차** ≈ 6.38e-7 / 4.51e-7 / 4.41e-7 → **2차 근사가 1차보다 ~2.7–3.4× 작음**(pooled-median 비율 seed별 2.75/3.43/2.89; 스텝별 t2≤t1 비율 0.79–0.81, 판정 `t2_better` 전 seed) = **2차(HVP) 항 추가가 물리적으로 근사를 개선**.
- 2차 잔차가 추정 3차항 크기의 ~21–37×(seed별 round-median의 중앙값 36.5/21.0/26.0) → 남은 잔차는 3차항이 아니라 노이즈/고차 누적이 지배 = 3차 확장의 이득 없음.
- **φ 정합**: closed-form vs Flirds(2차) max|Δφ| **3.4–5.5e-10**(seed별 3.41/5.38/5.54e-10; fp32 노이즈 바닥 2.4e-7의 ≥1/430), exact vs {1차, 2차, closed} Spearman **전부 1.000**.
- ⚠ `summary.json`의 sanity verdict "CHECK"는 bit-동일성 플래그 때문(closed↔flirds2 비트동일 아님) — 실차 ~1e-10로 물리 결론 무영향.

**출처**: `runs/measured_2026-07/taylor/llama1b_r10_seed{0,1,2}/summary.json` (커밋 b694f07). 배경 = [[irds-fl-math-rigor-2026-07/irds-fl-math-rigor]] (P1–P8; P1·P2 대수 1e-12는 gpt2 스모크, P3 물리는 여기서 확증).

## 5.6 수렴(Convergence) — overview 스코프 제외 (2026-07-22)

수렴 축은 본 문서에서 제외한다(clean-IID에서 arm 간 사실상 동률, 7B std20 차이도 seed-std와 중첩 — Yonghee 2026-07-22 결정). **데이터는 rundir 존속**: `runs/track_d/rundirs/*/metrics.json`·`runs/track_c/c2/*/metrics.json`의 `arms.*.{rounds_to_target,val_curve}` — 필요시 재집계.

## 5.7 각주 — E1–E7 ↔ 영어 목적명 매핑

> 본문은 영어 목적명만 사용. taxonomy의 E# 코드와의 대응: **E1**=Fidelity · **E2**=Selection→downstream performance · **E3**=Corrupt-client detection · **E4**=Fairness·reward · **E5**=Stability(replication) · **E6**=Cost·scalability · **E7**=Aggregation quality. (E-세션 실험 코드 E2/E4/E5/E7과는 별개 명명 — 그쪽 매핑은 §2.1.)

---

# 6. 부록

## 6.1 Foundational validation (`phase1`) — 첫 clean run

**(a) 세팅**: Llama-3.2-1B, **N=5 full**, K=3, 오염 주입(noisy=client0 answer-swap / free-rider=client1 zero-update), per_domain train=12000·val=200·test=2000, local 10 steps. **full** = R=50, lr∈{1e-3, 3e-3} × 3 seed, oracle_b off. 메트릭: AUROC(noisy/free-rider) + selection(K=3 keep) + arms(→ §3.2.5).

**(b) 결과 — AUROC + selection** (full run 3-seed)

| group | noisy AUROC ↑ | free-rider AUROC ↑ | flirds_keep (seed별) | random_keep |
|---|---|---|---|---|
| full lr1e-3 (3 seed) | 0.750±.000 | 1.000±.000 | [3,2,4] 매 seed (=clean) | seed별 가변 |
| full lr3e-3 (3 seed) | 1.000±.000 | 0.750±.000 | [2,3,4] 매 seed (=clean) | seed별 가변 |

> **lr 의존 반전**: full lr1e-3은 noisy 0.75/FR 1.0, lr3e-3은 noisy 1.0/FR 0.75 (AUROC가 lr에 의존). **selection**: flirds_keep이 매 seed에서 정확히 clean 클라 3개(client 0=noisy·1=free-rider 항상 드롭) → 안정적 분리. LR sweep(1-seed 4종)은 §6.3 검증-전용.

**(c) 출처**: `runs/phase1/rundirs/*/metrics.json` (`auroc_noisy`, `auroc_freerider`, `selection`, `arms`).

## 6.2 Caveats (주의)

1. **3B robustness = 1 seed** (`phase2_matrix/3B_silo5_*` seeds=[0]). 3-seed 미완(계획 P5; β0.3 잔여 18셀에 포함 — 루트 REMAINING §1.2). 3B robustness 수치는 단일 seed.
2. **(a) retrain oracle = 1B anchor5만**(track_d). 3B/7B anchor5는 fidelity·runtime만 있고 (a) 없음(⬚, 계획 P2/P3). *프로젝트 노트엔 별도 task6에서 3B (a)-valloss≈0.900 언급이 있으나 track_d rundir엔 없음 → 파일-only 원칙상 본 표엔 미수록.*
3. **device100 비-anchor truth = Flirds proxy reference** (정확 (b)가 칸당 ~25,000s라 α=0.5만 실측). 그 칸의 Spearman은 *vs Flirds*이지 vs exact oracle 아님 → 1.000은 "Flirds-1st·loss-heur가 Flirds와 동일 순위"의 뜻.
4. **CNN fidelity pool 평균은 `iid` 포함 → 깎임**. iid 셀은 오염·skew 신호가 없어 fidelity가 의미상 낮다(§3.1.2에 iid 제외 값 병기).
5. **CNN C2 / track_c 그룹 테이블은 partition·강도·dataset을 threat 내에서 pool** → std 큼. 셀별 30칸은 `RESULTS.txt`.
6. **7B anchor5 arm(MMLU/ROUGE) = 2026-06-26 추가 완료**(arm-only 재실행). LLM standard 6 셀 전부 arm 포함.
7. **tiny val** caveat: Robustness silo5 val=20 / device100 val=10 — 작은 검증셋이라 AUROC가 coarse(특히 noisy φ-as-detector).
8. **poison ASR**은 deployed-model 기준(silo5≈1.00, device100 α0≈1.00/α0.5≈0.50, 3B≈1.00).
9. **ShapleyFL β=0.3 통일 재실행 캠페인 상태**: **1B_silo5 오염 4셀 = β0.3 재실행 완주·착지(ce0b454, 2026-07-20)** — §3.3.1·§3.1.5·§5.4의 silo5 값은 그 정본(스위트 15종으로 확장, Fed-LOO·ComFedSV 추가; Flirds poison 0.917→0.500 변화는 §3.3.1 각주). **잔여 18셀**(device100 14 + 3B silo5 4; 루트 REMAINING §1.2, R4 뒤 큐 자동 재개) + **deferred 9셀**(7B_std20×3·7B_anchor5×3·device100-a0.5 anchor×3; §1.3) — §3.1.1 7B 열·§3.4.2 runtime·§3.3.2 anchor 행은 아직 이전 실행 기준. 재개법 = `runs/rerun_beta03/RESUME_AFTER_MIGRATION.md`. **β provenance 실측**(07-16 스캔): track_c CNN 120셀 = β0.5-era 산출물('CNN은 β-불변' 주장 canon 미확보 실측) — 상세·β 효과 크기 = §4.7.
10. **신호크기 probe(§4.2–4.3)**: A축 LLM = lr격자(st10)·std50k5(r16)·noise(r16) **3-seed 확정(커밋 454db39)**; rank probe·st20/30·r32/64는 seed0, std50k5 seeds 1-2는 Flirds·Flirds-1st만 채점(경량 스위트). CNN=3-seed. cross-seed ρ 붕괴(partial 참여)·Flirds-1st 붕괴는 CNN **R=10·클라당 참여 ~2회** 짧은 지평 특성(LLM std50k5 R=200선 Flirds-1st 1.0 유지) → "참여 분수"가 아니라 **클라당 참여 횟수**가 조건. C2 참여 sweep **f=0.2 결측**(shapleyfl arm 2²⁰/라운드 exact 불가). A축 probe는 (b) only(ORACLE_A=0). B축 매트릭스(§3.1.5)는 3-seed 완료, 단 `make_analysis.py`에 iid5/silo5_clean 미반영 → rundir 직접 집계(frdelta §3.3.4도 동일). figure는 `runs/probe_signal/make_figures.py` 재실행으로 재생성(수치 표가 정본).
11. **loss-heur runtime = C6 측정버그(2026-07-17 수정)**: `in_run_sv.py`의 singleton 유틸이 base U(P_r)를 클라마다 중복 평가 → fwd **2|P_r| → 1+|P_r|**로 수정(`in_run_singletons` 캐시; **φ 비트동일 = fidelity 무영향**, runtime만 과대였음). **pre-fix 측정치(본 문서 잔존)**: §3.4.2 track_d loss-heur 전 열(1B anchor5 1093s=1.66×·std20 2913s=1.32× 과대; 3B/7B 동일 구조)·§3.4.3 CNN 열. **post-fix 정본**: 1B anchor5 **657±19s**·std20 **2199±60s**(E4 rundir, §3.1.3)·silo5 **96.6/100.1/99.9s**(β0.3 재실행 rundir — §3.3.1의 ~99s; staging 재실험 CSV `runs/measured_2026-07/loss_heur_acct/`는 96.6/100.1/100.2s로 별도 실행 미세차) — op-count 예측(silo 96s, §3.4.1)과 정합. 3B/7B·기타 트랙 재측정 대기. (구 참조 `REMAINING_after_e_session_2026-07-19.md`는 루트 `REMAINING.md`로 개명·재작성됨.)
12. **Banzhaf·Ripple 비교군 제외(2026-07-22 Yonghee 결정)**: 두 방법의 행·열·언급을 본 문서 전 표·범례·baseline-set 노트에서 제거했다 — 원 데이터는 rundir 존속(track_c C1·track_d anchor5·phase2 silo5·probe 등에서 재집계 가능; Ripple 제외 근거 상세 = [[ripple-audit-2026-07/ripple-baseline-exclusion]], 제외 전 수치는 git 이력).
13. **재현성 정정(2026-07-21, 커밋 8598cea)**: ① **H1 — LLM LoRA adapter init이 unseeded RNG였음**(phase2_matrix·track_d·track_g 러너; 현재는 `seed_everything(0)` 고정) → 기존 **전 LLM rundir의 절대값(φ·AUROC·val-loss·error bar)은 재현 불가**(순위-fidelity 결론은 강건 예상); ② determinism 강화(단 CNN cuDNN TF32 비활성화는 `fa2c167`로 원복 — torch 기본 유지); ③ **M1 — 라이브러리 스택 혼재**(track_h 129/108·probe_signal 90/25 rundir가 torch 2.11/2.12 혼재). P0(LLM 전 그리드)·P1(CNN) 재실행 계획·우선순위 = 루트 `RERUN_AFTER_REPRO_FIX_2026-07-21.md`. 기존 수치는 당시 코드의 실측으로 유효(meta.json git_sha 증빙)하나 "재현 가능" 주장은 재실행 후에만.

## 6.3 검증-전용 기록 (논문 비게재 내부 검증; 1–2줄 요약 + 출처만)

- **phase1 LR sweep** (`phase1/...sweep-lr*`, 1-seed×4lr, R=20, (b) on): lr 1e-4~3e-3 전 칸 noisy AUROC 0.75 / FR 1.0, flirds_keep [3,2,4]·random_keep [2,3,4] 동일 — full run(§6.1)의 lr-반전과 달리 sweep 무대에선 lr 무감. 출처 = `runs/phase1/rundirs/*sweep*/metrics.json`.
- **TF32 A/B** (`measured_2026-07/tf32_ab/`, cifar10 {iid,label-flip}×{tf32on,off} seed0): cuDNN conv TF32 on/off의 final_acc 차 ≤0.001·Flirds spearman_b 비트동일 — 정밀도 축이 CNN 결론을 안 바꿈의 실측(§6.2 caveat 13 ②의 원복 근거).
- **E3 CNN cost 스모크** (`measured_2026-07/e3_cost_smoke/`, mnist·cifar10 iid seed0): 제외 baseline(§6.2 caveat 12)의 자기-궤적 재실행 비용을 별도 위상으로 분리 실측 — 셀 total의 ~95%(mnist 2151s·cifar10 4275s) vs 전 방법 공유 valuation 65.8/163.9s; Flirds 0.27/0.36s·Fed-LOO 0.28/0.63s.
- **FL 학습시간 위상분리** (`flirds/timing.py` → timing.json): device100 clean seed0 = client-training **2249s** vs valuation 2704s(peak 33.5/99.1 GiB; `measured_2026-07/timing_device100/`); E4 anchor5 client-training ~1821–2145s·std20 ~4470–4937s(셀당 2.1~5.5 GPU-h); E5 N=10 총 34.7 GPU-h 중 (b) 단독 94.1%. 롤업 = `experiments/aggregate_runs.py`.
- **microbench** (`measured_2026-07/microbench/summary.json`): B200 fp32 per-op = forward 1.60s·HVP 10.36s(비율 6.47); fp32/bf16 배율 fwd×5.33·HVP×4.09·GEMM×22.68 — §3.4.1 op-count와 C3 fp32 환산의 실측 입력.
- **acct** (`measured_2026-07/acct/acct_seed{0,1,2}.summary.txt`): B200 staging에서 silo5 전 방법 재실행 요약(FL train 434.6s + 방법별 φ·AUROC·runtime) — C6 fix 검증용 스테이징 기록.
- **loss_heur_acct** (`measured_2026-07/loss_heur_acct/loss_heur_silo5_runtime.csv`): loss-heur post-fix silo5 runtime 96.6/100.1/100.2s 영속본(§6.2 caveat 11).
- **gpt2 대수검증 스모크**: 명제 P1·P2의 1e-12 수준 대수 정확성(closed=flirds 5.83e-12·perround=2^N 3.93e-7) — 산출물·상세 = [[irds-fl-math-rigor-2026-07/irds-fl-math-rigor]] 폴더(`gpt2_smoke_weakdelta_*`).

## 6.4 상호 링크
- 선행연구 6축 분류 + 마스터 표: [[prior-work-taxonomy/README]] · [[prior-work-taxonomy/taxonomy]]
- 검증실험 카탈로그(CNN/LLM 트랙 분리): [[prior-work-taxonomy/validation-experiments]]
- metric·benchmark·ground-truth 출처: [[prior-work-taxonomy/metrics-and-benchmarks]]
- baseline 수치 ↔ 원 논문 대조: [[baseline-original-paper-verification]]
- 수학 엄밀성 검증(P1–P8): [[irds-fl-math-rigor-2026-07/irds-fl-math-rigor]] · 비용 방법론: [[cost-comparison-methodology-2026-07/cost-comparison-methodology]]
- **논문 실험 배치안(본문/ablation/appendix)**: [[paper-experiment-placement-plan]]
- 잔여 작업 정본: 루트 `REMAINING.md` · 재현성 재실행 계획: 루트 `RERUN_AFTER_REPRO_FIX_2026-07-21.md`

---

# 7. 유지보수 / 갱신

**미실행(⬚) 행을 채우는 법**
- 그 실험을 돌린 뒤 해당 rundir가 생기면, 아래 재집계로 수치가 채워진다. 마스터 표(§2)의 `status`를 ⬚→●로, 본문 표의 ⬚ 칸을 mean±std로 교체.
- 예: 7B anchor5 (a) retrain oracle(P3) → `ORACLE_A=1 REGIME=anchor5 SMOKE_MODEL=7B` 로 track_d 실행 → `make_fidelity.py` 재실행 → §3.1.1 anchor5 (a)oracle 7B 칸 채움.

**수치 갱신 (rundir만으로, GPU 불필요)**
- LLM standard: `python runs/track_d/make_fidelity.py` → `fidelity.csv` 재생성 → §3.1.1 표 재집계. ⚠ E4/E5 신규 root(`rundirs_e4_fedloo`/`rundirs_e5_n10`)는 make_fidelity 인자 지원 확인 필요 — §3.1.3 수치는 현재 rundir `metrics.json` 직접 집계.
- Robustness: `python runs/phase2_matrix/make_analysis.py` → `analysis/00_overview/master_metrics.csv` 재생성(`RESULTS.md`는 `make_report.py`; 둘 다 gitignored) → §3.3 표 재집계. iid5·silo5_clean·frdelta는 rundir 직접.
- CNN: `codes/slurm/scripts/merge_oracle_a.py` → `fidelity.csv`; Track C 결과 스크립트 → `RESULTS.txt`.
- track_g/track_h: 각 `make_analysis.py` → `analysis/*.csv`(+ scale/dyn 하위 analysis).
- **Figure**: 실험별 `python runs/<exp>/make_figures.py` → `runs/<exp>/figures/` PNG+CSV 재생성(rundir-only; 내용·스팟체크 = 각 `figures/MANIFEST.md`; 통합 인덱스 `runs/make_index.py`). 본 문서는 그림 임베드 없음 — 필요시 각 figures/ 참조. ⚠ tracked figures 중 β0.3 재실행 전 산출물(예 `matrix_cxni/figures/crossseed_rho.csv`)은 재생성 필요.

**새 실험 완료 시**: §2에 행 1개 추가(축 분류 + § 링크 + status ●) → **Main(§3)/Ablation(§4) 분류 판단**(논문 본문 주장을 직접 뒷받침하면 Main, 구성요소·lever·프로토콜 변형 검증이면 Ablation, 해석·감사·재분석이면 §5) → 해당 섹션에 (a)세팅/(b)결과/(c)출처·baseline-set 3블록 추가 → §2.1 매핑표·§8 커버리지 갱신. **문서 상단에 갱신 이력(⟳) 블록을 쌓지 말 것** — 본문 섹션을 직접 최신화하고 이력은 git 커밋 메시지로 남긴다(2026-07-19 Yonghee 지시).

---

# 8. 커버리지 자가점검 (runs/ 실제 폴더 ↔ 본 문서 1:1)

| runs/ 세트 | 디스크 rundir/셀 수 (2026-07-22 실사) | 본 문서 수록 | 비고 |
|---|---|---|---|
| `phase1` | 12 rundir (full 6 + sweep 4 + mini/smoke 2) | full 6 → §6.1·§3.2.5 / sweep 4 → §6.3 | mini/smoke 2 = 진단용, 미수록(명시) |
| `track_d` | rundirs 18 (3 scale × 2 stage × 3 seed) + E4 6 + E5 1 | 6 셀 전부(§3.1.1 fidelity + §3.2.1 arms + §3.4.2 runtime) + §3.1.3(E4·E5) | 수렴 데이터는 §5.6 포인터 |
| `track_c` | c1 30 + c1_oracle 30 + c2 90 = 150 rundir | C1 → §3.1.2(+탐지 §3.3.5) / C2 → §3.2.2 / runtime → §3.4.3 | C2는 그룹 평균(셀별=RESULTS.txt); c1_oracle = `*_aonly_*` 30셀(t_a만) |
| `phase2_matrix` | **rundirs 31**(25-셀 그리드 + B축 iid5 5 + silo5_clean 1; **1B_silo5 오염 4셀 = β0.3 재실행판 ce0b454**) + **rundirs_2026-07 23**(7월 재실행 배치 22 + frdelta 1) | 25셀 → §3.3.1–3(+§3.1.4 요지) / B축 6셀 → §3.1.5 / frdelta → §3.3.4 | rundirs_2026-07의 22셀은 7월 캠페인 중간 산출(canonical 아님 — 인용은 rundirs/ 기준); frdelta만 §3.3.4 인용 |
| `matrix_cxni` | rundir 0 (드라이버·figures 전용) | — (산출 셀은 `phase2_matrix/rundirs/1B_{iid5,silo5}_*`로 착지 = §3.1.5 귀속) | tracked figures는 β0.3 재실행 전 산출(재생성 필요) |
| `probe_signal` | rundirs 21 + noise_probe 4 + cnn_c1 66 + cnn_c2 24 | §4.2(LLM A축) · §4.3(CNN 72/30셀; track_c 기준점 재사용 포함) | lr격자 st10·std50k5 r16·noise r16 = 3-seed, 나머지 A축 seed0 |
| `removal_dose` | rundirs 75 (A2 12 + B 63) + rundirs_cnn 18 + rundirs_trackd 6 (A1 3 + D 3) | §4.4(A2·A3·poison) · §4.5(B) · §4.6(A1·D) | 전부 3-seed |
| `track_g` | rundirs 218 (LLM; std50k5 8런 포함) + rundirs_cnn 36 + rundirs_cnn_v3 12 + audit 파생(309 rundir 스캔, 73,288행) | §3.2.3(LLM 게이팅) · §3.2.4(CNN 그리드+V3) · §5.2(감사) | `rundirs_llm/` 폴더는 부재(경로 정정); std50k5 = seed0 파일럿 동결 |
| `track_h` | rundirs_cnn **204**(경쟁 96 + P5 108) + rundirs_llm 12 + rundirs_cnn_scale **21**(12+앵커 9) + rundirs_cnn_dyn 9 | §3.2.6(경쟁 Tier1·2·R2) · §4.8(P5·Scale·Dyn) | R4 gsm50k5 = rundir 미착지(서버 실행 중, §2 P11); Tier 3 std50k5 12런은 현행 REMAINING 큐 미등재 |
| `measured_2026-07` | taylor 3 + e3_cost_smoke 2 + timing_device100 1 + tf32_ab 4 + microbench + acct + loss_heur_acct + figures | §3.4.1(op-count·microbench) · §5.5(Taylor) · §6.3(나머지 계측 전부) | 실험 트랙 아님(계측 캠페인) |
| `rerun_beta03` | rundir 0 (드라이버·문서·figures 전용; 산출은 phase2_matrix/rundirs 등 원위치 착지) | §4.7(β 대조·provenance) · §6.2 caveat 9(캠페인 상태) | `RESUME_AFTER_MIGRATION.md`·`figures/{beta_provenance,beta_contrast_3b}.csv` |
| **계획·미실행** | – | P2–P6(수치 ⬚) · P11 R4(실행 중) | P1 ◐((b) 2¹⁰ seed0=§3.1.3; (a)·seeds1-2 ⬚)·P7–P9 완료(§4.4–4.6)·P10 ◐(§3.2.6) |

