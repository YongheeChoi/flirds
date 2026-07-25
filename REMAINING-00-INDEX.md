# REMAINING — 전역 배분 인덱스 (2026-07-25 · **R4 = R=100** 개편판)

> **정본** = `research-wiki/survey/flirds-paper-experiment-plan.md`(Yonghee 07-25 확정 수록목록 + 결손 G1–G13).
> 이 인덱스는 그 계획의 **결손(G) → 실행처** 배정과 예상 종료만 담는다. 실행 절차는 각 서버 파일.
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 기준시각 = **07-25 22:00**(잔여 ~74h).
> 규칙: 전 실험 **3-seed**(seed-major) · **push는 Yonghee 직접** · 수치는 rundir/analysis 재생성 값만.

## 0. ★ 이번 개편의 축 — R4 무대를 **R=200 → R=100** 으로 재정의

**결정(Yonghee 07-25).** R4(gsm50k5)의 라운드 수만 절반으로 내린다. **다른 무대(silo5·anchor5·std20·device100·CNN)는 그대로다.**

- **근거(수렴)**: vanilla val_curve 실측(noisy seed0) `r0 0.9987 → r50 0.6426 → r100 0.6209 → r200 0.6092`. **r100 이 전체 개선의 97.0%** — 마지막 100라운드는 3%.
- **R=50 은 안 간다**: N=50·K=5라 라운드 수가 곧 φ의 표본 수다. R=50이면 클라당 기대 참여 5회, **셀당 ~1.7명이 참여 0–1회** → 참여 0회 클라의 φ는 **정확히 0 = free-rider와 구별 불가**로 frzero 열과 §5.4 탐지가 오염된다. R=100이면 그 기대값이 0.02명.
- **효과**: LLM 물량이 절반 → **G4c(§5.3 retrain renorm-4)가 되살아나** 마감 안에 전부 들어간다.
- **⚠ 코드 조치 완료(C-c)**: `rounds` 를 rundir **IDENTITY 필드로 승격**(`track_g` · `phase2_matrix`). 승격 전에는 R=100 실행이 R=200 canonical 을 **아무 경고 없이 덮었다** — 이름에 R이 없어서다(β 0.5→0.3 이 조용히 덮인 것과 같은 실패 모드). 이제 R=200 셀을 덮으려면 `RUNDIR_REPLACE=1` 이 필요하고, 전 sbatch·큐에 배선돼 있다.
- **⚠ 운용 원칙 예외**: "도는 잡은 죽이지 않는다"는 원칙에도 불구하고 **R=200 으로 도는 R4 잡은 중단·재제출**한다. 산출물이 무대 밖이라 살려도 못 쓴다(HJ L11 63셀이 해당).

## 1. 삭제된 작업 (재제안 금지)

| 폐기 | 사유 |
|---|---|
| **L9 frrand** 전량 | frrand = 축 밖 위협 → 07-25 전량 중단 확정 |
| **L7 P1w-LLM** | 계획서에 LLM P1w 항목 없음(§5a 옵션으로만 존치) |
| **L10 strmain-dose · L5 비등n · L6 graded-noisy** | 축 밖 |
| **fmnist competition · c2fid fmnist** | fmnist 파티션 제외 → **mnist 무대가 대체** |
| **W-B P1w obsf 잔여** | P1w는 G3·G10 rundir에 동반 산출(추가 런 0) |
| **전용 탐지기 4종 · Fed-LOO** | 집계에서만 제외(실행 0) |
| **forward 전용 val-chunk 분리(코드)** | `make_llm_loss` docstring: 이미 프로파일 **~1.0× = 효과 없음**(FLOP-bound). 대신 **코드 0**으로 forward-only 소스의 `VAL_CHUNK` 제약만 해제(§3) |

## 2. 배정표 — 결손 G × 실행처 (**단가 = R=100 기준**)

| G | 작업 | 셀 | 단가 | GPU-h | 실행처 | HW |
|---|---|---|---|---|---|---|
| **G1** | R4-L2 주무대 (b) 오라클 | 9 | **9.7h**(op-count×microbench) | **87** | **B200 세션 A** | HVP·canonical timing |
| **L1** | R4 flirds 개입 **R=100 재실행** | 18 | 1.9–9.0h(실측÷2) | **84** | **B200 세션 B** | HVP |
| **G4c** | R4 retrain renorm-4 — **부활** | 9 | B200 13.9h / A6000 ~50h | **83**+**150** | **B200 s0·s1** / **JB s2** | 48GB+ |
| **G4a** | R4 online same-game+FedIF | 27 | 3.7–4.65h(HJ 재측정÷2) | **109** | **HJ** s0·1 / **JB** s2 | A6000 |
| **G4b** | R4 online renorm-4 | 36 | ~12.5h | **450** | **HJ** s0·1 / **JB** s2 | A6000 |
| **G5** | 2차항 LLM 레그 seed 보강 | 4 | ~5h | ~20 | **YH**(후반) | 48GB |
| **G12** | A축 lever probe seed 보강 | 19 | ~2–4h | 50–90 | **YH**(후반·여유) | 48GB |
| **G3** | cifar10/iid 점수원 7종 + obs | 96 | ~0.4h·obs ~2h | ~80 | **YH** | 3090 |
| **G8** | mnist 부분참여 fidelity(+탐지) | 24 | 1.05h(실측) | ~25 | **YH** | 3090 |
| **G10** | mnist downstream 8점수원 | 216 | 추정 | ~135 | **YH** | 3090 |
| **G6** | Removal-curve CNN 오염축 | 9 | 추정 | ~15 | **YH** | 3090 |
| **G2** | cifar10 vs (a) 오염축 정렬 | 24 | **9.1h**(c1_oracle 실측) | **225** | **JW** | 3090 |
| **G9** | mnist vs (a) | 24 | **11.4h**(실측) | **280** | **JW** | 3090 |
| **G7** | op-count N·R·K 파라메트릭 | 0 | 문서 | 0 | (집필) | — |

- **LLM ≈ 1,050 GPU-h**(R=200이면 ~2,300) · **CNN ≈ 760**(R 무관).
- **LLM은 24GB 불가**(1B full-SFT peak 26.3 GiB 실측) → **A6000/RTX6000Ada/B200 전용**. CNN은 전량 3090(여유 21장).
- **G4c 를 B200 이 2/3 가져가는 이유**: 이 셀은 관찰자가 매 라운드 renorm-4 를 채점하고 그게 셀의 ~92%다 → B200 13.9h vs A6000 ~50h(**3.6×**). seed2 만 A6000(JB).

## 3. 코드 조치 (3건 — 2건은 CNN 착수 게이트)

| 코드 | 무엇 | 상태 |
|---|---|---|
| **C-c** | `rounds` 를 rundir IDENTITY 로 승격 (`track_g`·`phase2_matrix`) | ✅ **적용 완료** — R=100 전환의 전제 |
| **C-a** | `track_c2.py:157` `MODEL_FN` 에 `"mnist": LeNet5` (**1줄**) | ⬚ YH — **G8·G10** 을 막고 있음 |
| **C-b** | `track_c1.py` 에 `C1_PARTITION`(iid\|dir1) · `C1_THREAT`(clean\|label_flip\|free_rider\|grad_noise) · `C1_FLIP_RATE` 도입 | ⬚ YH — **JW 의 505 GPU-h** 를 막고 있음 → **최우선** |

**코드 0짜리 개선(적용 완료)**: L11 sbatch 가 `VAL_CHUNK` 를 소스별로 준다 — grad 경로(flirds1st·fedif)만 3, forward-only(lossheur·renorm-4)는 기본 10. 청크 합산이 exact 라 **φ 동일**이므로 무위험이고, A6000 OOM 가드가 필요 없던 5개 소스에 걸려 있던 제약을 푼다. (속도 이득은 **미측정** — HJ 재측정상 same-game 은 학습이 지배하고, renorm 셀에서만 여지가 있다.)

## 4. 실행처별 배분 · 예상 종료 (07-25 22:00 기준 · 잔여 74h)

| 실행처 | 슬롯 | 담당 | GPU-h | wall | 종료 |
|---|---|---|---|---|---|
| **B200 세션 A** | 2 (CID 1·2) | **G1** 9셀 + G4c seed1 3셀 | 129 | 65h | 07-28 오후 |
| **B200 세션 B** | 2 (CID 3·4) | **L1 R=100 재실행** 18셀 + G4c seed0 3셀 | 126 | 63h | 07-28 오후 |
| **HJ** | 8 (48GB) | L11 seed0·1 42셀 (**R=100 재제출**) | 372 | 47h | 07-27 후반 |
| **JB** | 8 (48GB) | L11 seed2 21셀 + G4c seed2 3셀 | 336 | 42h | 07-27 후반 |
| **JW** | 8 (3090) | G2 + G9 = (a) 오라클 48셀 | 505 | 63h | 07-28 오후 |
| **YH** | 8 (3090→48GB) | 코드 C-a·C-b → G3·G8·G6·G10 → G5·G12 | 255 + 70–110 | 43h | 07-27 후반 |

- **B200 두 세션은 서로 독립**이다 — 한쪽이 늦게 잡혀도 다른 쪽 산출물은 그대로 쓴다. 세션 A만 살면 §5.2/§5.4/§5.5 가, 세션 B만 살면 §5.3 이 각각 자립한다.
- **A6000 실가용 10장 / 98**(가동률 90%) → HJ·JB 가 상시 16슬롯을 못 채운다. **`asus_6000ada`(RTX6000Ada 48GB 8장)를 파티션 목록에 추가**해 뒀다(전 LLM sbatch).
- **3090 여유 21장** → JW·YH 16슬롯은 물리적으로 확보된다. **4090(여유 0)은 배정하지 않는다.**
- **YH 가 07-27 오전에 먼저 빈다** → JW 의 G9 꼬리 또는 HJ 의 L11 꼬리를 work-steal.

## 5. 지금 당장 할 조치 (순서대로)

1. **HJ**: R=200 으로 도는 L11 63셀 **전량 `scancel` → R=100 재제출**(sbatch 에 `ROUNDS=100`·`RUNDIR_REPLACE=1` 배선 완료). R=200 산출은 무대 밖.
2. **JB**: L9 frrand 전량 `scancel`(이미 결정) → **L11 seed2** 제출 → 이어서 **G4c seed2**(`--array=6-8`).
3. **YH**: **C-b 코드**(최우선, JW를 막고 있음) → C-a → G3 착수.
4. **JW**: C-b 착지 즉시 `sbatch_c1_axis.sh` seed0(cifar10 0-7 → mnist 8-15).
5. **B200**: 세션 A = `CID=1`·`CID=2`, 세션 B = `CID=3`·`CID=4` 로 각각 제출.

## 5a. 여유가 생기면 최우선 추가 = **LLM P1w**(가중 게이팅)

> Yonghee 07-25: "추가 실험을 한다면 이것부터". CNN P1w 는 G3·G10 rundir 에 **동반 산출**되어 자동 완비되므로, LLM 레그만 얹으면 P1w 의 "CNN·LLM 전 범위 3-seed 승격" 판정이 성립한다. 지금 LLM P1w 가 그 판정의 유일한 공백이다.

| 옵션 | 런 | GPU-h @R=100 | 채우는 것 |
|---|---|---|---|
| **① T1(online 가중 게이팅)만** ★ | 9 = `ARMS=flirds_gatew_v2` × 3위협 × 3seed | **~18** | §5.3 online P1w 행 |
| ② T1+T2 | 27 (T1 + 관찰자 + `t2_signw_flirds`) | ~50–60 | online + retrain 양 표 |

- **전제 충족**: 코드·테스트 커밋 완료(`ec6cbd5`) + H-14 사전등록 완료 → **실행만 남음**. 신규 arm 코드 0.
- **자리**: B200(HVP). ①은 관찰자 불요라 canonical `rundirs_llm` 에 직접 착지(arm 명이 신규라 충돌 0). R=100 이면 18 GPU-h 라 **세션 A·B 어느 쪽 꼬리에도 들어간다.**

## 6. 서버 파일

| 파일 | 담당 | HW |
|---|---|---|
| `REMAINING-b200.md` | G1 · L1 재실행 · G4c s0·s1 | B200 2세션 × 2 GPU |
| `REMAINING-slurm-YH.md` | 코드 C-a·C-b · G3 · G8 · G10 · G6 · G5 · G12 | 3090 → 48GB |
| `REMAINING-slurm-JW.md` | G2 · G9 (a)-오라클 | 3090 |
| `REMAINING-slurm-HJ.md` | L11 seed0·1 (R=100 재제출) | A6000 |
| `REMAINING-slurm-JB.md` | L11 seed2 + G4c seed2 | A6000 |

## 7. 완료 후 공통

1. rundir 커밋(push는 Yonghee) — 계정별 root 분리, `make_analysis.py` 가 dup-win 병합(`rundirs_llm_g4c` 포함).
2. 분석 재생성 → 축별 결과 페이지(`flirds-results-*`) → paper.
3. **스택 캐비엇**: Slurm(torch 2.11) vs canonical(B200 torch 2.12) — fidelity·recovery 는 recovery 정규화로 병치(mean|Δ|≤0.006). **`timing.json` 은 §5.5 cost 에 B200 실측만.**
4. **⚠ 표에 R=200 셀이 섞이지 않게** — `rounds` 가 config/IDENTITY 에 있으니 집계 시 확인할 것.
