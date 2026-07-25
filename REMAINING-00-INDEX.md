# REMAINING — 전역 배분 인덱스 (2026-07-25 전면 개편)

> **정본** = `research-wiki/survey/flirds-paper-experiment-plan.md`(Yonghee 07-25 확정 수록목록 + 결손 G1–G13).
> 이 인덱스는 그 계획의 **결손(G) → 실행처** 배정과 예상 종료만 담는다. 실행 절차는 각 서버 파일.
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 기준시각 = **07-25 20:00**(잔여 ~72h).
> 규칙: 전 실험 **3-seed**(seed-major = seed0 먼저) · **push는 Yonghee 직접** · 수치는 rundir/analysis 재생성 값만 · 기존 rundir read-only.
> **운용 원칙(07-25 Yonghee): 지금 돌고 있는 잡은 죽이지 않는다.** 회수는 **대기(PD) 상태**와 **미제출분**에서만 한다.

## 0. 이번 개편으로 **삭제된** 작업 (재제안 금지)

계획서 §0.1 오염축 5종({CNN: lf@0.70, frzero, grad-noise} / {LLM: swap@0.7, frzero}) + §0.2 비교군 9행 확정으로 아래가 스코프 밖이 됐다. **rundir·러너 산출은 존속, 표에서만 뺀다.**

| 폐기 | 있던 곳 | 사유 |
|---|---|---|
| **L9 frrand**(비-flirds arms + flirds 레그) | JB 전량 · B200 P3 | frrand = 축 밖 위협 → **07-25 전량 중단 확정**(실행 중 포함; ~580 GPU-h 회수) |
| **L7 P1w-LLM** 18런 | B200 P4 | 계획서에 LLM P1w 항목 없음(P1w는 CNN 부록만) |
| **L10 strmain-dose · L5 비등n · L6 graded-noisy** | B200 §2·§3 | 축 밖 |
| **fmnist competition 288런 · c2fid fmnist** | YH §3.5 | fmnist 파티션 제외 → **mnist 무대로 대체** |
| **W-B P1w obsf 잔여**(비-dir1 obsf) | YH §2 | P1w는 G3·G10 rundir에 **동반 산출**(추가 런 0) |
| **전용 탐지기 4종 · Fed-LOO** | 전 무대 | 집계에서만 제외(실행 0) |

## 1. 배정표 — 결손 G × 실행처

| G | 작업 | 런 | 단가(근거) | GPU-h | 실행처 | HW |
|---|---|---|---|---|---|---|
| **G1** | R4-L2 주무대 (b) 오라클 | 9 | ≥7h(측정 대기) | **63–90** | **B200** | HVP 95–106GiB = B200 전용 |
| **L1c** | R4 clean flirds 개입 seed1·2 | 4 | 3.8–5.5h(실측) | **18.6** | **B200** | HVP |
| **G5** | 2차항 LLM 레그 seed 보강 | 4 | 추정 5–10h | ~20–40 | **B200** | HVP |
| **G12** | A축 lever probe LLM seed 보강 | 23 | 추정 2–4h | ~50–90 | **B200** | HVP |
| **G4a** | R4 online same-game+FedIF 3종 | 27 | **7.4–9.6h**(HJ **재측정** 07-25 19:00) | **~217** | **HJ** s0·1 / **JB** s2 | A6000 |
| **G4b** | R4 online renorm-4 | 36 | 23–28h(JB 실측) | **~900** | **HJ** s0·1 / **JB** s2 | A6000 |
| ~~G4c~~ | ~~R4 retrain renorm-4 T2~~ | ~~9~~ | ~100h/셀(JW 실측) | ~~900~~ | **미실행**(§4) | — |
| **G3** | cifar10/iid 점수원 7종 + obs | 96 | 소스 ~0.4h·obs ~2h(추정) | **~60–100** | **YH** | 3090 |
| **G8** | mnist 부분참여 fidelity(+탐지) | 24 | 1.05h(c2fid 실측) | **~25** | **YH** | 3090 |
| **G10** | mnist downstream 8점수원 | 216 | 〃(추정) | **~110–160** | **YH** | 3090 |
| **G6** | Removal-curve CNN 오염축 | 9 | 추정 1–2h | ~10–20 | **YH** | 3090 |
| **G2** | cifar10 vs (a) 오염축 정렬 | 24 | **9.1h**(c1_oracle 실측 32,808s) | **~225** | **JW**(LLM→CNN 전환) | 3090 |
| **G9** | mnist vs (a) | 24 | **11.4h**(41,168s) | **~280** | **JW** | 3090 |
| **G7** | op-count N·R·K 파라메트릭 | 0 | 문서 | 0 | (집필) | — |
| G13 | *(선택)* loss-heur 3B/7B 재측정 | 12 | — | — | 미배정 | P3 |

- **LLM ≈ 1,240 GPU-h**(G4c 제외 후) · **CNN ≈ 710–810 GPU-h**.
- **⚠ G4a 단가 정정(07-25 19:00, HJ)**: 종전 `2.5–3.2h` 는 HJ 오측정 — 8셀 동시 가동분 실측이 **flirds1st·fedif 7.4h · lossheur 9.3h**(±2%)로 **~3×**. G4a 소계 80→**217**, HJ 총량 654→**700–820**(§4 갱신). renorm-4(G4b) 23–28h 는 JB 실측이라 그대로 두되 **미검증**이며, 같은 방향으로 어긋날 여지가 있다.
- **LLM은 24GB 불가**(1B full-SFT peak 26.3GiB 실측 · retrain-scoring 32GiB) → **A6000/B200 전용**. CNN은 전량 3090(여유 21장).

## 2. 서버 재배치 요약 (07-25 상황 반영)

| 서버 | 종전 | **개편 후** | 조치 |
|---|---|---|---|
| **B200** | L1/L2/L7/L9-flirds | **G1 · L1-clean · G5 · G12** | 큐 전면 교체(컨테이너 4개 = 각 1 GPU → 큐 4분할) |
| **YH** | CNN 완주(전부 done) | **G3 · G8 · G10 · G6** + 코드 C-a·C-b | 신규 착수 |
| **JW** | L4(G4c) — 시간부족 배제 | **G2 · G9** = CNN (a)-오라클 | **A6000 → 3090 전환**(같은 torch 2.11 env + torchvision 데이터만 추가) |
| **HJ** | L11 seed0·1 42셀 제출됨 | **유지**(=G4a+G4b 의 s0·1) | 손대지 않음. seed-major라 seed0부터 착지 |
| **JB** | L9 frrand 24셀 제출됨 | **frrand 전량 중단 → L11 seed2 21셀 인수** + HJ 꼬리 work-steal | `scancel` 후 §JB §2 로 전환 |

> **핵심 재배치 3건**:
> ① **JW 8슬롯을 CNN으로** — LLM에서 빼도 A6000 총량은 손해가 없다(어차피 클러스터 여유 10장이라 HJ·JB가 그 대역을 다 쓴다). 반면 **3090 21장이 놀고 있었다** → 놀던 자원이 CNN 최대 물량(505 GPU-h)을 흡수한다.
> ② **JB 를 frrand 에서 L11 seed2 로** — 같은 A6000·같은 sbatch·arm-level idempotent라 셋업 마찰 0.
> ③ **⚠ 발견한 구멍: L11 seed2 21셀이 무주공산이었다.** L11 63셀은 seed 로 쪼개져 HJ가 s0·1(42셀)만 제출했고, **YH 의 seed2 몫은 취소·미제출**. 그대로 두면 §5.3 online 표 7행이 **2-seed** 로 끝난다(3-seed 규칙 위반) → JB 가 인수한다.

## 3. 선행 코드 변경 2건 (CNN 착수 게이트)

| 코드 | 무엇 | 막는 것 | 크기 |
|---|---|---|---|
| **C-a** | `codes/experiments/track_c2.py:157` `MODEL_FN` 맵에 `"mnist": LeNet5` 추가 | **G8·G10** | **1줄** — `flirds/data/cnn.py`에 mnist 로더·정규화 이미 존재(l.11·24), `track_c2_fid.py`는 `c2.MODEL_FN` 참조라 자동 파급 |
| **C-b** | `codes/experiments/track_c1.py` `C1_SCENARIO`에 `free_rider`·`grad_noise` 추가 + 파티션 축(`dir1`) | **G2·G9**(+G6 공유) | 중 — 현 SCENARIO = iid\|label_skew\|quantity_skew\|label_flip\|feature_noise (l.79) |

> **C-b가 CNN 최대 물량(G2+G9 = 505 GPU-h)을 막는다** → 최우선. C-a는 1줄.

## 4. 예상 종료 (07-25 20:00 기준)

| 서버 | 슬롯 | 물량 | wall | 예상 종료 |
|---|---|---|---|---|
| **B200** | 4 (컨테이너 각 1 GPU) | G1·L1c·G5·G12 = 150–240 GPU-h | 38–60h | **G1 seed0 = 07-26 오전** · 전체 **07-27~28** |
| **YH** | 8 (3090) | G3·G8·G10·G6 = 205–305 | 26–38h | **07-27 오전** |
| **JW** | 8 (3090) | G2·G9 = 505 | 63h | **07-28 오후** ⚠ C-b 착지만큼 밀림 |
| **HJ** | 8 (A6000) | L11 s0·1 = **~700–820**(재측정) | **88–103h** | **07-29 오전~07-30** ⚠⚠ |
| **JB** | 8 (A6000) | L11 s2 = ~326 (+ HJ 꼬리) | 41h | **07-27** → 이후 HJ 흡수 |

- **A6000 실가용이 8슬롯보다 작다** — 클러스터 여유 10장/98(가동률 90%) → HJ·JB 합산이 상시 16슬롯을 못 채운다. 위 wall은 낙관치.
- **3090은 여유 21장** → YH·JW 16슬롯이 물리적으로 확보된다(가동률 89%지만 여유 절대수가 충분).
- **4090(0장 여유·가동률 100%)은 배정하지 않는다.**

## 5. ⚠ 스코프 결손 2건 (기록·판단)

**(1) G4c = R4 retrain renorm-4 미실행** (= 종전 "L4", JW 담당이던 것)

> **G4c 가 뭔가**: §5.3 LLM 개입은 CNN 과 마찬가지로 **표가 두 개**다 — ① **online**(학습 중 매 라운드 φ 부호로 배포 게이팅) ② **retrain**(학습을 끝내고 φ>0 클라만 남겨 **처음부터 재학습**). 각 표가 8–9행(점수원)이다.
> · online 표 = flirds(B200 L1 ✅) + 7 비-flirds(**L11** = HJ·JB) → 채워진다.
> · retrain 표 = same-game 4종(Flirds·Flirds-1st·loss-heur·FedIF)은 **L1 에서 3-seed 완료** + **renorm-4(GTG·FedSV·ComFedSV·ShapleyFL) 4칸 = G4c**. 이 4칸만 비는 것이다.

JW 시간부족으로 배제 확정(07-25 Yonghee). 실측 셀당 **~100h**(관찰자 arm 이 라운드마다 renorm-4 부분집합을 평가 → same-game 대비 라운드당 ~13×; clean 21.7분/R·noisy 19.7·frzero 29.4 × R=200 = 66–98h + T2 재학습 4arm ~24h) → 9셀 **~900 GPU-h** = A6000 으로 마감 내 불가.
- **결과**: §5.3 **retrain 표의 renorm-4 4칸이 공백**으로 남는다. retrain 표의 나머지 5행(Flirds·Flirds-1st·loss-heur·FedIF의 exact-0 계열)은 L1으로 이미 3-seed 확보.
- **완화**: renorm 붕괴는 ① **CNN §5.3**(8점수원 × online·retrain 양 표 3-seed 완비) ② **LLM §5.2 fidelity**(G1이 9방법 전량 산출)로 이미 시연된다 → LLM downstream retrain 칸의 공백은 "무대별 커버리지 차이"로 각주 처리 가능.
- **되살리려면**: 마감을 07-30으로 밀거나 A6000 여유가 20장 이상으로 회복돼야 한다.

**(2) G4b(online renorm-4) 3-seed 완주가 마감에 걸린다** — 36셀 × 23–28h = ~900 GPU-h.
- **clean 컷은 하지 않는다 (2026-07-25 Yonghee: "clean 은 필수").** ~300 GPU-h 를 아낄 수 있었으나, renorm 의 clean 칸은 **오발화(false-firing) 결과 그 자체**다 — flirds 와 달리 renorm 은 clean 에서도 음수 φ 로 발화해 `equals_vanilla` 스킵이 안 되고 실제 개입이 일어난다.
- **대신 회수한 것**: JB 의 **L9 frrand 전량 중단**(~580 GPU-h · 8슬롯) → 그 슬롯이 L11 seed2 와 HJ 꼬리를 흡수한다.
- **판정 시점**: **07-27 아침** — seed0 착지 + seeds1·2 진척으로 완주 여부 결정.

## 5a. 여유가 생기면 최우선 추가 = **LLM P1w**(가중 게이팅) — 규모 가늠

> Yonghee 07-25: "추가 실험을 할 경우의 **최우선 옵션**". 계획서 본표에는 없지만(부록 P1w는 CNN만) **CNN 쪽 P1w가 G3·G10 rundir에 동반 산출되면서 자동 완비**되므로, LLM 레그만 얹으면 **P1w의 "CNN·LLM 전 범위 3-seed 승격" 판정이 성립**한다. 지금 LLM P1w가 그 판정의 유일한 공백이다.

| 옵션 | 셀·런 | 단가(근거) | GPU-h | 채우는 것 |
|---|---|---|---|---|
| **① T1만 (online 가중 게이팅)** ★권고 | **9런** = `ARMS=flirds_gatew_v2` × {clean,noisy,frzero} × 3seed | **3.7–3.9 GPU-h/런**(B200 실측 `flirds_gate_v2` seed2 gpu_h 3.71·3.86) | **~35** | §5.3 online P1w 행 |
| **② T1+T2 (관찰자 동반)** | 27런 = 9 T1 + 9 관찰자 + 9 `t2_signw_flirds` | 관찰자 ~4.5h + T2 재학습 ~3h | **~100–120** | online + retrain 양 표 |

- **전제 충족**: 코드·테스트 **커밋 완료**(`ec6cbd5`) + **H-14 사전등록 완료** → **실행만 남음**. 신규 arm 코드 0(`build_arm`의 `_gatew_v2` 분기 기존).
- **자리**: **B200 전용**(flirds 스코어 = HVP 95–106 GiB). ①은 관찰자 불요라 `rundirs_llm`에 직접 착지 가능(P1w arm명이 신규라 충돌 0) — 별도 root 불필요.
- **어디서 빼나**: B200 용량 288 GPU-h(4컨테이너×72h) 대비 현 배정 150–240 → **①(35)은 `G12`(부록 P2·50–90)와 맞교환하면 그대로 들어간다.** ②(100–120)는 G12 전량 + G5 일부를 밀어야 하므로 마감 연장 시에만.
- **판정**: `make_analysis.parse_arm`이 `gatew_v2`→policy P2로 매핑 → CNN P1w와 같은 정책으로 묶여 승격 규칙(전 범위 승/동률/미수록) 자동 대조.

## 6. 서버 파일

| 파일 | 담당 | HW |
|---|---|---|
| `REMAINING-b200.md` | G1 · L1-clean · G5 · G12 | B200 ×4 컨테이너(각 1 GPU) |
| `REMAINING-slurm-YH.md` | G3 · G8 · G10 · G6 + 코드 C-a·C-b | 3090 |
| `REMAINING-slurm-JW.md` | G2 · G9 (a)-오라클 | 3090 (LLM→CNN 전환) |
| `REMAINING-slurm-HJ.md` | G4a · G4b (진행 중 유지) | A6000 |
| `REMAINING-slurm-JB.md` | L9 실행분 완주 → L11 work-steal | A6000 |

## 7. 완료 후 공통 절차

1. rundir 커밋(push는 Yonghee) — 계정별 root 분리(`rundirs_llm_{hj,jb}` 등), `make_analysis.py`가 dup-win 병합.
2. 분석 재생성 → 축별 결과 페이지(`flirds-results-*`) 갱신 → paper.
3. **스택 캐비엇**: Slurm(torch 2.11) vs canonical(B200 torch 2.12) — fidelity·recovery는 stack-robust(mean|Δ|≤0.006)로 병치하되 **`timing.json`은 §5.5 cost에 쓰지 않는다**(B200 실측만).
