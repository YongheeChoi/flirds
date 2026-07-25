# REMAINING (B200) — HVP 전용: LLM 주무대 fidelity·탐지 + clean 개입 + probe 보강

> 배분 정본 = **`REMAINING-00-INDEX.md`** · 수록목록 정본 = `research-wiki/survey/flirds-paper-experiment-plan.md`.
> **역할 = HVP(flirds 2차 φ, 95–106 GiB) 전용 + canonical timing.** 48GB 카드로는 기본 knob 불가 → 아래 작업은 **B200 아니면 못 한다**.
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 전 실험 3-seed(seed-major). push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만. 기존 rundir read-only.

## 0. 환경 · 구성 (2026-07-25 현재)

**B200 4장 = 컨테이너 4개(각 GPU 1장).** 4-GPU 단일 드라이버가 아니라 **컨테이너마다 별도 큐 1개**로 돈다.

```
CID=1 bash <repo>/runs/track_h/run_b200_batch.sh     # 컨테이너 1..4 에서 CID만 바꿔 제출
```

- 런처가 env·sed·스모크를 전부 내장(`REPO`/`BATCH` 기본값은 파일 상단에서 확인·수정).
- `BATCH=…/flirds_batch` 기준: `PY=$BATCH/venv/bin/python`(**torch 2.12.0+cu130 = canonical**), `HOME=$BATCH/home`, `HF_HOME=$BATCH/hf_home`, `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.
- venv는 기존 rundir `meta.json`과 버전 고정(transformers 5.9.0 · trl 1.5.1 · peft 0.19.1 · accelerate 1.13.0 · datasets 4.8.5 · numpy 2.4.6). 모델 캐시 검증 체인 = `$BATCH/PROVENANCE.md`.

### 컨테이너 ↔ 큐

| CID | 레인 | 큐 | 물량 |
|---|---|---|---|
| 1 | **noisy** | `runs/track_h/queue_b200_c1.txt` | G1 noisy ×3 → G12-A ×6 |
| 2 | **clean** | `queue_b200_c2.txt` | G1 clean ×3 → G12-B ×6 |
| 3 | **free-rider-zero** | `queue_b200_c3.txt` | G1 frzero ×3 → G12-C ×6 |
| 4 | **개입·probe** | `queue_b200_c4.txt` | L1-clean ×4 → G5 ×4 |

### 드라이버 운용 (하드룰)

- **큐 정지는 줄 삭제가 아니라 `#` 주석.** 드라이버는 매 루프 큐를 다시 읽고 `consumed` **인덱스**로 위치를 추적한다 — 줄을 지우면 인덱스가 밀려 오배치.
- **러너는 셀 단위 원자적** — 중도 kill = 그 셀 전손. 정지는 드레인(실행 중 셀 완주 후 종료).
- **마지막 셀은 `done[ok]` 줄이 안 남는다** → 완주 판정은 셀 로그의 `MATRIX DONE` / `TRACK G DONE` + rundir mtime 교차 확인.
- **영속 단위 차이**: `phase2_matrix`(G1) = **cell-end 1회 persist** → 중도 컷 = 전손(직전 컨테이너 실사례). `track_g`(L1-clean) = **arm 단위 영속** → 컷에도 완료 arm 생존. 컨테이너 컷이 예상되면 G1 신규 투입을 먼저 끊는다.

## 1. G1 — R4-L2 주무대 (b) 오라클 ★최우선

> **논문 최대 병목·대체 불가.** 이 9셀 하나가 **세 축**을 동시에 연다: §5.2 LLM fidelity(본문) · §5.4 LLM 탐지(부록) · §5.5 **canonical timing**. 축-지도에서 LLM fidelity·탐지가 ⬚로 남은 유일한 칸이고, 로컬 `phase2_matrix/rundirs` 에 `1B_gsm50k5_*` **0개**로 미실행 확인됨.

- **무엇**: `phase2_matrix.py REGIME=gsm50k5`(Llama-3.2-1B · N=50 · 5/50 · R=200 · GSM8K) — (b) per-round 2⁵ exact + 9방법 φ + `timing.json`.
- **셀**: {clean, noisy, freerider_zero} × seed{0,1,2} = **9**. 코드 변경 불필요(regime 구현 완료).
- **THREAT 토큰 주의**: `phase2_matrix` 는 **`freerider_zero`**(rundir 이름만 `frzero` 로 축약). `track_g` 의 `frzero` 와 **다른 토큰**이다. `NOISY_RATE` 는 gsm50k5 에서 0.7 이 기본값이라 생략.
- **비용**: **≥7h/셀**(FL 학습 ~2h + valuation ≥5h; 상한 미확정) → 9셀 **63–90 GPU-h**. 3 컨테이너 병렬 → **seed0 3셀 = 07-26 오전**, 전체 ~07-27.
- **완료 판정**: 셀 로그 `MATRIX DONE` + rundir `1B_gsm50k5_{clean,noisy_nr0.7,frzero}_s{seed}`.
- **채우는 것**: 계획서 §2.1 1A-LLM · §3.5 LLM 주무대 탐지 · 부록 D.2 `LLM-Main ⬚`.

## 2. L1-clean — R4 clean 개입 seed1·2

> §5.3 **clean 열**(무해성 parity·오발화 대조)의 3-seed 정본이 seed0 밖에 없다. clean 개입 EM 은 `track_g THREAT=clean` 에서만 나온다 — L2(=phase2_matrix)는 fidelity·탐지만 산출하고 게이트 arm 을 만들지 않는다.

- **셀 4개**(CID=4 큐): `clean_obs` seed1·2 + `clean_online` seed1·2. seed0 는 완료·커밋 → **재실행 금지**.
- **비용 실측**: `clean_obs` **5.5h** · `clean_online` **3.8h** → 계 **18.6 GPU-h**.
- clean 은 kept=전원이라 T2 가 `equals_vanilla` 로 스킵된다 → 실질 신규 = online 게이트 arm = 저비용.

## 3. G5 — 2차항(HVP) LLM 레그 seed 보강 (본문 ablation §5.6①)

- **셀 4개**(CID=4 큐): `1B_std50k5_r{32,64}_seed{1,2}`. r16 은 이미 3-seed.
- 러너 `track_d.py`, `REGIME=std20 N_CLIENTS=50 K_ABS=5 LORA_R=<r> ORACLE_A=0`, 착지 `runs/probe_signal/rundirs`.
- **왜 본문**: std50k5(5/50 부분참여)에서 Flirds +1.000 vs 1차계열 음수 붕괴 = 2차항 존재 이유의 LLM 레그. rank 축이 seed0 뿐이라 3-seed 규칙 미달.

## 4. G12 — A축 lever probe seed 보강 (부록 P2 · 최저 우선)

- **19셀**(C1·C2·C3 큐 꼬리에 6/6/6 분산). anchor5 lr·steps 스윕 seed1·2 + anchor5 rank r32·r64 seed1·2 + `noise_1B_r64` seed1·2.
- **핵심 미확인 질문** = "lr 로 커진 φ가 cross-seed 실재 신호인가"(현 예측 ρ≈0) → 그 검증에 필요한 건 `lr{2,3}e-3` 계열 seed1·2.
- `1B_anchor5_lr1e-3_st10` 기준칸의 seed0 은 기존 `1B_anchor5_seed0` 로 **대체(별칭)** — 유효하면 실행 불요(C4 큐에 주석으로 대기).

## 5. (옵션) LLM P1w — 여유 시 최우선 추가

> Yonghee 07-25: **추가 실험을 한다면 이것부터.** CNN P1w 는 YH·JW 의 G3·G10 rundir 에 **동반 산출**되어 자동 완비되므로, LLM 레그만 얹으면 P1w 의 "CNN·LLM 전 범위 3-seed 승격" 판정이 성립한다. 지금 LLM P1w 가 그 판정의 유일한 공백.

| 옵션 | 런 | 단가 | GPU-h |
|---|---|---|---|
| **① T1(online 가중 게이팅)만** ★ | **9** = `ARMS=flirds_gatew_v2` × 3위협 × 3seed | **3.7–3.9 GPU-h/런** (실측: seed2 `flirds_gate_v2` gpu_h 3.71·3.86) | **~35** |
| ② T1+T2 | 27 = 9 T1 + 9 관찰자 + 9 `t2_signw_flirds` | 관찰자 ~4.5h + T2 ~3h | ~100–120 |

- **전제 충족**: 코드·테스트 커밋 완료(`ec6cbd5`) + **H-14 사전등록 완료** → 실행만 남음. 신규 arm 코드 0(`build_arm` 의 `_gatew_v2` 분기 기존).
- **①은 관찰자 불요** → canonical `rundirs_llm` 에 직접 착지(P1w arm 명이 신규라 충돌 0). 큐 C1·C2·C3 상단에 **주석 3줄씩** 준비돼 있다 — 해제하고 같은 큐의 G12 6줄을 주석 처리하면 맞교환(≈12h ↔ 12–24h).
- **판정**: `make_analysis.parse_arm` 이 `gatew_v2`→policy P2 로 매핑 → CNN P1w 와 같은 정책으로 묶여 승격 규칙 자동 대조.

## 6. 우선순위 · 예상 종료

| P | 무엇 | 셀 | GPU-h | 근거 |
|---|---|---|---|---|
| **P0** | G1 seed0 (3 레인 동시) | 3 | 21–30 | 비어 있는 축(LLM fidelity·탐지)을 **여는** 유일한 셀 |
| **P1** | G1 seed1·2 | 6 | 42–60 | 3-seed 규칙 |
| **P2** | L1-clean seed1·2 | 4 | 18.6 | §5.3 clean 열 유일 소스 · arm 영속이라 컷에 강함 |
| **P3** | G5 | 4 | 20–40 | 본문 ablation seed 미달 |
| **P4** | G12 (또는 **P1w 와 맞교환**) | 19 | 50–90 | 부록 · 최저 우선 |

- **총 150–240 GPU-h** vs 4 컨테이너 × 72h = 288 → 들어간다.
- **예상 종료**: **G1 seed0 = 07-26 오전** · G1 전량 = 07-27 · 전체 = **07-27~28**.

## 7. 완료 후

1. rundir 커밋(push는 Yonghee).
2. `runs/track_h/make_analysis.py` + `runs/phase2_matrix/make_analysis.py` 재생성 → 축별 결과 페이지(`flirds-results-fidelity`·`-detection`·`-cost`) 갱신 → paper.
3. **canonical timing 은 이 B200 산출만** 쓴다(§5.5). Slurm(torch 2.11) 산출의 `timing.json` 은 cost 표에 금지.
