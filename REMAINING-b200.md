# REMAINING (B200) — HVP 전용: 주무대 (b) 오라클 + R=100 재실행 + renorm-4 재학습

> 배분 정본 = **`REMAINING-00-INDEX.md`** · 수록목록 정본 = `research-wiki/survey/flirds-paper-experiment-plan.md`.
> **역할 = HVP(flirds 2차 φ, 95–106 GiB) 전용 + canonical timing + renorm-4 재학습**(A6000 대비 3.6× 빠름).
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 3-seed. push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만.

## 0. ★ R4 = **R=100** (2026-07-25)

R4(gsm50k5)의 라운드 수만 200 → 100. 다른 무대는 불변. 근거·판단은 INDEX §0.
**여기서 실무적으로 중요한 것 두 가지**:

1. **`rounds` 가 rundir IDENTITY 필드로 승격됐다**(코드 적용 완료). 이름에 R 이 없어서 승격 전에는 R=100 실행이 R=200 canonical 을 **경고 없이 덮었다**. 이제 덮으려면 **`RUNDIR_REPLACE=1`** 이 필요하고 **큐 전 줄에 박혀 있다**.
2. **L1(§2)은 R=200 으로 이미 완주·커밋된 셀의 재실행**이다. R=200 산출은 무대 밖 → superseded(git 이력에 보존).

## 1. 환경 · 세션 구성

**B200 = 세션 2개 × GPU 2장, 컨테이너는 GPU마다 별개** → **컨테이너당 큐 1개**(총 4개).

```
CID=1 bash <repo>/runs/track_h/run_b200_batch.sh     # 세션 A 컨테이너 1
CID=2 …                                              # 세션 A 컨테이너 2
CID=3 …                                              # 세션 B 컨테이너 1
CID=4 …                                              # 세션 B 컨테이너 2
```

| 세션 | CID | 담당 | GPU-h | wall |
|---|---|---|---|---|
| **A** | 1·2 | **G1**(주무대 (b) 오라클 9셀) + G4c seed1 | 129 | 65h |
| **B** | 3·4 | **L1 R=100 재실행** 18셀 + G4c seed0 | 126 | 63h |

- **두 세션은 서로 의존하지 않는다.** 한쪽이 늦게 잡혀도 다른 쪽 산출물은 그대로 쓴다 — 세션 A만 살면 §5.2/§5.4/§5.5 가, 세션 B만 살면 §5.3 이 각각 자립한다.
- 런처가 env·sed·스모크를 내장(`REPO`/`BATCH` 기본값은 파일 상단 확인). `PY=$BATCH/venv/bin/python`(**torch 2.12.0+cu130 = canonical**), `HOME=$BATCH/home`, `HF_HOME=$BATCH/hf_home`, offline 플래그. 모델 캐시 검증 = `$BATCH/PROVENANCE.md`.

### 드라이버 운용 (하드룰)

- **큐 정지는 줄 삭제가 아니라 `#` 주석.** 드라이버가 `consumed` **인덱스**로 위치를 추적한다 — 줄을 지우면 밀린다.
- **러너는 셀 단위 원자적** — 중도 kill = 그 셀 전손. 정지는 드레인.
- **마지막 셀은 `done[ok]` 줄이 안 남는다** → 완주 판정은 `MATRIX DONE` / `TRACK G DONE` + rundir mtime.
- **영속 단위**: `phase2_matrix`(G1) = **cell-end 1회** → 중도 컷 = 전손. `track_g`(L1·G4c) = **arm 단위** → 완료 arm 생존. 컨테이너 컷이 예상되면 G1 신규 투입부터 끊는다.

## 2. G1 — R4-L2 주무대 (b) 오라클 (세션 A · 9셀 · 87 GPU-h)

> **논문 최대 병목·대체 불가.** 이 9셀 하나가 **세 축**을 동시에 연다: §5.2 LLM fidelity(본문) · §5.4 LLM 탐지(부록) · §5.5 **canonical timing**. 축-지도에서 LLM fidelity·탐지가 ⬚로 남은 유일한 칸이고, 로컬 `phase2_matrix/rundirs` 에 `1B_gsm50k5_*` **0개**로 미실행 확인됨.

- `phase2_matrix.py REGIME=gsm50k5`(Llama-3.2-1B · N=50 · 5/50 · **R=100** · GSM8K) — (b) per-round 2⁵ exact + 9방법 φ + `timing.json`.
- **셀**: {clean, noisy, freerider_zero} × seed{0,1,2} = **9**. 코드 변경 불필요.
- **⚠ THREAT 토큰**: `phase2_matrix` 는 **`freerider_zero`**(rundir 이름만 `frzero` 로 축약). `track_g` 의 `frzero` 와 **다른 토큰**. `NOISY_RATE` 는 gsm50k5 기본 0.7 이라 생략.
- **단가 ~9.7h/셀**(op-count 177.5 fwd-등가/round × 100 × microbench 1.60 s + FL ~1.8h). 큐 배치상 **3위협 seed0 이 ~19h 안에 착지**한다.
- **완료 판정**: `MATRIX DONE` + rundir `1B_gsm50k5_{clean,noisy_nr0.7,frzero}_s{seed}`.

## 3. L1 — R4 flirds 개입 R=100 재실행 (세션 B · 18셀 · 84 GPU-h)

> §5.3 의 flirds 열 + same-game retrain 4행 + **recovery 분모(vanilla·oracle_excl·random_excl)** 가 전부 여기서 나온다. **L11(HJ·JB)의 분모이기도 해서, 이게 착지해야 L11 559 GPU-h 가 분석 가능해진다.**

- 셀 = {noisy, frzero} × {obs_t2, online} × 3seed + {clean} × {obs, online} × 3seed = **18**.
- **단가(R=200 실측 ÷2)**: noisy obs_t2 9.0h · frzero obs_t2 5.0h · noisy online 4.8h · frzero online 4.4h · clean obs 2.8h · clean online 1.9h.
- clean 은 kept=전원이라 T2 가 `equals_vanilla` 로 스킵 → 저비용. clean online 에선 `oracle_excl`/`random_excl` 가 자동으로 빠진다(제외 대상 부재; `track_g` l.618).

## 4. G4c — R4 retrain renorm-4 (seed0·seed1 = B200 · 6셀 · 83 GPU-h)

> **R=100 으로 부활한 레그.** R=200 에선 ~100h/셀(9셀 900 GPU-h)이라 배제했는데, R=100 에서 B200 은 **~13.9h/셀**이다. §5.3 **retrain 표의 renorm-4 4칸**을 채운다(나머지 5행 = same-game 계열은 L1 이 담당).

- **왜 B200 이 2/3 를 가져가나**: 관찰자가 매 라운드 renorm-4 를 채점하고 그게 셀 비용의 **~92%** → B200 13.9h vs A6000 ~50h = **3.6×**. seed2 만 JB(A6000).
- **⚠ 별도 root `rundirs_llm_g4c`**: 이 셀의 `observer` arm 은 L1 의 same-game observer 와 **rundir 이름이 같다**(`regime_threat_arm_seed`) → 같은 root 면 서로 덮어쓴다. `make_analysis` 로더에 추가 완료.
- **⚠ `VAL_CHUNK` 를 낮추지 않는다**: renorm-4 는 forward-only(@no_grad)라 grad 경로의 OOM 가드가 적용되지 않는다. 청크 합산은 exact → φ 동일.

## 5. (옵션) LLM P1w — 여유 시 최우선 추가

- **①T1(online 가중 게이팅)만 = 9런 · R=100 이면 ~18 GPU-h** → 세션 A·B 어느 꼬리에도 들어간다. 관찰자 불요라 canonical `rundirs_llm` 에 직접 착지(arm 명이 신규라 충돌 0).
- ②T1+T2 = 27런 ~50–60 GPU-h.
- 전제 충족: 코드·테스트 커밋(`ec6cbd5`) + H-14 사전등록 완료 → **실행만 남음**. 근거·수록 규칙은 INDEX §5a.

## 6. 우선순위

| P | 무엇 | 세션 | 근거 |
|---|---|---|---|
| **P0** | G1 seed0 (3위협) | A | 비어 있는 축(LLM fidelity·탐지·canonical timing)을 **여는** 유일한 셀 |
| **P0** | L1 noisy·frzero 재실행 | B | L11 559 GPU-h 의 recovery 분모 |
| **P1** | G1 seed1·2 · L1 clean | A·B | 3-seed 규칙 · §5.3 clean 열 |
| **P2** | G4c seed0·1 | B·A | §5.3 retrain 표의 빈 4칸 |
| **P3** | (옵션) P1w T1 | 여유 | INDEX §5a |

## 7. 완료 후

1. rundir 커밋(push는 Yonghee).
2. `runs/track_h/make_analysis.py` + `runs/phase2_matrix/make_analysis.py` 재생성 → `flirds-results-{fidelity,detection,downstream,cost}` → paper.
3. **canonical timing 은 이 B200 산출만** 쓴다(§5.5). Slurm(torch 2.11) `timing.json` 은 cost 표 금지.
4. **⚠ 집계 시 `rounds` 확인** — R=200 잔재가 표에 섞이지 않게(이제 config·IDENTITY 에 기록된다).
