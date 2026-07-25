# REMAINING — 전역 배분 인덱스 (2026-07-25 · **LLM downstream 스코프 컷** 확정판)

> **정본** = `research-wiki/survey/flirds-paper-experiment-plan.md`(07-25 확정 수록목록 + 결손 G1–G13).
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 기준시각 = **07-25 23:00**(잔여 ~73h).
> 규칙: 3-seed · **push는 Yonghee 직접** · 수치는 rundir/analysis 재생성 값만.

## 0. ★ 이번 결정 — LLM downstream 을 {vanilla · oracle · random · flirds류}로 축소

**결정(Yonghee 07-25).** R4 §5.3(LLM 개입)의 비교 대상을 **이미 계산이 끝난 것**으로 한정하고, 주장을 **"vanilla·random 보다 낫다"** 수준으로 타협한다.

**디스크 확인 결과 — 생각보다 많이 완결돼 있다** (`rundirs_llm`, R=200):

| threat | observer(vanilla) | oracle_excl | random_excl | flirds online | **t2_sign ×4**(flirds·1st·loss-heur·FedIF) |
|---|---|---|---|---|---|
| noisy | 3 | 3 | 3 | 3 | **각 3** |
| frzero | 3 | 3 | 3 | 3 | **각 3** |
| clean | 1 | — | — | 1 | 각 1 |

⟹ **retrain 표는 이미 "4 추정량 + 앵커, 3-seed" 완결**이다(flirds류만이 아니라 loss-heur·FedIF 포함). 비는 곳은 **① online 표가 flirds 단독** ② **clean 열이 seed0 뿐** 둘뿐이다.

**따라오는 결정 — R4 는 `R=200` 유지(R=100 전환 철회).** R=100 은 L11·G4c 를 마감에 넣으려던 조치였다. 그 둘이 빠진 지금 R=100 은 **이미 완결·커밋된 L1 3-seed 를 재실행하게 만들 뿐**이다(재실행 84 GPU-h + 데이터 손실 위험). 클라당 참여도 20회로 더 방어적이다.
> `rounds` 를 rundir IDENTITY 로 승격한 코드 변경(C-c)은 **유지**한다 — 앞으로 R 이 다른 셀이 한 표에 섞이면 실행 시점에 실패하게 만드는 안전장치이고, 지금 무대에는 영향이 없다.

## 1. 삭제된 작업 (재제안 금지)

| 폐기 | 사유 |
|---|---|
| **L11 online 7종 중 6종**(loss-heur·FedIF·renorm-4) | LLM downstream 스코프 컷. **flirds1st 만 살린다**(§3) |
| **G4c retrain renorm-4** 9셀 | 〃 — `sbatch_l4_renorm_t2.sh` 는 ⛔ 배너 달고 존치(마감 이동 시 부활용) |
| **L9 frrand** 전량 | 축 밖 위협 |
| **L10 · L5 · L6** | 축 밖 |
| **fmnist competition · c2fid fmnist** | fmnist 파티션 제외 → mnist 무대가 대체 |
| **W-B P1w obsf 잔여** | P1w 는 G3·G10 rundir 에 동반 산출(추가 런 0) |
| **전용 탐지기 4종 · Fed-LOO** | 집계에서만 제외(실행 0) |
| **forward 전용 val-chunk 분리(코드)** | `make_llm_loss` docstring: 이미 프로파일 **~1.0×**(FLOP-bound) |

**빠지는 renorm-4 의 근거 보전**: cross-game 붕괴는 ① **CNN §5.3**(8점수원 × online·retrain 양 표 3-seed 완비) ② **LLM §5.2 fidelity**(G1 이 9방법 φ 전량 산출)에서 그대로 시연된다. LLM **downstream 표**에만 빠지므로 "무대별 커버리지 차이" 각주로 정직하게 처리한다.

## 2. 잔여 물량 (R=200 기준)

| G | 작업 | 셀 | 단가 | GPU-h | 실행처 |
|---|---|---|---|---|---|
| **G1** | R4-L2 주무대 (b) 오라클 | 9 | **19.6h**(op-count×microbench) | **176** | **B200** |
| **L1c** | R4 clean 개입 seed1·2 | 4 | 1.9–5.5h(실측) | **19** | **B200** |
| **G5** | 2차항 LLM 레그 seed 보강(본문) | 4 | ~5h | **20** | **B200** |
| **G12** | A축 lever probe seed 보강(부록) | 19 | 2–4h | 50–90 | **B200 꼬리**(droppable) |
| **L11′** | **online Flirds-1st 만** | 9 | **~4.2h**(B200; A6000 은 7.4h) | **38** | **B200 c4** |
| **G3** | cifar10/iid 점수원 7종 + obs | 96 | ~0.4h·obs ~2h | ~80 | **YH**(3090) |
| **G8** | mnist 부분참여 fidelity(+탐지) | 24 | 1.05h(실측) | ~25 | **YH** |
| **G6** | Removal-curve CNN 오염축 | 9 | 추정 | ~15 | **YH** |
| **G10** | mnist downstream 8점수원 | 216 | 추정 | ~135 | **HJ**(3090) |
| **G2·G9** | cifar10·mnist vs (a) 오라클 | 48 | **9.1 / 11.4h**(실측) | **505** | **JW** 0-23 / **JB** 24-47 |
| **G7** | op-count N·R·K 파라메트릭 | 0 | 문서 | 0 | (집필) |

- **B200 ≈ 253 GPU-h**(LLM 전량) · **Slurm ≈ 747**(CNN 전량).
- **LLM 은 전부 B200 으로 모았다** — HVP·canonical timing 은 물론이고 flirds1st online 도 B200 이 **~1.6× 빠르다**(이 셀은 FL 학습이 지배: A6000 7.4h → B200 ~4.2h). 그 덕에 **Slurm 4계정을 전부 3090 에 붙일 수 있다**(3090 여유 21장 vs A6000 여유 10장·가동률 90%).

## 3. L11′ — online Flirds-1st 를 B200 으로 (Yonghee 07-25)

online 표가 flirds 단독이면 "vanilla·random 보다 낫다"까지만 말할 수 있다. **Flirds-1st online 9셀**을 얹으면 online 표가 retrain 표와 같은 계열로 맞물려 **§5.6① 2차항 주장이 online 에서도 성립**한다. **B200 세션 B 의 2번째 GPU(CID 4)** 가 맡는다 — `queue_b200_c4.txt`.

> ⚠ **HJ 에서 이미 착지한 flirds1st 셀은 그대로 유효하다**(같은 R=200 무대). HJ 의 실행 중 셀은 완주시키고(~7.4h), **착지 목록을 받아 `queue_b200_c4.txt` 의 해당 줄을 주석 처리**할 것 — root 가 달라 충돌은 없지만 중복 실행은 GPU 낭비다. HJ 의 나머지 array 원소(loss-heur·FedIF·renorm-4)는 스코프 밖이라 취소.

## 4. 코드 조치

| 코드 | 무엇 | 상태 |
|---|---|---|
| **C-c** | `rounds` 를 rundir IDENTITY 로 승격(`track_g`·`phase2_matrix`) | ✅ 적용(안전장치로 유지) |
| **C-a** | `track_c2.py:157` `MODEL_FN` 에 `"mnist": LeNet5` (**1줄**) | ⬚ **YH** — G8·G10 게이트 |
| **C-b** | `track_c1.py` 에 `C1_PARTITION`·`C1_THREAT`·`C1_FLIP_RATE` 도입 | ⬚ **YH** — **G2·G9 505 GPU-h 게이트 → 최우선** |

## 5. 실행처별 배분 · 예상 종료 (07-25 23:00 기준 · 잔여 73h)

**B200 = LLM 전량 (컨테이너 4개 · 세션 2개 × 2 GPU)**

| 세션 | CID | 담당 | wall |
|---|---|---|---|
| **A** | 1 | G1 noisy ×3 + L1 clean_online s1 | 60.7h |
| **A** | 2 | G1 clean ×3 + L1 clean_online s2 | 60.7h |
| **B** | 3 | G1 frzero ×3 + L1 clean_obs s1 | 64.3h |
| **B** | **4** | **L1 clean_obs s2 → online Flirds-1st ×9 → G5 ×4** | 63.5h |

**Slurm = CNN 전량 (4계정 × 8슬롯, 전부 3090)**

| 실행처 | 담당 | GPU-h | wall | 종료 |
|---|---|---|---|---|
| **YH** | **코드 C-b·C-a** → G3(80)·G8(25)·G6(15) | 120 | 15h + 코드 | **07-26 후반** |
| **HJ** | 실행 중 flirds1st 완주 → **G10** mnist downstream | 135 | 17h(+7h) | 07-27 오전 |
| **JW** | c1축 `--array=0-23` | 237 | 30h | 07-27 오전 |
| **JB** | c1축 `--array=24-47` | 255 | 32h | 07-27 후반 |

- **전체 = 07-28 오후**(B200 최장 64.3h) — 마감(07-28 24:00) 대비 **~9h 여유**.
- **임계경로는 B200 의 G1**(셀당 19.6h × 9). LLM downstream 을 줄여도 wall-clock 은 크게 안 줄어든다 — 줄어든 건 자원과 리스크다.
- **C-b 가 505 GPU-h(JW·JB)를, C-a 가 216런(HJ G10)을 막는다** → YH 의 첫 작업이 코드인 이유. 착지 지연분만큼 뒤가 밀린다.
- **3090 여유 21장**이라 4계정 32슬롯 중 ~21–24 가 물리적으로 확보된다. **A6000·4090 은 배정하지 않는다**(각각 여유 10장·0장).

## 6. 지금 할 조치 (순서)

1. **YH**: **C-b 코드**(JW·JB 505 GPU-h 게이트) + **C-a 1줄**(HJ G10 게이트) → 그다음 G3(코드 불요라 병행 가능).
2. **HJ**: `squeue` 로 R/PD 구분 → **스코프 밖 원소(loss-heur·FedIF·renorm-4) 취소**, **실행 중 flirds1st 는 완주**. 착지 목록을 B200 에 전달(c4 큐 주석용) → C-a 착지 후 3090 에서 **G10**.
3. **B200**: 세션 A = `CID=1`·`CID=2`, 세션 B = `CID=3`·`CID=4`. **c4 큐의 flirds1st 줄 중 HJ 착지분은 주석 처리**하고 가동.
4. **JW·JB**: C-b 착지 즉시 `sbatch_c1_axis.sh` — JW `--array=0-23%8`, JB `--array=24-47%8`. 둘 다 **cifar10(본문 G2) 먼저**.
5. **JB(L9)** 는 이미 전량 중단 확정 — 잔재만 확인.

## 7. 여유가 생기면 — 우선순위

| 순위 | 무엇 | 비용 | 왜 |
|---|---|---|---|
| 1 | **LLM P1w T1** 9런 | ~35 (B200) | CNN P1w 가 G3·G10 에 동반 산출되므로, LLM 레그만 얹으면 "전 범위 3-seed 승격" 판정이 성립. 큐 c4 에 주석으로 대기 |
| 2 | **online loss-heur·FedIF** 18런 | ~150 (A6000) | online 표를 retrain 표와 완전 대칭(4 추정량)으로 |
| 3 | G12 잔여 | — | 부록·최저 |
| — | ~~G4c renorm-4 retrain~~ | 900 | 마감이 07-30 이후로 밀릴 때만 |

## 8. 서버 파일

| 파일 | 담당 |
|---|---|
| `REMAINING-b200.md` | G1 · L1 clean 보강 · G5 · G12 |
| `REMAINING-slurm-YH.md` | 코드 C-a·C-b · G3 · G8 · G10 · G6 |
| `REMAINING-slurm-JW.md` | G2·G9 seed0·1 |
| `REMAINING-slurm-JB.md` | G2·G9 seed2 (LLM→CNN 전환) |
| `REMAINING-slurm-HJ.md` | L11′ online Flirds-1st → CNN work-steal |

## 9. 완료 후 공통

1. rundir 커밋(push는 Yonghee) → `make_analysis.py` 재생성 → `flirds-results-*` → paper.
2. **§5.3 LLM 표의 스코프를 캡션에 명시**: 비교 대상 = vanilla · oracle_excl · random_excl · Flirds · Flirds-1st(+retrain 에 loss-heur·FedIF). renorm-4 는 CNN §5.3 과 LLM §5.2 가 담당한다고 각주.
3. **스택 캐비엇**: Slurm(torch 2.11) vs canonical(B200 torch 2.12) — recovery 정규화로 병치(mean|Δ|≤0.006). **`timing.json` cost 는 B200 실측만**.
