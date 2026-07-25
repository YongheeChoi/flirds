# REMAINING — 전역 배분 인덱스 (2026-07-25 · **LLM downstream 스코프 컷** 확정판)

> **정본** = `research-wiki/survey/flirds-paper-experiment-plan.md`(07-25 확정 수록목록 + 결손 G1–G13).
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 기준시각 = **07-25 23:00**(잔여 ~73h).
> 규칙: 3-seed · **push는 Yonghee 직접** · 수치는 rundir/analysis 재생성 값만.

## 0′. 현장 상태 (07-25 22시 · 4계정 보고 + 코드 대조)

**✅ 블로커 해소 — C-a·C-b 가 `origin/main` 에 착지했다**(`d09e528`, 07-25). 로컬·origin 동기
(`db2a713`). JW·JB 워처(`git show origin/main:…track_c1.py | grep -q C1_THREAT`)가 풀렸다.
**정본 = YH 버전**(Yonghee 판정) · JW 의 중복 구현 `989f5ca` 는 철회(`f056b16`).

| 계정 | 상태 | 남은 것 |
|---|---|---|
| **B200** | ⚪ **미기동**(4-GPU 단일 서버로 재통합) | 26셀 249 GPU-h → **63.8h · 07-28 오후** |
| **YH** | 🟢 G3 10/96 완료 · 8 실행 · 큐 재정렬 완료 | c1축 9 · G8 24 · G6 9 → **07-26 밤~07-27 오전** |
| **HJ** | 🟢 G12 제출 `1878707`(**15셀**, 0-4 실행 중 · OOM 0) · L11 정리 완료(`1876764_0,1,2` 잔존) | G10 216런(게이트 해소 → 즉시) |
| **JW** | 🟢 c1축 8셀 실행 중 · torchvision·파티션 이슈 해결 | reset → 잔여 13셀 |
| **JB** | ⏸ 워처 대기 → **즉시 착수 가능** | c1축 18셀 |

**0″. 오염-집합 규약이 무대별로 **셋**이다 — 확정 사항 + 논문에 **2문장 추가 필요**.

전 러너 전수 확인(07-25). `Bernoulli(ρ=0.4)` 는 코드에 **딱 한 곳** 있다 — `track_c2.py:258` 의
`label_flip` 분기.

| 무대 | 러너 | label_flip | fr · gn | LLM noisy · frzero | 실현 수 |
|---|---|---|---|---|---|
| **CNN N=100 주무대** | `track_c2`(+`_fid`) | **Bernoulli(ρ=0.4)** = FedCorr 공식 구현 재현 | 랜덤 고정 40 | — | lf **39/48/47** · fr·gn 40 |
| **CNN N=10 (a) 무대** | `track_c1` | 랜덤 고정 4 | 랜덤 고정 4 | — | **4/4/4** |
| **LLM 전 무대** | `phase2_matrix`·`track_g` | — | — | **결정적 인덱스 `0..19`** | **정확히 20 = 40%** |

- **Bernoulli 는 CNN 주무대 전체의 규약이고, 논문 본문 CNN 표 전부가 그 위에 있다**(§5.2 fidelity
  = c2fid · §5.4 탐지 = 같은 rundir · §5.3 downstream = dir1 기존 + G3 + G10 · G8). `track_c2_fid`
  는 `track_c2` 를 read-only 재사용하고, track_g CNN 게이트 그리드도 `run_cnn_grid.sh:30` 이
  `track_c2.py` 를 호출하므로 **같은 draw** → G3 의 arm 병합(track_g flirds + track_c2 7소스)이
  같은 오염 집합 위에서 이뤄진다. ✔ 단 **적용 대상은 `label_flip` 셀뿐**이다(4위협 중 1열).
- **LLM 결정적 인덱스는 일반성을 잃지 않는다** — `build_gsm8k_iid`(`flirds/data/llm.py:311-313`)가
  공식 train 을 **`shuffle(seed)` 후 등분**하므로 클라 `0..19` 가 쥔 데이터가 시드마다 바뀐다.
  noisy·frzero 가 **같은 인덱스 집합**을 쓰는 것도 설계다(위협만 바꾼 대조).
- **N=10 에서 Bernoulli 를 안 쓰는 판단의 정량 근거**: 재현 결과 **3/7/6**(평균 5.33 vs 명목 4,
  sd 1.55). 같은 draw 를 N=100 에 적용하면 논문 기재값 **39/48/47 을 정확히 재생산**하므로 신뢰
  가능. N=100 은 39–48(상대변동 ~12%)이라 무해하지만 N=10 은 3–7(**~39%**)로 label_flip 열이
  다른 도즈가 되어 4위협이 한 fidelity 표를 공유할 수 없다(`track_c1.py:188-192`).
- **Yonghee 판정(07-25) = YH 버전 그대로 통일.** ⟹ 코드 변경 0 · **JW 실행 중 8셀 전부 유효
  (재실행 0)** · JW 중복 구현 `989f5ca` 철회(`f056b16`).
  기대 corrupt 집합(N=10 · 전 위협 공통 · **seed-only**): seed0 `[1,4,6,7]` · seed1 `[0,4,6,7]` ·
  seed2 `[3,4,6,9]` (`default_rng(1000+seed)` 첫 소비 = `track_c2` 와 같은 스트림/오프셋).
- **미수록 확인**: `std50k5` 의 오염 규약(noisy `{0-4}`·fr `{5-9}` = **각 10%**)은 수록 대상이
  아니다 — 수록되는 std50k5 셀은 **G5(clean, `track_d`)** 뿐이고 `std50k5 mixed` 는 미수록 목록에
  있다. 따라서 위 3종이 전부다.
- **⚠ 남은 작업 = 논문 부록 B 에 2문장**(§9). 현 `paper-ko.md:863-869` 는 CNN label-flip 의
  Bernoulli 와 update-level 의 고정 `⌊ρN⌉` 만 서술하므로 **그 자체로는 맞지만 두 무대가 빠져 있다**:
  ① **LLM 전 무대** = 결정적 인덱스 `0..19` = 정확히 40%(클라 샤드가 시드-셔플이라 WLOG) ·
  ② **CNN N=10 (a) 무대** = 전 위협 고정 `⌊ρN⌉`=4(도즈 0.70 고정이라 FedCorr 재현 대상이 아니고,
  N=10 Bernoulli 는 3/7/6 으로 흔들려 4위협이 한 표를 공유할 수 없다).
  **코드가 아니라 서술이 유일한 미결 항목이다.**

**0‴. 이 문서군의 오류 2건 정정(07-25).**
1. **"C-b 착지 전에는 제출해도 실패한다" → 틀렸다.** 실패하지 않고 **조용히 틀린다** — 구 러너는
   미지 env 를 무시하고 `C1_SCENARIO=iid` 로 돌지만 `C1_RUN_NAME`(`track_c1.py:432`)은 먹혀
   위협 라벨 붙은 rundir 에 iid-clean 데이터가 들어가고 EXIT=0. 4위협이 한 셀로 붕괴한다.
   JW 지적 · 코드로 확인. 전 문서·sbatch 헤더 수정 완료. **방어 = 워처 술어**(JB 방식).
2. **`runs/track_c/c1/make_analysis.py` 는 존재하지 않는 파일이었다**(sbatch 헤더가 가리킴).
   다만 **집계기를 새로 쓸 필요는 없다** — `runs/track_c/make_figures.py` l.99-152 `load_c1()` 이
   이미 c1 rundir ↔ (a) 페어링 + `phi_a` 음수화 + Spearman 을 한다(= G2 표 그 자체). 막힌 건 l.36
   `SCENARIOS` 상수가 구 5축이라는 것뿐 → `{PARTS}×{THREATS}` + 이름 패턴
   `{ds}_{part}_{ttag}_seed{seed}`(sbatch l.65 고정)로 교체. **이름이 이미 고정이라 지금 써도 된다.**

**0⁗. 3090 파티션 확장(전 sbatch 반영 완료).** `base_suma_rtx3090` 단독은 07-25 여유 0
(총 71장 · 빈 6장은 draining node01) — 최초 표의 "21장/186" 은 3090 **풀 전체** 집계였다.
→ 4개 sbatch 전부 `--partition=base_suma_rtx3090,dell_rtx3090`. 같은 RTX3090 이라 스택 캐비엇 동일.
`torchvision` 은 계정마다 없을 수 있다(JW 는 `0.26.0+cu128` 별도 설치 · torch 2.11.0+cu128 불변).

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

online 표가 flirds 단독이면 "vanilla·random 보다 낫다"까지만 말할 수 있다. **Flirds-1st online 9셀**을 얹으면 online 표가 retrain 표와 같은 계열로 맞물려 **§5.6① 2차항 주장이 online 에서도 성립**한다. **B200 이 맡는다** — `queue_b200.txt` 블록 ④(seed1·2 6셀; seed0 3셀은 HJ 완주분).

> ⚠ **HJ 에서 이미 착지한 flirds1st 셀은 그대로 유효하다**(같은 R=200 무대). HJ 의 seed0 3셀(`1876764_0,1,2`)이 완주하면 B200 큐 블록 ④ 는 그대로 두면 되고(seed1·2 만), **실패했으면 큐 하단 폴백 3줄을 해제**한다. HJ 의 나머지 array 원소(loss-heur·FedIF·renorm-4)는 스코프 밖이라 취소 완료.

## 4. 코드 조치

| 코드 | 무엇 | 상태 |
|---|---|---|
| **C-c** | `rounds` 를 rundir IDENTITY 로 승격(`track_g`·`phase2_matrix`) | ✅ 적용(안전장치로 유지) |
| **C-a** | `track_c2.py:157` `MODEL_FN` 에 `"mnist": LeNet5` (**1줄**) | ✅ **착지 `d09e528`** — G8·G10 게이트 해소 |
| **C-b** | `track_c1.py` 에 `C1_PARTITION`·`C1_THREAT`·`C1_FLIP_RATE` 도입 | ✅ **착지 `d09e528`**(YH 버전 = 정본) — c1축 게이트 해소 |

> **C-b 의 오염-집합 규약 = 전 위협 고정 `⌊ρN⌉`=4/10**(`track_c1.py:187-196`). §0″ 의 위협별
> 규약(label_flip = Bernoulli)과 **의도적으로 다르고**, l.188-192 에 근거가 달려 있다:
> ① 이 무대는 도즈를 0.70 으로 고정해 FedCorr 를 재현하는 게 아니다(τ~U(0.5,1) 미사용) →
> "공식 구현 준수"의 대상이 아니다 ② N=10 에서 Bernoulli 는 3/7/6 으로 흔들려 **label_flip 열이
> 다른 도즈가 되고, 4위협이 한 fidelity 표를 공유할 수 없다**. **Yonghee 판정 = 이대로 통일.**
> ⟹ **JW 의 실행 중 8셀은 전부 유효**(재실행 0) · **논문에 stage-분리 1문장 필요**(§9).
> 기대 corrupt 집합(전 위협 공통, seed-only): seed0 `[1,4,6,7]` · seed1 `[0,4,6,7]` · seed2 `[3,4,6,9]`.

## 5. 실행처별 배분 · 예상 종료 (07-25 23:00 기준 · 잔여 73h)

**B200 = 단일 서버 4-GPU 통합** (07-25 구성 변경 · **현재 돌고 있는 실험 없음**)

종전 "세션 2개 × GPU 2장, 컨테이너 별개(CID=1..4)" 가 **4장 동시 제어**로 통합됐다 →
큐도 `queue_b200_c{1..4}.txt` 4개에서 **`queue_b200.txt` 1개**로 병합, 기동은 **한 번**:

```
bash <repo>/runs/track_h/run_b200_batch.sh
```

드라이버가 비는 GPU 에 큐 순서대로 배정한다(`GPUS="0 1 2 3"`) = 자동 부하분산. 레인 고정을
없앤 게 실이득이다 — 종전 c4 는 72.8/74h 로 아슬아슬했고 c1–c3 는 15h씩 놀았다(**유휴 6h 로 감소**).

| # | 블록 | 셀 | GPU-h | 마지막 셀 완주 |
|---|---|---|---|---|
| ① | **G1 seed0** (3위협) | 3 | 58.8 | 19.6h |
| ② | **L1 clean seed1·2** | 4 | 18.6 | **18.6h** |
| ③ | **G1 seed1·2** | 6 | 117.6 | **58.8h**(G1 전량) |
| ④ | online Flirds-1st seed1·2 | 6 | 25.2 | 52.3h |
| ⑤ | G5 | 4 | 20.0 | 62.3h |
| ⑥ | G12 앞 3셀(부록·droppable) | 3 | 9.0 | 63.8h |
| | **합계** | **26** | **249.2** | **63.8 wall-h** |

- **07-25 23:00 기동 → 07-28 15:00 완주 · 마진 ~9h**(실측 단가 list-scheduling 시뮬레이션).
- **순서 근거**: ① G1 seed0 3위협을 맨 앞에 둬 어떤 컷에도 **전 위협 1-seed 행**이 먼저 완성 →
  ② 값싼 L1 clean 을 붙여 남는 1장이 놀지 않게(18.6h 에 §5.3 clean 열 종료) → ③ G1 전량이
  **58.8h = 가능한 최조기**에 종료(cell-end persist 라 일찍 끝나는 게 보험) → ④–⑥ 은 arm 단위
  영속이라 꼬리에 둬도 컷 손실이 작다.
- **G12 는 앞 3셀만** — 나머지 15셀은 HJ(A6000)가 이미 돌리고 있다.
- **flirds1st seed0 3셀은 HJ 완주분 사용** → 큐는 seed1·2 만. 폴백 3줄은 큐 하단에 주석 대기.
- **★ 제출 전 확인**: `vicgalle/alpaca-gpt4`·`cais/mmlu` HF 캐시(⑤⑥ = track_d 가 없으면 오프라인
  기동 불가; HJ 계정에서 실제로 막혔다). 런처가 경고를 내지만 중단은 안 한다 — ①–④ 는 무관.

**Slurm = CNN 747 + LLM 넘침 G12 64 = 811 GPU-h · 4계정 균등**

| 실행처 | 담당 | c1축 array | GPU-h | wall |
|---|---|---|---|---|
| **YH** | **코드 C-b·C-a** + G3(80)·G8(25)·G6(15) + c1 cifar10 | `0-7,16` | **202** | **25.2h** |
| **HJ** | flirds1st 3셀 완주 → **G12 16셀**(64·A6000) → **G10**(135·3090) | — | **199** | **24.9h** |
| **JW** | c1 cifar10 s2·s1잔여 + mnist 일부 | `32-39,17-23,8-13` | **205** | **25.6h** |
| **JB** | c1 mnist 전량(부록 G9) | `14-15,24-31,40-47` | **205** | **25.7h** |

> **균등화**: 종전 15/17/30/32h → **전부 24.9–25.7h(편차 0.8h)**. 단가 차이(cifar10 9.1h vs mnist 11.4h)를 반영해 셀 수가 아니라 **GPU-h 로 맞췄다**.
> **본문 우선**: cifar10(본문 G2)을 YH·JW 가 나눠 갖고 mnist(부록 G9)는 JB 가 몰아 받는다 → 본문 표가 먼저 닫힌다.
> **HJ 만 c1축이 없다** — G10(216런)+G12(16셀)로 이미 몫을 채웠고, A6000 에서 G12 를 이어받는 게 전환 마찰이 0 이기 때문이다.

- **전체 = 07-28 오후**(B200 63.8h) — 마감(07-28 24:00) 대비 **~9h 여유**. Slurm 은 전부 **07-27 오전**에 끝난다.
- **임계경로는 B200 의 G1**(셀당 19.6h × 9). Slurm 을 균등화해도 전체 종료는 B200 이 정한다 — 균등화의 실익은 **07-27 오전에 CNN 전량이 동시에 닫혀** 분석·집필을 하루 앞당길 수 있다는 것.
- **C-b 가 c1축 492 GPU-h 를(4계정 전부), C-a 가 216런(HJ G10)+G8 을 막는다** → YH 의 첫 작업이 코드인 이유. **G3(80)만 게이트가 없어 코드 작성 중 병행 가동**한다.
- **3090 여유 21장**이라 4계정 32슬롯 중 ~21–24 가 물리적으로 확보된다. **A6000·4090 은 배정하지 않는다**(각각 여유 10장·0장).

## 6. 지금 할 조치 (순서)

1. **HJ**: `scancel <jobid>_[3-20]` — **flirds1st(0·1·2)만 남기고** lossheur·fedif·renorm-4 전량 중단. 착지한 rundir 은 지우지 않는다(집계는 canonical dup-win).
2. **YH**: **G3 즉시 착수**(게이트 없음) + 병행해 **C-b·C-a 코드**.
3. **B200**: `bash runs/track_h/run_b200_batch.sh` **한 번**(4-GPU 통합 · HF 캐시 확인 후).
4. **HJ**: flirds1st 완주(≈23:00) → 같은 A6000 에서 **G12 16셀** → C-a 착지 후 3090 에서 **G10**.
5. **C-a 착지 후**: YH → `sbatch_fid_mnist.sh`(G8) · HJ → `sbatch_cnn_mnist_comp.sh`(G10).
6. **C-b 착지 후**: `sbatch_c1_axis.sh` — YH `0-7,16` · JW `32-39,17-23,8-13` · JB `14-15,24-31,40-47`.
7. **JB(L9)** 는 이미 전량 중단 확정 — 잔재만 확인.

## 7. 여유가 생기면 — 우선순위

| 순위 | 무엇 | 비용 | 왜 |
|---|---|---|---|
| 1 | **LLM P1w T1** 9런 | ~35 (B200) | CNN P1w 가 G3·G10 에 동반 산출되므로, LLM 레그만 얹으면 "전 범위 3-seed 승격" 판정이 성립. **4-GPU 통합으로 실현 가능해졌다**: ⑥(G12 3셀)과 **맞교환**하면 275.2 GPU-h → **69.6 wall-h**(마진 +3.4h). ⑥ 을 남긴 채 추가하면 73.5h = 초과. 큐 하단에 주석 대기 — **①–③ 이 58.8h 에 정상 종료된 걸 본 뒤 결정** |
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
