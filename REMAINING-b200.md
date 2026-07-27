# REMAINING (B200) — **단일 서버 4-GPU 통합**: 주무대 (b) 오라클 + clean 열 보강 + probe

> 배분 정본 = **`REMAINING-00-INDEX.md`** · 수록목록 정본 = `research-wiki/survey/flirds-paper-experiment-plan.md`.
> **역할 = HVP(flirds 2차 φ, 95–106 GiB) 전용 + canonical timing.** 48GB 로는 기본 knob 불가 → 여기 있는 건 **B200 아니면 못 한다**.
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 3-seed. push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만.
> **B200 이 이번 계획의 임계경로다** — 전체 종료 시각을 여기가 정한다.
> ~~**현재 돌고 있는 실험 없음**(07-25 기준)~~ → **가동 중**. 현황은 **§0′.4–§0′.7**(07-27 갱신)만 읽는다.

## 0′. ⚠ 07-26 재계획 — **아래 §0~§6 의 단가·배분은 폐기됐다**

> **읽는 순서**: §0′ 만 유효하다. §0~§6 의 숫자(19.6 h/셀 · 26셀 249 GPU-h · L1 clean 4셀 ·
> §6 P1w 맞교환 표)는 **전부 폐기된 추정치**이므로 인용 금지 — 실측 단가는 §0′.4 다.

> 운용 절차·교체 런북 = `runs/track_h/B200_SWAP_RUNBOOK.md`. 이 절은 **무엇을 왜 바꿨는지**만 남긴다.

### 0′.1 구성 정정 — 단일 4-GPU 서버가 아니다

실측: **컨테이너 4대 `yong-1..4` × B200 1장**. §0 의 "4장을 한 서버에서 동시 제어"는 이 서버에 해당하지 않는다.
`queue_b200.txt`(단일 서버 `GPUS="0 1 2 3"` 전제)를 그대로 띄우면 슬롯 1·2·3 이 즉사하고 드라이버가
큐를 소진한다. → 26셀을 원문 그대로 `queue_b200_lane{1..4}.txt` 로 분할 + `run_b200_lane.sh`(`GPUS="0"`)
+ `b200_fleet.sh`(4대 제어). **정본 `queue_b200.txt`·`run_b200_batch.sh` 는 무수정 보존.**

### 0′.2 G1 단가 — 19.6h 는 틀렸다 (해석 추정치의 상수 오류)

§4 의 `valuation ~15.8h = 177.5 fwd-등가 × 200 × microbench 1.60 s` 에서 **1.601 s 는 `val=100`
에서 잰 값**(`runs/measured_2026-07/microbench/summary.json`)인데 gsm50k5 는 **`val=200`** 이다
(`phase2_matrix.py:171`). 청크 수가 2배 → per-pass 도 ~2배. **정정 단가 ≈ 34.5h/셀**(FL 2.95h 실측
+ valuation ~31.6h). 07-25 22:03 기동분이 23.5h 경과에도 미착지한 것이 이것과 정합한다.
> 교훈: op-count 상수는 **레짐의 val 크기마다 재환산**해야 한다. silo5(val=20)·device100(val=10)에서만
> 교차검증돼 있어 gsm50k5(val=200)에서 드러나지 않았다.

### 0′.3 결정 (Yonghee 07-26)

| # | 결정 | 효과 |
|---|---|---|
| ① | **G1 채점 = (b)oracle + Flirds + Flirds-1st 만** (`MIN_METHODS=1`) | **34.5h → ~10.2h/셀** |
| ② | **L11 6셀 → HJ(A6000)** job `1885728` | B200 부하 −25h |
| ③ | **G5(4셀)·G12(3셀) 제외** | B200 부하 −29h(ARMS=0 기준) |
| ④ | 레인4 진행 중 셀(17.6h 경과)을 **kill 후 MIN 으로 재시작** | −6.9h |

①의 근거: cross-game 4종(GTG·FedSV·ShapleyFL·ComFedSV)이 valuation 의 **~71%** 를 먹는데,
축-지도 §1A 원칙상 **in-run (b) 본문 채점은 same-game 만**이라 이들의 R4 열은 표에 안 간다.
그 3-seed fidelity 는 silo5·device100·std20 에 **이미 있다**. FedIF·loss-heur·전용탐지기 4종
(후자는 plan §0.2 로 논문 전면 제외)도 같은 컷에 포함.
구현 = `phase2_matrix.py:143` `MIN_METHODS` + Flirds-1st 직후 early-return(2줄). gpt2 스모크로
3방법만 산출·`MATRIX DONE` 확인.

> **⚠ provenance**: 진행 중이던 seed0 3셀(noisy·clean·frzero)은 **9방법 전량**으로 착지한다
> (그대로 두는 게 재시작보다 빠름 + cross-game seed0 근거 확보). seed1·2 는 3방법.
> (b)/Flirds/Flirds-1st 의 φ 는 early-return 이 **계산 후**라 비트동일 — 3-seed 행은 유효하다.
> cross-game 4종은 **R4 에서 seed0 1-seed** 로만 존재 → 3-seed 규칙상 표에 못 넣는다(의도된 결과).

### 0′.4 진행 현황 (2026-07-27 15:00) — **13셀 중 7 착지 · 4 진행 중 · 2 대기**

**G1 = 6/9 착지**

| rundir | 방법 | 실측 | |
|---|---|---|---|
| `1B_gsm50k5_clean_nr0.7_s0` | 9방법 | **30.65h** | 07-27 04:42 |
| `1B_gsm50k5_noisy_nr0.7_s0` | 9방법 | **30.88h** | 07-27 04:56 |
| `1B_gsm50k5_noisy_nr0.7_s2` | MIN | **9.42h** | 07-27 07:17 |
| `1B_gsm50k5_frzero_nr0.7_s1` | MIN | **9.37h** | 07-27 13:29 |
| `1B_gsm50k5_clean_nr0.7_s1` | MIN | **9.79h** | 07-27 14:30 |
| `1B_gsm50k5_noisy_nr0.7_s1` | MIN | **9.88h** | 07-27 14:49 |
| `..._clean_s2` | MIN | 진행 중 → ~16:53 | 레인4 |
| `..._frzero_s0` **재실행** · `..._frzero_s2` | MIN | phase 2 | |

→ **MIN 실측 9.37–9.88h**(평균 ~9.6h). 예측 9.46h 대비 +1.5%. 9방법 대비 **3.2× 절감**.

**L1 clean = 2/4 착지 · 3셀 동시 진행 중**

| 셀 | 상태 |
|---|---|
| `gsm50k5_clean_observer_seed0` · `_seed1` | 착지(seed1 = 07-26 04:05, 6.03h) |
| `L1_clean_obs_s2`(레인3 13:29~) | → **19:30** |
| `L1_clean_online_s1`(레인2 14:30~) | → 18:41 |
| `L1_clean_online_s2`(레인1 14:49~) | → 19:01 |

### 0′.5 남은 일정 — 교체 1회 + phase 2 2셀

| 시각 | |
|---|---|
| 07-27 **19:30** | phase 1 종료(레인3 `obs_s2` 가 마지막) |
| 07-27 **~19:45** | **컨테이너 교체** — 만료 21:11 까지 1.7h. 드레인·kill 불요(자연 종료) |
| — | **다음 컨테이너는 2장이면 충분**: phase 2 = `frzero_s0`(재실행) + `frzero_s2` 2셀, 서로 독립 |
| 07-28 **~05:45** | 캠페인 종료. 마감 24:00 대비 마진 **~18h** |

> phase 2 두 셀은 **서로 다른 GPU 에 배치**할 것. 레포 큐 그대로면 레인1 이 둘 다 지고 6h 늦어진다
> (`frzero_s2`=레인1, `frzero_s0`=레인3). seal 시점(드라이버 정지)이 옮길 수 있는 유일한 창이다.

### 0′.6 최종 산출 전망

| 블록 | 결과 |
|---|---|
| **G1** | **9/9 · 3-seed** — `clean_s0`·`noisy_s0` 만 9방법, 나머지 7셀은 (b)+Flirds+Flirds-1st |
| **L1 clean** | **4/4 · 3-seed** → §5.3 clean 열 완결 |
| L11 (HJ) | **9셀 전량 origin 반영 완료**(`fdad460`) |

제외분(G5 4셀·G12 3셀)은 레인 큐에 **주석으로 존속**(삭제 아님) — 되살리려면 주석만 풀면 된다.
07-27 Yonghee 최종 결정 = **둘 다 포기**(a6000 이전도 하지 않음, HJ 통보 완료).

### 0′.7 디스크 대조 (07-27 · 레포 워킹트리 기준) — **§0′.4 전량 일치**

`timing.json` 실측이 §0′.4 표와 ±0.15 h 안에서 맞는다(반올림 차). **G1 6/9 · L1 clean 2 착지**가
레포에 이미 커밋돼 있다 — 서버 보고와 레포가 어긋난 곳 없음.

| rundir | `timing.json total` |
|---|---|
| `1B_gsm50k5_clean_nr0.7_s0` · `noisy_nr0.7_s0` (9방법) | **30.64 h** · **30.87 h** |
| `..._noisy_s1` · `_clean_s1` · `_noisy_s2` · `_frzero_s1` (MIN) | 9.86 · 9.77 · **9.57** · 9.35 h |

> `noisy_s2` 만 §0′.4 의 9.42 h 와 0.15 h 차 — 표 쪽이 반올림 실수로 보인다. **MIN 평균 = 9.64 h**
> (§0′.4 의 9.6 h 와 정합).

**확정 잔여 = 6셀.** G1 `clean_s2`(진행 중) · `frzero_s0`(재실행) · `frzero_s2` / L1 clean
`obs_s2` · `online_s1` · `online_s2`(3셀 모두 진행 중). 디스크에 `gsm50k5_clean_flirds_gate_v2`
는 **seed0 뿐**, `clean_observer` 는 **seed0·1** 뿐인 것으로 재확인.

### 0′.8 phase 2 슬랙 — **R-clean 3셀만 채운다** (Yonghee 결정 2026-07-27)

phase 2 는 `frzero_s0`·`frzero_s2` **2셀뿐**이라 4-GPU 컨테이너면 **2장이 07-27 19:45 ~ 07-28
24:00 내내 논다(≈ 56 GPU-h 유휴)**. 그 슬랙에 **silo5 removal-curve clean 3셀만** 넣는다.

| 후보 | 판정 |
|---|---|
| **R-clean** — silo5 removal clean × 3seed | ✅ **넣는다.** T13 LLM removal 표가 noisy·frzero 2행뿐이라 **위협-특이성이 미증명**이다(clean 대조 부재). CNN 짝은 clean rundir 이 이미 있어 표에 행만 넣으면 되는 상태 → LLM 만 비면 비대칭. 큐 = `queue_b200_lane2.txt`(seed0·1) · `lane4.txt`(seed2), **주석 상태**로 대기 |
| **device100 clean 3셀** | ⛔ **버린다** (07-27 Yonghee). ~23 GPU-h 로 슬랙엔 들어가지만, 이 무대는 near-additive 라 clean 도 +1.000 근처가 확실해 **얻는 게 §0.1 형식 준수뿐**이다. → 논문은 캡션 각주("이 무대는 탐지·비용 전용이라 clean 대조를 두지 않았다")로 처리 |
| **gsm5(`LLM-Small`) 6셀** | ⛔ **2026-07-24 Yonghee 보류 결정이 유효** — IID·near-additive·ρ≈0 **축퇴 무대**라 `Anchor` 0.933 과 정보 중복. 조치는 실행이 아니라 **논문에서 [F3]·세팅표 행 삭제**(→ 실험정리 세션) |
| **LLM P1w T1 9런** | ⛔ 옵션(§6). 단가 미실측 + `A7` 은 CNN 레그만으로도 표가 선다 → 마감 앞 신규 리스크 불필요 |

**R-clean 실행 명세**

| 항목 | 값 |
|---|---|
| 셀 | `1B_silo5_clean_removal_seed{0,1,2}` → `runs/removal_dose/rundirs/` |
| 프로토콜 | `run_full_sweep.sh` §[1] 의 `matrix … REMOVAL=1` 과 **동일**(threat 만 clean). 코드 변경 0 |
| 스택 | 기존 4위협 셀과 **동일**(둘 다 torch 2.12.0+cu130 canonical, meta.json 대조 확인) → **캐비엇 없음** |
| **단가** | **2–3 h/셀 추정** = valuation 3,447 s + 재학습 **317.7 s × distinct-subset(≈8–20)** + silo5 학습. ※ 계획서 §4.3 의 "**<1 GPU-h**" 는 재학습 1회분만 센 **과소평가**다 |
| 기동 | **phase 2 신규 컨테이너에서 주석 해제** — 현 컨테이너(만료 21:11)에선 레인2 가 18:41 에야 비어 셀이 잘린다. `phase2_matrix` 는 cell-end persist 라 중도 kill = 전손 |
| 배치 | phase 2 는 **4레인 전부** 띄운다: 레인1 `frzero_s2` · 레인3 `frzero_s0` · 레인2 R-clean s0·s1 · 레인4 R-clean s2 → **07-28 ~01:00 종료**(G1 은 05:45 그대로) |

> **하드룰**: 컨테이너 교체 시 phase 2 두 G1 셀은 **서로 다른 GPU** 에 둔다(§0′.5). 큐를 그대로 쓰면
> 레인1 이 둘 다 져서 6 h 늦는다.

## 0. 구성 변경 (2026-07-25) — **4장을 한 서버에서 동시 제어**

종전 "세션 2개 × GPU 2장, 컨테이너 GPU마다 별개" 가 **GPU 4장을 한 서버에서 동시 제어**하는 구성으로 통합됐다. 그래서:

| | 종전 | **현재** |
|---|---|---|
| 큐 | `queue_b200_c{1,2,3,4}.txt` 4개 (레인 고정) | **`queue_b200.txt` 1개** (병합) |
| 기동 | `CID=1..4` 로 4번 | **1번** (`bash runs/track_h/run_b200_batch.sh`) |
| GPU | 컨테이너당 `GPUS="0"` | **`GPUS="0 1 2 3"`** |
| 배정 | 사람이 미리 쪼갠 고정 레인 | **드라이버가 비는 GPU 에 순서대로** = 자동 부하분산 |

- 레인 고정을 없앤 게 실제 이득이다: 종전 c4(72.8h)가 74h 창에 아슬아슬했고 c1–c3 는 ~15h씩 유휴였다. 병합하면 유휴가 **총 6h** 로 줄고 c4 의 아슬아슬함이 사라진다.
- 드라이버(`runs/rerun_beta03/run_multi_driver.sh`)는 원래 4-GPU 스케줄러다(`GPUS` 기본값이 `0 1 2 3`) — 새 코드 없음.
- 실행 중 GPU 를 빼고 싶으면 `GPUS_FILE` 을 편집하면 된다(재시작 불요; 빠진 GPU 의 실행 중 셀은 완주하고 신규 배정만 멈춘다).

## 1. 무대 = **R=200 유지**

LLM downstream 을 {vanilla·oracle·random·flirds류}로 줄이면서 L11·G4c 가 빠졌고, 그 둘을 넣으려던 **R=100 전환도 함께 철회**했다. 이제 **이미 완결·커밋된 L1 3-seed 를 재실행 없이 그대로 쓴다.** 큐 어디에도 `ROUNDS` 를 주지 않는다(레짐 기본 200).

> `rounds` 를 rundir IDENTITY 로 승격한 코드 변경은 유지한다 — R 이 다른 셀이 한 표에 섞이면 실행 시점에 실패하게 만드는 안전장치이고, 지금 무대에는 영향이 없다(신규 셀은 `rounds: 200` 으로 기록된다).

## 2. 기동

```
bash <repo>/runs/track_h/run_b200_batch.sh
```

- 런처가 env·sed·스모크 내장. `PY=$BATCH/venv/bin/python`(**torch 2.12.0+cu130 = canonical**), `HOME=$BATCH/home`, `HF_HOME=$BATCH/hf_home`, offline 플래그. 캐시 검증 = `$BATCH/PROVENANCE.md`.
- ~~**★ 제출 전 확인 1건 — track_d 용 HF 캐시**~~ (**G5·G12 취소로 무의미** — 07-27): `vicgalle/alpaca-gpt4` · `cais/mmlu` 가 없으면 **§5 G5·G12(track_d)가 오프라인에서 시작조차 못 한다**. HJ 계정에서 실제로 둘 다 없어 막혔다(공유 캐시에도 없었다). B200 은 과거 anchor5·probe_signal 을 돌렸으니 있을 가능성이 높지만 **확인하고 띄운다**. 없으면 `flirds/hf_pin.py`(REVISIONS 비어 있음 = 최신 커밋 OK, 둘 다 public, +238MB)로 받는다. G1·L1(§3·§4)은 gsm8k 계열만 쓰므로 이 확인과 무관하다.
- GPU 수를 바꾸려면 `GPUS="0 1"` 처럼 넘긴다(런처가 기본값 `0 1 2 3` 을 덮어쓴다).

### 드라이버 운용 (하드룰)

- **큐 정지는 줄 삭제가 아니라 `#` 주석** — 드라이버가 `consumed` 인덱스로 추적한다.
- **가동 후 줄 순서 변경 금지** (같은 이유).
- **러너는 셀 단위 원자적** — 중도 kill = 그 셀 전손. 정지는 드레인(실행 중 셀 완주 후 자동 종료).
- **마지막 셀은 `done[ok]` 줄이 안 남는다** → 완주 판정은 `MATRIX DONE` / `TRACK G DONE` + rundir mtime.
- **⚠ 영속 단위**: `phase2_matrix`(G1) = **cell-end 1회** → 셀당 ~19.6h 를 통째로 잃을 수 있다(07-24 실사례). `track_g`·`track_d` = arm/셀 단위라 컷에 강하다.
  → **컷이 예상되면 G1 신규 투입부터 끊고 드레인.** `$BATCH/runlogs/seal_watchdog.sh` 재가동 권장.
- **★ `BATCH` 를 export 하지 말 것** — 세 러너 모두 `BATCH` 를 batch-size 노브로 읽는다(`phase2_matrix.py:185`·`track_g.py:136`·`track_d.py:101`). export 하면 경로 문자열이 들어가 전 셀이 15초 만에 `ValueError` 로 죽고 드라이버가 큐를 그대로 소진한다(07-25 실사례). 런처가 `export -n` 으로 막아 두었다.

## 3. 스케줄 — 26셀 · 249 GPU-h · **4-GPU 63.8 wall-h**

큐 순서 = 드라이버 배정 순서(list scheduling)라 순서가 곧 스케줄이다. 아래는 실측 단가로 시뮬레이션한 결과다:

| # | 블록 | 셀 | 단가 | GPU-h | 마지막 셀 완주 |
|---|---|---|---|---|---|
| ① | **G1 seed0** (3위협) | 3 | 19.6h | 58.8 | 19.6h |
| ② | **L1 clean seed1·2** | 4 | 4.65h | 18.6 | **18.6h** |
| ③ | **G1 seed1·2** | 6 | 19.6h | 117.6 | **58.8h** |
| ④ | online Flirds-1st seed1·2 | 6 | 4.2h | 25.2 | 52.3h |
| ⑤ | G5 (r32·r64 seed1·2) | 4 | 5.0h | 20.0 | 62.3h |
| ⑥ | G12 앞 3셀 (부록) | 3 | 3.0h | 9.0 | 63.8h |
| | **합계** | **26** | | **249.2** | **63.8h** |

- **07-25 23:00 기동 → 07-28 15:00 완주 · 창 73h 대비 마진 ~9h.** 유휴 6h(이상적 하한 62.3h).
- **순서를 이렇게 잡은 이유**: ① G1 seed0 3위협을 맨 앞에 둬서 **어떤 컷이 나도 전 위협 1-seed 행이 먼저 완성**되게 하고, ② 값싼 L1 clean 4셀을 그 뒤에 붙여 남는 1장이 놀지 않게 하고(18.6h 에 §5.3 clean 열 종료), ③ 그 다음 G1 6셀 → **G1 전량이 58.8h 에 끝난다**(가능한 최조기; G1 은 cell-end persist 라 일찍 끝나는 게 곧 보험이다). ④–⑥ 은 arm 단위 영속이라 꼬리에 두어도 컷 손실이 작다.
- **⑥ 은 부록·최저**라 마감에 걸리면 그냥 안 돌아도 된다.

## 4. G1 — R4-L2 주무대 (b) 오라클 (9셀 · 176 GPU-h) ★임계경로

> **논문 최대 병목·대체 불가.** 이 9셀 하나가 **세 축**을 동시에 연다: §5.2 LLM fidelity(본문) · §5.4 LLM 탐지(부록) · §5.5 **canonical timing**. 축-지도에서 LLM fidelity·탐지가 ⬚ 로 남은 유일한 칸이고, 로컬 `phase2_matrix/rundirs` 에 `1B_gsm50k5_*` **0개**로 미실행 확인됨.
> **스코프 컷과 무관하게 9방법 φ 를 전부 산출**하므로, downstream 표에서 뺀 renorm-4 의 LLM-스케일 근거가 여기서 나온다.

- `phase2_matrix.py REGIME=gsm50k5`(Llama-3.2-1B · N=50 · 5/50 · **R=200** · GSM8K) — (b) per-round 2⁵ exact + 9방법 φ + `timing.json`.
- **셀**: {clean, noisy, freerider_zero} × seed{0,1,2} = **9**. 코드 변경 불필요.
- **⚠ THREAT 토큰**: `phase2_matrix` 는 **`freerider_zero`**(rundir 이름만 `frzero` 로 축약) — `track_g` 의 `frzero` 와 다르다. `NOISY_RATE` 는 gsm50k5 기본 0.7 이라 생략.
- **단가 ~19.6h/셀**: FL ~3.8h + valuation ~15.8h(라운드당 177.5 fwd-등가 × 200 × microbench 1.60 s). 직전 컨테이너에서 20h+ 돌고도 valuation 을 못 끝낸 관측과 일치한다.
- **완료 판정**: `MATRIX DONE` + rundir `1B_gsm50k5_{clean,noisy_nr0.7,frzero}_s{seed}`.

## 5. 나머지 블록

| 블록 | 무엇 | 왜 필요한가 |
|---|---|---|
| **L1 clean seed1·2** (4셀) | `clean_obs`·`clean_online` seed1·2 | 디스크 확인(07-25): **noisy·frzero 는 이미 3-seed 완결**(observer·oracle_excl·random_excl·flirds online + `t2_sign`×4). **clean 만 seed0** → 이 4셀이 §5.3 clean 열(무해성 parity·오발화 대조)을 닫는다. clean 은 kept=전원이라 T2 가 `equals_vanilla` 로 스킵되고 online 에선 `oracle_excl`/`random_excl` 가 자동으로 빠진다(`track_g` l.618) → 저비용. **noisy·frzero 는 절대 재실행하지 않는다.** |
| **online Flirds-1st seed1·2** (6셀) | `flirds1st_gate_v2` | online 표가 flirds 단독이면 "vanilla·random 보다 낫다"까지만 말할 수 있다. Flirds-1st 를 넣으면 **§5.6①(2차항) 주장이 online 에서도 성립**한다(retrain 표엔 이미 flirds·flirds1st·loss-heur·FedIF 가 3-seed 로 있다). **seed0 3셀은 HJ 완주분**(`rundirs_llm_hj`, 같은 R=200 무대라 유효) — 실패했으면 큐 하단 폴백 3줄 해제. |
| ~~**G5** (4셀)~~ ⛔ **취소 (2026-07-27 Yonghee)** | ~~`1B_std50k5_r{32,64}_seed{1,2}`~~ | **돌리지 말 것.** 2차항 LLM 레그를 **논문에서 뺐다** → 2차항 논지는 CNN 레그만으로 간다. |
| ~~**G12** (3셀)~~ ⛔ **취소 (2026-07-27 Yonghee; HJ 큐도 07-26 취소)** | ~~anchor5 lever probe `lr{2,3}e-3`~~ | **돌리지 말 것.** lever LLM 레그를 **논문에서 뺐다** → lever 논지는 CNN 레그만으로 간다. (구 메모: 부록·최저. 핵심 질문 = "lr 로 커진 φ가 cross-seed 실재 신호인가"(예측 ρ≈0) → `lr{2,3}e-3` 을 먼저 배치했다. |

- G5·G12 는 **R4 무대가 아니다** → `ROUNDS` 를 주지 않는다(track_d 가 자기 레짐 값을 쓴다).

## 6. (옵션) LLM P1w T1 — 여유 시 최우선 추가

- 9런 · **~35 GPU-h**. CNN P1w 는 G3·G10 rundir 에 동반 산출되므로, LLM 레그만 얹으면 **"CNN·LLM 전 범위 3-seed 승격" 판정이 성립**한다. 지금 그게 유일한 공백.
- 관찰자 불요 → canonical `rundirs_llm` 에 직접 착지(arm 명이 신규라 충돌 0). **큐 하단에 주석으로 대기.**
- **4-GPU 재통합으로 실현 가능해졌다** (같은 시뮬레이션):

  | 안 | GPU-h | 4-GPU wall | 창 73h 마진 |
  |---|---|---|---|
  | 기본 26셀 | 249.2 | 63.8h | **+9.2h** |
  | **⑥(G12 3셀) ↔ P1w 9런 맞교환** | 275.2 | 69.6h | **+3.4h** ← 가능하나 얇다 |
  | G12 유지 + P1w 추가 | 284.2 | 73.5h | **−0.5h** ← 불가 |

  → **맞교환만 성립**한다(G12 3셀은 부록·최저라 내려도 된다; 나머지 15셀은 HJ 가 돌린다).
  마진 3.4h 면 G1 셀 하나(19.6h)의 재시도 여유가 없으므로, **①–③ 이 예정대로(58.8h) 끝난 걸 확인한 뒤에** 결정한다.
- 전제 충족: 코드·테스트 커밋(`ec6cbd5`) + H-14 사전등록 완료 → 실행만 남음.

## 7. 완료 후

1. rundir 커밋(push는 Yonghee).
2. `runs/phase2_matrix/make_analysis.py` + `runs/track_h/make_analysis.py` 재생성 → `flirds-results-{fidelity,detection,cost,downstream}` → paper.
3. **canonical timing 은 이 B200 산출만**(§5.5). Slurm(torch 2.11) `timing.json` 은 cost 표 금지.
