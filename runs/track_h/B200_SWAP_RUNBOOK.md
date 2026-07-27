# B200 — 48h 컨테이너 교체 런북 (2026-07-25 기동분)

> 이 문서는 **컨테이너 48h 만료를 무손실로 넘기는 절차**다. 배분·수록 정본은 건드리지 않는다
> (`REMAINING-00-INDEX.md` · `REMAINING-b200.md` · `queue_b200.txt`).

## 0. 확정 사실 (실측)

| 항목 | 값 |
|---|---|
| 구성 | **컨테이너 4대 `yong-1..4` × B200 1장** (단일 4-GPU 서버 아님) |
| 컨테이너 기동 | 07-25 **21:11:06** (4대 동시 = 48h 시계 동기) |
| **강제 만료** | **07-27 21:11** |
| 캠페인 기동 | 07-25 **22:03** |
| 마감 | 실험 07-28 24:00 |

`queue_b200.txt`(정본)와 `run_b200_batch.sh`는 **단일 서버 GPU 0–3**을 가정한다. 이 서버에서 그대로
띄우면 슬롯 1·2·3 셀이 `CUDA_VISIBLE_DEVICES=1,2,3`을 못 찾아 즉사하고 드라이버가 큐를 소진한다
(BATCH export 사고와 같은 전손 형태). 그래서 정본 26셀을 **원문 그대로** 레인 4개로 나눴다:
`queue_b200_lane{1..4}.txt` + `run_b200_lane.sh`(GPUS="0") + `b200_fleet.sh`(4대 동시 제어).

## 1. 왜 "셀 경계에서만" 끊어야 하나 — 러너별 영속 단위 (코드 확인)

| 블록 | 러너 | 영속 지점 | 중도 컷 시 |
|---|---|---|---|
| **G1** (①③) | `phase2_matrix.py` | `_persist()` **733행, 끝 1회** | **셀 전손 (19.6h)**, 재개 불가 |
| L1 obs (②) | `track_g.py` | `persist(arm,…)` arm 단위 (observer + T2 파생) | 끝난 arm 은 디스크에 남지만 **자동 스킵 없음** → `ARMS=` 로 수동 재개 |
| L1 online·L11 (②④) | `track_g.py`, **단일 arm** | 사실상 셀 끝 1회 | 전손 (3.8 / 4.2h) |
| G5·G12 (⑤⑥) | `track_d.py` | 437행 fidelity 직후 **체크포인트 1개** + 말미 | fidelity 는 살아남고 arms 만 재실행 |

- `RunLogger.precheck`(= `check_identity` 별칭)는 **identity 가드일 뿐 done-skip 이 아니다** — 재제출하면 처음부터 다시 돈다.
- 결론: **중간 재개가 되는 러너는 없다.** 교체는 셀 경계에 맞춰야 무손실이다.
- 참고로 현행 정본 큐 순서 그대로 4-GPU 로 돌렸다면 t=48h 에 두 장이 G1 한복판(8.8h·9.8h 경과)
  이라 **19.4 GPU-h 전손**이었고, 전 GPU 공통 유휴 지점은 t=0 말고 없었다.

## 2. 배치 — phase 1 이 만료 전에 끝나도록 설계했다

3×G1 = 58.8h 는 47h 창에 못 들어가므로 **G1 9셀 중 1셀은 반드시 phase 2**로 넘어간다.
각 레인에 G1 2셀(39.2h)을 고정하고 남는 여유에 L1 을 한 셀씩 얹어 **8/9 G1 + 4/4 L1** 을 만료 전에 닫는다.

| 레인 | 노드 | PHASE 1 (셀 종료 시각) | 종료 |
|---|---|---|---|
| 1 | yong-1 | `G1_noisy_s0` 07-26 17:39 → `G1_noisy_s1` 07-27 13:15 → `L1_clean_obs_s2` | **07-27 18:45** |
| 2 | yong-2 | `G1_clean_s0` 07-26 17:39 → `G1_clean_s1` 07-27 13:15 → `L1_clean_online_s1` | **07-27 17:03** |
| 3 | yong-3 | `G1_frzero_s0` 07-26 17:39 → `G1_frzero_s1` 07-27 13:15 → `L1_clean_online_s2` | **07-27 17:03** |
| 4 | yong-4 | `L1_clean_obs_s1` 07-26 03:33 → `G1_noisy_s2` 07-26 23:09 → `G1_clean_s2` | **07-27 18:45** |

→ **4장이 07-27 17:03~18:45 사이에 함께 빈다.** 만료(21:11)까지 **2.4h 여유**.

| 레인 | PHASE 2 (교체 후) | 소요 |
|---|---|---|
| 1 | `G1_frzero_s2` | 19.6h |
| 2 | `G5_r32_s1` → `G5_r64_s1` → `L11_clean_s1` → `L11_noisy_s1` | 18.4h |
| 3 | `G5_r32_s2` → `G5_r64_s2` → `L11_frzero_s1` → `L11_clean_s2` | 18.4h |
| 4 | `L11_noisy_s2` → `L11_frzero_s2` → `G12` ×3 | 17.4h |

→ 교체 ~07-27 19:00 시작 시 **완주 07-28 14:36 · 마감 대비 마진 ~9.4h**.

## 2b. ⚠ 07-26 재배치 — §2 표는 폐기됐다

**G1 단가 추정(19.6h)이 틀렸다.** 07-25 22:03 기동한 G1 3셀이 **21.1h 경과에도 미착지**.
`REMAINING-b200.md:76` 의 19.6h 는 op-count×microbench **해석값**이고, 착지한 gsm50k5
phase2_matrix rundir 이 레포에 0개라 검증된 적이 없었다 — 같은 줄의 "직전 컨테이너 20h+ 미완"이
유일한 실측이었고 지금 그것과 일치한다. **E(G1 실단가)는 여전히 미확정, 하한만 21.1h.**

### Yonghee 결정 (07-26 19:xx)

1. **L11 6셀 → HJ/A6000 이관.** seed0 3셀을 그쪽이 이미 A6000 에서 완주(7.53–8.00 GPU-h)했고,
   seed1·2 도 거기서 돌면 3-seed 전체가 한 스택이라 정합성이 오히려 낫다.
   런타임 큐 + **레포 레인 큐 둘 다 주석**(재기동해도 안 되살아난다).
2. **G1 2번째 셀 전부 보류.** E 확정 전엔 투입하지 않는다 — E>23.6h 면 만료를 걸쳐 전손이다.
   **런타임 큐에만 주석**(레포 무수정 → 재기동 시 되살아나 phase 2 에서 돈다).
3. **가벼운 셀을 4레인 균형 배치**, G12 는 **최후순위 꼬리**(빼도 균형이 유지되게).

### 현 배치 (07-26 22:00 확정 — 남은 실험 = G1 9 + L1 3)

`MIN_METHODS=1` 로 G1 단가가 **34.5h → ~10.2h** 가 되어 G1 을 phase 1 로 되돌렸다.
G5·G12 는 제외(주석 존속), L11 6셀은 HJ.

| 레인 | 진행 중 | 착지 | 다음 (MIN) | phase 1 종료 |
|---|---|---|---|---|
| 1 | `noisy_s0` (**9방법**) | E = 07-27 08:33 예측 | `noisy_s1` | **18:42** |
| 2 | `clean_s0` (**9방법**) | 〃 | `clean_s1` | **18:42** |
| 3 | `frzero_s0` (**9방법**) | 〃 | `frzero_s1` | **18:42** |
| 4 | `noisy_s2` (MIN, 07-26 21:52 재시작) | **08:01 확정** | `clean_s2` | **18:10** |

레인4 는 두 셀 다 MIN 이라 **E 와 무관하게 결정적**이다. 레인1~3 만 E 에 종속된다.

→ phase 1 = **G1 8/9**. `frzero_s2` + L1 3셀만 phase 2 로 넘어간다(24.6h).

### 교체 타이밍

| | |
|---|---|
| 4레인 동시 유휴 시작 | **07-27 18:42** (레인4 는 18:10) |
| 강제 만료 | 07-27 21:11 |
| **창** | **2.5h** |
| **권장 교체** | **07-27 19:00** — 만료까지 2.2h 여유 |

만료까지 기다리지 말 것. 18:42 에 드레인 없이 자연 종료하므로 **19:00 에 교체하면 phase 2 를
2.1h 일찍 시작**한다(19:30 착수 → 완주 07-28 05:39, 마감 대비 마진 18.4h).

**E 슬립 시**: 레인1~3 의 두 번째 G1 은 21:11 전에 끝나야 하므로 **11:02 까지 착수**해야 한다
= **E ≤ 37.0h**. 넘으면 그 레인은 투입하지 말고(드레인) 해당 셀을 phase 2 로 넘긴다 —
phase 2 는 24.6h 뿐이라 G1 이 3셀 더 와도(총 55h/4레인) 여유가 있다.

### phase 2 배치는 seal 시점에 재분배할 것

레포 큐 그대로면 레인1 이 `obs_s2`+`frzero_s2` = 16.2h 를 혼자 지고 레인4 는 빈다(완주 11:41).
드라이버 정지 상태에서 `frzero_s2` 를 레인4 로 옮기면 **완주 05:39** 로 6h 당겨진다(§5 규정 허용).

### 이 배치가 안전한 이유

G1 은 cell-end 1회 persist 라 만료를 걸치면 전손인데, 단가가 10.2h 로 줄어 **만료 전에 두 셀이
온전히 들어간다**. E 가 빗나가도 판정선(37.0h)에서 드레인하면 손실 0 이고, 넘어간 셀은
phase 2 의 넉넉한 창이 흡수한다.

### 큐 조작 규약 (드라이버 가동 중 적용분)

- **런타임 큐**(`$BATCH/runlogs/logs_lane*/queue.run.txt`)만 수정 — 드라이버가 매 루프 `mapfile`
  로 재읽기하고 `case "$line" in ''|'#'*) continue` 로 주석을 건너뛴다.
- **주석만·삭제 없음** → 줄 수 보존 → `consumed` 인덱스 불변. **추가는 파일 끝 append 만**
  (기존 인덱스 안 밀림). 소진된 인덱스는 불가침. 원자적 쓰기(tmp→replace). 백업 `.bak*_07-26`.
- **셀 4개가 레포 큐와 다른 레인에서 돈다**(`L1_clean_online_s1`→L1, `L1_clean_online_s2`→L4,
  `G12 lr3e-3_s1`→L1, `G12 lr2e-3_s2`→L2). 그래서 §3-④ `seal` 이 **전 레인 로그를 뒤지도록**
  고쳤다 — 자기 레인만 보면 이 4개가 미완료로 남아 재기동 때 중복 실행된다.

## 2c. 07-27 04:46 실측 — E 확정 + `frzero_s0` 크래시

### 실측 단가 (`G1_L2_clean_s0`, 04:42:02 착지)

| 방법 | runtime | |
|---|---|---|
| **(b)oracle** | **18,426 s** | 5.12h — MIN 비용의 78% |
| **Flirds** | 3,791 s | 1.05h |
| **Flirds-1st** | 1,229 s | 0.34h |
| GTG / FedSV / ShapleyFL | 18,573 / 18,443 / 18,418 s | 각 ~5.1h ← **잘라낸 부분** |
| ComFedSV / loss-heur | 4,217 / 3,440 s | |
| FLDetector / STD-DAGMM / FLTrust / FedDQC | 5,733 / 4,654 / 1,244 / 252 s | 논문 제외분 |

- **9방법 셀 = 30.65h**(예측 34.5h 대비 −11%) · **MIN 셀 = FL 2.95h + 6.51h = 9.46h**
- **결과**: Flirds **+1.000** · Flirds-1st **+0.999** vs (b) — R4 주무대 fidelity 헤드라인.
  대조군은 GTG +0.988 · FedSV +0.917 · loss-heur +0.999 / **FedIF −0.055 · ShapleyFL −0.018 ·
  ComFedSV −0.101**(1차·renorm 계열 붕괴). clean 셀이라 AUROC 는 `nan`(오염 클라 0 = 정의 불가).

### ⚠ `frzero_s0` 전손 — 논문에서 뺀 탐지기에서 죽었다

```
flirds/baselines/std_dagmm.py:151  L = torch.linalg.cholesky(cov_reg)
torch._C._LinAlgError: ... input is not positive-definite (leading minor of order 7)
```

**30.05h 를 돌고 `_persist` 직전에 죽었다**(cell-end 1회 persist). 죽인 코드는 STD-DAGMM =
plan §0.2 로 **논문 전면 제외된 전용 탐지기 4종 중 하나**다. `MIN_METHODS=1` 이었다면 early-return
으로 도달조차 안 했다 — **남은 셀은 전부 MIN 이라 재발 불가**.
> 잠재 버그로 존치: `std_dagmm.py` 의 GMM 공분산이 비-PSD 가 될 때 cholesky 가 터진다.
> `phase2_matrix` 를 `MIN_METHODS` 없이 돌리는 다른 무대(silo5·device100)도 같은 경로를 탄다.

### `noisy_s0` 04:56:36 착지 (9방법 생존) — 탐지축 동반 산출

같은 STD-DAGMM 경로를 **통과**했다(4,791 s) → 07-27 04:06 의 크래시는 frzero 데이터 고유의
비-PSD 공분산이지 코드 전면 결함은 아니다. noisy 셀이라 AUROC 가 정의된다:

| | AUROC | Spearman/(b) | | | AUROC | Spearman/(b) |
|---|---|---|---|---|---|---|
| (b)oracle | 1.000 | (truth) | | ShapleyFL | 1.000 | +0.678 |
| **Flirds** | **1.000** | **+0.999** | | **ComFedSV** | **0.583** | **−0.022** |
| **Flirds-1st** | **1.000** | **+0.996** | | **FLDetector** | **0.483** | 논문 제외 |
| loss-heur | 1.000 | +0.997 | | **STD-DAGMM** | **0.682** | 〃 |
| GTG | 1.000 | +0.985 | | FLTrust | 1.000 | 〃 |
| FedSV | 1.000 | +0.948 | | FedDQC | 1.000 | 〃 |
| FedIF | 1.000 | +0.651 | | | | |

- φ 계열은 **전부 AUROC 1.000**, 전용 탐지기는 **FLDetector 0.483(우연 이하)·STD-DAGMM 0.682** 로
  갈린다 — 제외 결정(plan §0.2)과 배치되지 않는 결과다.
- clean 에서 붕괴했던 FedIF(−0.055)·ShapleyFL(−0.018)이 noisy 에선 +0.651/+0.678 로 올라온다
  (오염이 실신호를 만들기 때문). **ComFedSV 만 양쪽에서 붕괴.**

### ⚠ G1 rundir 이름 — `clean` 에도 `nr0.7` 이 붙는다 (문서와 불일치)

```
실제:  1B_gsm50k5_clean_nr0.7_s0     1B_gsm50k5_noisy_nr0.7_s0
문서:  1B_gsm50k5_{clean, noisy_nr0.7, frzero}_s{seed}      ← REMAINING-b200.md:86, 틀림
```
`phase2_matrix` 는 `track_g`(`:558`, threat∈{noisy,mixed} 에만 부착)와 달리 **nr 태그를 무조건**
붙인다. `1B_gsm50k5_clean_s*` 로 글롭하면 clean·frzero 셀을 통째로 놓친다 — L11 의 dose-태그 함정과
같은 계열이다. **완료 판정은 이름 글롭이 아니라 `MATRIX DONE` + `phi.parquet` 존재로 한다.**

### ⚠ 공유 브랜치 동기화 — **양방향 공백** (07-27 06:00 확정)

| 쪽 | 미반영분 |
|---|---|
| B200(여기) | **rundir 7개 미커밋**: `1B_gsm50k5_{clean,noisy}_nr0.7_s0`(G1 seed0 2셀!) · `gsm50k5_clean_observer_seed1` · `t2_sign_{flirds,flirds1st,fedif,lossheur}_seed1` |
| HJ | **7 커밋 미푸시** (L11 6셀 + 집계 재생성) |

`origin/main` 확인: `runs/track_h/rundirs_llm/` 아래 `clean_observer` 는 **seed0 하나뿐**.
→ 앞서 "HJ 뷰에 없어서 난 NaN" 이라 한 설명은 **틀렸다**. 이쪽 seed1 이 애초에 커밋되지 않았다.
**push 는 Yonghee 전담**이므로 양쪽 다 대기 상태다. `L1_clean_obs_s2`(19:15) 착지 후
**양쪽을 한 번에 맞추는 것**이 깔끔하다(HJ 제안, 동의).

### 갱신 일정 (측정 기반 · 07-27 05:00)

| 레인 | 진행 중 → 착지 | 꼬리 → 착지 |
|---|---|---|
| 1 | `noisy_s1` → **14:26** | `L1_clean_online_s2` → **18:38** |
| 2 | `clean_s1` → **14:10** | `L1_clean_online_s1` → **18:22** |
| 3 | `frzero_s1` → **13:12** | `L1_clean_obs_s2` → **19:15** |
| 4 | `noisy_s2` → **07:20** | `clean_s2` → **16:48** |

- 착지 완료: `clean_s0`(04:42, 9방법) · `noisy_s0`(04:56, 9방법). `frzero_s0` 는 크래시 → phase 2.
- **phase 1 종료 19:15 → 교체 ~19:30**(만료까지 1.7h) → phase 2 = `frzero_s0` 재실행 + `frzero_s2`
  2셀뿐 → **완주 07-28 04:36 · 마감 대비 마진 19.4h**.

## 3. 교체 절차

### ① 07-27 16:00경 — 상태 점검
```bash
bash runs/track_h/b200_fleet.sh status
```
마지막 G1(`*_s1`, `G1_clean_s2`)이 예정(13:15 / 18:45)대로 가는지 본다.
**단가가 밀려 마지막 셀이 21:11 을 넘길 것 같으면 그 셀을 새로 시작시키지 않는다** → ②로.

### ② 신규 투입 차단 (드레인)
```bash
for c in 1 2 3 4; do bash runs/track_h/b200_fleet.sh drain $c; done
```
런타임 큐의 남은 줄을 **주석 처리만** 한다(줄 수 보존 = 드라이버 `consumed` 인덱스 안 밀림).
진행 중 셀은 완주하고 드라이버가 스스로 종료한다. **`kill` 금지** — G1 이면 19.6h 전손이다.

### ③ 전 레인 종료 확인
```bash
bash runs/track_h/b200_fleet.sh status     # driver=idle × 4
```

### ④ 완료 셀을 레포 레인 큐에 확정(seal)
```bash
bash runs/track_h/b200_fleet.sh seal
```
셀 로그에 `MATRIX DONE` / `TRACK G DONE` / `TRACK D DONE` / `[persist]` 가 있는 줄만 `#` 주석한다
(**삭제 아님**). 미완료 줄은 남아서 재기동 시 거기서 이어간다.

### ⑤ 컨테이너 교체 (Yonghee)
- lustre(`/NHNHOME/.../yonghee`)는 공유·영속이라 rundir·로그·큐는 그대로 살아남는다.
- 홈(`/home/edgeai_lab`)은 노드-로컬 overlay 라 사라진다 — 런처가 `HOME=$BATCH/home` 으로
  갈아끼우므로 무관하다.
- 노드 이름이 바뀌면: `FLEET_NODES="새이름1 새이름2 새이름3 새이름4" bash …/b200_fleet.sh up`

### ⑥ 재기동
```bash
bash runs/track_h/b200_fleet.sh nodes      # 4대 도달·GPU 확인
bash runs/track_h/b200_fleet.sh up         # PHASE 2 부터 이어간다
bash runs/track_h/b200_fleet.sh status
```

## 4. 하드룰 (정본과 동일)

- 큐 **줄 삭제 금지** — `#` 주석만. 가동 후 **순서 변경 금지**.
- **`BATCH` 를 export 하지 말 것** — 세 러너가 batch-size 노브로 읽는다. 런처가 `export -n` +
  `env -u BATCH` 로 막아 뒀다.
- **`VAL_MAXLEN`·학습 batch 는 어떤 이유로도 변경 금지**(φ·궤적이 달라진다).
  `VAL_CHUNK` 는 청크 합산이 exact 라 φ 불변이지만, B200(180GB)에선 조정 불요다
  (A6000 48GB 고유 제약이라 HJ 쪽만 3 으로 낮춰 쓴다).
- 어디에도 `ROUNDS` 를 주지 않는다(R4 = R=200, 레짐 기본).
- **기동 후 `git pull` 금지** — 셀이 다른 sha 로 갈린다.
- push 는 Yonghee 직접. 논문 수치는 rundir/analysis 재생성 값만.
- **완주 판정에 exit code 를 쓰지 않는다** (HJ 07-26 실사례). `sbatch_l11_online.sh` 말미의 `echo` 가
  python 종료코드를 덮어써서 **8초 만에 전멸한 6셀이 전부 `ExitCode=0:0`** 으로 보였다. 또 Track G
  배너(`=== Track G | 1B … ===`)는 `_load()` **앞에서** 찍히므로 배너 출력 ≠ 모델 로드 성공이다.
  실행 증거는 **`grep -c train_runtime <log> >= 1`**.
  → **B200 은 이 함정에 면역**(확인): `run_multi_driver.sh:39` · `b200_fleet.sh:32` 모두 exit code 를
  안 보고 `MATRIX DONE|TRACK G DONE|TRACK D DONE|[persist]` 마커로만 판정한다. 다만 남의 로그를
  넘겨 읽거나 착지를 보고할 때는 `train_runtime` 카운트를 함께 본다.

## 5. 예정 밖 상황

| 상황 | 조치 |
|---|---|
| G1 단가가 19.6h 를 크게 넘김 | 만료 전 마지막 셀 투입을 끊고(②) 그 셀은 phase 2 로. 레인 큐에서 미완료로 남으므로 재기동만 하면 이어간다 |
| 컨테이너가 예고 없이 죽음 | `$BATCH/runlogs/seal_watchdog.sh` 재가동 권장. 죽은 뒤엔 ④ seal → ⑥ up |
| 특정 레인만 빨리 빔 | 다른 레인 큐 하단의 미착수 줄을 옮겨 붙일 수 있다(드라이버 정지 상태에서만) |
| ④ L11 seed0 (HJ 몫) 실패 | 정본 `queue_b200.txt` 하단 폴백 3줄을 해제해 레인에 추가 — **07-25 23:40 해소: HJ 3/3 완주(EXIT=0), 폴백 잠금 유지 확정.** 큐 변경 없음 |

## 6. L11 집계 주의 — 9셀 전부 HJ/A6000 (07-26 이관 확정)

⑤L11 은 **seed0·1·2 전부 HJ(A6000)** 가 돈다. seed1·2 6셀은 07-26 B200 에서 이관했다
(job **`1885728`**, array `21-23,42-44%8`; 첫 제출 `1885698` 은 HJ 계정의 `HF_HOME` 기본값
`/scratch/chyoyhr/hf_home` 의 `token` 이 mode 600 이라 6셀 전부 8–10초 만에 사망 → `HF_HOME` 명시로
재제출). **착지 루트 = `runs/track_h/rundirs_llm_hj/`
9셀 전부 — 루트 하나다.** (B200 큐 원문의 `RUNDIR_ROOT=…/rundirs_llm` 은 B200 착지 전제였고 폐기됐다.)

- ⚠ **`sbatch_l11_online.sh:28` 은 `SEED==2 → rundirs_llm_yh` 로 자동 분기**한다. HJ 가 `RUNDIR_ROOT`
  를 명시해 seed2 3셀도 `_hj` 로 보냈다. 다음에 이 스크립트를 쓸 때 명시하지 않으면 seed2 만 `_yh`
  로 흩어진다. (`make_analysis.py:95` 가 `_hj`·`_yh` 둘 다 로드하므로 집계 자체는 되지만 경로가 갈린다.)
- ⚠ **`--export` 는 chyoyhr 외 계정에선 필수**다 — `:13` 의 `REPO=${REPO:-$HOME/projects/flirds}`
  기본값이 HJ 홈(`~/flirds`)에 없고, `PY` 도 chyoyhr 하드코딩이라 700 권한으로 못 읽는다.
- **중복 충돌 없음**(검증함): `make_analysis.py:105` 가 `groups[key][cfg["arm"]]` 로 **arm 을 내부
  키**로 쓴다 → (regime,threat,nr,seed) 가 같아도 arm 이 다르면 공존한다. 기존 `rundirs_llm` 의
  flirds1st seed1·2 는 `t2_sign_flirds1st`(다른 arm)라 `flirds1st_gate_v2` 를 덮지 않는다.

0. **⚠ 동기화 격차 (07-27 05:40 확인)** — HJ 가 6/6 완주를 통보했으나(job `1885728`, 46.5 GPU-h,
   3-seed 패턴 seed0 재현) **이 파일시스템엔 seed0 3개뿐**이고 그쪽 커밋 `19f80d7`·`593eb88`·
   `9e30f40`·`38aabc4`·`28c046b`·`eccde71` 이 `git cat-file` 미존재다(seed0 배치의 `381a5fd`·
   `17315c3`·`ae4f212` 는 HEAD 조상으로 존재 → seed0 은 흘러왔고 이번 6셀만 안 왔다).
   **최종 집계 전 push/동기화 필수.** 안 되면 `llm_competition.csv` 318행이 이쪽에서 재현되지 않는다.
   > 파생 오진 1건: HJ 가 "clean 열 `delta`·`recovery` NaN = `observer` clean 이 seed0 뿐" 이라 했으나
   > 이쪽엔 `gsm50k5_clean_observer_seed1`(07-26 04:05 착지, `final_val_loss=0.6068`)이 **있다**.
   > 그쪽 뷰에 없어서 난 NaN 이다. seed2 는 `L1_clean_obs_s2`(오늘 19:15)가 닫는다.

1. **noisy 셀 이름에 dose 태그가 붙는다** — `gsm50k5_noisy_**nr0.7**_flirds1st_gate_v2_seed*`.
   `gsm50k5_noisy_flirds1st_*` 로 글롭하면 **noisy 3셀 전부 놓친다**(HJ 감시자 오탐 사례).
   `track_g.py:558` — `gsm50k5` 는 `NOISY_RATE` 기본 0.7 이라 `THREAT∈{noisy,mixed}` 에서 항상 붙는다.
2. **clean 의 `precision=0.0 / recall=None` 은 실패가 아니라 축퇴값**이다 — 오염 클라 0 → 참양성 미정의,
   배제 전건이 정의상 오배제. 읽을 값은 **오발화율**(seed0 = 1.03%, 103 pair).
3. **noisy recall 이 낮은 건 §2.1 예측과 방향이 맞다** — `runs/track_g/README.md:42` 등록 예측
   "noisy@canon 부호-게이트 = parity(게이트 침묵); nr∈(0,1] 에 0-교차 없음". **미스가 아니라 히트로 읽는 칸**이고
   회수는 z-게이트/V2w 몫이다. (등록 예측은 Flirds 기준 · 이 셀은 flirds1st — 그 차이만 명시.)
4. **스택은 3-seed 전부 동일**(이관의 부수 이득) — 9셀 모두 A6000/torch 2.11.0+cu128.
   당초 우려했던 seed 간 스택 혼재(2.11 vs 2.12)가 사라져 cross-seed 분산이 깨끗하다.
   단 §5.5 canonical timing 은 B200 산출만 쓰므로 **L11 `timing.json` 은 cost 표에 넣지 않는다**
   (원래도 L11 은 timing 소스가 아니다).
