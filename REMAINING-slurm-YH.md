# REMAINING (Slurm · YH = chyoyhr) — 선행 코드 변경 + CNN 점수원 경쟁 신설

> 배분 정본 = **`REMAINING-00-INDEX.md`** · 수록목록 정본 = `research-wiki/survey/flirds-paper-experiment-plan.md`.
> **역할 = (1) 선행 코드 변경 2건**(JW 를 막고 있음) **+ (2) CNN 점수원 경쟁 신설**(cifar10/iid · mnist).
> **하드웨어 = RTX3090 24GB**(클러스터 여유 21장 · 8-GPU QOS). CNN 은 전량 3090 — LLM 1B 은 24GB 불가라 여기 오지 않는다.
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 전 실험 3-seed(seed-major). push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만. 기존 rundir read-only.
> **현재 YH 큐는 비어 있다**(종전 CNN 주무대 전량 완주).

## 0. 환경

- conda `lora4cl`(`/home/chyoyhr/anaconda3/envs/lora4cl/bin/python`, **torch 2.11.0**) · partition `base_suma_rtx3090` · 8-GPU QOS(`QOSMaxGRESPerUser`).
- 리포 = `/home/chyoyhr/projects/flirds/` · `codes/` 에서 `PYTHONPATH=.` · `HF_HUB_OFFLINE=1`.
- **스택 고정**: CNN 산출물 전량이 torch 2.11 이다. 신규 셀도 2.11 에서 돌려야 셀 내부에서 recovery **분모**(vanilla·oracle_excl)와 소스 arm 이 같은 스택에 놓인다. A6000(2.12)로 옮기면 스택이 쪼개진다.

## 1. 완료 — 손댈 것 없음

| 실험 | 상태 |
|---|---|
| **1A-CNN 부분참여 fidelity(+φ-AUROC)** = c2fid | ✅ **144/144**(로컬 rundir 전수 대조 2026-07-25). 확정 스코프분(cifar10 {iid,dir1} × 4위협 × 3seed = 24셀) 전부 존재 → **잔여 0** |
| **2-CNN 점수원 경쟁 — cifar10/dir1** | ✅ 3-seed 완비 |
| **W-B P1w 관측자(T2) leg** | ✅ **90/90** |
| **C1 β0.3 재실행 30셀** | ✅ 완주·커밋 `47680ec` |

**W-B 결과(rundir-only 재생성 `make_p1w_cnn_table.py`; flirds rows 804 · T2 240)** — P1w 승격 판정의 CNN 근거:
online 오염평균 acc P1 +0.650 / P1w +0.653 → **gap +0.003**(dir1 참조 +0.007) · retrain P1 +0.667 / P1w +0.660 → **gap −0.007**(dir1 참조 −0.015). recovery(guard, 4셀 드롭) online **+0.152** / retrain **−0.050**. clean dAcc online −0.006/−0.007 · retrain −0.004/**−0.017**(밴드 ±0.006 — retrain P1w 만 이탈).
> **⚠ W-B 단독으로 P1w 를 판정하지 않는다** — 승격 규칙이 "CNN·LLM 전 범위"라 **LLM 레그가 있어야 판정이 성립**한다(그 레그 = `REMAINING-b200.md` §5 옵션).

**취소·폐기(재제안 금지)**: fmnist competition Part B(취소 시점 seed0 45/96 보존·seeds1-2 미착수) · C-fr frrand(완주했으나 frrand 는 축 밖) · LLM downstream 보조(미제출). 전부 **표에서만 제외**, rundir 은 존치.

## 2. ✅ 선행 코드 변경 2건 — **착지 완료 (`d09e528`, 07-25). 게이트 전부 해소**

> C-a·C-b 가 `origin/main` 에 있다. JW·JB 워처가 풀렸고 **c1축 492 GPU-h(4계정)와 G10 216런이
> 착수 가능**하다. 정본 = **YH 버전**(Yonghee 판정) · JW 의 중복 구현 `989f5ca` 는 철회(`f056b16`).
>
> ### 오염-집합 규약 — 전 러너 전수 확인 결과 **3종**. 논문 부록 B 에 **2문장 추가**가 남았다
>
> `Bernoulli(ρ=0.4)` 는 코드에 **딱 한 곳** — `track_c2.py:258` 의 `label_flip` 분기.
>
> | 무대 | 러너 | label_flip | fr · gn | LLM noisy · frzero | 실현 수 |
> |---|---|---|---|---|---|
> | CNN N=100 주무대 | `track_c2`(+`_fid`) | **Bernoulli(ρ=0.4)** | 랜덤 고정 40 | — | lf **39/48/47** |
> | **CNN N=10 (a) 무대** | `track_c1` | **랜덤 고정 4** | **랜덤 고정 4** | — | **4/4/4** |
> | LLM 전 무대 | `phase2_matrix`·`track_g` | — | — | **결정적 `0..19`** | **정확히 20 = 40%** |
>
> - **Bernoulli 는 CNN 주무대 전체의 규약**이고 **논문 본문 CNN 표 전부가 그 위에 있다** —
>   §5.2 fidelity(c2fid) · §5.4 탐지(같은 rundir) · §5.3 downstream(dir1 기존 + **G3** + G10) · **G8**.
>   `track_c2_fid` 는 `track_c2` 를 read-only 재사용하고 track_g CNN 그리드도 `run_cnn_grid.sh:30`
>   이 `track_c2.py` 를 호출하므로 **같은 draw** → G3 의 arm 병합이 같은 오염 집합 위에서 이뤄진다.
>   단 **적용 대상은 `label_flip` 셀뿐**(4위협 중 1열).
> - `track_c1` 이 고정을 쓰는 근거(l.188-192, 채택됨): ① 도즈를 0.70 으로 고정해 **FedCorr 를
>   재현하는 게 아니다**(τ~U(0.5,1) 미사용) ② N=10 Bernoulli 는 **3/7/6** 으로 흔들려(평균 5.33 vs
>   명목 4 · sd 1.55) label_flip 열이 다른 도즈가 되고 **4위협이 한 fidelity 표를 공유할 수 없다**.
>   N=100 은 39–48 = ~12% 라 무해하지만 N=10 은 **~39%** 다. (검증: 같은 draw 를 N=100 에 적용하면
>   논문 기재값 39/48/47 을 정확히 재생산한다.)
> - LLM 의 결정적 인덱스는 **WLOG** — `build_gsm8k_iid`(`flirds/data/llm.py:311-313`)가 공식 train 을
>   `shuffle(seed)` 후 등분하므로 클라 `0..19` 가 쥔 데이터가 시드마다 바뀐다. noisy·frzero 가 같은
>   인덱스를 쓰는 것도 설계다(위협만 바꾼 대조).
> - **미수록 확인**: `std50k5` 오염 규약(각 10%)은 수록 대상이 아니다 — 수록 std50k5 는 **G5(clean,
>   `track_d`)** 뿐이고 `std50k5 mixed` 는 미수록 목록에 있다. 위 3종이 전부다.
>
> **✅ 해결 = 논문에서 FedCorr 주장을 걷어냈다**(Yonghee 07-25). **코드 변경 0 · 재실행 0.**
> "c1축을 Bernoulli 로 맞출까"를 검토하다 근거 ①이 **주무대에서도 성립하지 않음**을 확인했다 —
> 주무대 rundir 도 `label-flip_fr0.70` 로 도즈를 0.70 에 고정하면서 Bernoulli 를 유지한다.
> 게다가 FedCorr 와 맞은 건 4개 축 중 1개뿐이었다: 집합 draw ✔ / 강도 0.70 고정 ✗ /
> relabel = 참 라벨 제외 K−1 균일 ✗(FedCorr 는 전 K 균일 = (K−1)/K 희석) / 3무대 중 1 ✗.
> ⟹ "공식 구현 그대로"는 과대주장 → **주장을 걷어내고 우리 프로토콜을 서술**한다.
> `paper-ko.md` 부록 B.2 문단 **재작성 완료**(FedCorr 언급 2줄이 그 문단에만 있었다):
> 강도는 전 무대 상수 고정(원칙) + 오염 집합은 무대별 실현 수 병기 표(`CNN-Main` 39/48/47·40 ·
> `CNN-Small` 4 · `LLM-Main` 20 · **`LLM-Device` 5(5%)** · **`Silo` 1(20%)**).
> **⚠ `paper-ko.md` 는 미커밋** — 다른 세션의 미커밋 변경 916줄이 함께 들어간다. 충돌 확인 후 커밋.
>
> **기대 corrupt 집합**(N=10 · 전 위협 공통 · seed-only · `default_rng(1000+seed)` 첫 소비 =
> `track_c2` 와 동일 스트림): seed0 `[1,4,6,7]` · seed1 `[0,4,6,7]` · seed2 `[3,4,6,9]`.
>
> ### 착지한 6파일 중 공용 코어 2개 — B200 과의 관계
>
> `flirds/fl/server.py` · `flirds/oracle/exact_sv.py` 는 **전 러너 공용**이다(G3 track_c2 ·
> B200 G1 phase2_matrix · track_g). **B200 은 아직 미기동**이라 착지 이후에 시작하면 전 셀이
> 같은 sha 로 기록된다 = 쪼개짐 없음. 반대로 **기동 후에는 pull 하지 않는다**(같은 표의 셀이
> 다른 코드로 갈린다). C-a 는 dict 키 추가라 cifar10/fmnist 경로에 원리적으로 무해하므로
> 이미 도는 G3 는 무영향이다.

### C-a — mnist 무대 개방 (**1줄**) — 착지 완료, 아래는 명세 기록

```
codes/experiments/track_c2.py:157
  MODEL_FN = partial({"cifar10": FedSVCNN, "fmnist": LeNet5, "mnist": LeNet5}[DATASET], width=WIDTH)
```
- `flirds/data/cnn.py` 가 mnist 로더·정규화를 이미 갖고 있다(l.11 `"mnist": ((0.1307,), (0.3081,))` · l.24 분기).
- `track_c2_fid.py` 는 `import experiments.track_c2 as c2` 후 `c2.MODEL_FN`·`c2.DATASET` 을 쓰므로 **이 1줄이 fidelity 러너까지 파급**된다.
- **여는 것**: §4 G8(mnist fidelity+탐지) · §5 G10(mnist downstream).

### C-b — C1 을 논문 오염축·파티션으로 정렬 — 구현 완료, 아래는 명세 기록

현 `track_c1.py:79` 는 `C1_SCENARIO` 하나에 **파티션과 오염을 섞어** 두고(iid\|label_skew\|quantity_skew\|label_flip\|feature_noise) free-rider·grad-noise 구현이 없다 → 확정 오염축 3종과 **한 칸도 겹치지 않는다**.

**요구 env 계약**(`runs/track_c/c1/sbatch_c1_axis.sh` 가 이 이름으로 호출한다):

| env | 값 |
|---|---|
| `C1_PARTITION` | `iid` \| `dir1` |
| `C1_THREAT` | `clean` \| `label_flip` \| `free_rider` \| `grad_noise` |
| `C1_FLIP_RATE` | `0.70` (label_flip 도즈 — 기존 pair-ladder 대체) |

- **이식 소스는 이미 있다**: 오염 = `flirds/data/corruptors.py`(`CNN_CORRUPTORS`, track_c2 가 쓰는 것) · dir1 = `flirds/fl/partition.py`. 새 로직이 아니라 **C1 로 옮기는 작업**.
- **여는 것**: **JW 의 G2+G9 = 505 GPU-h(CNN 최대 물량)** + §6 G6. **JW 는 이게 착지할 때까지 착수 불가.**

## 3. G3 — cifar10/iid 점수원 경쟁 (본문) · 96런 · **코드 불요 → 즉시**

> 본문 downstream "2-CNN P1 부호-게이트 — cifar10/iid". dir1 은 8점수원 3-seed 완비인데 iid 는 **flirds 만** 있다.
> **부록 P1w 는 추가 런 0** — 같은 rundir 이 P1(`gate_v2`)과 P1w(`gatew_v2`) arm 을 함께 낳는다(계획서 §7.2).
> **관측자에 `C2_OBS_SRCS` 를 지정하지 않는다** → 기본값 = **8소스 전량 T2**. 기존 iid 관측자(`_obsf`)가 flirds 만 담아 retrain 열이 비었던 게 이 결손의 원인이다.

- 4위협(clean·lf@0.70·free_rider·grad_noise) × (7 비-flirds + 관측자) × 3seed = **96**. flirds online arm 은 `track_g/rundirs_cnn` 의 cifar10 iid 그리드에 이미 있다.
- **비용 추정 ~60–100 GPU-h**(소스런 ~0.4h · 관측자 ~2h; W-B obsf 실측 0.35h/런에서 유추).
```
mkdir -p runs/track_h/_logs
sbatch --array=0-31%8   runs/track_h/sbatch_cnn_iid_comp.sh
sbatch --array=32-95%8  runs/track_h/sbatch_cnn_iid_comp.sh
```

## 4. G8 — mnist 부분참여 fidelity(+탐지) · 24런 (C-a 이후)

- cifar10 본문 무대와 **동일 세팅, 데이터셋만 mnist**. fidelity(부록)와 φ-AUROC(부록)가 **같은 rundir**.
- {mnist × [iid, dir1]} × 4위협 × 3seed = **24** · **~25 GPU-h**(1.05 GPU-h/셀 실측).
```
mkdir -p runs/track_c/c2fid/_logs
sbatch --array=0-7%8   runs/track_c/c2fid/sbatch_fid_mnist.sh
sbatch --array=8-23%8  runs/track_c/c2fid/sbatch_fid_mnist.sh
```

## 5. G10 — mnist 점수원 경쟁 (부록) · 216런 (C-a 이후)

- 2파티션 × 4위협 × (**8**소스 + 관측자) × 3seed = **216**. mnist 는 track_g 그리드가 없어 **flirds 소스도 여기서 생성**(7이 아니라 8인 이유).
- **비용 추정 ~110–160 GPU-h**. P1·P1w 동시 산출.
```
sbatch --array=0-71%8    runs/track_h/sbatch_cnn_mnist_comp.sh
sbatch --array=72-215%8  runs/track_h/sbatch_cnn_mnist_comp.sh
```

## 6. G6 — Removal-curve CNN 오염축 정렬 · 6~9런 (C-b 이후)

- 현 removal 시나리오({feature-noise, label-flip 사다리, iid})가 확정 오염축과 불일치 → **frzero·grad-noise** 에서 worst-first 제거 → acc 분리(순위→성능 인과)를 다시 낸다.
- frzero·grad-noise × 3seed = **6**(+ lf@0.70 재실행 시 3) · 추정 ~10–20 GPU-h.
- 러너 = `runs/removal_dose/run_cnn_removal.sh` — **C-b 의 위협 토큰 확장을 공유**(별도 코드 없음).

## 7. 우선순위 · 예상 종료

| P | 무엇 | 물량 | 상태 | 근거 |
|---|---|---|---|---|
| ✅ | **코드 C-a·C-b + push** | 6파일 | ✅ **완료 `d09e528`** | 4계정 게이트 전부 해소 |
| **P1** | G3 `hiidcomp` | 96런 · ~80 | 🟢 10/96 완료 · 8 실행 · 86 남음 | 본문 downstream 의 빈 절반 |
| **P2** | **c1축 `c1axis` `0-7,16`** | 9셀 · ~82 | 🟡 PD(셀 ~9.1h) | **본문 G2 의 P0 seed** — C-b 를 쓴 사람이 직접 |
| **P3** | G8 `c2fidmn` | 24런 · ~25 | 🟡 PD | 부록 fidelity+탐지가 한 rundir |
| **P4** | G6 `c1rmax` | 9런 · ~15 | 🟡 PD | 본문 ablation |
| — | ~~G10 216런~~ | — | ✅ **HJ 로 이관 → 취소(전량 PD·mnist rundir 0)** | 개정 배분 §5 |
| ✅ | **논문 부록 B.2 오염-집합 문단 재작성**(FedCorr 주장 철회 + 무대별 실현 수 표) | — | ✅ 초안 완료(**미커밋** — 타 세션 916줄과 겹침) | §2 |

- 실측 재집계: **128 태스크 ~172 GPU-h**(G3 일부 선행분 반영). 임계경로 = G3 배수 ~6.5h 후
  c1축 8슬롯 2웨이브 ~18h → **07-26 밤~07-27 오전**. 마감 07-28 24:00 대비 여유.
- G3 가 10/96 인 이유 = 지난 세션 편집 창 사고로 62셀 사망 → 재제출(`1878494`), 그 62셀은 큐에 존재.
- **c1축은 P2 로 올라와 G8·G6 보다 앞선다**(본문 > 부록). YH 가 이미 취소·재제출로 정렬 완료(손실 0).

```
sbatch --array=0-7,16%8  runs/track_c/c1/sbatch_c1_axis.sh    # cifar10 seed0 8셀 + seed1 1셀 (C-b 착지 후)
```

- **YH 몫 ~202 GPU-h** / 8슬롯 → **~25 wall-h** → **07-27 오전**. 코드 작성 중에도 **G3 는 게이트가 없어 병행 가동**할 수 있다.
- **c1축 cifar10 seed0(0-7)을 YH 가 맡는 이유**: 본문 G2 의 최우선 seed 인데, C-b 를 쓰는 사람이 직접 돌려야 게이트 해제 즉시 착수된다.
- **G10 은 HJ, G5·G12 는 B200 c4** — 여기가 아니다.
- 슬롯이 남으면 JW·JB 잔여를 work-steal(남은 `--array` 범위만).

## 9. 완료 후 · 미해결 배선

1. rundir 커밋(push는 Yonghee) → `make_analysis.py` 재생성 → `flirds-results-{downstream,fidelity,detection}` → paper.
   - **G2/G9 (a)-fidelity 집계기는 새로 쓰지 않는다** — `runs/track_c/make_figures.py` l.99-152
     `load_c1()` 이 이미 c1 rundir ↔ (a) 페어링 + `phi_a` 음수화 + Spearman 을 한다(= G2 표).
     막힌 건 l.36 `SCENARIOS` 상수가 구 5축이라는 것뿐 → `{PARTS} × {THREATS}` 격자 +
     이름 패턴 `{ds}_{part}_{ttag}_seed{seed}`(sbatch l.65 고정)로 교체. 이름 패턴이 이미
     고정이라 **첫 셀 착지를 기다리지 않고 지금 써도 되고**, 착지 후 검증하면 된다.
     확인 1건: `C1_ORACLE_A=1` 의 `phi_a` 가 같은 rundir metrics.json 인지 별도
     `c1_oracle/*_aonly_*/` 인지(l.174 각주는 후자 가정) — 페어링 경로가 여기서 갈린다.
   - (구 세션이 `sbatch_c1_axis.sh` 헤더에 적은 `runs/track_c/c1/make_analysis.py` 는
     **존재하지 않는 파일**이었다. 07-25 헤더 정정 완료.)
2. **rundir 정체성 잔여**: `track_c1`·`track_c2`·`track_c2_fid`·`track_d`·`phase1_*` 는 아직 `identity=None`(C1 재실행이 `*_<hash>` 를 낸 원인). **C-b 작업 중 `track_c1` 만이라도 정합**시키면 G2·G9 착지가 깨끗해진다.
