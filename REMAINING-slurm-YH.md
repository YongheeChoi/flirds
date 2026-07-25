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

## 2. ★ 선행 코드 변경 2건 (최우선 — 다른 계정을 막고 있다)

### C-a — mnist 무대 개방 (**1줄**)

```
codes/experiments/track_c2.py:157
  MODEL_FN = partial({"cifar10": FedSVCNN, "fmnist": LeNet5, "mnist": LeNet5}[DATASET], width=WIDTH)
```
- `flirds/data/cnn.py` 가 mnist 로더·정규화를 이미 갖고 있다(l.11 `"mnist": ((0.1307,), (0.3081,))` · l.24 분기).
- `track_c2_fid.py` 는 `import experiments.track_c2 as c2` 후 `c2.MODEL_FN`·`c2.DATASET` 을 쓰므로 **이 1줄이 fidelity 러너까지 파급**된다.
- **여는 것**: §4 G8(mnist fidelity+탐지) · §5 G10(mnist downstream).

### C-b — C1 을 논문 오염축·파티션으로 정렬 (중)

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

## 7. CNN 완주 후 — LLM probe 2건 (48GB 파티션)

> YH 의 CNN 몫은 ~32 wall-h 라 **07-27 오전에 슬롯이 빈다.** 그때 48GB 파티션으로 넘어가 아래 2건을 맡는다(둘 다 1B LoRA → 24GB 불가). R4 무대가 아니므로 **`ROUNDS` 를 주지 않는다**(각자 자기 레짐 값).

| 항목 | 무엇 | 셀 | GPU-h |
|---|---|---|---|
| **G5** | 2차항(HVP) LLM 레그 seed 보강 — **본문 §5.6①** | 4 (`1B_std50k5_r{32,64}_seed{1,2}`) | ~20 |
| **G12** | A축 lever probe seed 보강 — **부록**·최저 우선 | 19 | 50–90 |

```
# G5 (본문). ORACLE_A=0, 착지 = runs/probe_signal/rundirs
REGIME=std20 N_CLIENTS=50 K_ABS=5 LORA_R=<32|64> SEED=<1|2> ORACLE_A=0 \
  RUN_NAME=1B_std50k5_r<r>_seed<s> RUNDIR_ROOT=$REPO/runs/probe_signal/rundirs \
  VAL_CHUNK=2 PYTHONPATH=. $PY -u experiments/track_d.py
```
- **`VAL_CHUNK=2`**: flirds HVP 경로라 48GB 에선 낮춰야 한다(청크 합산 exact → **φ 동일**).
- **G12 내역**(19셀): anchor5 `lr{1,2,3}e-3 × st{20,30}` seed1·2 · anchor5 `r{32,64}` seed1·2 · `noise_1B_r64` seed1·2. 핵심 질문은 "lr 로 커진 φ가 cross-seed 실재 신호인가"(예측 ρ≈0)라 **`lr{2,3}e-3` 계열을 먼저** 돌린다.
- 여유가 없으면 **G12 는 버린다**(부록·P2). G5 는 본문이라 우선.

## 8. 우선순위 · 예상 종료

| P | 무엇 | 물량 | 근거 |
|---|---|---|---|
| **P0** | **C-b**(+C-a 1줄) 코드 | — | JW 의 505 GPU-h 를 막고 있다 |
| **P1** | G3 seed0 → 전량 | 96런 · ~80 | 본문 downstream 의 빈 절반 · 코드 불요 |
| **P2** | G8 | 24런 · ~25 | 부록 fidelity+탐지가 한 rundir |
| **P3** | G10 | 216런 · ~135 | 부록 downstream(P1·P1w 동시) |
| **P4** | G6 | 9런 · ~15 | 본문 ablation |
| **P5** | G5 → G12 (48GB) | 4 + 19런 · 70–110 | 본문 ablation seed 미달 → 부록 |

- **CNN 계 ~255 GPU-h** / 8슬롯 → **~32 wall-h** → **07-27 오전** → 이후 §7(G5·G12) → **07-27 후반**.
- 슬롯이 남으면 **JW 의 G9(mnist (a)-오라클) 꼬리를 work-steal** — 같은 3090·같은 env·셀 단위 idempotent.

## 9. 완료 후 · 미해결 배선

1. rundir 커밋(push는 Yonghee) → `make_analysis.py` 재생성 → `flirds-results-{downstream,fidelity,detection}` → paper.
2. **rundir 정체성 잔여**: `track_c1`·`track_c2`·`track_c2_fid`·`track_d`·`phase1_*` 는 아직 `identity=None`(C1 재실행이 `*_<hash>` 를 낸 원인). **C-b 작업 중 `track_c1` 만이라도 정합**시키면 G2·G9 착지가 깨끗해진다.
