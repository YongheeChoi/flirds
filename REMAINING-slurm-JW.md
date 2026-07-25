# REMAINING (Slurm · JW) — **LLM → CNN 전환**: (a) 재학습 오라클 무대

> 배분 정본 = **`REMAINING-00-INDEX.md`** · 수록목록 정본 = `research-wiki/survey/flirds-paper-experiment-plan.md`.
> **역할 변경(2026-07-25)**: L4(renorm-4 T2, LLM)는 **시간부족으로 배제** → JW 의 8슬롯을 **RTX3090 CNN** 으로 돌린다.
> **새 역할 = (a) 재학습 오라클 무대 전량**(G2 본문 cifar10 + G9 부록 mnist) = **CNN 최대 물량 505 GPU-h**.
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 3-seed(seed-major). push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만. 기존 rundir read-only.

## 0. 왜 3090으로 옮기나

| | A6000(종전) | **3090(신규)** |
|---|---|---|
| 클러스터 여유 | **10장 / 98**(가동률 90%) | **21장 / 186**(89%) |
| 담당 작업 | L4 = LLM 1B renorm-4 T2 | **CNN N=10 (a) 2¹⁰ 오라클** |
| 메모리 | 32 GiB 필요 → 48GB 필수 | LeNet5/FedSVCNN — 24GB 충분 |
| 마감 내 완주 | **불가**(§1 각주) | **가능**(63 wall-h) |

- LLM 을 빼도 A6000 총량은 손해가 없다 — 어차피 클러스터 여유가 10장이라 HJ·JB 가 이미 그 대역을 다 쓴다. 반면 **3090 21장은 놀고 있었다.**
- CNN 은 B200·A6000 어느 것에도 의존하지 않는다(자체완결).

> **L4 배제의 근거(실측)**: 셀당 **~100h**(observer arm 이 라운드마다 renorm-4 부분집합 평가 → same-game 대비 라운드당 ~13×; clean 21.7분/R · noisy 19.7 · frzero 29.4 × R=200 = 66–98h + T2 재학습 ~24h). 9셀 = **~900 GPU-h** → 8슬롯이어도 ~8.4일. `--time=24:00:00` 로는 `persist()` 가 arm 완료 후에만 호출되어(`track_g.py:704`) **산출물 0으로 소멸**한다. 되살리려면 마감을 07-30 이후로 밀어야 한다(`REMAINING-00-INDEX.md` §5).

## 1. 셋업 (LLM 계정에서 CNN 으로 — 추가분만)

1. **conda env**: 기존에 만든 torch 2.11 스택 그대로 쓴다(`torch 2.11.0+cu128`). CNN 은 transformers/trl/peft 불요.
2. **데이터**: torchvision `mnist` · `cifar10`(HF 캐시 아님). YH 의 데이터 디렉토리를 복사하거나 최초 1회 다운로드(수백 MB). `HF_HOME` 무관.
3. **파티션/QOS**: `--partition=base_suma_rtx3090` + 기본 QOS(8-GPU/user). sbatch 에 내장돼 있다.
4. 공통: `codes/` 에서 `PYTHONPATH=.`.
5. sbatch 상단 `REPO`/`PY` 기본값이 YH 경로다 → 다르면 `REPO=… PY=… sbatch …` 로 오버라이드.

## 2. G2 · G9 — (a) 재학습 오라클을 논문 오염축·파티션으로 정렬

> **왜 필요한가**: 현 C1 시나리오({iid, label_skew, quantity_skew, label_flip 사다리, feature_noise})는 확정 오염축 3종(lf@0.70·free-rider-zero·grad-noise)과 **한 칸도 겹치지 않는다**. (a) 오라클은 방법-중립 참값이라 **전 방법을 채점하는 유일한 무대**인데, 그 무대가 논문 위협축과 어긋나 있다.
>
> **비교불가성은 그대로 남는다**: (a)는 2^N 재학습이라 **N=100 에서 원리적으로 불가** → 1A-CNN(N=100 부분참여)과 **N·참여율은 못 맞춘다**. 맞출 수 있는 건 **오염축과 파티션뿐**이고 이 잡이 그걸 한다. 이 한계는 논문에도 명시한다.

- **셀**: {cifar10(본문 G2), mnist(부록 G9)} × {iid, dir1} × 4위협(clean·lf@0.70·free_rider·grad_noise) × 3seed = **48**.
  셀 내부 = (a) 2¹⁰ 재학습 오라클 + (b) 2¹⁰ in-run + 9방법 φ (N=10 full · R=10).
- **비용(실측)**: (a) 2¹⁰ 재학습 `t_a` = **cifar10 32,808 s ≈ 9.1 h** · **mnist 41,168 s ≈ 11.4 h**(`runs/track_c/c1_oracle/*/metrics.json`). 궤적 ~103 s·전 방법 합 ~8분은 무시 가능 → **셀 ≈ t_a**.
  → **48셀 ≈ 505 GPU-h** · 8슬롯 **~63 wall-h**.
- **⚠ 착수 게이트 = C-b 가 `origin/main` 에 있는 것.** 요구 env 계약 = `C1_PARTITION`(iid\|dir1) · `C1_THREAT`(clean\|label_flip\|free_rider\|grad_noise) · `C1_FLIP_RATE=0.70`.
  **정정(07-25, JW 지적 · 코드 확인)**: 구 러너로 제출하면 **실패하지 않고 조용히 틀린다.**
  `os.environ.get` 은 아는 키만 읽어 위 3개를 무시하고 `C1_SCENARIO=iid` 로 도는데,
  `track_c1.py:432` 의 `C1_RUN_NAME` 은 그대로 먹혀 **`cifar10_iid_free-rider_seed0` 이름에
  iid-clean 데이터**가 들어가고 EXIT=0. 4위협이 한 셀로 붕괴하고 로그로는 구별이 안 된다.
  → 제출 전 술어: `git show origin/main:codes/experiments/track_c1.py | grep -q C1_THREAT`.

### ✅ 오염-집합 규약 = **전 위협 고정 `⌊ρN⌉`=4/10 으로 확정** → **재실행 0, 8셀 전부 유효**

Yonghee 판정(07-25) = **YH 버전 그대로 통일**. 착지 = `d09e528`(`track_c1.py:187-196`).
JW 가 자기 버전으로 돌린 8셀은 **같은 고정-개수 규칙**이라 전부 유효하다.

- **JW 의 판단 근거가 채택됐다** — N=10 Binomial 이 2~6 으로 흔들려 위협 간 비교가 깨진다는
  진단이 그대로 코드 주석(l.188-192)의 근거가 됐다. 실제 재현값 **3/7/6**(평균 5.33 vs 명목 4,
  sd 1.55)이 그 판단을 정량으로 뒷받침한다.
- **N=100 주무대(`track_c2`)와는 의도적으로 다르다** — 거기선 label_flip 만 FedCorr 공식 구현의
  Bernoulli(ρ=0.4)를 써서 개수가 변동한다(실현 39/48/47). 이 무대에서 고정을 쓰는 근거:
  ① 도즈를 0.70 으로 고정해 **FedCorr 를 재현하는 게 아니다**(τ~U(0.5,1) 미사용) → "공식 구현
  준수"의 대상이 아니다 ② N=100 은 39–48 = 상대변동 ~12% 라 무해하지만 **N=10 은 3–7 = ~39%** 로
  label_flip 열이 다른 도즈가 되어 4위협이 한 fidelity 표를 공유할 수 없다.
- 남은 것은 **논문 부록 B 에 stage-분리 1문장**이며 코드·실행 변경은 없다.

**기대 corrupt 집합 (전 위협 공통 · seed-only = dataset/partition/dose 무관 · 스모크 대조용)**

| seed0 | seed1 | **seed2 (JW 몫)** |
|---|---|---|
| `[1,4,6,7]` | `[0,4,6,7]` | **`[3,4,6,9]`** |

스트림/오프셋이 `track_c2` 와 동일하다(`default_rng(1000+seed)` 의 첫 소비).

**조치**: ① `git fetch && git reset --hard origin/main` — `989f5ca` 폐기(같은 결론에 도달한 별도
구현이라 잃는 내용 없음; 철회 기록 = `f056b16`) ② 도는 8셀은 **건드리지 않는다** ③ 잔여 13셀 제출.
스모크로 확인하려면 `C1_ORACLE_A=0` 빌드-only 로 위 seed2 집합과 `corrupt=` 를 대조.
`rates=[0,…]` = free_rider 라벨 무접촉(update-level 이 맞다).

**JW 몫 = 21셀 ~205 GPU-h** — Slurm 4계정을 **GPU-h 로 균등화**한 결과다(전 계정 ~25 wall-h; YH `0-7,16` · JB `14-15,24-31,40-47` · HJ 는 c1축 없이 G12+G10).

```
mkdir -p runs/track_c/c1/_logs
sbatch --array=32-39%8   runs/track_c/c1/sbatch_c1_axis.sh   # cifar10 seed2 (본문 G2 먼저)
sbatch --array=17-23%8   runs/track_c/c1/sbatch_c1_axis.sh   # cifar10 seed1 잔여 7셀
sbatch --array=8-13%8    runs/track_c/c1/sbatch_c1_axis.sh   # mnist seed0 6셀 (부록 G9)
```
인덱스 규약: `SEED=IDX/16` · 그 안에서 `0-7`=cifar10, `8-15`=mnist · 파티션 `iid,dir1` × 4위협.
**cifar10 을 먼저** 제출한다(본문 G2 > 부록 G9).
- `--time=24:00:00` 내장(최장 셀 11.4h + 여유). 완료 판정 = rundir + 로그 EXIT=0.
- **채우는 것**: 계획서 §2.1 "1B-CNN 소형 교차-사일로 vs (a)"(본문) · §3.1 "1B-CNN mnist vs (a)"(부록) · §3.4 φ 부호 감사의 **CNN 레그 재감사**(현 감사 스냅샷에 frzero·grad-noise 가 없다).

## 3. 우선순위 · 예상 종료

| P | 무엇 | 셀 | GPU-h | 근거 |
|---|---|---|---|---|
| **P0** | G2 cifar10 seed2 (32-39) | 8 | ~73 | **본문** fidelity 표 |
| **P1** | G2 cifar10 seed1 잔여 (17-23) | 7 | ~64 | 〃 |
| **P2** | G9 mnist seed0 일부 (8-13) | 6 | ~68 | 부록 fidelity |

- **예상 종료 = 07-27 오전**(205 GPU-h / 8슬롯 ≈ **26 wall-h**; **C-b 착지 시각만큼 밀린다**).
- **HJ 가 07-26 오전, JB 가 07-26 후반에 비므로 그쪽이 꼬리를 work-steal** 한다 — 같은 3090·같은 env·셀 단위 idempotent 라 안전하다. 제출 시 남은 `--array` 범위만 지정할 것(중복 = GPU 낭비).
- 셀 하나가 ~9–11h 라 **중도 컷 = 그 셀 전손**(rundir 은 셀 종료 시 기록). `--time` 을 줄이지 말 것.

## 4. 완료 후

1. rundir 커밋(push는 Yonghee).
2. C1 집계 재생성 → `flirds-results-fidelity`(vs (a) 절) · `flirds-results-ablation`(부호 감사 CNN 레그) → paper §5.2 sub·부록 C.
3. **스택 캐비엇 없음** — (a)는 재학습 오라클이라 하드웨어 독립이고, 이 잡은 기존 C1 과 같은 torch 2.11 이다. 다만 **`timing.json` 은 §5.5 cost 표에 쓰지 않는다**(canonical = B200 실측만).
