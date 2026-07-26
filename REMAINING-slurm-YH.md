# REMAINING (Slurm · YH = chyoyhr) — 선행 코드 변경 + CNN 점수원 경쟁 신설

> 배분 정본 = **`REMAINING-00-INDEX.md`** · 수록목록 정본 = `research-wiki/survey/flirds-paper-experiment-plan.md`.
> **역할 = (1) 선행 코드 변경 2건**(JW 를 막고 있음) **+ (2) CNN 점수원 경쟁 신설**(cifar10/iid · mnist).
> **하드웨어 = RTX3090 24GB**(클러스터 여유 21장 · 8-GPU QOS). CNN 은 전량 3090 — LLM 1B 은 24GB 불가라 여기 오지 않는다.
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 전 실험 3-seed(seed-major). push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만. 기존 rundir read-only.
> **진행 상황 (07-27 02:30)**: G3 ✅ 96/96 · **G8 ✅ 24/24 `bac5cff`** · **G6 ✅ 9/9 `93dbedd`** · **G14 ✅ 18/18 `3111ca1`** · c1축 🟢 **8/9 `61bd93a`** — **잔여 = idx16 단 1셀**(07-27 ~10:34 착지 예정).
> **실패 0 · 미커밋 산출물 0 · 계 156셀 착지.** 마감(07-28 24:00) 대비 ~37h 여유. `origin/main` 은 `057506a` 까지 push 됨(Yonghee), 이후 4커밋 미push.

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

## 3. ✅ G3 — cifar10/iid 점수원 경쟁 (본문) · **96/96 착지 완료 (`4395028`, 07-26)**

> 본문 downstream "2-CNN P1 부호-게이트 — cifar10/iid". dir1 은 8점수원 3-seed 완비인데 iid 는 **flirds 만** 있었다 → **이 착지로 채워짐**.
> **부록 P1w 는 추가 런 0** — 같은 rundir 이 P1(`gate_v2`)과 P1w(`gatew_v2`) arm 을 함께 낳는다(계획서 §7.2).
> **관측자에 `C2_OBS_SRCS` 를 지정하지 않는다** → 기본값 = **8소스 전량 T2**. 기존 iid 관측자(`_obsf`)가 flirds 만 담아 retrain 열이 비었던 게 이 결손의 원인이다.

- 4위협(clean·lf@0.70·free_rider·grad_noise) × (7 비-flirds + 관측자) × 3seed = **96 · 실패 0**. flirds online arm 은 `track_g/rundirs_cnn` 의 cifar10 iid 그리드에 이미 있다.
- **실측 ~89 GPU-h**(추정 60–100 적중). 셀당 로그 78셀 평균: flirds1st/fedif 18.0분 · lossheur 18.4 · fedsv 35.9 · comfedsv 36.2 · gtg 88.6 · shapleyfl 108.9 · 관측자 117.1.
- 집계 재생성 = `python runs/track_h/make_analysis.py` (커밋에 포함). 결과 = `runs/track_h/analysis/cnn_competition.csv`.

**결과(rundir-only 재생성값) — cifar10/iid 오염셀 평균 recovery, online**:

| policy | 순위 |
|---|---|
| P1 `gate_v2` | lossheur .887 > **flirds .710** > fedif .556 > flirds1st .543 ≫ gtg −.049 > fedsv/comfedsv −.572 > shapleyfl −.581 |
| P2 `gatew_v2` | fedif .889 ~ lossheur .879 > **flirds .764** > flirds1st .536 |

clean parity `|dAcc|`(밴드 .006): fedif .0003 · flirds1st .0003 · lossheur .0021 **통과** / flirds .0060 **경계** / gtg .0101 · shapleyfl .0248 · fedsv·comfedsv .0290 **이탈** — MC 계열이 clean 에서 오발화한다.

> ⚠ **FedSV ≡ ComFedSV — iid 한정, 버그 아님. 표 작성 시 필수 캐비엇.**
> 이 무대의 두 열은 φ 가 raw/cum **비트동일**이다. per-round surrogate 의 유틸리티가
> 서로 affine(`u_com(S) = base − u_fed(S)`)이고 Shapley 가 유틸리티에 선형이라, 각
> provider 가 적용하는 부호 반전까지 합쳐 정확히 상쇄된다. 단 이는 서브셋 파라미터
> 빌더가 일치할 때만인데 `_llm_subset_params`(크기가중 `n_c/Σn_c`) vs
> `_uniform_subset_params`(균등 `1/|S|`) 는 **클라 크기가 같을 때만** 같다 —
> iid = 5000×10 균등이라 성립. 예측대로 **dir1(3834·4910·… 불균등)에서는 갈라진다**
> (기존 dir1 셀 실측 `max|Δcum|` 0.74~2.56).
> ⟹ **iid 표에서 comfedsv 를 독립 baseline 으로 보고하지 말 것**(dir1 에서는 유효).
> 리뷰어에게는 복붙 오류로 보이므로 표에 각주가 필요하다.

provenance: 96셀 sha 스팬 `ca5934e`(07-25 18:58)~`5c9934d`(07-26 00:30). 그 구간 `codes/` 변경 중 cifar10 경로에 닿는 것 없음 — `track_c2.py` 는 mnist dict 키 1줄(C-a)뿐, `exact_sv.py` 는 `delta_transform=None` 기본값 비트동일, `track_c1`/`track_g`/`phase2_matrix` 는 다른 러너.

```
# (기록) 제출 이력 — 재실행 불요
sbatch --array=0-31%8   runs/track_h/sbatch_cnn_iid_comp.sh
sbatch --array=32-95%8  runs/track_h/sbatch_cnn_iid_comp.sh
```

## 4. G8 — mnist 부분참여 fidelity(+탐지) · 24런 (C-a 이후)

- cifar10 본문 무대와 **동일 세팅, 데이터셋만 mnist**. fidelity(부록)와 φ-AUROC(부록)가 **같은 rundir**.
- {mnist × [iid, dir1]} × 4위협 × 3seed = **24** · **~9 GPU-h**(0.38h/셀 실측 — 아래 정정).
- ✅ **24/24 완주**(커밋 `b160c85`+`bac5cff` · 전부 EXIT=0 · 결손 0) · **실측 9.32 GPU-h(셀당 0.39h)**.
- **★ 단가 정정 — 1.05h/셀이 아니라 0.39h (총 ~25–38 → 실측 9.32 GPU-h)**
  종전 1.05h 는 **cifar10** 실측이었다. 같은 러너·같은 참여 스케줄에서
  cifar10 1.00–1.19h · fmnist 0.35–0.41h · **mnist 0.35–0.41h** — mnist 는 fmnist 와
  동일 대역(둘 다 LeNet5)이고 cifar10 의 ~1/2.8. phase 는 valuation 12–15분 ·
  client-training 5.0분 · oracle-b 4.3–4.5분.
  ⚠ **이 클러스터의 1.5× 계수는 여기 안 걸린다** — 그 계수는 fedavg 1024회 호출당
  고정 오버헤드라 **c1축 (a) 전용** 현상이고, G8 은 (a) 재학습이 없다.
- **무료 확인 1건 통과**: `n_rounds` 100-벡터가 cifar10·fmnist·mnist 전 셀에서 **동일**
  (13,13,9,8,12,5,…) = 부분참여 draw 가 데이터셋 무관 seed 함수라는 설계대로 → 무대 간 비교 성립.
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

## 6b. **G14 — mnist 기준 arm(앵커) · 18런 전량** (2026-07-26 신규 배분)

> **왜 생겼나**: G10(HJ, 216런)이 mnist 점수원·관측자를 다 채웠는데 **`oracle_excl`(천장)·`random_excl`(통제)만 없다**. cifar10/fmnist에선 이 둘을 `track_g/rundirs_cnn/*_g_seed*` 그리드가 낳는데 거기 mnist가 없다. G10 계획서(§5 L162)가 그 부재를 알고 **flirds 소스만** track_h로 옮겨 보정했고, 같은 그리드가 낳는 두 기준 arm은 옮기지 않았다. ⇒ **현재 mnist는 recovery·정책축·parity를 원리적으로 못 낸다**(절대 acc 대조까지만).
> 근거·판정: `research-wiki/survey/flirds-paper-experiment-plan.md` §4.4.

- **셀 = `--array=0-17%8` = mnist × {iid, dir1} × {lf@0.70, free-rider, grad-noise} × seed{0,1,2} = 18 전량.**
  ⓘ 07-26 배분 변경 — 처음엔 JB와 9셀씩 나눴으나 **총 3–5.5 GPU-h라 쪼갤 실익이 없고** JB는 G2·G9 seed2 16셀 × ~17.6 h로 07-27 09:40까지 차 있다. 인덱스는 **파티션-바깥**(`0-8`=iid · `9-17`=dir1, 각 range 안 seed-major)이라 중간에 끊겨도 **완성된 파티션 하나는 3-seed**로 쓸 수 있고, 쪼갤 일이 생기면 그 경계로 자르면 된다.
- **arm = `vanilla,oracle_excl,random_excl`** · **valuation(φ) 없음** = 세 arm 전부 순수 학습 런.
  clean은 오염 클라가 0이라 두 arm이 **정의되지 않아** 4위협이 아니라 3이다(cifar10도 `_g_seed` 96셀 중 84셀만 `oracle_excl` 보유 = 4파티션 × 7오염 × 3seed).
- **`vanilla`를 같이 도는 이유**: `runs/track_h/make_analysis.py::analyze_cnn`이 분모를 **`arms["vanilla"]` 이름으로만** 찾고 `observer`로 폴백하지 않는다. vanilla가 없으면 mnist 전 행 `delta_acc`/`recovery`가 None이고 `skipped=equals_vanilla` 칸의 `final_acc`도 안 채워진다.
- **단가 추정 ~10–18분/셀 → 18셀 ≈ 3–5.5 GPU-h**. 근거 = §3 G3 최저가 셀(flirds1st/fedif) 18.0분인데 거긴 valuation 포함이고 mnist는 cifar10보다 싸다. **첫 셀 로그로 실측 교체할 것.**
- **스택 = torch 2.11 그대로**(§0 스택 고정). 분모(이 잡)와 분자(G10 소스 arm, HJ 3090 torch 2.11)가 같은 스택에 있어야 recovery가 성립한다.
- **우선순위**: 총 ~3–5.5 GPU-h로 §4 G8(25)·§6 G6(10–20)보다 싸다. **G8·G6 뒤 아무데나 붙여도 종료시각(07-27 ~10:50)에 영향 없음.**

```
mkdir -p runs/track_g/_logs
sbatch --array=0-17%8 runs/track_g/sbatch_cnn_mnist_anchor.sh
```

✅ **18/18 완주 (07-27 01:10 · 커밋 `3111ca1`)** — job `1886441` · EXIT=0 · arm 결손 0 · 셀당 ~17분.
제출 전 확인 4건:
- **덮어쓰기 0** — 목표 이름 18개(`mnist_{iid,dir1}_{label-flip_fr0.70,free-rider,grad-noise}_g_seed{0,1,2}`)를 인덱스 산술로 전개해 `runs/track_g/rundirs_cnn/` 과 대조 → **18/18 유니크·기존 0건**. (그 디렉토리의 mnist 매치 48개는 전부 `fmnist` 다.)
- **머지 경로** — `make_analysis.py:179` 가 `track_g/rundirs_cnn` 을 CNN 소스로 스캔하고(`_load(RUNS/"track_g"/"rundirs_cnn") + _load(ROOT/"rundirs_cnn")`), lf 도즈는 이름의 `_fr0.70_` 토큰으로 키를 맞춘다(l.168-179) → G10 track_h 행과 붙는다. CNN 분모는 `arms.get("vanilla")` **단독**(l.190)으로 observer 폴백이 없음도 재확인 — vanilla 를 같이 도는 이유가 코드로 확정.
- **env 계약** — `track_c2.py` 가 `C2_ARMS`(l.493) · `oracle_excl`/`random_excl`(l.371/376) · `C2_RUN_ROOT`(l.145) · `C2_RUN_NAME`(l.672) · `mnist`(l.73, C-a) 전부 지원.
- **`--time=04:00:00` 여유** — valuation 0 · (a) 재학습 0 이라 c1축 같은 컷 위험 없음. 추정 10–18분 대비 13–24× 헤드룸이라 이 클러스터의 1.5× 계수를 먹어도 안전.

**✅ 착지 후 확인 2건 전부 통과 (07-27 02:30 · 커밋 `3111ca1`)**

1. 집계 재생성 → `cnn_competition.csv` mnist 행 **recovery 997/1330 채워짐(G14 전 0)**.
   오염평균 recovery(online · n18):

   | 무대 | 순위 |
   |---|---|
   | mnist/iid P1 | **flirds +0.948** > lossheur +0.896 > gtg +0.652 > fedif +0.618 > flirds1st +0.564 > shapleyfl +0.434 > comfedsv/fedsv +0.424 |
   | mnist/dir1 P1 | lossheur +0.968 > **flirds +0.892** > flirds1st +0.728 > fedif +0.698 > gtg +0.131 > fedsv −0.408 > comfedsv −0.476 > shapleyfl −0.585 |
   | mnist/iid P2 | fedif +0.937 > **flirds +0.887** > lossheur +0.885 > gtg +0.855 |
   | mnist/dir1 P2 | fedif +1.007 > **flirds +0.967** > lossheur +0.914 > gtg +0.654 |

2. **무료 검증 — `vanilla`(G14) vs `observer`(G10) 18/18 bit-identical(`|Δ|=0.000000`)**.
   fmnist 선례(36쌍 전부 0)와 동일 거동이고 cifar10 의 grad_noise 0.024 는 나타나지 않았다
   ⇒ **두 루트(track_g 분모 · track_h 분자) 병합 전제가 실측으로 확인**됐다.

> **FedSV≡ComFedSV 캐비엇이 mnist 에서 독립 재현** — mnist/iid 에서 두 열이 P1 +0.424 /
> P2 +0.692 로 동일하고 dir1 에서는 갈린다(P1 −0.408 vs −0.476). §3 의 등n 축퇴 진단이
> cifar10 밖에서도 성립 = 각주 근거 보강. **iid 표에서는 두 열을 독립 baseline 으로 쓰지 말 것.**

> ⚠ **free-rider 열은 recovery 분모가 좁다** — `oracle_excl − vanilla` 가 +0.006~0.007
> (grad-noise 는 +0.053~0.075). 분모가 작아 잡음에 민감하니 해석·표기 시 유의.

## 6. G6 — Removal-curve CNN 오염축 정렬 · 6~9런 (C-b 이후)

- 현 removal 시나리오({feature-noise, label-flip 사다리, iid})가 확정 오염축과 불일치 → **frzero·grad-noise** 에서 worst-first 제거 → acc 분리(순위→성능 인과)를 다시 낸다.
- frzero·grad-noise × 3seed = **6**(+ lf@0.70 재실행 시 3) · 추정 ~10–20 GPU-h.
- 러너 = `runs/removal_dose/run_cnn_removal.sh` — **C-b 의 위협 토큰 확장을 공유**(별도 코드 없음).
- ✅ **9/9 완주 (커밋 `93dbedd`)** · **실측 1.4 GPU-h(셀당 0.15~0.16h)** — 추정 10–20 은 (a) 2¹⁰ 을 가정한 값이었으나 `C1_ORACLE_A=0` 이라 재학습이 없다.
  오염집합 3-seed 전수 정본 일치(seed0 `[1,4,6,7]`·seed1 `[0,4,6,7]`·seed2 `[3,4,6,9]`).
  기존 `runs/removal_dose/rundirs_cnn/cifar10_iid_seed{0,1,2}`(07-20 clean 앵커)와 **이름 규약이 달라
  덮어쓰기 없음**(`cifar10_iid_{위협}_seed*`) — read-only 규약 유지 확인.

## 7. 우선순위 · 예상 종료

| P | 무엇 | 물량 | 상태 | 근거 |
|---|---|---|---|---|
| ✅ | **코드 C-a·C-b + push** | 6파일 | ✅ **완료 `d09e528`** | 4계정 게이트 전부 해소 |
| ✅ | G3 `hiidcomp` | 96런 · **89 실측** | ✅ **96/96 · 실패 0 · 커밋 `4395028`**(§3) | 본문 downstream 의 빈 절반 |
| **P2** | **c1축 `c1axis` `0-7,16`** | 9셀 · **~125 실측** | 🟢 **8/9 커밋**(`…`·`61bd93a`) · **idx16 1셀 실행중** | **본문 G2 의 P0 seed** — C-b 를 쓴 사람이 직접 |
| **P3** | G8 `c2fidmn` | 24런 · **9.32 실측** | ✅ **24/24 `bac5cff`** | 부록 fidelity+탐지가 한 rundir |
| **P4** | G6 `c1rmax` | 9런 · **1.4 실측** | ✅ **9/9 `93dbedd`** | 본문 ablation |
| **P5** | **G14 `gmnanch`**(신규 07-26) | 18런 · **~5 실측** | ✅ **18/18 `3111ca1`** | mnist recovery **분모** — 없으면 mnist 정책축 자체가 안 나옴(§6b) |
| — | ~~G10 216런~~ | — | ✅ **HJ 로 이관 → 취소(전량 PD·mnist rundir 0)** | 개정 배분 §5 |
| ✅ | **논문 부록 B.2 오염-집합 문단 재작성**(FedCorr 주장 철회 + 무대별 실현 수 표) | — | ✅ 초안 완료 → **타 세션 이관**(paper 는 YH 가 더 손대지 않음) | §2 |

**★ c1축 실단가 정정 — 9.1h 가 아니라 14.6~17.4h (승계 기준의 1.5×)**

JB 가 착지시킨 seed2 8셀(`0875976`)이 YH 의 seed0 8셀과 **인덱스까지 1:1 대응**이라 추정
대신 그 `timing.json` 을 쓴다:

| 셀 | 실측 | 셀 | 실측 |
|---|---|---|---|
| iid clean | 16.94h | dir1 clean | **17.44h** |
| iid label-flip@0.70 | 17.04h | dir1 label-flip@0.70 | 16.80h |
| iid free-rider | 14.62h | dir1 free-rider | 14.56h |
| iid grad-noise | 14.89h | dir1 grad-noise | 14.94h |

JB 의 진단: (a) 2¹⁰ 재학습 **50,983s = 49.79s/retrain, 레거시 대비 1.54×**인데 traj 는
1.12× 에 그침 → **느린 축은 GPU 가 아니라 fedavg 1024회 호출당 고정 오버헤드**다. 즉
노드 경합이 아니라 클러스터 단가이며, `--time=24:00:00` 대비 최장 17.4h 라 컷 위험 없음
(**이 값을 줄이지 말 것** — 종전 9.1h 를 근거로 `--time` 을 깎으면 전손).
seed0 오염집합 `rates=[0,.7,0,0,.7,0,.7,.7,0,0]` → **[1,4,6,7]** 정본 일치 확인.

**★ 노드 계수 확정 — 4셀 실측으로 0.90× (07-26 21:25)** — YH 착지 4셀 대 JB 대응셀:

| 셀 | YH | JB(seed2) | 비 |
|---|---|---|---|
| iid free-rider (`be53009`) | 12.99h | 14.62h | 0.888 |
| iid grad-noise | 13.67h | 14.89h | 0.918 |
| dir1 free-rider | 12.96h | 14.56h | 0.890 |
| dir1 grad-noise | 13.42h | 14.94h | 0.898 |

| iid clean (`1ea3856`) | 15.14h | 16.94h | 0.894 |

평균 **0.899** → 종전 0.91 을 **0.90** 으로 확정. 위 seed2 실측표에 ×0.90 이 YH 예측이다.
오염 3셀 phase 분해는 동일: **(a) 2¹⁰ 재학습 98.8~98.9%** · client-training 0.03h ·
valuation 0.13h — 이 축의 비용은 사실상 2¹⁰ 단독이고, 그래서 `--time` 을 깎으면 전손이다.

**★ clean·label-flip 셀이 ~2h 비싼 이유 = Ripple** — `iid clean` 만 methods **11종**(+Ripple)
이고 `ripple-own-trajectory` phase **1.88h** 가 붙는다(그래서 (a) 비중이 98.8% 가 아니라 86.5%).
오염 2셀(fr·gn)은 10종·해당 phase 없음. JB seed2 에서 clean/lf 가 ~17h, fr/gn 이 ~14.6–14.9h
로 갈린 것도 같은 원인이다 ⇒ **이 축의 비용 = 2¹⁰ 재학습 + (clean·lf 한정) Ripple 궤적**.

**오염집합 정본 대조 — 통과**: 오염 3셀 전부 `corrupt=[0,1,0,0,1,0,1,1,0,0]` = **[1,4,6,7]**
(seed0 정본, §2) · clean 전 0 · `dose={flip_rate 0.7, mal_frac 0.4, grad_noise_std 0.1}`.

**예상 종료 (07-26 21:25 기준 재계산 · G14 반영)**

| 시점 | 상태 |
|---|---|
| ✅ 07-26 22:18~23:01 | c1축 1차 웨이브 완주 — **seed0 8/8**(`61bd93a`) |
| ✅ 07-26 21:45~07-27 01:10 | **G14 18/18** 완주(`3111ca1`) |
| ✅ 07-27 ~01:30 | **G8 24/24**(`bac5cff`) · **G6 9/9**(`93dbedd`) 완주 |
| **07-27 ~10:34** | **idx16 착지 = YH 전량 완주**(임계경로 · 유일한 잔여) |

**실측 대 추정 — 3건 전부 과대추정이었다**

| 잡 | 추정 | 실측 | 원인 |
|---|---|---|---|
| G8 | ~25–38 | **9.32** | 1.05h/셀 이 cifar10 값이었음(mnist=fmnist 대역) |
| G6 | ~15–27 | **1.4** | `C1_ORACLE_A=0` — 2¹⁰ 재학습 없음 |
| G14 | ~3–5.5 | **~5** | 적중 |

그래서 07-26 22:30 에 7슬롯이 열린 뒤 3잡 계 51셀이 ~3h 만에 소화됐다(예상 4h).

> idx16 = **seed1 iid clean** 이라 방금 착지한 seed0 iid clean **15.14h** 가 그대로 예측치다
> (19:26 착수 + 15.14h). Ripple 포함 셀이므로 fr/gn 13.x h 를 쓰면 과소추정이 된다.

- 임계경로 = 그 **9번째 셀 하나**. 9셀/8슬롯이라 2차 웨이브가 불가피하고 QOS 8장
  상한 때문에 당길 수단이 없다. G8·G6·**G14** 는 전부 그 그늘에 들어가 makespan 무영향 —
  **G14 를 꼬리에 붙여도 종료시각이 안 변하는 이유가 이것**이다(§6b 우선순위 항과 동일 근거).
- 마감 07-28 24:00 대비 **~37h 여유**.
- G3 가 한때 10/96 이던 이유 = 지난 세션 편집 창 사고로 62셀 사망 → 재제출(`1878494`)로 **전량 회수, 최종 손실 0**.
- **c1축은 P2 로 올라와 G8·G6 보다 앞선다**(본문 > 부록). Slurm 이 job age 순으로 뽑아 이 정렬이 자동 유지된다.

```
sbatch --array=0-7,16%8  runs/track_c/c1/sbatch_c1_axis.sh    # cifar10 seed0 8셀 + seed1 1셀 (C-b 착지 후)
```

- **YH 몫 ~202 GPU-h** / 8슬롯 → **~25 wall-h** → **07-27 오전**. 코드 작성 중에도 **G3 는 게이트가 없어 병행 가동**할 수 있다.
- **c1축 cifar10 seed0(0-7)을 YH 가 맡는 이유**: 본문 G2 의 최우선 seed 인데, C-b 를 쓰는 사람이 직접 돌려야 게이트 해제 즉시 착수된다.
- **G10 은 HJ, G5·G12 는 B200 c4** — 여기가 아니다.
- 슬롯이 남으면 JW·JB 잔여를 work-steal(남은 `--array` 범위만).

## 9. 완료 후 · 미해결 배선

0. **G3 는 이 경로를 이미 통과했다**(rundir+analysis 커밋 `4395028`). 남은 3건(c1축·G8·G6)도
   같은 순서로 처리한다. **push 는 Yonghee 직접** — YH 는 커밋까지만.
   - downstream 표로 옮길 때 **§3 의 FedSV≡ComFedSV(iid) 캐비엇을 반드시 각주로 달 것.**
     각주 없이 두면 두 열이 같은 값이라 복붙 오류로 읽힌다.
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
