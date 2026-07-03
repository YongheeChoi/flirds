# Flirds 신호 크기 진단 — "val-loss 변화량이 너무 작다" 가설 검증

- 작성: 2026-07-02 (Phase 1 진단 완료; Phase 2 probe는 승인 대기)
- 가설: 학습에서 일어나는 val-loss 절대 변화량이 너무 작아 (1) 기여도 fidelity와
  (2) 기여도-가중 학습(intervention)의 성능 변화 정밀도가 떨어진다. 핵심 의심:
  이 세팅의 SFT가 validation 기준으로 실제로 유의미한 학습을 하는가.
- 데이터: `runs/track_d/rundirs` 18셀(1B/3B/7B × anchor5/std20 × 3seed),
  `runs/phase2_matrix/rundirs` 25셀, `runs/track_c` 150셀 — **전부 기존 산출물, 재실행 없음**.
- 서술 순서 = 핵심 질문 위계(1차 fidelity → 2차 성능/수렴/탐지).

## 0. 결론 요약

병목은 셋 중 어느 하나가 아니라 **층위가 다른 두 개**다.

1. **측정 정밀도(fp32)는 병목이 아니다.** 관측되는 최소 신호(클라 간 φ 차이
   ~1e-4–1e-3)도 fp32 플로어(ulp ~1.7e-7 @loss≈1.4)보다 2–4 자릿수 위에 있다.
2. **절대 학습량 부족은 절반만 사실.** val-loss 감소(−0.03~−0.13)와 ROUGE-L
   상승(+5~+13pp)은 실재하지만, **capability 축(MMLU)은 향상 0 또는 하락** —
   이 SFT는 포맷/스타일을 학습하지 소재 능력을 학습하지 않는다.
3. **더 근본적인 병목: IID-clean 무대에는 측정할 클라 간 '진짜 신호'가 구조적으로
   없다.** 클라들이 교환 가능하게 설계되어 있어 (b) oracle 자신의 클라 순위조차
   seed 간 재현되지 않고(cross-seed ρ≈0), 게임이 사실상 가산적이라 모든 semivalue가
   같은 순위로 붕괴하며(방법 구별 실패), intervention 이득의 정답 자체가 ~0
   (do-no-harm parity)이다. 대조군이 이를 확증한다: CNN Track C는 오염/비IID 셀에서
   oracle 순위 안정성 0.51~0.97, IID 셀은 −0.04이고, **LLM도 같은 코드로 무대만
   5-domain 비IID(silo5)로 바꾸면 ρ가 −0.37→+0.93~1.00으로 산다**(§1.4). 즉 신호원은
   학습 강도(lr·epoch·rank = A축)가 아니라 클라 간 실제 차이(비IID·오염 = B축)다.
   → 다음 실험은 이 B축을 직접 여는 **오염축 × 비IID축 2×2 매트릭스**(§2.4).
4. seed 노이즈는 **unpaired 비교(벤치마크 축)에서만** 병목이다. MMLU는 효과
   크기(~0.001)가 표본 SE(±0.004)보다 작아 원리적으로 검출 불가. 반면 paired
   설계의 val-loss 축에서는 intervention 효과(−0.001~−0.004)가 SNR 2.4–4.5로
   일관되게 검출된다 — 방향은 실재, 크기만 0.07–0.3%.

rank·참여수 lever에 대한 예상(Phase 2에서 검증): rank↑는 per-round Δ와 φ 규모를
키울 수 있으나 IID-clean인 한 클라 간 순위의 '실재성'은 만들어주지 않는다.
참여수↑(round당 2→5)는 (b) per-round 서브게임을 2²→2⁵로 키워 방법 구별력 축을
움직인다.

## 1. Phase 1 진단 (기존 산출물)

### 1.1 val-loss 동역학 (항목 1)

Track D vanilla val 곡선(`arms.vanilla.val_curve`, val=200 고정 셋):

| cell | init→final (3seed 범위) | ΔTotal | Δ/round | Δ/round 말미 10% |
|---|---|---|---|---|
| 1B anchor5 (R=30) | 1.399→1.321 등 | −0.074~−0.079 | −0.0025 | −0.0011~−0.0012 |
| 1B std20 (R=200) | 1.399→1.288 등 | −0.104~−0.112 | −0.00052~−0.00056 | −3e-5~−5e-5 |
| 3B anchor5 | 1.304→1.231 등 | −0.071~−0.080 | −0.0024~−0.0027 | −0.0015~−0.0017 |
| 3B std20 | | −0.114~−0.131 | −0.00057~−0.00065 | −5e-5~−7e-5 |
| 7B anchor5 | 1.152→1.125 등 | −0.027~−0.033 | −0.0009~−0.0011 | −0.0015~−0.0017 |
| 7B std20 | | −0.085~−0.096 | −0.00042~−0.00048 | −3e-5~+2e-6 |

- anchor(R=30)는 아직 수렴 전(말미에도 −0.001/round), std20(R=200)은 플래토
  (7B seed0는 말미 +2e-6 = 미세 과적합 시작).
- fp32 기준: loss≈1.4에서 ulp ≈ 1.7e-7. per-round Δ(5e-4~2.6e-3), per-coalition
  Δ(φ로 역산 ~1e-4~1e-2) 모두 플로어의 10²~10⁴배. **fp32 정밀도는 병목 아님**
  (bf16이었다면 φ 클라 간 차이 ~1e-3이 prec ~8e-3에 묻힘 — fp32 관례가 맞았음).

### 1.2 절대 성능: base vs trained (항목 2, 결정적 테스트)

Track D arms(base vs vanilla; MMLU full-test 0-shot n=14042, Alpaca-test 1k ROUGE-L):

| cell | MMLU base→vanilla (Δ) | ROUGE-L base→vanilla (Δ) | val-loss Δ |
|---|---|---|---|
| 1B anchor5 | 0.4822→0.4801 (**−0.2pp**) | 0.217→0.273 (+5.6pp) | −0.077 |
| 1B std20 | 0.4822→0.4742 (**−0.8pp**) | 0.217→0.284 (+6.7pp) | −0.109 |
| 3B anchor5 | 0.6230→0.6215 (**−0.15pp**) | 0.222→0.275 (+5.3pp) | −0.075 |
| 3B std20 | 0.6230→0.6149 (**−0.8pp**) | 0.222→0.302 (+8.0pp) | −0.123 |
| 7B anchor5 | 0.4175→0.4206 (**+0.3pp**) | 0.150→0.165 (+1.5pp) | −0.030 |
| 7B std20 | 0.4175→0.4038 (**−1.4pp**) | 0.150→0.278 (+12.8pp) | −0.089 |

(MMLU 이항 SE ≈ ±0.42pp → 1B/3B anchor의 −0.2pp는 SE 이내, std20의 −0.8pp와
7B std20의 −1.4pp는 유의한 **하락**. base가 seed 무관 동일값인 것은 base 모델이
seed와 무관하기 때문.)

**판정**: "향상이 실재하는가"의 답은 축에 따라 갈린다 — val-loss·ROUGE-L(같은 분포
표면 지표)은 실재하는 소폭 학습, **capability(MMLU)는 0/음수**. 이 무대의 SFT는
alpaca 포맷/스타일 적응이지 능력 획득이 아니다. 즉 "모든 미세함이 학습량 0에서
설명된다"는 극단 가설은 기각되지만, intervention의 상한이 되는 '학습으로 얻는
전체 파이' 자체가 val-loss 기준 6–10%, capability 기준 0이다.

### 1.3 미세함 분해 (항목 3) — 1차 질문(fidelity) 관점

(track_d `fidelity.csv`, 3-seed mean±std)

- **(a) 순위 fidelity 포화 여부**: anchor5(N=5, full)에서는
  Flirds/Flirds1st/GTG/Banzhaf/loss-heur 전부 Spearman **+1.000±0** (포화).
  std20(N=20, 2/round)에서는 구별이 생긴다:
  Flirds +1.000 / loss-heur ~+1.000 / Flirds1st +0.999 / GTG +0.975±0.02 /
  FedSV +0.91±0.09 / ShapleyFL +0.19 / ComFedSV +0.09 / FedIF +0.16.
- **(b) 방법 간 구별 가능성**: anchor에서는 차별화 실패(1.5절의 가산성 때문 —
  제대로 계산하면 누구나 같은 순위). 차별화는 부분참여(std20)가 만든다.
- **(c) 값-수준 Pearson**: Flirds vs (b) = 0.99999+ 전 스케일·전 레짐. 추정기는
  oracle을 순위가 아니라 **값까지** 재현한다. (a)-vs-(b)도 0.86–0.98.
- **(d) φ 절대 크기·분산** (φ 단위 = val-loss 변화 귀속):

| cell | Σφ_(b) | φ̄ | 클라 간 spread | spread/|φ̄| |
|---|---|---|---|---|
| 1B anchor5 | −0.074~−0.078 | −0.0149~−0.0157 | 0.0012~0.0033 | 8–22% |
| 3B anchor5 | −0.071~−0.079 | −0.0143~−0.0159 | 0.0009~0.0022 | 6–15% |
| 7B anchor5 | −0.027~−0.033 | −0.0055~−0.0066 | 0.0012~0.0016 | 18–30% |
| 1B std20 | −0.104~−0.112 | −0.0052~−0.0056 | 0.0070~0.0106 | 126–190% |
| 3B/7B std20 | 유사 | | 0.0051~0.0097 | ~120–180% |

std20의 큰 spread는 참여 추첨(어느 라운드에 뽑혔나)이 지배 — 클라 속성이 아니다.

### 1.4 신호 vs 노이즈 (항목 4)

**(ii) seed SNR — (b) oracle 자기 순위의 cross-seed 재현성** (track_d, 3-seed 쌍별
Spearman 평균): 1B anchor −0.37 / 1B std20 −0.11 / 3B anchor +0.27 / 3B std20
−0.24 / 7B anchor +0.73 / 7B std20 +0.16. 모든 방법이 oracle과 같은 불안정성을
공유한다(예: 3B std20에서 Flirds −0.24, loss-heur −0.24 — 추정기는 oracle의
노이즈까지 충실히 재현).

- caveat: seed가 데이터 파티션도 바꾸므로 클라 정체성이 seed 간에 유지되지
  않는다. 그러나 이것이 곧 진단이다 — **IID-clean 무대에선 클라가 설계상 교환
  가능하고, φ의 클라 간 차이는 '어떤 표본이 어느 클라에 떨어졌나'의 추첨 노이즈**다.
- **통제된 대조군 — 무대에 심긴 클라 간 차이가 있으면 oracle 자기 순위가 안정된다.**
  (b) oracle cross-seed 안정성 ρ를 무대별로 보면(재실행 없이 phi.parquet에서 계산):

  | 무대 (같은 방법·같은 코드, 무대만 다름) | (b)oracle ρ_xseed | 비고 |
  |---|---|---|
  | CNN cifar10 **IID** | **−0.042** | 클라 균질 → 신호 없음 |
  | CNN mnist feature_noise | −0.123 | (mnist 과easy) |
  | CNN cifar10 label_skew | 0.511 | 비IID |
  | CNN cifar10 feature_noise | 0.693 | 오염 |
  | CNN cifar10 **label_flip** | **0.968** | 오염 |
  | CNN cifar10 **quantity_skew** | **0.968** | **품질 동일, 양만 차이** |
  | LLM 1B **track_d IID-alpaca** | **−0.37~−0.11** | 위 (ii) — 균질 |
  | LLM 1B **silo5 (5-domain non-IID)** | **+0.93~1.00** | 같은 코드, 무대만 비IID |

  (`runs/track_c/RESULTS.txt` C1 stability; `runs/phase2_matrix/rundirs/1B_silo5_*`.)
  심긴 신호가 있으면 oracle 순위는 안정, 없으면 oracle도 추첨 노이즈를 랭킹한다.
  **핵심: CNN quantity_skew(품질 동일·양만 차이)와 LLM silo5(오염 섞였으나 도메인
  분리)가 둘 다 강한 신호 → 신호원은 학습 강도(lr·epoch·rank)가 아니라 클라 간
  실제 차이다.** 스케일을 1B→7B로 키워도 IID면 ρ≈0으로 남는 것(위 (ii))과 정합적.
  단 silo5는 오염이 섞여 "도메인 분리만의 순수 신호"를 오염 신호와 분리하지 못한다
  → §2의 clean×non-IID 셀이 이를 확정한다.

**intervention Δ vs seed 노이즈** (track_d arms, vanilla 대비 paired 3-seed):

- unpaired 관점: vanilla final val-loss의 seed-std = 0.021–0.027. 효과(~0.001)의
  20배 → 셀 독립 비교로는 영원히 검출 불가.
- paired(같은 seed·init·데이터) 관점:

| cell | flirds_w Δval-loss (±seed std) | |SNR| | flirds_w ΔMMLU | ΔROUGE |
|---|---|---|---|---|
| 1B anchor5 | −0.0012±0.0005 | 2.4 | +0.0001 (n.s.) | +0.0016 |
| 3B anchor5 | −0.0009±0.0003 | 3.2 | −0.0002 (n.s.) | +0.0006 |
| 7B anchor5 | **−0.0036±0.0008** | **4.4** | +0.0005 (n.s.) | +0.0028 |
| 1B std20 | −0.0000±0.0001 | 0.02 | +0.0003 (n.s.) | +0.0007 |
| 3B std20 | −0.0004±0.0001 | 3.2 | −0.0004 (n.s.) | −0.0001 |
| 7B std20 | −0.0009±0.0002 | 4.5 | −0.0012 (n.s.) | +0.0003 |

  → val-loss 축에서는 flirds_w가 vanilla보다 일관되게 낮다(방향 실재). 크기는
  0.07–0.3% — **같은 게임(val-loss)에선 보이고, 다른 게임(MMLU: 효과 0.001 <
  SE 0.004; ROUGE: SE ~0.005)에선 원리적으로 분해능 밖**.
- 수렴 속도(rounds-to-target): 1B/3B는 해상도 없음(198–200/None). 7B std20만
  유의미: vanilla 159 vs flirds_w 127 / shapleyfl_w 124 / fedif_w 142 —
  vanilla가 플래토(과적합 시작)에 들어간 무대에서만 개입이 라운드를 벌어준다.

**(i) val 측정 노이즈**: 체크포인트가 보존되지 않아(런디렉토리는 φ/metrics만)
trained 모델 고정 bootstrap은 기존 산출물로 불가 → **Phase 2 probe에 per-chunk
val-loss dump를 넣어 coalition Δ의 bootstrap SE를 직접 측정**하도록 설계에 포함
(아래 §2). 분석적 참고: val=200의 unpaired SE ≈ 0.04–0.06으로 coalition Δ보다
크지만 모든 비교가 같은 val 셋에 paired라 상쇄되고, 남는 질문("val을 다시 뽑으면
순위가 재현되는가")은 (ii)의 cross-seed 불안정과 같은 방향일 것으로 예상.

### 1.5 비가산성 (항목 5, 묘사 — 결함 아님)

기존 phi.parquet만으로 계산 가능: loss-heur의 φᵢ = U({i})(singleton 효용,
`in_run_utility(logs,[i])`, U(∅)=0)이고 Σφ_(b) = U(N)이므로,

- **가산성 갭** v(N) − Σv({i}) = Σφ_(b) − Σφ_loss-heur:
  1B anchor +0.0005~+0.0006 (Σφ의 0.6–0.8%), 3B +0.0001~+0.0002 (0.1–0.3%),
  7B anchor −0.0002~−0.0003 (−0.9%), std20 전 스케일 0.0~0.5%.
- **S-의존성**: Banzhaf−Shapley 차 d∞ ≈ 2e-6 (클라 간 spread의 ~0.2%),
  singleton 순위 vs Shapley 순위 ρ = +1.00 (전 셀).

→ 게임이 사실상 가산적. marginal 기여가 S에 거의 무관하므로 **모든 제대로 계산된
semivalue(Shapley/Banzhaf/singleton)가 같은 순위/거의 같은 값으로 붕괴**한다.
anchor에서 방법들이 전부 +1.000인 이유이며, 이 무대(스무스한 SFT, 10 step의 작은
per-round 이동, momentum=0)의 특성이지 계산 결함이 아니다. 방법 간 차이는
가산성이 아니라 **부분참여 하의 추정 전략 차이**(std20의 GTG/FedSV 하락)에서 나온다.

참고(phase2_matrix, 오염 무대와의 대조): corrupt−clean φ 분리는 silo5 noisy
+0.0019 / free-rider +0.0039 (clean 클라 간 spread ~0.0033의 0.6–1.2배),
poison +0.048~+0.063 (**9–18배**). 오염이 있으면 φ 신호가 노이즈 위로 올라온다 —
IID-clean에서 신호가 없는 것과 정합적.

### 1.6 병목 판정

| 후보 병목 | 판정 | 근거 |
|---|---|---|
| fp32 측정 정밀도 | **아님** | 최소 신호가 플로어의 10²⁺배 (1.1) |
| 절대 학습량 부족 | **부분적** | val-loss/ROUGE 실재(작음), MMLU 0/음수 (1.2) |
| val/seed 측정 노이즈 | **축에 따라** | unpaired 벤치마크 축은 노이즈 아래, paired val-loss 축은 SNR>2 (1.4) |
| 방법 비구별(비가산) | **무대 특성** | 갭 ≤1%, semivalue 붕괴; 구별은 부분참여가 만듦 (1.5) |
| **클라 간 진짜 신호 부재 (IID-clean 설계)** | **주 병목** | oracle 자기 순위 cross-seed ~0; CNN 대조군 0.97 vs −0.04 (1.4) |

따라서 "신호를 키우는" lever는 두 종류로 갈린다: (A) **신호의 크기**(per-round Δ,
φ 규모, intervention Δ)를 키우는 lever — rank·참여수가 여기 해당, Phase 2에서 검증;
(B) **신호의 실재성**(클라 간 순위의 ground-truth)을 만드는 lever — 클라 간 실제
품질/분포 차이(오염, 비IID, 데이터 품질 격차). (B)는 이번 스코프 밖(무대 변경)이며
§4에서 판단거리로만 제시.

## 2. Phase 2 probe 계획 (2026-07-02 승인·실행 개시: seed0 파일럿 먼저, **full 11종 스위트**, CNN C1+C2; 실행 상세 `runs/probe_signal/README.md`)

lever 2개만 변경(LoRA rank, round당 참여 수). lr·R·steps·모델·task·데이터 고정.
설정 변경은 전부 env-gated(기본값=현행)로 main 파이프라인 동작 불변; 커밋은 요청 시.

### 2.1 LLM (1B, 이 박스 GPU 직접 실행)

| 그룹 | 셀 | 신규/재사용 |
|---|---|---|
| A. anchor5 (N=5 full, R=30) | rank 32, 64 × 3 seed | 신규 6 (rank16 = 기존 재사용) |
| B. std-참여변경 (N=50, 5/round, R=200, (b)=per-round 2⁵) | rank 16, 32, 64 × 3 seed | 신규 9 (전부 신규 — N=50/5는 기존에 없음) |
| C. noise-probe (4-i) | anchor rank16/64, seed0, per-chunk (b) 효용 dump → bootstrap SE | 신규 2 (경량) |

- LoRA α는 α=2r 유지(현행 r16/α32의 스케일 관례; α 고정 시 rank lever에 α/r
  스케일 변화가 섞임).
- 방법 스위트: **full 11종 + 전체 arm** (Yonghee 결정 2026-07-02; trimmed 옵션
  기각). probe에서도 기존 셀과 완전 동형 비교.
- 예상 GPU-h (1B, B200 1장; 기존 셀 runtime에서 스케일링, full 스위트):
  - A(anchor): ~10h/seed → 6셀 ≈ **60h**
  - B(std50/5): vanilla 3.2h + (b) 6.7h + 방법 ~15h + arms ~18h ≈ ~43h/seed
    → 9셀 ≈ **390h** ((b) 2⁵/round·k=5 학습·코얼리션 방법이 지배)
  - C(noise-probe): ≈ **4h**
  - **합계 ≈ 450 GPU-h**; 파일럿(seed0) ≈ 150h ≈ 4 GPU ~1.5일.
  - 단계화: **파일럿 = 전 셀 seed0 먼저** → 결과 확인 후 seeds 1–2.
    파일럿에서 rank 효과가 0으로 나오면 중단 가능.
- 코드 변경(승인 후 착수, 전부 기본값=현행): `track_d.py`에 `LORA_R`/`LORA_ALPHA`
  env, RCFG env-override에 `n_clients`/`k_abs` 추가, `FID_METHODS`/`ARMS_LIST`
  트리밍 env; `RUNDIR_ROOT=runs/probe_signal`로 기존 rundirs와 분리;
  `experiments/probe_val_noise.py` 신규(체크포인트 재학습 1회 + chunk-resolved
  coalition 효용 → bootstrap SE·순위 CI).

### 2.2 CNN (yonsei SLURM — 이 박스엔 sbatch 없음 → 스크립트/그리드 작성, 제출은 Yonghee)

CNN엔 LoRA rank가 없으므로 용량 파라미터 = **모델 폭 배수 w**(FedSVCNN 채널
32/64/FC512 × w; disanalogy caveat: LLM rank는 업데이트 부분공간만, CNN 폭은
모델 자체를 바꿈 — 문서에 명시하고 해석 시 분리).

| 그룹 | 그리드 | 셀 수 | 예상 |
|---|---|---|---|
| C1-probe (fidelity·φ 규모) | cifar10 × {iid, label_flip(대조군)} × w{0.5,1,2,4} × k{2,5,10}/N=10 × 3seed, Ripple 제외, ORACLE_A=0 | 72 | ~10–40min/셀, 총 ~25 GPU-h (3090 array) |
| C2-probe (intervention Δ) | cifar10 iid × {clean, label_flip} × [w 4점 @C=0.1] + [C{0.05,0.2} @w=1] × 3seed | 36 | ~1–2h/셀, 총 ~50–70 GPU-h |

- 산출물: `slurm/grids/probe_c1.txt`, `probe_c2.txt` (기존 `run_array.sbatch`
  재사용, `RUN_NAME|ENV` 관례) + track_c1/c2에 `C1_WIDTH`/`C1_KFRAC`/`C2_WIDTH`
  env(기본 1.0 = 현행), `models/cnn.py`에 width 인자(기본값 현행).
- 제출 예: `sbatch -J probe_c1 -t 4:00:00 --array=0-71 scripts/run_array.sbatch grids/probe_c1.txt experiments/track_c1.py`

### 2.3 각 조건에서 측정 (승인된 지표 셋)

val-loss 절대 감소폭 / per-round·coalition Δ 크기 / fidelity(Spearman·Kendall·
Pearson, estimator vs (b)) / intervention Δ(paired) / 절대 성능(MMLU·ROUGE, LLM) /
Taylor 타당성(rank↑ 시 estimator-vs-(b) fidelity 하락 여부 — HVP는 LoRA 파라미터
공간에서 크기 2r∝ 이므로 rank 64에서 2차항 비중 변화 관찰) / (C) val bootstrap SE.

### 2.4 오염축 × 비IID축 매트릭스 (신호 실재성 = B축; Yonghee 2026-07-02)

§1.4·§0이 보인 것: fidelity 신호를 만드는 건 학습 강도(A축)가 아니라 클라 간 실제
차이(B축)다. rank·참여 probe(§2.1–2.3)는 A축을 마저 확인하는 값이고, 이 매트릭스가
B축을 직접 연다. 오염축(clean↔오염)과 비IID축(IID↔5-domain)을 분리해 각각의
fidelity·탐지 기여를 정량화한다(기존 silo5는 둘이 항상 결합돼 분리 불가).

| 무대 \ 클라 | clean | noisy | free-rider(rand/zero) | poison |
|---|---|---|---|---|
| **IID** (alpaca 균질) | 신설 | 신설 | 신설 | 신설 |
| **non-IID** (5-domain) | **신설** | ✓ 기존 silo5 | ✓ 기존 | ✓ 기존 |

- 규모: silo5 급 통일(val20/R10, N=5 full, (b)=exact 2⁵) — 전 칸 같은 규모라
  IID↔비IID, clean↔오염 직접 비교. 신규 6셀×3seed(iid5 {clean,noisy,frrand,frzero,
  poison} + silo5 clean); silo5 오염 3셀은 기존 재사용.
- 측정(위계 순): **1차 fidelity** = (b)oracle 자기 순위 cross-seed ρ — 관심 미지수:
  non-IID clean이 오염 없이도 ρ 높은가(도메인 순수 신호), IID clean은 ρ≈0 재현되는가.
  **2차-③ 탐지 AUROC** = 오염 클라 이진 탐지 — 핵심 대조 IID+noisy vs non-IID+noisy
  ("도메인 이질성이 탐지를 돕나/방해하나"; IID 균질 배경은 오염 클라만 순수하게 튀어
  탐지 신호를 배경 효과와 분리). detector 4종(FedDQC/STD-DAGMM/FLTrust/FLDetector)
  + valuation φ 이미 붙음.
- 구현: `phase2_matrix.py`에 `REGIME=iid5`(build_alpaca_iid, silo5-matched totals)
  + `THREAT=clean` 분기(2026-07-02, env-gated·기존 동작 불변) + `build_alpaca_iid`에
  backdoor 인자. 스모크 6/6 green(iid5 clean/noisy/frzero/poison + silo5 clean;
  poison real-install은 실행 시 확인). poison은 별도 config(LR=2e-3 BATCH=8 EPOCHS=5
  POISON_FRAC=0.8 POISON_TRAIN=1000).
- 신규 셀명(`1B_iid5_*`, `1B_silo5_clean`)이라 **기존 결과 안 덮어씀**;
  `runs/phase2_matrix/rundirs`에 추가해 make_analysis 통합 분석.

## 3. probe 결과 (실행 후 기입)

### 3.1 rank probe — anchor5 (seed0, 파일럿 진행 중)

| | rank16 (기존 재사용) | rank32 | rank64 |
|---|---|---|---|
| (b)oracle φ range | 0.00119 | 0.00102 | 0.00106 |
| Flirds Spearman vs (b) | **+1.000** | **+1.000** | **+1.000** |
| Flirds1st / Banzhaf | +1.000 | +1.000 | +1.000 |
| FedSV | +0.700 | +0.700 | +0.500 |

seed0 3점(rank 16→64, 4배)이 §0·§1.6 예측을 확정한다: **(1) estimator fidelity가 안
깨진다** — Flirds/Flirds1st/Banzhaf가 (b) 값-수준 +1.000 유지(HVP가 크기 2r인 LoRA
공간에서 2차항 비중이 바뀌어도 fidelity 무영향); **(2) 신호(클라 간 φ 차이)가 안 커진다**
— φ range가 0.001 근처에서 평평(rank32/64에서 오히려 미세 감소; IID-clean이라 φ는 추첨
노이즈, rank는 그 크기를 못 바꿈). FedSV만 rank64에서 0.5로 하락(부분참여 방법, seed0
단일 노이즈 가능). rank↑는 A축 lever이지만 IID-clean에선 B축 신호를 만들지 못한다는
§1.4 판정과 정합. → seeds 1–2로 굳히기; 방법 구별은 참여 probe(§3.2)가, 신호 실재성
B축은 §2.4 매트릭스가 검증.

### 3.2 참여 probe (std50k5) — TBD (파일럿 실행 중)

### 3.3 오염축×비IID축 매트릭스 (§2.4) — TBD (std50k5 후 자동 실행)

### 3.4 CNN probe (C1 fidelity + C2 intervention) — **완주** (A축 CNN 확정)

CNN이 A축 probe의 완주 파트다(LLM §3.1은 seed0 파일럿, §3.2는 실행 중). **C1** = 폭 w{0.5,1,2,4}×참여
k{2,5,10}/N=10×{iid, label-flip}×3seed = 72셀(66 신규 + (w1,k10) track_c 재사용). **C2** = 폭·참여 sweep ×
{clean, label-flip}×3seed = 30셀(f0.2는 shapleyfl arm의 2²⁰/라운드 exact 불가로 제외 → 계획 36→30). 전체 표·
수치 = overview [[flirds-experiment-results-overview-2026-06-25]] §3.6; 여기선 진단 판정만. (커밋 `d2e7ed6`.)

**(1) 신호 실재성 — (b)oracle 자기순위 cross-seed ρ** (full 참여 k=1.0, 폭별):

| 시나리오 | w=0.5 | w=1 | w=2 | w=4 |
|---|---|---|---|---|
| iid | +0.034 | −0.042 | +0.038 | +0.123 |
| label-flip | +0.976 | +0.968 | +0.859 | +0.923 |

**폭을 8×(0.5→4) 키워도 iid ρ는 0 근처 불변**, 오염(label-flip)은 ρ≈0.9(역시 폭 무관). §1.4 CNN 대조
(iid −0.042 vs label_flip 0.968; = 이 표의 w=1 칸, `track_c` RESULTS.txt와 교차검증 일치)를 폭 그리드로 확장 재현
— **신호 실재성은 A축(용량)이 아니라 B축(오염)이 만든다**(§0-3·§1.6 주 병목 판정을 CNN이 직접 확정). 단
partial 참여(k<1.0)면 label-flip도 ρ→0 붕괴: N=10·R=10에선 클라당 참여 ~2회라 φ per-round 분해가 참여
추첨에 지배됨(§1.3d). φ range도 동일 — iid ~0.05 vs label-flip ~0.12(2–4×), 둘 다 폭 평평.

**(2) method fidelity — 2차항이 부분참여에서 값을 한다.** Flirds(2차)·Banzhaf는 폭·참여·시나리오 전반 0.9+
(전 72셀 pool Flirds +0.953±.080). **Flirds-1st는 참여↓서 붕괴**(label-flip k=0.2→+0.305, full→+0.940; Flirds
2차는 k=0.2도 +0.904 유지). → §1.5의 "방법 차이는 부분참여가 만들고 2차항이 방어"를 CNN이 확증. **caveat**:
이 붕괴는 CNN R=10(짧은 지평) 특성 — LLM std20(R=200, 클라당 ~20회 참여)에선 2/20이어도 Flirds-1st +0.999.
"참여 분수"가 아니라 **클라당 참여 횟수**가 1차항 정확도의 조건.

**(3) intervention(C2) — clean parity vs 오염 이득, 둘 다 폭 무관.** clean은 전 폭·참여에서 Δacc≈0(do-no-harm
parity, |Δ|<0.006); label-flip은 flirds_mult Δ≈+0.09·shapleyfl +0.08(폭 무관), vanilla가 오염에 눌려 target 0.6
미달인데 개입 arm은 도달(폭↑일수록 빠름, r2t 82→55). 탐지 AUROC 0.93~0.99. → **개입 효과 크기도 A축이 아니라
B축(오염)이 지배**(§4-2 예비 권고 확정). 예외 sfedavg(softmax 선택): AUROC 높아도 개입 Δ≈0 = 탐지≠좋은 개입.

**CNN probe 종합**: A축 lever(용량 폭 8× + 참여)는 IID-clean 신호를 못 만든다(ρ≈0, φ 작음, 개입 parity). 신호는
오염이 만든다(ρ≈0.9, φ 2–4×, 개입 Δ≈+0.09; 전부 폭 무관). → 다음은 §2.4 B축 매트릭스(오염축×비IID축).
LLM rank/참여 probe(§3.1–3.2)가 seeds 1–2·std50k5로 굳히면 A축 판정 완결.

## 4. "신호를 키우려면 무엇을 바꿔야 하는가" — Phase 1 기반 예비 권고

probe 결과로 확정하되, Phase 1만으로 이미 말할 수 있는 것:

1. **(1차 fidelity)** 추정기의 fidelity 측정 자체는 병목이 없다(값-수준 0.9999+).
   fidelity 실험의 남은 정보량은 "방법 간 구별"인데, 이는 무대의 비가산성이 아니라
   부분참여에서 나온다 → 참여수 lever(N=50/5)가 rank lever보다 fidelity 축 정보량에
   직접적일 것.
2. **(2차 성능/수렴)** intervention 효과가 '커지는' 무대의 공통점은 vanilla가
   비효율적인 곳(7B std20: 플래토+과적합에서 r2t −32라운드). IID-clean + 균질
   클라에서는 정답이 parity — 효과 크기를 키우려면 (i) 학습이 아직 진행 중인
   구간에서의 가중이 아니라 (ii) 클라 간 실제 품질 격차가 필요하다.
3. **(무대 판단거리 — 이번 스코프 밖, 데이터 근거만 제시)** 클라 간 진짜 신호가
   있는 무대(품질 격차·비IID·오염 혼합)로 가면 φ 신호가 노이즈 위로 올라오는 것은
   phase2_matrix(poison 9–18×)와 Track C(label_flip 안정성 0.97)가 이미 보여줌.
   "clean × non-IID" 칸(기존 next 항목의 분리 실험)이 자연스러운 다음 후보.
   base model·task 변경은 이번에 다루지 않음.

---
부록: 분석 스크립트는 세션 스크래치(`phase1_diag*.py`)에서 실행 — 수치는 rundir에서
재계산 가능. 3B_std20 metrics는 진단 당시 working tree의 β=0.3 재실행본(캠페인이
의도대로 canonical에 덮어쓴 것)을 읽었으나 방법 공통 수치의 차이는 1e-3 이하로 결론
불변(β=0.5 원본은 git 히스토리에 보존). **이번 probe는 기존 결과를 일절 덮어쓰지
않는다**: LLM은 `runs/probe_signal/rundirs/`에 신규 셀명(`1B_anchor5_r{32,64}_seed*`,
`1B_std50k5_r*_seed*`), CNN은 `runs/probe_signal/cnn_c{1,2}/`에 `pc{1,2}_*` 셀명으로
저장하고, 기존 그리드와 겹치는 기준점 셀(rank16 anchor, CNN w=1)은 재실행 없이 기존
rundir을 재사용한다.
