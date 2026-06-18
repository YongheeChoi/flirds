# Track D 재설계 — 검토용 정리 (2026-06-13, 갱신 2026-06-15)

> **이 문서의 성격**: Yonghee가 시간 날 때 찬찬히 검토하기 위한 임시 문서입니다 (읽고 삭제 예정).
> 영구 기록은 `research-wiki/wiki/flirds-implementation-plan.md` §3.11 + `research-wiki/raw/conversations/flirds/2026-06-13-track-d-redesign-iid-clean.md`에 따로 있습니다.
> "추천대로 모두 진행" 지시에 따라 아래 설계대로 **구현+검증 완료**, **1B 본실행 진행 중**(2026-06-15: anchor5 3-seed + std20 3-seed, GPU 0–3).

## ⭐ 검토 빠른 요약 (먼저 읽기)

**현재 열려 있는 결정 = 3건** (전부 "순수 추가형" 또는 단일 baseline → 본 그리드를 막지 않음; 나중에 덧붙이거나 한 method만 재계산):
1. **bridge arm** — vanilla를 문헌 레시피(AdamW 5e-5 cosine, r32/α64)로 1회 더 돌려 "SGD mom=0 caveat"을 *측정된 갭*으로 전환할지. (성능 비교 전용; valuation 불가) → §4 + §9.
2. **FlowerTune-채점 모드** — MMLU를 likelihood(현행) 외에 생성-추출식으로도 채점해 FlowerTune published 숫자와 채점방식까지 맞춘 참조 컬럼 추가할지. → §4.
3. **ShapleyFL β = 0.5 vs 0.3** — 논문은 0.3, 우리 wiring은 0.5(기존 C2와 일관). ShapleyFL 1행 + shapleyfl_w arm 1개에만 영향. → §10 끝.

**이미 결정/진행된 것** (참고): (a) seed 수 = **3-seed 전부**(2026-06-15 결정) · std20 = **진행 중**(greenlight) · LoRA r16 유지 · MMLU full-test 0-shot · D-옵1/옵2 제외.

**설계가 타당한지 확인할 핵심 포인트**: §1(재정의=IID·clean 3축) · §2(무대=OpenFedLLM verbatim + caveat 3) · §4(직접비교 가능성: caveat-free 셋업 없음) · §10(파일럿 실측: Flirds vs (b) +1.000, coalition류 1/5 비용).

---

## 1. Track D의 재정의 (이 세션에서 확정)

Yonghee 정의(요지): *"이 실험은 특수 상황(Non-IID, 오염) 말고 일반적인 학습 상황(IID)에서
SV 계산, benchmark accuracy, 수렴 속도 측면에서 우리 방법론이 다른 방법론에 비해 얼마나
잘 하는지를 검증하기 위한 것. 오염 클라 탐지는 전혀 아님. val/test 데이터도 IID."*

→ **오염축 전면 제거**: answer_swap·noisy 클라·AUROC·오염-필터링 arm(mixed/clean_oracle/
flirds_filtered/random_drop)·filter-q 결정 모두 Track D에서 삭제. 질문 위계(루트 CLAUDE.md에
새김) 순서대로 3축만:

| 축 | 질문 | 지표 |
|---|---|---|
| ① SV 계산 (1차, 헤드라인) | 각 방법이 정확한 Shapley를 얼마나 잘 근사하나 | Spearman·Kendall + GTG 거리 3종(cosine/Euclid/max-diff) + wall-clock |
| ② benchmark accuracy (2차-①) | 각 방법의 점수로 온라인 개입(가중/선택)하면 최종 성능이 어떤가 | MMLU full-test 0-shot + 같은분포 Alpaca-test ROUGE-L |
| ③ 수렴 속도 (2차-②) | 같은 개입 arm들의 수렴 | round별 val-loss 곡선, rounds-to-target, per-round 오버헤드 |

---

## 2. 무대 — OpenFedLLM 대표 레시피 verbatim 미러

출처: 로컬 참조클론 `codes/external/OpenFedLLM/training_scripts/run_sft.sh` (직접 확인).
그들의 대표 SFT 스크립트가 정확히 이 무대입니다:

| 항목 | OpenFedLLM 원값 | 우리 | 비고 |
|---|---|---|---|
| 데이터 | `vicgalle/alpaca-gpt4`, **dataset_sample=20000** | 동일 (IID 균등 샤드) | 그들 코드 기본값 |
| 템플릿 | alpaca (`### Instruction:\n{} \n\n### Response:`) | 동일 (verbatim, `utils/template.py` 미러) | `data/llm.py:_ALPACA_PROMPT` |
| 클라/참여 | num_clients=20, sample_clients=2 | 동일 (std20 레짐) | |
| 라운드 | num_rounds=200 | 동일 | 총 64k 샘플 ≈ 3.2 epoch |
| 로컬 | max_steps=10, batch=16, seq=512 | 동일 | seq 768→512로 변경됨 |
| 모델 | meta-llama/Llama-2-7b-hf | 1B/3B=Llama-3.2-Instruct + **7B=Llama-2-7b-hf 동일** | task8 사다리 |
| LoRA | r32/α64 | **r16/α32** | 전 트랙 일관성(기존 결정); α=2r 관행 동일 — **caveat ①** |
| optimizer | AdamW 5e-5, 라운드별 cosine | **SGD mom=0, lr 1e-3 상수** | Taylor/per-step 가정 lock — **caveat ②** (FedIT-SGD 전례: FedHDS Table 2) |
| 정밀도 | 8bit 로드 | **fp32** | oracle/estimator 정밀도 바닥(coalition diff ~1e-2 < bf16 ~8e-3) — **caveat ③** |

데이터 분할: train 20,000 / val 200(utility용) / test 1,000(같은분포 ROUGE-L) — 전부
같은 Alpaca-GPT4 분포에서 상호 disjoint carve. **train/val/test 모두 IID** (요구 충족).
외부 벤치마크 = MMLU full-test 14,042문항 0-shot likelihood 채점(기존 결정 유지).

### 레짐 2개

| 레짐 | 구성 | 역할 |
|---|---|---|
| **std20** | N=20, 2/round, R=200 | 문헌 표준형. 축②③의 메인 무대. GT=(b) per-round exact(라운드당 2²) |
| **anchor5** | N=5, full participation, R=30 | oracle 정밀 지점: exact (b) 2⁵ + **(a)-retrain oracle**(32 retrains, val-loss utility·fp32 = task6 교훈) + Banzhaf 포함 전 coalition 방법 exact. (a)+(b) 듀얼 GT는 조사 확인된 문헌 공백 |

---

## 3. 축별 상세

### 축① SV 계산 fidelity (1차)
- vanilla FedAvg 궤적 **1개를 얼리고** 전 방법이 같은 로그에서 φ 계산 (기존 phase2 관행).
- **방법 11종**: (b)oracle(GT) / (a)oracle(anchor5, GT₂) / Flirds-2nd / Flirds-1st / GTG /
  FedSV / ComFedSV / ShapleyFL / FedIF / Banzhaf(anchor5만) / loss-heur.
  부호 규약은 phase2_matrix 그대로(전부 "val-loss attribution, good→LOW"로 정렬;
  ShapleyFL/FedIF/ComFedSV/(a)는 negate).
- Ripple은 LLM eigsh 이슈로 제외 관행 유지(LLM 교정은 별도 세션 결정 존속) — caveat 한 줄.
- 거리 metric은 같은 단위(val-loss 게임)일 때만 절대 해석 가능 — min-max/EMA류(ShapleyFL 등)는
  rank 컬럼으로 해석(`eval/metrics.py` docstring에 명시된 기존 caveat). C1과 같은 metric 세트라
  CNN↔LLM 트랙 간 표가 나란히 섭니다.
- IID 무대 특성 명시: 클라 교환가능 → 진짜 φ 격차가 작은 **near-tie 무대** = rank fidelity의
  분해능 시험대 (GTG가 IID 시나리오에서 거리 metric을 쓴 이유와 같은 논리).

### 축② benchmark accuracy (2차-①)
온라인 개입 arm — `fl/intervene.py`(C2에서 빌드·검증, CNN/LLM 공용 설계) 이식:

| arm | 메커니즘 | 비고 |
|---|---|---|
| base | 학습 전 모델 | 바닥 참조 |
| **vanilla** | plain FedAvg | **단순 baseline** = phase① 궤적 재사용(추가 비용 0) |
| flirds_w | Flirds 곱셈가중 w∝n·s (β=0.5) | equal-n IID라 대체형과 항등 → 규칙 ablation 생략 |
| flirds_sel | Flirds softmax-selection | **std20만** (full 참여에선 선택이 퇴화 — k=n) |
| shapleyfl_w | ShapleyFL 대체가중 (β=0.5) | 그들 논문 메커니즘 (C2 wiring 동일) |
| fedif_w | FedIF 대체가중 (β=0.7=1−γ) | 〃 |

각 arm → **MMLU full-test 0-shot + Alpaca-test ROUGE-L**.
기대치 정직 명시: clean IID에서 개입 이득 ~0이 정상(**do-no-harm 검증**; ShapleyFL/FedIF의
이득 주장은 non-IID/오염 무대였음) — 차이가 나면 그게 발견.
(removal curve는 권장대로 **제외** — IID라 정보량 낮음; 필요 시 추가 쉬움.)

### 축③ 수렴 속도 (2차-②)
- 각 arm 로그에서 post-hoc으로 round별 val-loss 곡선 (`_val_curve`; 추가 학습 0).
- rounds-to-target: vanilla의 최종 val-loss를 target으로, 각 arm이 도달한 첫 라운드.
- per-arm 학습 wall-clock(개입 오버헤드 포함) — FedIF의 집계시간 보고 관례와 정합.

### 시드·스케일
- 3 seeds (우리 관행; 문헌 4–5 caveat).
- 스케일 사다리: 1B(전체 그리드) → 3B(축소) → 7B(fidelity 우선; **Llama-2 pad-token 첫 smoke 확인** 메모 유지).

---

## 4. 직접 비교 가능성 분석 (요청하신 확인)

**결론부터: caveat 없이 숫자-옆-숫자가 성립하는 published 셋업은 존재하지 않습니다.**
이유와 가장 가까운 후보들:

1. **OpenFedLLM (KDD'24)** — 학습 무대는 우리와 **완전 동일**하게 맞췄지만(위 표), 그들의
   general 트랙 평가는 **MT-Bench/Vicuna = GPT-4-judge(API)** 라서 우리가 같은 지표를 만들 수
   없습니다(로컬 클론 `evaluation/` 직접 확인: open_ended judge 스크립트 + FinGPT close-ended뿐,
   MMLU 없음). → **무대-직접, 지표-불가**.
2. **FlowerTune LLM Leaderboard (NeurIPS'25 벤치마크, arXiv:2506.02961)** — 같은 데이터
   (alpaca-gpt4) + 같은 벤치마크(MMLU 0-shot, STEM/인문/사회 카테고리)로 published 숫자가
   있고 Llama-3.2-1B-Instruct(평균 1.03%)/3B(24.92%) 행도 존재합니다. 그러나:
   - FL 설정 상이: 20클라 중 **4/round, R=15**, 로컬 10 steps, **DoRA r32/α64**, AdamW 5e-5 cosine.
   - 1B가 1.03%라는 것은 **생성-추출식 채점**(답 문자를 못 뽑으면 오답)을 시사 — 우리
     likelihood 채점(4지선다 최저 ~25%)과 채점 방식이 다름.
   - → **지표-유사, 설정·채점-상이**: "같은 데이터·같은 벤치마크의 참조점" 수준으로 병기 가능,
     절대수치 비교는 부적절.
3. **축① fidelity** — LLM-scale FL valuation 선행이 없음(06-12 조사 확정: 직접 경쟁자 0) →
   비교는 구조적으로 내부(oracle 대비)일 수밖에 없고, **이 공백 자체가 기여**. 숫자-옆-숫자
   fidelity 비교의 무대는 CNN **Track C1**(GTG 무대 미러)이 담당 — 이미 빌드 완료.

**옵션 2개 (승인 시 추가 구현; 현재 미구현)**:
- **bridge arm**: vanilla를 문헌 레시피(AdamW 5e-5 cosine, r32/α64)로 1회 더 돌려 MMLU 비교
  → "SGD caveat"을 **측정된 갭**으로 전환. 비용 = 궤적 1개 × seed (valuation 불가, 성능 비교
  전용). `llm_server`에 optimizer/lr-schedule override 추가 필요(기본값 보존).
- **FlowerTune-채점 모드**: MMLU를 생성-추출식으로도 채점(기존 `generate` + `extract_choice`
  재사용) → FlowerTune 숫자와 채점-방식까지 맞춘 참조 컬럼.

---

## 5. 구현 내역 (이 세션, 전부 미커밋 — 커밋은 요청 시)

| 파일 | 변경 | 성격 |
|---|---|---|
| `codes/flirds/data/llm.py` | `_fmt_alpaca`+`build_alpaca_iid`(IID 샤드+val/test carve, OpenFedLLM 템플릿) · `build_domain_iid`(FiQA/AQUA IID — **현 설계에선 미사용**, D-옵1 잔재로 존치) | 신규 함수 (기존 함수 무수정) |
| `codes/flirds/eval/mmlu.py` | **신규** — MMLU 0-shot likelihood 채점(문항당 forward 1회, letter-token argmax, left-pad/left-trunc, MMLU_LIMIT 서브샘플) | 신규 파일 |
| `codes/flirds/fl/llm_server.py` | `run_llm_fedavg_logs`에 `select_fn/weights_fn` 개입 seam 통과 (기본 None=비트 동일) | 시그니처 확장 |
| `codes/experiments/track_d.py` | **전면 재작성** — 위 설계 그대로 (REGIME=std20/anchor5; 축①②③; RunLogger persist) | 러너 |

실행법 (`codes/`에서):
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. REGIME=anchor5 SEED=0 python -u experiments/track_d.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. REGIME=std20  SEED=0 python -u experiments/track_d.py
# env: ROUNDS/TOTAL_TRAIN/VAL/TEST/MAX_STEPS/LR/MAXLEN/BATCH/VAL_CHUNK/VAL_MAXLEN
#      MMLU_LIMIT(스모크용)/MMLU_BATCH/ARMS=0(fidelity만)/ORACLE_A(anchor5 기본 on)/SEED/RUN_NAME
```

구현 디테일 메모:
- **(a) oracle** = `make_a_utility`(러너 내 val-loss-only utility; `exact_shapley` 재사용,
  `llm_subset_utility`의 ROUGE 생성 비용 회피) — task6 "same-game val-loss/fp32" 교훈 그대로.
- **훅 위생**: SFTTrainer가 라운드/리트레인마다 train모드+임베딩 훅을 남김 → functorch HVP가
  금지(기존 "LLM 3 musts") → 온라인 점수기는 `_guard`(매 호출 eval()+훅 클리어), (a) 직후에도
  동일 처리. 스모크에서 실제로 이 모드 재현·수정 확인됨.
- 수렴 곡선은 R=200 로그(LoRA 상태 ×200, 1B ≈ 한 궤적 ~18GB GPU 상주)에서 post-hoc 평가 —
  B200 단독 GPU에서 여유; arm마다 사용 후 `del`.
- ComFedSV는 저R에서 행렬완성 퇴화(스모크 R=4에서 NaN — task7d 확인된 특성; R=200 정상 예정).

## 6. 검증 결과 (스모크; 값은 tiny-config 노이즈, 코드경로 검증 목적)

| 스모크 | 결과 |
|---|---|
| gpt2 std20 (R=4, 20클라) | 11-method 테이블+arm 5종+persist green; Flirds vs (b) Spearman **+1.000**; flirds_sel 정상 등장 |
| gpt2 anchor5 (R=2, (a) 포함) | **(a) 32 retrains 경로 green**(181s); Banzhaf 등장, flirds_sel 자동 제외; 전 arm green |
| 1B anchor5 (R=2, (a) 포함) | **(a) vs (b) Spearman/Kendall +1.000/+1.000**(1B fp32 듀얼오라클 일치 — gpt2의 +0.1은 모델-스케일 노이즈였음 확인); Flirds +1.000(cos-d 0.0000, **5.9s**) vs coalition류 ~26s vs **(a) 474.5s** = 비용 서열 스모크에서도 가시; base MMLU 0.480(limit=50); arm parity(R=2 미학습 정상, ROUGE는 base 0.1716→arm 0.1740 이동) |
| CNN guard | green + import-graph 증명(수정 모듈이 CNN 경로에 부재) |

이전(오전) D-메인 빌드의 1B 스모크 참고치: base MMLU 0.530(limit=100; 공칭 ~49% 정합),
Spearman +1.000 — MMLU 하니스·fidelity 파이프는 실모델에서 이미 확인됨.

## 7. 비용 추정 (1B, B200 1장; 스모크 외삽 — real run 전 1-seed 파일럿으로 확정 권장)

| 항목 | 추정 |
|---|---|
| std20 궤적 1개 (R=200) | ~1.5–2h |
| std20 fidelity 전 방법 | ~30–60min ((b)-perround는 R200×2²로 수 분) |
| std20 arm 4종 추가 학습 | ~6–8h (arm당 1.5–2h; 개입 점수 오버헤드 포함) |
| MMLU full-test 1회 | ~10–20min (×arm 6 + base) |
| anchor5 전체 ((a) 포함) | ~3–5h ((a) 32 retrains ≈ R30 기준 ~2–4h 단일 GPU; 4-GPU 샤딩 가능) |
| **합계** | 1B 3-seed 2-레짐 ≈ **40–60 GPU-h**; 3B ≈ ×3, 7B ≈ ×7(fidelity 우선) |

## 8. 오늘 무효화/보류된 것 (경위 추적용)

- 오전 결정 중 **supersede**: noisy 레짐 구성(silo5 2-noisy/device20 10-noisy), filter-q=오염수
  매칭, D-옵1(FedDQC Table-1 미러), D-옵2(FedHDS 미러; A/B/C fork 질문 자체 소멸 — FedHDS
  무대는 비IID+OOD-eval이라 IID 요구와 충돌).
- **보존**: q-sweep follow-up 메모(오염 실험용), `build_domain_iid`·corruptor 코드(분리축 실험 재료),
  MMLU full-test 0-shot 결정, LoRA r16 유지 결정.
- **NEW 후속 설계 항목**: **오염 탐지축 ↔ 비IID축 분리 실험** — 현 LLM main 실험은 두 축이 항상
  결합(5-도메인+오염). "오염 없는 순수 비IID" 칸이 빈칸(CNN C2엔 있음). FedHDS 무대(비IID·clean)가
  그 후보. 별도 세션에서 설계.
- **ShapleyFL β 불일치 (결정 필요)**: 06-12 PDF 재추출(log.md note)에서 ShapleyFL 논문이 **β=0.3**
  채택임이 확인됨 — 우리 C2/D wiring은 β=0.5(초기 추출 기준, `intervene.py` docstring도 0.5를
  ShapleyFL 값으로 기술). shapleyfl_w arm과 fidelity의 `shapleyfl_from_logs(beta=0.5)` 호출에 공통.
  0.3으로 통일할지(논문 충실) / 0.5 유지(기존 실험과 일관)할지 Yonghee 결정 항목.

## 9. 다음 단계

1. (이 세션) 1B 스모크 완료 확인 → 기록(plan §3.11/raw/log) 갱신 [완료 시 채팅 보고]
2. (Yonghee) 이 문서 검토 → bridge arm / FlowerTune-채점 모드 추가 여부 결정 → 문서 삭제
3. (본 실험 세션) real run: 1B 파일럿 1-seed로 비용 확정 → 3-seed 2-레짐 → 3B/7B stage-gate
4. (별도 세션) 오염축·비IID축 분리 실험 설계

참고 링크: [FlowerTune 벤치마크 논문](https://arxiv.org/abs/2506.02961) ·
[FlowerTune general-NLP 리더보드](https://flower.ai/benchmarks/llm-leaderboard/nlp/) ·
OpenFedLLM 로컬 클론 `codes/external/OpenFedLLM/`

---

## 10. 파일럿 실측 + arm 버그 수정 (2026-06-13 추가; §6·§7 추정치를 실측으로 대체)

**anchor5 1-seed 파일럿(1B, 실제 config N=5/R=30/train20k/val200/MMLU-full)** — fidelity(1차 축) 완주, 실측:

| 방법 | Spearman vs (b) | Kendall | cos-거리 | runtime |
|---|---|---|---|---|
| (a)oracle (retrain GT) | **+0.900** | +0.800 | 0.0002 | **31,519s ≈ 8.75h** |
| **Flirds** | **+1.000** | **+1.000** | **0.0000** | **725s** |
| Flirds1st | +1.000 | +1.000 | 0.0000 | 237s |
| GTG | +1.000 | +1.000 | 0.0010 | 3,647s |
| Banzhaf | +1.000 | +1.000 | 0.0000 | 3,623s |
| loss-heur | +1.000 | +1.000 | 0.0000 | 1,123s |
| ComFedSV | +0.900 | +0.800 | 0.24 | 2,255s |
| ShapleyFL | +0.900 | +0.800 | 0.16 | 3,609s |
| FedSV | +0.700 | +0.600 | 0.005 | 3,609s |
| FedIF | +0.700 | +0.600 | 0.34 | 238s |
| (b)oracle | (truth) | — | — | 3,624s |

- **헤드라인(1차 질문)**: clean-IID(클라 교환가능 = near-tie 난도)에서 **Flirds가 exact (b) Shapley를 완벽 재현(+1.000), coalition류의 ~1/5 · (a)의 ~1/43 비용**. FedSV/FedIF는 IID서도 (b) 불완전 추종(+0.700) = 우리 분해능 우위 지점.
- **(a) vs (b) = +0.900**: task6의 3B 결과와 동일(인접 1쌍 swap = retrain 노이즈; 듀얼 GT 합치 재확인).

**⚠ 실측 비용 정정 (§7 대체)**: **(a)-retrain oracle = 8.75h/seed** (1B anchor5; 추정 ~2h를 크게 초과 — 32 coalition × R30 × SFTTrainer 재인스턴스화 오버헤드). (b)/coalition류 각 ~1h, Flirds 12min, Flirds1st 4min. → **(a) seed 수 = 3-seed 전부로 결정**(Yonghee, 2026-06-15; (a) 자체의 seed 분산도 논문에 제시). anchor5 3-seed가 GPU 1·2·3에서 병렬 실행 중(~17h).

**arm 단계 OOM → 수정 완료 (이 세션)**:
- 원인: 온라인 Flirds 점수기 `intervene.flirds_round_raw`가 **val-set 청킹 누락** → 200개 val에 단발 eager HVP(175GB) (FedIF 경로는 이미 청킹, ShapleyFL은 forward-only라 무사).
- 수정: `intervene.py`에 `loss_chunks` 통과(**기본 None = CNN 비트동일, guard CLEAN**) + `track_d` build_arms에서 `lc` 전달. 메모리 증거로 동작 확인(수정 후 프로세스 peak **14.75GB**, 175GB 폭주 소멸).
- 견고화: fidelity를 **arm 전에 체크포인트 persist**(8.75h (a)를 arm 크래시로 다시 안 잃도록) + arm try/except + **`FIDELITY=0` 토글**(비싼 (a)/fidelity 없이 arm만 싸게 재검증/재실행).
- **파일럿 run-dir 손실**: OOM이 persist 직전 발생 → 위 표는 .log에만 존재(체크포인트 persist 도입으로 재발 방지).

**arm 완주 미확인 = GPU 경합**: 재검증 중 **GPU 0에 외부 잡 난입**(타 사용자 llama2-7b AdvBench 안전성 평가, `CUDA_VISIBLE_DEVICES=0`, 순간 161GB 점유)으로 arm end-to-end 완료는 아직 못 봄(제 프로세스는 15.6GB로 정상). 현 GPU: 0=외부 잡 / 1–3=real grid(각 ~82GB 여유) / 4–7 금지. → **어디서 돌릴지 Yonghee 결정 대기**.

**결정 항목 (열림 3건)**: bridge arm / FlowerTune-채점 / ShapleyFL β(0.5↔0.3). [(a) seed 수 = 3-seed로 결정됨; 상단 ⭐요약 참조]

## 11. ✅ 1B 본실행 완료 — 3-seed 최종 (2026-06-15)

6셀(anchor5×3 + std20×3) 전부 완주, rundirs `1B_{anchor5,std20}_s{0,1,2}` 영속화. **arm 청킹 수정 검증됨**(armcheck 6 arm OOM 0; 본실행도 전 arm 완주).

**1차 fidelity — Spearman vs (b) oracle, 3-seed mean±std:**

| 방법 | anchor5(N=5) | std20(N=20, 2/round) | 비고 |
|---|---|---|---|
| **Flirds** | **+1.000±0.000** | **+1.000±0.000** | 양 무대 완벽+무분산; anchor5 726s(=b의 1/5, a의 1/43) |
| Flirds1st | +1.000±0.000 | +0.999±0.001 | std20서 2차항이 마지막 갭 메움 |
| GTG | +1.000±0.000 | +0.975±0.018 | exact-coalition |
| FedSV | +0.700±0.163 | +0.910±0.073 | |
| ComFedSV | +0.500±0.432 | **+0.093±0.146** | std20 붕괴 |
| ShapleyFL | +0.700±0.283 | **+0.194±0.351** | std20 붕괴+시드 요동 |
| FedIF | +0.067±0.531 | **+0.157±0.303** | 양 무대 약함+불안정 |
| Banzhaf | +1.000±0.000 | (N=5 전용) | |
| loss-heur | +1.000±0.000 | +1.000±0.000 | |
| (a)oracle | +0.933±0.047 | (anchor5 전용) | 듀얼 GT 합치 |

→ **헤드라인**: 근사 baseline(ComFedSV·ShapleyFL·FedIF)은 부분참여(std20)에서 무너지고 시드 분산도 큰데, **Flirds만 두 무대 모두 +1.000±0.000 — 완벽·안정·최저비용**. (loss-heur·GTG·Banzhaf도 exact 계열이라 IID-clean near-additive서 +1.000이나, loss-heur는 상호작용항 없는 N-forward, GTG/Banzhaf는 2^N 비용 — 차별점은 비용+부분참여 안정성, 그리고 비IID/오염 무대[main 실험].)

**2차 benchmark accuracy + 수렴 — clean-IID do-no-harm parity 확인(3-seed):**
- anchor5: 전 arm MMLU ~0.480 동급, ROUGE-L vanilla 0.2725 → flirds_w **0.2741**(미세 우위).
- std20: 전 arm MMLU ~0.474 동급, ROUGE-L vanilla 0.2841 → flirds_w **0.2848**. rounds-to-target 전 arm ~동일(개입 무해).
- 해석: 개입법 이득 주장은 비IID/오염 무대 것 → IID-clean에서 "해를 안 끼침"이 정직한 결과(예상대로).

**남은 것**: 3B/7B stage-gate(3B seed0 = 2026-06-15 시작, (a) 제외 빠른 fidelity 먼저) · 시드는 독립 rundir라 나중에 1B/3B에 seed 추가 가능 · 열린 결정 3건(특히 ShapleyFL β: 본 그리드 β=0.5, 0.3이면 ShapleyFL만 재계산).
