---
type: conversation
date: 2026-06-13
topic: flirds
participants: [Yonghee, Claude]
tags: [track-d, llm-standard-setting, iid-clean, redesign, core-question-hierarchy, openfedllm, flowertune, mmlu, implementation]
---

# Track D 구현 세션 — D-메인 빌드 → 핵심 질문 위계 명시 → IID·clean 전면 재설계

세션 흐름이 3막: (1) 06-12 설계(오염 50% swap 포함)대로 D-메인을 빌드·검증 → (2) Yonghee가
프로젝트 핵심 질문 위계를 명시하며 detection-중심 프레이밍 교정 → (3) Track D를 "일반 학습
상황(IID·clean)" 실험으로 전면 재설계, 재구현, 스모크 완료. 검토용 전체 정리 =
`TRACK_D_REVIEW_2026-06-13.md` (프로젝트 루트; Yonghee 검토 후 삭제 예정).

## 1막 — 원설계대로 D-메인 빌드 (오전; 이후 §3 재설계로 대부분 supersede)

- fork 3건 확정: **레짐 프레이밍**(Yonghee 제안: "5짜리를 silo, 20짜리를 device로" — N=5
  full/2-noisy 보조 + N=20 2-per-round/10-noisy 헤드라인) · **MMLU full-test 57과목 0-shot**
  (likelihood 채점; 설명 후 결정) · **filter q=실제 오염 수 매칭** + "나중에 sweep 추가 실험하는걸
  고려해보자. 기억해놔줘" → 메모리 기록.
- 빌드: `build_alpaca_iid`(OpenFedLLM 알파카 템플릿 verbatim — `utils/template.py` +
  `alpaca_format` 미러) · `eval/mmlu.py`(0-shot likelihood, 문항당 forward 1회, letter-token
  argmax, left-pad/left-trunc) · `experiments/track_d.py`(AUROC+Spearman+필터링 4-arm) ·
  `build_domain_iid`(D-옵1 FiQA/AQUA).
- 검증: gpt2 e2e(silo5/device20/math) + **1B tiny: base MMLU 0.530**(공칭 ~49% 정합 = 하니스
  실측 확인), AUROC 1.000, Spearman +1.000, flirds_filtered가 noisy={0,1} 정확 제거. CNN guard
  green(+import-graph 증명). FiQA 중복 질문 57% 발견(데이터 속성; row-disjoint 유지).
- D-옵2(FedHDS) 스펙 PDF 추출: Dolly 8-task 중 마지막(summarization 1,188)=test 전용, 7개
  카테고리 Dir(α∈{0.5,5}) 200클라, 5%/round, R=60(3B)/40(1.3B), LoRA r8/α16/d0.05, 4-seed.
  fork ②=둘 다(top-k%+bottom-q%), ③=**우리 r16 유지** 결정받음 (①은 §3에서 소멸).

## 2막 — 핵심 질문 위계 (Yonghee 명시; 프로젝트 전체 새김)

D-옵2 fork 질문에 대한 답변에서 (verbatim 요지):

> "우선 오해가 있는데, 우리가 던지는 메인 질문은 우리 방법이 오염된 클라이언트를 얼마나 잘
> 필터링 하느냐가 아니야. **우리의 핵심 질문은 우리 방법론이 FL에서 기여도를 얼마나 정확히 잘
> 측정하느냐야.** 이건 정말 중요한거니까 우리 프로젝트 전체에 이 기록을 꼭 새겨놔야해. 이게
> 가장 기본이 되는 질문이야. 그리고 그 다음 우리가 측정한 기여도가 얼마나 실제로 의미가 있고
> 실효성이 있냐를 검증하기 위해 **일반적인 성능 향상, 수렴 속도, 오염 클라 탐지** 같은 것도
> 부차적인 검증을 위해 수행하는거지. 그리고 **오염 클라 탐지는 성능 향상이나 수렴 속도 다음**으로
> 고려되는 사항이야. 왜냐하면 기여도가 오염 클라 탐지랑 완전 직접적으로 연결되는건 아니거든.
> 이건 중요한거니까 프로젝트 전체를 탐색해서 잘 못 기록된게 있다면 고쳐줘."

→ 새김 위치: 루트 `CLAUDE.md` "핵심 질문 위계" 섹션(신규) · `wiki/flirds.md` "핵심 질문 위계
(locked 2026-06-12)" 섹션(신규) · memory `core-question-hierarchy.md`(신규) · 순서-역전 표현
교정 3곳(plan §3.11 D 평가 서술, memory track-cd, 루트 CLAUDE.md next) · `track_d.py` 출력
컬럼 순서(Spearman을 AUROC 앞으로 — 이후 재설계에서 AUROC 자체 제거).

**협업 피드백 (durable)**: "계속 질문창에만 내용을 담는 이유가 뭐야?" — fork 설명은 본문
메시지에 충분히 풀어서, AskUserQuestion 창에 압축하지 말 것. 질문창 한글 깨짐(앵커→앨커)도
확인 → 용어는 본문에서 풀어 설명. memory에 기록.

- "단순 baseline(FedAvg)이 포함되나" 질문 → 답: valuation 비교(fidelity 표)에는 per-client
  점수를 내는 방법만 행이 됨(Yonghee 자답 정확); plain FedAvg는 ①다운스트림 arm의 대조군
  ②모든 valuation의 기판(frozen 궤적)으로 이미 배치. random/균등/크기비례 φ는 equal-size
  설계에서 수학적 퇴화 → 문헌처럼 random은 다운스트림 쪽 hard bar(이미 보유).

## 3막 — Track D 전면 재설계 (IID·clean)

Yonghee (verbatim 요지): "이 세션에서 실험하는 내용은 오염된(noise, freerider, poison)
클라이언트를 탐지하는 것이 전혀 아니야. … 일반적인 학습 상황(IID)에서 SV 계산, benchmark
accuracy, 수렴 속도 측면에서 우리 방법론이 다른 방법론에 비해 얼마나 잘 하는지를 검증하기
위한거야. … D-메인, D-1옵션에 오염축이 포함되어 있으면 그것도 없애야해. … 난 이 실험은
iid에서 진행되길 원해 validation data나 test data도 똑같이." + **NEW 후속 설계 항목**: "오염된
client 탐지축과 non-iid 상황은 서로 다른 특성으로 검증되어야 하는데 지금 완전 두 개가 중복된
실험만 하나 설계돼버렸네… 두 축이 분리된 실험도 설계를 해야할 것 같아"(분리는 쉬움 — 결합
설계 존재; CNN C2엔 clean×non-IID 셀 있음, LLM이 빈칸). "아예 처음부터 다시 설계… 기존
연구들을 살펴보고 추천하는 실험 세팅을 다시 정리해줘" → 제안 후 "**우선 추천대로 모두** 해주고
세부사항을 따로 정리해놔줘(검토 후 삭제). **선행연구 baseline들과 직접 비교 가능한 실험 세팅이
있는지 확인**해줘."

### 확정 설계 (추천 전부 채택)

- **무대 = OpenFedLLM run_sft.sh verbatim**(로컬 클론 확인): alpaca-gpt4 **20k**(그들
  dataset_sample 기본값), N=20·2/round·R=200·10steps×batch16·**seq512**·alpaca 템플릿·
  Llama-2-7b-hf(=우리 7B 사다리와 동일). deviation caveat 3: SGD mom=0 lr1e-3 상수(vs AdamW
  5e-5 cosine; FedIT-SGD 전례), LoRA r16/α32(vs r32/α64), fp32(vs 8bit).
- **레짐**: **std20**(N=20, 2/round, R=200; (b)=per-round exact) + **anchor5**(N=5 full, R=30;
  exact (b) 2⁵ + **(a)-retrain oracle**(val-loss·fp32, 32 retrains; (a)+(b) 듀얼 GT=문헌 공백) +
  Banzhaf 포함 전 coalition exact).
- **축① fidelity(1차)**: 11-method, 같은 frozen vanilla 로그; Spearman+Kendall+GTG 거리 3종+
  wall-clock; Ripple 제외 관행 유지. IID=near-tie 분해능 시험대 프레이밍.
- **축② benchmark acc**: 개입 arm 6종(base/vanilla/flirds_w 곱셈 β0.5/flirds_sel(std20만)/
  shapleyfl_w β0.5/fedif_w β0.7) → **MMLU full-test 0-shot + 같은분포 Alpaca-test(1,000)
  ROUGE-L**; clean-IID 기대=parity(do-no-harm) 명시. removal curve 제외(권장대로).
- **축③ 수렴**: round별 val-loss 곡선(로그 post-hoc) + rounds-to-target(vanilla 최종 loss) +
  per-arm wall-clock. 3 seeds.
- **직접 비교 분석**: caveat-free 직접 비교 셋업 **없음** — OpenFedLLM(무대 동일하나 지표=
  GPT-judge=API불가; 로컬 클론 evaluation/ 확인), FlowerTune(arXiv:2506.02961; alpaca-gpt4+
  MMLU 0-shot이나 4/20클라·R15·DoRA r32·AdamW + 생성-추출 채점 시사[1B 1.03%] → 참조점 수준),
  fidelity축은 LLM-scale 선행 0(=novelty). 옵션 2(미구현, 승인 대기): bridge arm(vanilla-AdamW
  레시피로 optimizer 갭 수치화), FlowerTune-채점 모드(생성-추출).

### 구현 (이 세션; 미커밋)

- `fl/llm_server.py`: `run_llm_fedavg_logs`에 `select_fn/weights_fn` seam 통과(기본 None=비트동일).
- `data/llm.py`: `build_alpaca_iid`에 n_test(같은분포 test carve, domain="alpaca") 추가.
- `experiments/track_d.py` 전면 재작성(위 설계). (a)=러너 내 `make_a_utility`(val-loss-only;
  `llm_subset_utility`의 ROUGE 생성 비용 회피, `exact_shapley` 재사용).
- **훅 위생 발견**: SFTTrainer가 retrain/라운드마다 train모드+임베딩 훅을 남겨 functorch HVP가
  막힘(기존 LLM 3-musts의 런타임 재발 형태) → `_guard`(온라인 점수기 매 호출 eval()+훅 클리어)
  + (a) 직후 동일 처리. anchor5 스모크에서 실제 재현→수정 확인.
- 스모크 green: gpt2 std20(11-method+arm5+persist; Flirds vs (b) +1.000) · gpt2 anchor5((a) 32
  retrains 경로; Banzhaf 등장·flirds_sel 자동제외) · 1B anchor5(백그라운드; 결과는 plan §3.11
  기록 참조) · CNN guard green. ComFedSV는 저R 스모크에서 행렬완성 퇴화(NaN; task7d의 R≥30
  특성 — real R=200 정상 예정).

### supersede/보존

- supersede: 1막 fork 3건 전부(noisy 레짐/filter-q/MMLU만 유지), D-옵1·D-옵2(코드는 존치;
  FedHDS 무대는 "비IID·clean 분리축" 실험 후보로 파킹).
- 보존: q-sweep follow-up(오염 실험용), corruptor/`build_domain_iid` 코드, r16 유지, MMLU
  full-test 0-shot.
