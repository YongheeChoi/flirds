# 프로젝트: flirds

## Pipeline Status

```yaml
stage: implementation # idle | idea-discovery | implementation(/experiment-bridge) | training | review | paper
idea: "client-level FL Shapley via 1st+2nd Taylor of validation loss (Flirds)"  # 현재 idea 한 줄 요약
contract: "research-wiki/wiki/flirds-implementation-plan.md"  # operational plan (wiki 기반; 별도 research_contract 없음)
current_branch: "main"   # feature/flirds-phase-0 → main 병합 후 브랜치 삭제; main 직접 작업
baseline: "Valuation-baseline LLM 1B N=5 3-seed(2026-06-07): **ALL methods Spearman +1.000 vs (b)oracle** (Flirds·Flirds-1st·GTG·FedSV·Banzhaf·ShapleyFL·loss-heur). runtime Flirds-1st~35s/Flirds~107s/loss-heur~164s/GTG~537s/FedSV~532s/Banzhaf~531s/ShapleyFL~531s/Ripple~4515s/(b)oracle~531s; AUROC noisy0.75/FR1.0 (coarse@N=5; Ripple noisy0.50±0.20). free-rider φ exact-0: Flirds/oracle/Banzhaf/loss-heur (GTG/FedSV renorm≠0). N=5 near-additive→Flirds dominates frontier 5–15× cheaper. #7 selection works both lr. **Detection (task5 FLDetector, 06-07)**: model-free server-side L-BFGS detector = **cheapest ~24s** but **weakest** AUROC noisy0.50/FR0.75 (vs valuation 0.75/1.0) — clean math client tops suspicious score every seed = non-IID erosion (headline N=10/100). **Dual-oracle (task6, 06-07, N=5@1B fp32)**: (a)-retrain-val-loss = (b) in-run = estimator **Spearman +1.000** (method validated); (a)-ROUGE diverges (different game, corruption-fooled). [CNN Phase0: ComFedSV Spearman {1.0,0.96,0.85,0.84}] **Cross-device(task7,06-08,N=100 α=0.5,1B)**: Flirds vs exact-per-round (b) **Spearman +1.000**(near-additive holds at scale); oracle 771ms/fwd(fp32-B200, no-tensor-core)→~11h/4-GPU. **3B (a)-valloss vs (b)=+0.900**(estimator +1.000, AUROC 동일)."         # 비교용 baseline 숫자
training_status: idle  # idle | running(위치/GPU 명시) | complete | failed
language: 한국어      # 스킬 출력 언어 — english | 한국어 (값은 영어 유지). 자세히: shared-references/output-language.md
code_dir: codes      # 로컬 코드 디렉토리; OpenFedLLM 은 codes/external/ 참조 클론(reference-guided self-build, gitignored). CNN track 은 자체 시뮬레이터
active_tasks: []     # 백그라운드 작업
next: "**Real grid 실행 중**(06-10 시작: tier1 silo5 4-threat 3-seed 1B DONE, tier2 device100 α-sweep 진행; 상세=memory checkpoint-2026-06-10; ⚠ 초기 tier .log-only 주의). **Track C/D 추가 실험 설계 확정(06-12, 이 세션)**: 선행 13편 실험 조사 기반 — **C1** CNN fidelity&cost(MNIST+LeNet5/CIFAR-10+FedSVCNN, N=10 full, GTG 5-시나리오(graded label-flip ladder), 듀얼 oracle (a)2^10 retrain val-loss+(b)exact, Spearman/Kendall+GTG 거리 metric+wall-clock, 3-5 seeds, 9-method+Ripple(eigsh guard)) → **C2** CNN 일반성능=추가분 메인(N=100 C=0.1 T=100-150, {IID,Dir(1),2-shard}×{clean,label-flip(ρ,τ),FR,grad-noise}; 개입 3종: ①가중집계[**곱셈형 w∝n·s 메인**=Yonghee 고유규칙(비교군 무전례)+대체형(FedIF/ShapleyFL 관례)+additive λ0.5(Ripple 관례)]+②selection(S-FedAvg식)+③bottom-q%(FedSV식); baseline은 각자 논문 메커니즘; 평가 공통 AUROC/최종acc±seed/acc-vs-round/rounds-to-target) — C1→C2 순차 stage-gate → **C3** cross-seed stability(Banzhaf+Volatility 응답, 비용0) → **D** LLM 표준세팅 전부 API-free(D-메인 Alpaca-GPT4 20k IID N=5 answer_swap50%→AUROC+Spearman vs (b)+필터링후 MMLU(random-q% 대조)/D-옵1 FedDQC Table-1 미러 FiQA·AQUA(judge 컬럼 로컬 대체)/D-옵2 FedHDS Dolly category-Dir 200클라 Rouge-L; 모델 1B/3B=Llama-3.2, **7B=Llama-2-7b-hf**=task8 원결정=FL-LLM 문헌 표준 일치, HF 접근 OK). 결정(전부 Yonghee): mom=0 전 트랙(baseline 포함)·기존 모델·**label_flip corruptor 신규**(per-client rate로 ladder/(ρ,τ) 표현; label_shuffle은 LLM-정합용 존치)·SGD 유지(AdamW 문헌차 caveat). 조사 핵심: FedDQC 50% response-swap=answer_swap 동등(정당화 인용원), 우리 medical 셋=OpenFedLLM/FlowerTune medical 학습셋, LLM-scale FL valuation 직접 경쟁자 없음(인접: LM-arith DPO Shapley/TraceFL/CLAIR=인용용), GTG는 label-skew서만 오차>1e-2(공략점). **구현=다음 세션**(label_flip→C1 러너→개입 루프→C2→D; φ음수→min-max+EMA 후 가중, 곱셈형은 equal-n서 대체형과 동일=size-skew서만 차이, 7B 첫 smoke서 Llama-2 pad-token 확인). 이 세션 변경(미커밋→Yonghee push): phase2_matrix.py SCALE 7B=Llama-2 파싱+docstring, plan §3.11+raw 기록. 상세: plan §3.11 + raw [[2026-06-12-track-cd-additional-experiments-design]]. 환경: python /home/korea_bupj/miniconda3/envs/flirds/bin/python, GPU 0-3만, codes/서 PYTHONPATH=."  # 다음 단계
last_updated: "2026-06-12"  # YYYY-MM-DD
```

## 프로젝트 제약

- (미정 — 추후 채워주세요)

## 비목표 (Non-Goals)

- (미정)

## 컴퓨팅 예산

- Nvidia DGX B200 x4

<!-- KARIS:BEGIN -->
## KARIS Skill Scope
KARIS skills installed in this project: 55 entries.
Manifest: `.karis/installed-skills.txt` (lists every skill KARIS installed and its upstream target).
For KARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/NHNHOME/WORKSPACE/26msit001_A/edge_ai_lab/yonghee/karis`).
Update with: `bash /NHNHOME/WORKSPACE/26msit001_A/edge_ai_lab/yonghee/karis/tools/install_karis.sh`  (re-runnable; reconciles new/removed skills).
<!-- KARIS:END -->