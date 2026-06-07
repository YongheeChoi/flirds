# 프로젝트: flirds

## Pipeline Status

```yaml
stage: implementation # idle | idea-discovery | implementation(/experiment-bridge) | training | review | paper
idea: "client-level FL Shapley via 1st+2nd Taylor of validation loss (Flirds)"  # 현재 idea 한 줄 요약
contract: "research-wiki/wiki/flirds-implementation-plan.md"  # operational plan (wiki 기반; 별도 research_contract 없음)
current_branch: "main"   # feature/flirds-phase-0 → main 병합 후 브랜치 삭제; main 직접 작업
baseline: "Valuation-baseline LLM 1B N=5 3-seed(2026-06-07): **ALL methods Spearman +1.000 vs (b)oracle** (Flirds·Flirds-1st·GTG·FedSV·Banzhaf·ShapleyFL·loss-heur). runtime Flirds-1st~35s/Flirds~107s/loss-heur~164s/GTG~537s/FedSV~532s/Banzhaf~531s/ShapleyFL~531s/Ripple~4515s/(b)oracle~531s; AUROC noisy0.75/FR1.0 (coarse@N=5; Ripple noisy0.50±0.20). free-rider φ exact-0: Flirds/oracle/Banzhaf/loss-heur (GTG/FedSV renorm≠0). N=5 near-additive→Flirds dominates frontier 5–15× cheaper. #7 selection works both lr. **Detection (task5 FLDetector, 06-07)**: model-free server-side L-BFGS detector = **cheapest ~24s** but **weakest** AUROC noisy0.50/FR0.75 (vs valuation 0.75/1.0) — clean math client tops suspicious score every seed = non-IID erosion (headline N=10/100). **Dual-oracle (task6, 06-07, N=5@1B fp32)**: (a)-retrain-val-loss = (b) in-run = estimator **Spearman +1.000** (method validated); (a)-ROUGE diverges (different game, corruption-fooled). [CNN Phase0: ComFedSV Spearman {1.0,0.96,0.85,0.84}]"         # 비교용 baseline 숫자
training_status: idle  # idle | running(위치/GPU 명시) | complete | failed
language: 한국어      # 스킬 출력 언어 — english | 한국어 (값은 영어 유지). 자세히: shared-references/output-language.md
code_dir: codes      # 로컬 코드 디렉토리; OpenFedLLM 은 codes/external/ 참조 클론(reference-guided self-build, gitignored). CNN track 은 자체 시뮬레이터
active_tasks: []     # 백그라운드 작업
next: "**Phase 1 DONE + Phase 2 tasks 1–5 DONE**(2026-06-07). **Tasks 2–4 완료+검증**(1B N=5 3-seed, RIPPLE=0, 전부 Spearman vs (b)oracle +1.000): **Data Banzhaf**(`baselines/banzhaf.py` = (b)-oracle coalition util을 균등 1/2^{n-1}로 재가중; `in_run_sv._coalition_utilities` 헬퍼 추출, in_run_shapley bit-identical; free-rider φ=정확0; ~531s) · **ShapleyFL**(`baselines/shapleyfl.py` surrogate-FSV = uniform submodel+per-round exact Shapley+min-max+EMA, 논문 정독 재현; DMC estimator→cross-device task7; ~531s) · **loss-heuristic**(singleton in-run util `in_run_utility([k])`; ~164s) · **Flirds-1st-only**(`second_order=False`; **~35s≈15× cheaper**). `phase1_baseline_compare.py` 9-method로 확장(print=method-order 리스트). N=5 near-additive→전부 동일 랭킹→**Flirds 프론티어 지배**(5–15× cheaper). 핵심통찰: Shapley linearity로 exact+우리utility는 (b)oracle과 degenerate-동일 → ShapleyFL의 uniform-util+min-max+EMA가 구별 만듦. **검출 regime split LOCKED(새 결정 — raw·distill 둘다 없던것, full raw 검색 확인)**: FLDetector→cross-silo, STD-DAGMM→cross-device(task7; N=5선 5벡터로 DAGMM 학습=degenerate). **task5 FLDetector DONE+검증**(`baselines/fldetector.py` = model-free server-side from-logs L-BFGS detector: Byrd-Nocedal compact HVP + Cauchy-MVT 예측잔차 ‖ĝ−g‖ ℓ1-norm; **model/loss_fn 불필요→AUROC 표만, Spearman 없음**; gᵢᵗ=raw delta, wᵗ−wᵗ⁻¹=n_c-weighted aggregate, w_r 미사용; ≥1 secant pair부터 score(R<10 대응), Gap+2-means skip. 합성 AUROC=1.0/CNN bit-identical/LLM smoke PORT OK/compare 9→10-method. 1B N=5 3-seed: **최저비용 ~24s**(Flirds-1st 35s보다도 쌈)·**최약 검출** noisy0.50/FR0.75 vs valuation 0.75/1.0 = clean math 매seed 최고 score=non-IID erosion→헤드라인 N=10/100). **task6 (a)-retrain LLM N=5@1B VALIDATED**(06-07; `oracle/exact_sv_llm.py`+`experiments/phase2_llm_a_oracle.py`): (a)-**val-loss**=(b) in-run=estimator **Spearman +1.000**(fp32, 두 lr). 핵심교훈(Yonghee): (a)는 method 검증 위해 **val-loss(같은 게임)** 써야 — ROUGE는 다른 게임(미분불가→estimator-ROUGE 불가능); 약했던 건 **bf16 정밀도**(val-loss 차이~0.005-0.02 < bf16 prec 0.009 = (b) fp32 이유)지 신호크기 아님. ROUGE-divergence(기억용): (a)ROUGE vs (b) +0.4(1B)/−0.9(3B)=answer_swap 도메인-포맷에 속음, val-loss는 안 속음. cost ladder(a N=5): 47min(1B bf16)/126min(1B fp32)/90min(3B bf16); **N=10=retrain 64×+eval 32×→2-5일/1-GPU→실제 실험서 멀티-GPU 샤딩**. 3B (a)-valloss fp32 백그라운드 confirm 중. **NEXT (다음 세션): task7 cross-device N=100,K=10** — 분해+순서 결정됨: **7a** Dirichlet(α)-of-5도메인 N=100 로더(`fl.partition.dirichlet_partition` 도메인-id를 label로 재사용; α-sweep {0,0.01,0.1,0.5,5.0}; `data/llm.build`는 N∈{5,10}만 → 신규 `build_crossdevice`) → **7b** N=100 Flirds smoke(K=10=`sample_frac=0.1`, `_fedavg_core` 이미 지원·estimator partial-correct → 검증만; **Flirds@N=100=헤드라인**, oracle은 MC라 비쌈) → **7c** (b)-MC oracle(2^100 불가→permutation MC; 작은 M 500-1k으로 비용·수렴 먼저 가늠→5k-10k) → **7d** ComFedSV LLM 포팅(`comfedsv.py` CNN→loss_fn, GTG/FedSV 패턴) → **7e** STD-DAGMM(update벡터에 DAGMM+GMM, free-rider) → **7f** 실험(Flirds vs (b)-MC vs ComFedSV/STD-DAGMM, α-sweep). **Fork 결정: 데이터=Dirichlet(추천 채택), Fed-WildChat은 별도 natural 벤치 나중.** · task8 3B/7B 스케일업(7B bf16train/fp32eval) · task9 corruptor 확장 → Phase 3 매트릭스(144런). ⏳ **3B (a)-valloss fp32 confirm 백그라운드 실행 중**(`/tmp/a_oracle_3b_fp32.log`; 세션 후 완료 — 1B서 이미 +1.0라 confirmation, 죽었으면 재실행 불필요/스킵 가능). 상세: plan 'Next concrete action(2026-06-07)' + §3.9 + `raw/conversations/flirds/2026-06-07-phase2-task5-fldetector-cross-silo`. ⚠ git: main이 origin보다 앞섬(미푸시)→Yonghee push. ⚠ Ripple eigsh flaky(이번 smoke서 CPU spinning 멈춤; 알려진 수렴 이슈)→compare는 RIPPLE=0(Ripple 값은 06-06것). 환경: python `/home/korea_bupj/miniconda3/envs/flirds/bin/python`, GPU 0-3만, `codes/`서 PYTHONPATH=."  # 다음 단계
last_updated: "2026-06-07"  # YYYY-MM-DD
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