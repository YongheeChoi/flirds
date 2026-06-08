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
next: "**Phase 1 DONE + Phase 2 tasks 1–6 + task 7a–7d DONE**(2026-06-08). **Tasks 2–4 완료+검증**(1B N=5 3-seed, RIPPLE=0, 전부 Spearman vs (b)oracle +1.000): **Data Banzhaf**(`baselines/banzhaf.py` = (b)-oracle coalition util을 균등 1/2^{n-1}로 재가중; `in_run_sv._coalition_utilities` 헬퍼 추출, in_run_shapley bit-identical; free-rider φ=정확0; ~531s) · **ShapleyFL**(`baselines/shapleyfl.py` surrogate-FSV = uniform submodel+per-round exact Shapley+min-max+EMA, 논문 정독 재현; DMC estimator→cross-device task7; ~531s) · **loss-heuristic**(singleton in-run util `in_run_utility([k])`; ~164s) · **Flirds-1st-only**(`second_order=False`; **~35s≈15× cheaper**). `phase1_baseline_compare.py` 9-method로 확장(print=method-order 리스트). N=5 near-additive→전부 동일 랭킹→**Flirds 프론티어 지배**(5–15× cheaper). 핵심통찰: Shapley linearity로 exact+우리utility는 (b)oracle과 degenerate-동일 → ShapleyFL의 uniform-util+min-max+EMA가 구별 만듦. **검출 baseline → threat-matched REDESIGN(06-08; 아래 NEXT)**: 옛 FLDetector↔noisy는 위협 불일치(FLDetector=crafted-update poisoning 검출기, answer_swap=정직-나쁜데이터≠그것). **task5 FLDetector DONE+검증**(`baselines/fldetector.py` = model-free server-side from-logs L-BFGS detector: Byrd-Nocedal compact HVP + Cauchy-MVT 예측잔차 ‖ĝ−g‖ ℓ1-norm; **model/loss_fn 불필요→AUROC 표만, Spearman 없음**; gᵢᵗ=raw delta, wᵗ−wᵗ⁻¹=n_c-weighted aggregate, w_r 미사용; ≥1 secant pair부터 score(R<10 대응), Gap+2-means skip. 합성 AUROC=1.0/CNN bit-identical/LLM smoke PORT OK/compare 9→10-method. 1B N=5 3-seed: **최저비용 ~24s**(Flirds-1st 35s보다도 쌈)·**최약 검출** noisy0.50/FR0.75 vs valuation 0.75/1.0 = clean math 매seed 최고 score=non-IID erosion→헤드라인 N=10/100). **task6 (a)-retrain LLM N=5@1B VALIDATED**(06-07; `oracle/exact_sv_llm.py`+`experiments/phase2_llm_a_oracle.py`): (a)-**val-loss**=(b) in-run=estimator **Spearman +1.000**(fp32, 두 lr). 핵심교훈(Yonghee): (a)는 method 검증 위해 **val-loss(같은 게임)** 써야 — ROUGE는 다른 게임(미분불가→estimator-ROUGE 불가능); 약했던 건 **bf16 정밀도**(val-loss 차이~0.005-0.02 < bf16 prec 0.009 = (b) fp32 이유)지 신호크기 아님. ROUGE-divergence(기억용): (a)ROUGE vs (b) +0.4(1B)/−0.9(3B)=answer_swap 도메인-포맷에 속음, val-loss는 안 속음. cost ladder(a N=5): 47min(1B bf16)/126min(1B fp32)/90min(3B bf16); **N=10=retrain 64×+eval 32×→2-5일/1-GPU→실제 실험서 멀티-GPU 샤딩**. 3B (a)-valloss fp32 백그라운드 confirm 중. **task7a-7d DONE(06-08, 1B)**: **7a** `fl.partition.client_dirichlet_partition`+`data.llm.build_crossdevice`(per-CLIENT Dir(α) 도메인-혼합=**Option B**; 기존 per-class `dirichlet_partition`은 5도메인→100클라서 degenerate[α=0→5클라만,크기 0–12k]; Option B=고정크기·전부 non-empty·α=0=domain-disjoint·purity==A) · **7b** N=100 Flirds 검증완료(라이브러리 무변경; `sample_frac=0.1`→K=10, `flirds_values(n_clients=100)` 명시; FR φ exact-0) · **7c (b)oracle=EXACT per-round 분해**(φ_i=Σ_{r:i∈P_r} 2^{|P_r|}-Shapley = 2^N oracle, Δφ≈3e-16 증명; **MC 아님** — Yonghee: MC는 (a)-RETRAIN용, in-run은 싸서 exact): N=100 α=0.5 **Flirds vs (b) Spearman +1.000**, oracle **771ms/fwd**(fp32-B200)→~11h/4-GPU→1-2 α만 · **7d** ComFedSV LLM 포팅(`comfedsv_from_logs(loss_fn,pkeys)`, uniform-subset, partial=True; ==exact uniform-Shapley +1.000; CNN bit-identical). **NEXT (task7e+ = 검출 baseline SUITE, threat-matched 재설계 §3.9)**: data-quality(`answer_swap`)→**FedDQC** / free-rider→**STD-DAGMM**(독립 AE+std)+**FLTrust**(any-N cosine-to-root, cosine≈Flirds-1st라 보조) / **poisoning→FLDetector·FLTrust**(신규 위협=**Xu2023 instruction-trigger + Bagdasaryan plain-scaled** backdoor; DBA 제외); **두 검출기 다 양 regime**. STD-DAGMM ①per-(client,round) pooling ②feature-hash proj→256(std는 full벡터) ③random@benign-std+zero(delta→task9). **순서: STD-DAGMM → FLTrust → poisoning-corruptor+FLDetector-repoint → FedDQC → 매트릭스**(3위협×2regime×{검출기+Flirds+valuation}×α-sweep). LLM-FL 검증 검출기 전무→baseline은 CV 포팅→Flirds LLM-scale 분리가 신규. backdoor-vs-Flirds framing은 **실험 후 판단(pre-position 금지)**. · task8 3B/7B(7B bf16train/fp32eval); task9(corruptor)는 7e에 흡수 → Phase 3. ✅ **3B (a)-valloss fp32 CONFIRMED(06-08)**: vs (b)=+0.900(clean-client 1-swap=retrain noise), estimator=+1.000, AUROC 동일(noisy0.75/FR1.0); (a)ROUGE=+0.100(속음). 1B 검증이 3B서 유지. 상세: plan 'Next concrete action(2026-06-08)' + §3.9(재설계) + `raw/conversations/flirds/2026-06-08-phase2-task7-crossdevice-detection-redesign`. ⚠ git: main이 origin보다 앞섬(미푸시)→Yonghee push. ⚠ Ripple eigsh flaky(이번 smoke서 CPU spinning 멈춤; 알려진 수렴 이슈)→compare는 RIPPLE=0(Ripple 값은 06-06것). 환경: python `/home/korea_bupj/miniconda3/envs/flirds/bin/python`, GPU 0-3만, `codes/`서 PYTHONPATH=."  # 다음 단계
last_updated: "2026-06-08"  # YYYY-MM-DD
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