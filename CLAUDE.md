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
next: "**[07-02] 신호크기 진단 DONE + rank/참여 probe 실행 중**(가설 판정: fp32 병목 아님·MMLU 학습 0[포맷만]·주 병목=IID-clean에 클라 간 진짜 신호 구조적 부재[(b)oracle 자기순위 xseed ρ≈0, 가산 갭≤0.9%]; 문서 research-wiki/wiki/flirds-signal-size-diagnosis.md + raw [[2026-07-02-signal-size-diagnosis-probe-plan]]. probe[Yonghee 승인: seed0 파일럿 먼저→full 11종 스위트]: LLM 1B anchor r{32,64}+std N=50/5 r{16,32,64}, `runs/probe_signal/run_pilot.sh` 가동 중[GPU3 즉시, β0.3 캠페인 종료 후 GPU0-2 자동]; CNN w{0.5,1,2,4}×참여 66+30셀 grids/probe_c{1,2}.txt = **yonsei 제출 대기**[runs/probe_signal/PROMPT_yonsei_cnn.md를 새 세션에]. **β0.3 재실행 = canonical 덮어쓰기 금지(Yonghee 07-02 지시)** → `runs/track_d/rundirs_beta03/` 분리: 3B_std20 3-seed 분리+원본 복원 완료, 진행 중 3B_anchor5 3-seed는 preserve_beta03.sh 가드가 종료 시 처리, v2 큐 #PAUSED 라인 전부에 RUNDIR_ROOT 리다이렉트 삽입[재개해도 안전]). **[06-19] Pearson(값-수준 fidelity) 추가 = 전 fidelity 비교 공통**(metrics.pearson; phase2 make_analysis가 기존 25셀 phi.parquet 백필[+GTG 거리 3종] + 러너 3종[track_c1·track_d·phase2_matrix] 네이티브; Track C/D 기존 셀은 rundir 불변+파생 emitter `merge_oracle_a`[(a)+(b)×Spearman+Pearson]·`make_fidelity`[track_d]; gitignore. 스코프=기여도-vs-oracle만[spearman_vs_rate·track_c3 안정성·detection AUROC 제외]. **부수: merge_oracle_a reorg 경로버그 수정**[runs/track_c1→runs/track_c/c1 + _aonly 명명]. raw [[2026-06-19-pearson-value-level-fidelity]]). **✅ Real grid COMPLETE 2026-06-15 — 25/25 셀, 실패 0**(06-10 시작→06-15 ~03:37 `DRIVER DONE`; 5 카테고리=silo5 4 / device100_sweep 12 / device100_poison 2 / anchor 3 / 3B 4; anchor가 최고가 칸[셀당 ~63h]이라 마지막에 완주; 전 셀 rundir[config+meta(git/env)+phi.parquet+metrics] 영속화+커밋 8d364cc→b9113c4→a755149[author Yonghee, **push 대기**]; **결과정리 툴 NEW `runs/phase2_matrix/make_analysis.py`**=rundir만으로 5-카테고리 CSV+차트 재생성[재실행 가능], 결과수치는 RESULTS.md+analysis/에만. **다음=결과 분석/서술**[핵심 질문 위계 순: fidelity 1차→성능/수렴/탐지 2차]+발표+Track C/D 본런. 7B·N=10 오라클은 설계상 deferred. 상세 plan §3.9 'REAL GRID COMPLETE'+raw [[2026-06-15-phase2-grid-completion-analysis-tooling]]). **Track D 재정의+재구현 DONE(06-13)**: Yonghee 핵심 질문 위계 명시(1차=fidelity, 2차=성능→수렴→탐지; 루트 CLAUDE.md·wiki/flirds.md에 새김+기록 sweep 교정) 후 **D = 오염축 전면 제거한 IID·clean LLM 표준무대 실험으로 재설계** — 무대=OpenFedLLM run_sft.sh verbatim(alpaca-gpt4 20k IID, N=20 2/round R=200 10steps×b16 seq512, 7B=Llama-2-7b-hf 동일; deviation caveat: SGD mom=0 lr1e-3 상수/r16/fp32), 레짐 std20+anchor5(N=5 full R=30; exact(b) 2⁵+**(a)-retrain oracle** val-loss·fp32=듀얼 GT 문헌공백+Banzhaf), 축①=11-method fidelity(Spearman·Kendall·GTG거리3종·wall-clock; Ripple 제외) ②=개입 arm 6종(base/vanilla/flirds_w곱셈β.5/flirds_sel[std20만]/shapleyfl_w β.5/fedif_w β.7)→MMLU full-test 0-shot+같은분포 Alpaca-test(1k) ROUGE-L(clean-IID 기대=do-no-harm parity) ③=val-loss 곡선+rounds-to-target. 구현: track_d.py 전면 재작성+llm_server seam(select/weights_fn)+build_alpaca_iid(n_test)+eval/mmlu.py; **훅 위생 `_guard`**(SFTTrainer train모드/임베딩훅↔functorch HVP 충돌 — 온라인 점수기·(a) 직후 필수). 스모크 green(gpt2 std20/anchor5+(a) 32-retrain, 1B; ComFedSV 저R NaN=기지특성). D-옵1/옵2 제외(코드 존치; FedHDS는 분리축 후보). **직접 비교: caveat-free 셋업 없음**(OpenFedLLM=GPT-judge라 지표 불가, FlowerTune 2506.02961=설정·채점 상이→참조점; fidelity축 LLM-scale 선행 0=novelty). **다음**: ①Yonghee가 루트 `TRACK_D_REVIEW_2026-06-13.md` 검토(읽고 삭제; 결정 3건 대기 — bridge arm[vanilla-AdamW 문헌레시피로 optimizer 갭 수치화]/FlowerTune-채점 모드/ShapleyFL β 0.5↔0.3) ②D real run(1-seed 파일럿→3-seed 2-레짐→3B/7B; 1B ≈40–60 GPU-h) ③**NEW: 오염축↔비IID축 분리 실험 설계**(LLM에 clean×non-IID 칸 빈칸; FedHDS 무대 후보) 별도 세션. **Track D 코드·문서 커밋됨**(track_d.py·llm_server.py·data/llm.py·eval/mmlu.py·wiki·raw 06-13 = 17db38b/3fb5201/6c58002; 이 06-15 그리드-완주 세션 커밋들과 함께 **전부 Yonghee push 대기** — main이 origin보다 앞섬). 상세: plan §3.11 구현 세션 ② + raw [[2026-06-13-track-d-redesign-iid-clean]]. 환경: python /home/korea_bupj/miniconda3/envs/flirds/bin/python, GPU 0-3만(0 비어있음), codes/서 PYTHONPATH=."  # 다음 단계
last_updated: "2026-07-02"  # YYYY-MM-DD
```

## 핵심 질문 위계 (Yonghee 2026-06-12 명시 — 모든 실험·문서·발표에 적용)

1. **1차(핵심): 우리 방법론이 FL에서 기여도를 얼마나 정확히 측정하는가** — (a)/(b) oracle 대비
   fidelity(Spearman·Kendall·거리 metric). 가장 기본이 되는 질문.
2. **2차(측정한 기여도의 실효성 검증; 이 순서대로)**: ① 일반 성능 향상 → ② 수렴 속도 →
   ③ 오염 클라 탐지. **탐지는 마지막** — 기여도와 탐지는 완전 직결이 아님(예: clean-val-loss를
   낮추는 공격자는 φ가 '기여 높음'으로 나오는 게 valuation의 정직한 답).
   표·서술·발표의 순서와 "headline" 표현도 이 위계를 따른다.

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