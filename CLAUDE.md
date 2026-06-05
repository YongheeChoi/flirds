# 프로젝트: flirds

## Pipeline Status

```yaml
stage: implementation # idle | idea-discovery | implementation(/experiment-bridge) | training | review | paper
idea: "client-level FL Shapley via 1st+2nd Taylor of validation loss (Flirds)"  # 현재 idea 한 줄 요약
contract: "research-wiki/wiki/flirds-implementation-plan.md"  # operational plan (wiki 기반; 별도 research_contract 없음)
current_branch: "main"   # feature/flirds-phase-0 → main 병합 후 브랜치 삭제; main 직접 작업
baseline: ""         # 비교용 baseline 숫자 — Phase 0 재현 후 기입 (예: ComFedSV Spearman {1.0,0.96,0.85,0.84})
training_status: idle  # idle | running(위치/GPU 명시) | complete | failed
language: 한국어      # 스킬 출력 언어 — english | 한국어 (값은 영어 유지). 자세히: shared-references/output-language.md
code_dir: codes      # 로컬 코드 디렉토리; OpenFedLLM 은 codes/external/ 참조 클론(reference-guided self-build, gitignored). CNN track 은 자체 시뮬레이터
active_tasks: []     # 백그라운드 작업
next: "Phase 1 stage 3 데이터레이어 완료(2026-06-04, commit 7abf6fa/1a75a0b): 5도메인 **자유생성** 로더(data/llm.py — medical=medalpaca-flashcards / legal=ibunescu-legalQA / finance=FiQA / math=AQUA-RAT / general=Dolly) + validation 1000(§3.4) + **val 미니배칭**(make_llm_loss chunk_domains + estimator _chunked weighted-sum; eager-HVP가 val=1000에서 OOM → chunk별 exact; CNN bit-identical) + **per-domain 정규화**(token vs domain-macro, ablation 플래그). HVP 프로파일·est-vs-oracle 매트릭스 락(CNN{5,10} / LLM 1B N5+N10후순위 / 3B N5 / 7B (b)N5·(a)✗; 전부 3seed). ⚠ 자유생성 swap이 D3 medical=PubMedQA/legal=CaseHOLD를 **supersede** → plan §3.1/§3.4 + flirds.md reconcile 필요(동시진행 D3-distill 세션 e532258은 옛 framing; threads/dataset-format-uniformity 참고). **② LLM corruptor DONE+verified(2026-06-04 sess3)**: noisy=answer_swap(완성 셔플) + free-rider=free_rider(zero/random; Lin/STD-DAGMM taxonomy, delta/advanced=Phase2) → data/corruptors.py + data/llm.build(noisy=) + fl/llm_server(free_riders=,free_rider_mode=); estimator/oracle/CNN **bit-identical**; 실제 1B free-rider-zero φ=정확히0·est≈oracle 2.7e-7. prior-art: FedCorr data-side≈answer_swap, Lin update-side≈free_rider(1:1). **③↔#7 RESEQUENCED(Yonghee) → #7 first clean run을 SV baselines port보다 먼저(de-risk scale)**. **#7 downstream metric=FedHDS식 per-domain held-out ROUGE-L + math(AQUA) EM**(utility=val-loss와 분리; flirds/eval/metrics.py DONE+unit-tested). **#7 인프라 DONE + de-risk 완료(2026-06-04→05)**: orchestrator(phase1_clean_run.py)+run_logger+eval/{metrics,generate}+3-way data split 전부 빌드·검증. **(b) oracle 비용폭발(2ᴺ·R·val·seq)으로 9h 멈춤 → `ORACLE_B` env-toggle 기본 OFF**(est≈oracle은 R10/smoke서 검증, #7 헤드라인은 estimator만)+phase-print+`python -u`. **flat-training 발견**: plain-SGD@lr=2e-5 느림 + 대상이 **1B-Instruct라 headroom 제한**(val~2.43) + 5-disjoint 도메인 non-IID. **lr-sweep(4GPU)**: 학습∝lr, est≈oracle 1e-3까지 tight(6e-6)·3e-3서 8e-5(여전히 exact); free-rider=zero AUROC1.0, noisy0.75(answer_swap mild); selection 신호 real but small. **NOW RUNNING(4GPU, estimator-only)**: full **lr1e-3 s0/s1 + lr3e-3 s0/s1** → `runs/full_lr{1e-3,3e-3}`, ~6-10h. 결과읽기 `PYTHONPATH=. python experiments/read_runs.py`. **NEXT**: 결과 확인 → 남은 s2 each 런칭(`RUN_LR=.. RUN_SEED=2`) → lr선택+selection 판정(작으면 corruption 재고) → ③ SV baselines port. 매트릭스 전 durable opt: tokenize-once + bf16-train/fp32-eval(~2×). 7B bf16 train/fp32 eval 분리"  # 다음 단계
last_updated: "2026-06-05"  # YYYY-MM-DD
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