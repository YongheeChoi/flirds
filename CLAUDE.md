# 프로젝트: flirds

## Pipeline Status

```yaml
stage: implementation # idle | idea-discovery | implementation(/experiment-bridge) | training | review | paper
idea: "client-level FL Shapley via 1st+2nd Taylor of validation loss (Flirds)"  # 현재 idea 한 줄 요약
contract: "research-wiki/wiki/flirds-implementation-plan.md"  # operational plan (wiki 기반; 별도 research_contract 없음)
current_branch: "main"   # feature/flirds-phase-0 → main 병합 후 브랜치 삭제; main 직접 작업
baseline: "SV-baseline LLM 1B N=5 3-seed(2026-06-06): Flirds Spearman +1.000 vs (b)oracle; runtime Flirds~107s/GTG~537s/FedSV~532s/Ripple~4515s/(b)oracle~531s; AUROC noisy0.75/free-rider1.0 (Ripple noisy0.50±0.20). #7 selection works both lr. [CNN Phase0: ComFedSV Spearman {1.0,0.96,0.85,0.84}]"         # 비교용 baseline 숫자
training_status: idle  # idle | running(위치/GPU 명시) | complete | failed
language: 한국어      # 스킬 출력 언어 — english | 한국어 (값은 영어 유지). 자세히: shared-references/output-language.md
code_dir: codes      # 로컬 코드 디렉토리; OpenFedLLM 은 codes/external/ 참조 클론(reference-guided self-build, gitignored). CNN track 은 자체 시뮬레이터
active_tasks: []     # 백그라운드 작업
next: "**Phase 1 DONE + Phase 2 진행 중**(2026-06-06). **#7 first clean run 완료**(1B N=5, 두 lr 3-seed): Flirds client-selection 작동 — flirds_topk가 val_loss·ROUGE 양쪽서 random_k 이김 + 부패클라(noisy-medical/free-rider-legal) 정확 드롭(lr1e-3 2.3978/0.1485 vs random 2.4111/0.1462; lr3e-3 2.3926/0.1509 vs 2.4055/0.1449; `python experiments/read_runs.py runs/full_lr1e-3 runs/full_lr3e-3`). **Phase 2 task1 SV-baselines 포팅 완료+검증**(commit d5e06d2): GTG/FedSV backend-agnostic(공유 `_round_metrics`, CNN bit-identical) + Ripple LLM(`baselines/ripple_llm.py`, 자체 FedAvg 궤적). compare 1B N=5 3-seed: **Flirds가 exact (b)oracle ranking 재현(Spearman +1.000) + ~5× faster than GTG/FedSV, ~42× than Ripple**(Ripple noisy 검출 약함 0.50±0.20); free-rider φ: Flirds/oracle 정확0 vs GTG/FedSV within-subset renorm-dilution(≠0). ComFedSV defer(cross-device, task7). **Banzhaf baseline 추가 진행 중**(병렬 세션: `baselines/banzhaf.py` + compare/smoke + in_run_sv 수정 — 미커밋, 안 건드림). **NEXT(Phase 2 잔여)**: 검출 baseline 2종(FLDetector noisy / STD-DAGMM free-rider) · ShapleyFL · loss-heuristic · (a)-retrain LLM판(N10@1B/N5@3B; `oracle/exact_sv.py`는 CNN전용→LLM subset_utility 필요) · cross-device N=100,K=10 (b)-oracle MC + ComFedSV(새 로더; `data/llm.build`는 N∈{5,10}만) · **3B/7B 스케일업**(7B bf16train/fp32eval) → 그 후 Phase 3 매트릭스(144런). 상세: `MEMORY.md` + plan 'Next concrete action(2026-06-06)' + `raw/conversations/flirds/2026-06-06-sv-baseline-port-and-results`. ⚠ git: main이 origin보다 앞섬(미푸시)→Yonghee push. ⚠ Ripple 무겁고 느림: eager grad O(batch·seq²)→local batch 작게(4/512); nvidia-smi mem엔 재사용캐시 포함(~86GB=Flirds HVP 정상, OOM판정선~160GB). 환경: python `/home/korea_bupj/miniconda3/envs/flirds/bin/python`, GPU 0-3만, `codes/`서 PYTHONPATH=."  # 다음 단계
last_updated: "2026-06-06"  # YYYY-MM-DD
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