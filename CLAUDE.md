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
next: "Phase 1 진행(2026-06-03 sess2): estimator/oracle = partial-participation + per-layer 로깅(seam1) + loss_fn backend 추상화 완료(CNN 회귀 bit-identical), OpenFedLLM 정찰 완료 → 남음: LLM 2단계(backends/llm + FL루프 self-build + 5-domain data layer/seam2), 다른 세션에서. estimator/oracle 추가 변경 불필요"  # 다음 단계
last_updated: "2026-06-03"  # YYYY-MM-DD
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