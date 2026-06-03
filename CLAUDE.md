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
next: "Phase 1 stage 2 완료(2026-06-04): LLM backend(backends/llm:make_llm_loss) + FL loop(fl/server _fedavg_core 추출 + fl/llm_server, TRL SFTTrainer 1.x+forced SGD+completion-only) self-build, LLM-FL 스모크 green(Llama-3.2-1B real 궤적, est≈oracle 1.7e-6). LLM 3 musts(eager-attn / named-key state / embedding-hook clear). validation §3.4 확정(도메인당 200/총 1000, vs 2¹⁰=1024 subset 분리). seam2(a) CNN corruptor registry 최소구현 done(data/corruptors.py:label_shuffle, bit-identical). 남음: 3번 5-domain data layer(validation stratified loader + LLM text corruptor; seam2 풀 registry는 corruptor 쓸 때) + LLM baselines port. 7B는 bf16 train/fp32 eval 분리"  # 다음 단계
last_updated: "2026-06-04"  # YYYY-MM-DD
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