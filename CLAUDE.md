# 프로젝트: flirds

## Pipeline Status

```yaml
stage: idle          # idle(시작 전) | idea-discovery(/idea-discovery) | implementation(/experiment-bridge) | training | review(/auto-review-loop) | paper(/paper-writing)
idea: ""             # 현재 idea 한 줄 요약 (예: "이산 확산 LM의 분해된 attention gap")
contract: ""         # research_contract.md 경로 (예: idea-stage/docs/research_contract.md)
current_branch: ""   # 이 idea의 git 브랜치 (예: feature/factorized-gap)
baseline: ""         # 비교용 baseline 숫자 — 재현치 vs 논문치 함께 표기 (예: "WikiText-103 PPL=18.2 (논문 18.5)")
training_status: idle  # idle(학습 없음) | running(실행 중, 위치/tmux/GPU 명시) | complete | failed
language: 한국어      # 스킬 출력 언어 — english | 한국어 (값은 영어 유지). 자세히: shared-references/output-language.md
code_dir: codes      # 로컬 코드 디렉토리 — /experiment-bridge 가 새 코드 작성 위치, /run-experiment 가 rsync source. BASE_REPO 도 codes/base_repo/ 로 clone
active_tasks: []     # 백그라운드 작업 리스트 — 위치+tmux+검사 방법 포함 (예: ["training exp01 on b2 (GPU 0-3, tmux=exp01)"])
next: ""             # 구체적인 다음 단계 한 줄 (예: "학습 종료 후 테스트셋 eval 실행")
last_updated: ""     # YYYY-MM-DD HH:mm — 스킬이 출력 시 자동 갱신
```

## 프로젝트 제약

- (미정 — 추후 채워주세요)

## 비목표 (Non-Goals)

- (미정)

## 컴퓨팅 예산

- (미정 — 추후 채워주세요)

## Remote Server

`/run-experiment` 와 `/experiment-queue` 가 SSH 로 붙어 실험을 돌리는 데 사용합니다. 원격 GPU 서버를 쓰지 않으면 이 섹션을 삭제하고 `## Local Environment` 만 남기세요.

- gpu: remote               # 사전 등록된 SSH 서버 사용
- SSH: `ssh my-gpu-server`  # ~/.ssh/config 의 Host alias
- GPU: 4x A100 (80GB each)  # 참고용 (스킬이 직접 사용하진 않음)
- Conda: `eval "$(/opt/conda/bin/conda shell.bash hook)" && conda activate research`
- code_dir: codes           # 로컬 코드 디렉토리 (rsync source, 기본값: codes)
- Code dir: `/home/user/experiments/`  # 원격 sync 대상 경로
- code_sync: rsync          # 기본값. `git` 으로 두면 git push/pull 워크플로
- wandb: false              # `true` 로 두면 학습 스크립트에 W&B 로깅 자동 삽입
- wandb_project: my-project # wandb: true 일 때 필수 — W&B 프로젝트명
- wandb_entity: my-team     # 선택 — W&B 팀/유저 (없으면 기본 entity 사용)

> **W&B 셋업**: 원격 서버에서 `wandb login` 1회 (또는 `WANDB_API_KEY` 환경변수). 대시보드는 `https://wandb.ai/<entity>/<project>`.

## Local Environment

원격 서버 대신 로컬 GPU 로 실험을 돌릴 때 사용합니다.

- gpu: local                 # 로컬 GPU 사용
- Mac MPS / Linux CUDA       # 환경 참고용
- Conda env: `ml` (Python 3.10 + PyTorch)
<!-- KARIS:BEGIN -->
## KARIS Skill Scope
KARIS skills installed in this project: 55 entries.
Manifest: `.karis/installed-skills.txt` (lists every skill KARIS installed and its upstream target).
For KARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/c/Users/YH_Yonsei/Docs/Projects/karis`).
Update with: `bash /c/Users/YH_Yonsei/Docs/Projects/karis/tools/install_karis.sh`  (re-runnable; reconciles new/removed skills).
<!-- KARIS:END -->
