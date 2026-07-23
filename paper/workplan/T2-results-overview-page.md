# T2 — 논문-수록 실험 전용 결과 overview 페이지 + 시각화

> 목적: **논문에 넣기로 확정된 실험만**, **논문 §5 순서 그대로** 정리한 위키 페이지 신설(미완 실험은 ⬚ 빈 칸). Yonghee가 Obsidian에서 논문 진행 상황을 한눈에 보는 대시보드.
> 전제 정본 = `paper/workplan/00-INDEX.md`(구조·수록/제외 결정) + `T1-paper-section5.md`(표 스펙).

## 산출물

1. **페이지**: `research-wiki/survey/flirds-paper-results-overview.md`
   - frontmatter: `type: survey`, `sources: [flirds-experiment-results-overview]`, created/updated.
   - 구조 = 논문 미러: §5.1 세팅 요약표 → §5.2 fidelity(메인 쌍 ⬚ + retrain-(a) sub) → §5.3 개입 → §5.4 탐지 → §5.5 비용 → §5.6 ablation → 부록 B–E 대응 표.
   - 각 표: 확정 값은 기입(출처 = 기존 overview §번호 + rundir/analysis 경로), 미완은 ⬚ + "채울 소스" 주석(L1/L2/L7/L8/W-B/c2fid).
   - 값의 정본은 기존 `flirds-experiment-results-overview.md`(전량 카탈로그) — 이 페이지는 **파생·선별본**임을 헤더에 명시(이중 기입 시 정본 우선).
   - 갱신 규칙 절: 실험 착지 → 기존 overview 반영 → 이 페이지 ⬚ 채움 + figure 재생성.
2. **시각화**: `research-wiki/survey/flirds-paper-results-overview-figs/` 폴더
   - `make_figures.py`(커밋; matplotlib only, 입력=리포 내 CSV/rundir 경로, 출력=같은 폴더 PNG) — **python 있는 환경(GPU 서버/conda)에서 실행**. 로컬 Windows엔 python 없음.
   - 페이지에서 `![[flirds-paper-results-overview-figs/<name>.png]]`로 삽입.

## Figure 목록 (초기 세트; 데이터 없으면 스크립트가 해당 figure skip + ⬚ 표시 유지)

| # | 내용 | 입력 | 상태 |
|---|---|---|---|
| F1 | CNN C1 시나리오×방법 vs (a) Spearman heatmap | `runs/track_c/fidelity.csv` | 지금 가능 |
| F2 | anchor5 전 방법 vs (a) bar(±std) | `runs/track_d/rundirs/1B_anchor5_seed*/phi.parquet` 재계산 | 지금 가능 |
| F3 | 메인 쌍 fidelity heatmap(same-game × 무대) | c2fid `analysis/fidelity.csv` + L2 rundir | ⬚ |
| F4 | R4 개입 EM bar(T1/T2, vanilla·oracle·random 앵커선) | L1 `analysis/gsm50k5_*.csv` | ⬚ |
| F5 | CNN 경쟁 P1 acc bar(위협별, 8점수원) | `runs/track_h/analysis/cnn_competition.csv` (restack 확인 후) | 조건부 |
| F6 | 탐지 AUROC 표-히트(R4 φ+탐지기 / c2fid) | L2·c2fid metrics | ⬚ |
| F7 | 비용 log-scale bar(op-count 모델 vs 실측; N=10·device100 포함=부록 E) | `runs/measured_2026-07/op_counts.py` 산출 + rundir runtime | 지금 가능 |
| F8 | removal 곡선(silo5 val-loss·cifar10 acc, worst/best-first) | `runs/removal_dose/rundirs*` | 지금 가능 |
| F9 | 2차항 ablation: k-sweep(Flirds vs 1st) line | `runs/probe_signal/cnn_c1/` | 지금 가능 |

## 규약

- 이 페이지에는 **미수록 실험을 넣지 않는다**(제외 목록 = 00-INDEX §0) — 전량은 기존 overview가 담당.
- pre-fix R4 seed0 값 금지(⬚ 유지). CNN dir1 계열은 restack 드리프트 표 확인 전 "기존값(가라앉을 수 있음)" 라벨.
- figure 스타일: 값 라벨 표기, oracle=점선 상한, vanilla=실선 하한, 3-seed는 ±std 에러바.
- 완료 조건: 페이지 + F1·F2·F7·F8·F9 생성·삽입, ⬚ figure 자리 표시, `wiki/log.md`에 생성 기록 1줄(위키 관례).
