# 2026-06-19 — Pearson(값-수준 fidelity) 추가 + Track C/D 파생 emitter + merge_oracle_a 경로 수정

> 세션 유형: 메트릭 보강(코드) + 기록. Yonghee 질문에서 출발. 핵심 질문 위계 1차(fidelity) 보강.

## 발단 (Yonghee)
- "fidelity에서 oracle과 기여도 **값 자체**가 얼마나 비슷한지 — 순위만 보지 말고 값의 선형 일치를 보는
  Pearson 지표, 다른 연구들은 다 하는데 우리는 metric에 없지 않냐?"
- 결정: **B** — Pearson 추가 + GTG 거리 3종을 phase2 그리드에 백필.
- 스코프 단서: "기여도 비교 말고 다른 곳은 굳이 안 해도" → **φ-vs-oracle/truth fidelity 비교에만**.

## 스코프 (넣는 곳 / 안 넣는 곳)
- **넣음**(fidelity vs oracle/truth): `track_c1`((a)/(b)) · `track_d`((b)) · `phase2_matrix`(truth) ·
  `make_analysis`(phase2 25셀 백필) · Track C/D 파생 emitter.
- **안 넣음**: `track_c1 spearman_vs_rate`(φ vs 오염률 = 의미 검증) · `track_c3` 안정성(cross-seed 재현성) ·
  detection AUROC · `phase2_llm_a_oracle`(이미 끝난 검증 스모크).

## 구현
- `flirds/eval/metrics.py` `pearson(a,b)`: numpy corrcoef, 상수벡터→nan. affine-invariant —
  글로벌 스케일/오프셋을 용서(거리 metric이 벌하는)면서, 순위가 아닌 **선형 값 일치**를 보상.
  N=5 near-additive에서 순위가 +1로 포화될 때 그 아래 값 격차를 드러내는 게 핵심 용도.
- 러너 fidelity 블록(앞으로의 런이 자동 저장): `track_c1.py` `pearson_{a,b}`(loop가 (a)/(b) 모두 커버)+출력 /
  `track_d.py report_fidelity` `pearson`+출력 / `phase2_matrix.py report` `pearson`+GTG 거리 3종(cos_d/euc_d/max_diff)+출력.
- phase2 기존 25셀 = **재실행 없이 백필**: `make_analysis.py`가 phi.parquet에서 truth((b)oracle 또는
  Flirds proxy) 대비 pearson+거리 산출 → `master_metrics` 컬럼 + 카테고리별 `pearson_vs_*.csv` + pearson heatmap.
- **Track C/D = 파생 emitter**(rundir 불변 유지 = phase2 analysis/ 패턴; rundir mutate 안 함):
  - `slurm/scripts/merge_oracle_a.py` 확장: (a)+(b) × Spearman+Pearson, c1/c1_oracle rundir phi →
    `runs/track_c/fidelity.csv`.
  - `runs/track_d/make_fidelity.py` 신규: rundir phi.parquet → vs (b)oracle Spearman/Kendall/Pearson/거리 →
    `runs/track_d/fidelity.csv`. (a)oracle도 (b) 대비 채점 = dual-oracle 일치.
  - 두 CSV는 gitignore(파생·재생성). 러너 자체는 phase2/track_c/track_d 모두 앞으로의 런에 pearson 네이티브 저장.

## 부수 발견·수정 — merge_oracle_a 경로 버그 (reorg 회귀)
- `TRAJ="runs/track_c1"` / `ORACLE="runs/track_c1_oracle"` = **reorg 이전 경로**(현재 `runs/track_c/{c1,c1_oracle}`)
  → glob 0건, 깨져 있었음. + a_dir 이름도 `{base}_aonly`였으나 새 명명은 `{ds}_{scen}_aonly_seed{N}`(seed trailing)
  → `base.replace("_seed","_aonly_seed")`로 교정. **repo-루트 상대경로(__file__)**로 바꿔 standalone(어디서 실행해도 OK).

## 검증
- py_compile 7파일 OK. make_analysis 25셀 clean(34 CSV/31 charts). 두 emitter clean(c1 30셀 / track_d 10셀).
- 메트릭이 순위 포화 아래 값-구조를 드러냄(검증 증거; **수치는 analysis/·fidelity.csv에만**):
  - phase2 silo5 poison: Flirds1st Spearman 0.0 → **Pearson −0.95**(값 강한 음의 선형) = "Flirds 회피" 선명화.
  - Track C(N=10) feature_noise: FedIF rho_b 0.616 vs r_p_b 0.693 등 순위<값 일치 노출.
  - Track D anchor5(N=5): near-additive +1.0 포화 + (a)·(b) dual-oracle Pearson(0.86–0.98).

## 미해결/주의
- 미추적 3B rundir 4개(Track D 3B 본실행 결과)는 이 커밋에서 **제외**(내 작업 아님 → 별도 맥락 커밋 대상).
- Track C/D 러너는 앞으로의 런에 pearson 자동 저장; 기존 셀은 파생 emitter로 본다(rundir 불변).
