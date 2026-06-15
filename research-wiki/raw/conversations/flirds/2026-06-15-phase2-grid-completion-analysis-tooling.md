---
type: conversation
date: 2026-06-15
topic: flirds
participants: [Yonghee, Claude]
tags: [phase2, step5, real-grid, grid-completion, analysis-tooling, results-aggregation, restart-recovery, run-logger, persistence]
---

# Phase 2 step5 real grid — 추적 → 완주(25/25) + 결과정리 툴링

운영/추적 중심 세션 (06-13 → 06-15). 06-10 시작한 real grid를 Claude 재시작을 여러 번 겪으며
추적·복구하고, **모든 개별 run을 분류 체계대로 모으는 재실행 가능 결과정리 코드**(`make_analysis.py`)를
새로 만들고, **그리드를 25/25 완주까지 끌고 가** 최종 φ 데이터를 커밋했다. 설계·방법론 변경은 없음 —
06-09 LOCKED 그리드를 그대로 실행 완료한 것이 핵심 산출.

## Yonghee 요청 (시간 순)

1. "나 클로드 껐다켜서 지금 실험 트래킹 잘 되고 있는지 확인 한 번 해줘." (재시작 → 추적 재무장)
2. "잘 돌아가고 있지? 지금까지 돌아간 결과 커밋 한 번 해줘." (헬스체크 + 그때까지 φ rundir 커밋)
3. "실험 결과들 모아서 정리하는 폴더를 따로 만들고 그 안에다가 결과를 정리하는 코드를 생성해줘.
   디렉토리는 우리가 실험들 분류한 체계에 맞게 구성하고 그 안에 모든 개별 run의 결과들을 하나로 다
   모아줘. 실험 결과는 csv 형태의 표와 차트 같은 시각자료로 구성해줘." (결과정리 폴더+재실행 코드,
   분류 체계 미러, 전 run 집계, CSV 표 + 차트)
4. "커밋해줘." (분석 툴링 커밋)
5. "업데이트 때문에 껐다 켜졌는데 추적 다시 잘 해줘." (업데이트 재시작 → 추적 재무장)
6. "지금 설계된 전체 실험에서 어느 정도까지 돌아갔는지 확인해줘." (전체 설계 대비 진행 지도)
7. "진행상황 알려줘." ×3 (진행 점검)
8. "기록 진행하고 커밋까지 해줘." (이 KARIS 기록 + 커밋)

## 1. 재시작 복구 플레이북 (durable ops 지식)

Claude가 세션 도중 여러 번 재시작됨(수동 1회 + 업데이트 1회). 매번 확인된 패턴:

- **살아남는 것**: detached 드라이버(`run_driver.sh`) + 실험 python 프로세스 + 백그라운드 regen 루프
  (전부 nohup/detached) → 재시작과 무관하게 계속 돌아감.
- **죽는 것**: 하니스 측 watcher(Monitor 툴, run_in_background echo 루프)의 **알림 채널** — 프로세스는
  남아도 새 세션으로 출력이 안 옴 → **매 재시작마다 재무장 필요**. (1차 재시작: 파일 쓰는 regen orphan은
  계속 살아 파일은 갱신됐고 모니터만 끊김. 2차=업데이트 재시작: watcher 전멸.)
- **복구 절차**: ① `pgrep -f 'run_driver[.]sh'`로 드라이버 생존 ② `grep -c 'done\[' tier2/_driver.log`로
  완료 카운트 ③ GPU 표 + rundir mtime + RESULTS.md mtime ④ 고아 regen 루프 PID kill 후 재무장.
- 이번 세션 중 **고아 regen 루프 1개 발견·정리**(중복 실행). 같은 `while pgrep run_driver; do make_report;
  make_analysis; sleep 180; done` 루프가 두 개 떠 있었음 → 하나 kill.

## 2. 결과정리 툴링 — `runs/phase2_matrix/make_analysis.py` (NEW, ~590줄, 커밋 3a5c2a1)

요청 #3의 산출물. **재실행 가능**: `analysis/`를 매번 비우고 rundir만으로 재구축(파생물이므로 gitignore).

- **분류 체계 = 5 카테고리** (master_queue 단계 분류 그대로):
  `01_silo5` / `02_device100_sweep` / `03_device100_poison` / `04_device100_anchor` / `05_scale_3b`.
  `classify(cfg)`: silo5+1B→01, silo5+3B→05, poison threat→03, oracle_b→04(anchor), else→02(sweep).
- **Spearman 기준점**: anchor=`(b)perround`(device)/`(b)oracle`(silo), off-anchor=`Flirds(proxy)` —
  실행 시 사용한 proxy-truth와 일치(phase2_matrix.py `report()`의 truth 선택 로직 미러).
- **점수 방향 통일**: 저장된 모든 벡터는 higher=more-suspicious(persist 시 FedIF/ShapleyFL/ComFedSV
  부호 정렬됨) → AUROC=corrupt-high 한 축으로 모든 차트 공유, per-method flip 불필요.
- **출력**(25셀 기준): 카테고리별 `csv/`(metrics_long, auroc_table, spearman_vs_*, runtime_table,
  phi_long) + `charts/`; `00_overview/`(master_metrics.csv, master_phi.csv, runs_inventory.csv,
  auroc_table.csv + heatmap_all + frontier). 차트: AUROC×threat, Spearman heatmap, runtime(log),
  φ heatmap(z-score, corrupt 빨강), vs-α 패널, φ separation(benign box+corrupt 빨강점), poison
  per-cell 패널(제목에 ASR), scale 1B-vs-3B 페어, method×cell AUROC 그리드, frontier(runtime-vs-Spearman).
- 모든 figure에 config provenance 각주(`cfg_note()`: scale/regime/α/N/R/steps/lr/val/seeds/git sha+dirty).
- `analysis/README.md`(한국어): 분류 체계 + 미완료 셀(master_queue 파싱) + 방향/커버리지/기준점 노트 +
  재생성 커맨드.
- 부수: matplotlib 3.11.0를 flirds env에 신규 설치(additive only, 기존 패키지 업그레이드 없음 — GPU 잡
  돌아가는 중이라 안전 확인 후). Read 툴 PNG >2000px 실패 → PIL thumbnail로 /tmp 복사 검수.

## 3. RunLogger 영속화 정책 (확정)

φ rundir 추적 정책을 git에 새김 (`.gitignore` phase2 블록):
- **rundirs/ = TRACKED** (셀당 config.yaml + meta.json[git_sha/dirty/env_hash/versions] + phi.parquet +
  metrics.json; 작고 immutable한 실제 결과 → 커밋하면 재분석에 재실행 영영 불필요).
- **tier*/(raw 로그) + RESULTS.md + analysis/ = IGNORED** (전부 rundir에서 재생성되는 파생물).
- 커밋 이력: `8d364cc`(20셀) → `b9113c4`(poison 2셀, 22/25) → `a755149`(anchor 3셀, **25/25 완주**).
  전부 author Yonghee Choi, push는 Yonghee(Claude는 creds 없음).

## 4. 그리드 완주 — 25/25, 실패 0 (06-15 ~03:37 `DRIVER DONE`)

설계(06-09 LOCKED) 그대로 cost-tiered stage-gate 실행 완료. 5 카테고리 25셀:

| 카테고리 | 셀 수 | 내용 |
|---|---|---|
| 01_silo5 (1B N=5) | 4 | 4 threat × 3 seed, 전 메서드 + (b) exact 2⁵ |
| 02_device100_sweep | 12 | α∈{0,0.01,0.1,5.0} × 3 threat, cheap 메서드, Flirds proxy-truth |
| 03_device100_poison | 2 | α∈{0.5,0.0}, D2b 설치 config(LR=2e-3 R=60 EPOCHS=5 frac0.8) |
| 04_device100_anchor | 3 | α=0.5 + (b)-perround + coalition, 3 threat(noisy/frrand/frzero) |
| 05_scale_3b | 4 | 3B silo5 seed0, (b) 포함, coalition off |

- **anchor 3셀이 가장 비싼 칸**(셀당 3seed × [FL ~38분 + 밸류에이션 ~20.4h: (b)-perround +
  ShapleyFL + GTG + FedSV 순차, 무로그] ≈ 63h) → 나머지 22셀 끝난 뒤 GPU 1·2·3에서 마지막까지
  단독 진행. 큐 주석대로 "expensive, last".
- frzero(마지막 셀)는 드라이버 `done[ok]` 라인이 큐 소진과 겹쳐 누락됐으나 rundir 4파일 + 3seed
  metrics + 로그 `MATRIX DONE`으로 정상 완주 검증. 그리드 전체에 fail/CHECK/error/traceback 라인 0.
- 최종 리포트: `make_report.py` = "25 cells: 25 done, 0 running, 0 failed", `make_analysis.py` =
  04_device100_anchor 3셀 반영(핵심인 `spearman_vs_bperround.csv` 포함).

> 결과 수치 자체는 RESULTS.md + analysis/에만 (Yonghee 지시 "결과를 채팅에 풀지 말고 파일로"). 이
> 기록도 진행상황·구조만 담고 AUROC/Spearman 표는 파일 참조로 둠.

## 5. 노드 공유 상태 (ops 노트)

06-14 중 다른 랩원 잡(`dlilab/evaluator_student.train`, `verl_sglang` env)이 GPU 4–7 + GPU 0으로
확장됨. **우리 anchor 3셀은 GPU 1·2·3 단독 점유(각 96.8GB/183GB) → 충돌 없이 완주.** GPU 4–7은
원래 우리가 안 건드리는 영역. 그 잡이 GPU 1–3로 번지면 OOM 위험이라 모니터링했으나 끝까지 깨끗.

## 다음

- **real grid 실행 = COMPLETE.** 남은 것: 결과 분석/서술(fidelity 1차 → 성능/수렴/탐지 2차, 핵심
  질문 위계 순), 발표자료, Track C/D 본실험(D는 구현+스모크 done, 본런 미실행).
- 7B(Tier3)·N=10 (a)/(b) 오라클은 설계상 의도적 deferred(큐 미포함).
- 후속 분석 시: rundirs/만 있으면 `make_analysis.py` 재실행으로 모든 표·차트 재생성 — 메서드 재실행 불필요.
