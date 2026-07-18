# 남은 작업 — E-세션(2026-07-17~19) 이후 인수인계

> 이 세션(컨테이너 마감 대비 실험 실행 세션)이 **실험은 다 끝냈고**, 남은 것은 **문서화·부수분석·다음
> 컨테이너 이월분**이다. 실험 결과는 전부 rundir로 영속·커밋됨(아래 §0). 무맥락으로 이어갈 수 있게 정리.
> 실행 원본 프롬프트는 git 히스토리 `950678c^`의 `PROMPT_paper_followup_experiments.md`(삭제됨).

## 0. 이번 세션 완료 (전부 커밋됨 — 재실행 불필요)

| 실험 | 결과 | rundir | 커밋 |
|---|---|---|---|
| E4 Fed-LOO | std20·anchor5 3-seed, same-game 전원 (b) 대비 **Spearman +1.000**(Fed-LOO 포함); anchor5 Flirds 745s vs (b) 3735s | `runs/track_d/rundirs_e4_fedloo/` | 155324b |
| E5 (b) N=10 | 2¹⁰ exact, seed0. same-game +1.000; (b) **32.7h**(117649s) vs Flirds 733s = **160×** | `runs/track_d/rundirs_e5_n10/` | 3895396 |
| E7 frdelta | 3-seed. Flirds +1.000이나 **AUROC 0.25 = (b)oracle와 동일** → "φ=0 exact는 zero/random 한정" | `runs/phase2_matrix/rundirs_2026-07/1B_silo5_frdelta/` | 5ed9b9e |
| AdamW bridge | seed1·2 → 3-seed. (a)vs(b) = −0.10/−0.90/−0.60 = **−0.53±0.33**(전부 음수); Flirds vs (b) +0.77(SGD +1.000 대비 저하) | `runs/removal_dose/rundirs_trackd/1B_anchor5_adamw_seed{1,2}/` | faf0341 |
| probe seeds1-2 | std50k5 3-seed Flirds +1.000; lr격자 \|φ\|∝lr(cross-seed 신호 무); noise SE | `runs/probe_signal/{rundirs,noise_probe}/` | 454db39 |
| E3 cost | CNN mnist·cifar10 timing.json(Ripple phase 분리) | `runs/measured_2026-07/e3_cost_smoke/` | 5ed9b9e |
| 3.1 loss-heur | runtime **96.6/100.1/100.2s**(C6 fix 검증; 기존 170s=1.7× 버그) | logs only(§2에서 정본화 필요) | 5ed9b9e |
| 3.2 device100 | 학습시간 timing.json | `runs/measured_2026-07/timing_device100/` | 5ed9b9e |
| 인프라 | track_c1·track_d timing 배선(HANDOFF 3.3) + track_d `METHODS` 필터 | — | 171b5d5 |

E2(Taylor 잔차 1B)는 **이전 세션에 이미 완료**(`runs/measured_2026-07/taylor/`, b694f07) — 문서 랜딩만 남음.

## 1. 남은 것 A — 문서 갱신 (GPU 불필요; 최우선)

**파일-canon 원칙**: rundir → overview → paper-ko 순.

1. **overview 반영** — `research-wiki/survey/flirds-experiment-results-overview-2026-06-25.md`:
   - E4 Fed-LOO 행(std20·anchor5), E5 N=10 열, E7 frdelta(§탐지), AdamW §3.7.4 3-seed(−0.53±0.33),
     probe seeds1-2(§6.1 std50k5·§6.3 lr/noise), 3.1 loss-heur runtime, 3.2 device 학습시간.
2. **표1 Fed-LOO 재집계** — `python runs/track_d/make_fidelity.py` (신규 root `rundirs_e4_fedloo` 인자 확인 필요).
3. **tab:cost** — `paper/sections/results.tex`:
   - loss-heur silo runtime **170→~99s** 교체(3.1 실측 96.6/100.1/100.2), device overhead%(3.2), E3 CNN cost.
   - anchor loss-heur는 E4 anchor5 rundir(`rundirs_e4_fedloo/1B_anchor5_seed*`)의 runtime에서.
   - HANDOFF 3.4 tab:cost end-to-end/overhead% 2블록(`\pending` 해소) — device 학습시간 이제 有.
4. **paper-ko 마커 해소** — E2·E3·E4·E5·E7·E11(probe seeds) 🔴TODO/🟣VERIFY 닫기.
   §3.7.4는 AdamW 3-seed로 "−0.10 단일관측" → "−0.53±0.33" 갱신.

## 2. 남은 것 B — 무GPU 부수분석 (PROMPT §3; 선택)

- **3.1 정본화**: loss-heur runtime이 acct 로그에만 있음(`flirds_batch/logs/cells/acct_fix_seed{0,1,2}.log`).
  R1/R2 재발방지 위해 CSV/rundir로 영속 권장 (또는 overview 각주에 seed별 명기).
- **oracle noisy AUROC 0.604 vs 0.660 불일치** — `runs/phase2_matrix/analysis/00_overview/master_metrics.csv` 확정, overview §3.4.2 각주.
- **bootstrap CI(B=1000)** — 기존 rundir 재집계로 헤드라인 표 CI(§5).
- **momentum 열화 수치(CNN 0.73 vs 0.81) 정본 rundir 위치** 확인(§5 🟣VERIFY).

## 3. 남은 것 C — 다음 컨테이너 (이 컨테이너에서 물리적 불가)

- **β0.3 deferred 9셀**: 7B_std20×3(70-90h) + device100-a0.5 anchor×3(63h) + 7B_anchor5×3(35-45h).
  재개법 = `runs/rerun_beta03/RESUME_AFTER_MIGRATION.md` + `flirds_batch/scripts/run_deferred.sh`
  (⚠️ GPU 5→4장이므로 `GPUS="0 1 2 3"`). 완료 후 overview 7B 열·§3.4 phase2 ShapleyFL 행 갱신.
- **E5 seed1·2**: 셀당 33h(=(b) 2¹⁰) → 마감 초과로 seed0만 함(Yonghee 결정 2026-07-18). N=10 3-seed 원하면 다음 컨테이너.
- **CLAUDE.md 장기 대기**: lr·steps intervention 2차검증, 1B·CNN β-불변 canon 확인.

## 4. 병렬 진행(타 세션, 이 세션 소관 아님)

- **Track G**(φ 부호-게이팅 V1/V2/V3): 커밋 6623fdf·7055f98로 진행 중. `track_c1.py`에 `C1_V3` 등 추가됨
  (이 세션의 timing 배선과 공존). 별도 세션이 소유.

## 5. E-세션 실행물 위치 (참고)
- 실행 스크립트·체인 로그·재개 노트: `/NHNHOME/WORKSPACE/26msit001_A/flirds_batch/scripts/e_session_0717/`
  (`RESUME_NOTE.md` = 실험 재개용; done-마커 `flirds_batch/state/done_e0717/`).
- 이 파일과 RESUME_NOTE의 차이: RESUME_NOTE=실험 재발사, 이 파일=실험 후 문서화·이월 작업.
