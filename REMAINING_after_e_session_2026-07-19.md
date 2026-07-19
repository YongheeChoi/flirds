# 남은 작업 — E-세션(2026-07-17~19) 이후 인수인계
# [갱신 2026-07-20 00:00, F-세션] Track G Phase-B 완주·β0.3 부분 재개 반영 + 전면 일시정지 상태

> E-세션 원문 위에 F-세션(2026-07-19 04:40~, 신규 컨테이너 GPU 5장) 결과를 덧기입.
> **현재 상태 = §6 (전 GPU 일시정지 예약 — Yonghee 신규 실험 코드 우선 지시, 07-19 밤).**
> 실행 원본 프롬프트는 git 히스토리 `950678c^`의 `PROMPT_paper_followup_experiments.md`(삭제됨).

## 0-F. F-세션(07-19) 완료 — Track G Phase-B 전체 + β0.3 앞부분

| 블록 | 결과 | rundir | 커밋 |
|---|---|---|---|
| 스모크 진단·수정 | assert 'excluded=FR단독' 가정 오류(게이트·로깅은 설계 정상; §4 구가설 (a)/(b) 둘 다 아님) → 함의형 수정, ALL GREEN | `runs/track_g/_smoke/` | 69cb6bf |
| CNN 그리드 | 48/48 무실패. **V2w 승격 = DO NOT PROMOTE**(clean parity 5/6 위반 Δacc −0.01~−0.03 + corrupt 2/6 V2w<V2) → LLM ARMS 불변 | `runs/track_g/rundirs_cnn{,_v3}/` | 5a1d2bb |
| LLM silo5·iid5 | silo5 4threat×3seed + iid5 2threat×3seed + nr0.75, 전 셀 arms+V3+per-round phi_rounds. **frzero flirds_gate_v2 회수 1.0000 정확**(=lossheur=oracleb 천장; P=R=1.0, fx=0)·noisy 게이트 침묵(P3 예측 적중)·clean 27행 max\|Δ\|=0.00056 | `runs/track_g/rundirs/` | aa1286d |
| std50k5 mixed s0 | 5/5 arm 완주. GPU-h: flirds 4.55/vanilla 4.39/oracle 3.34/random 3.14/**shapleyfl 9.94** = **셀당 25.4 GPU-h** | 〃 (shapleyfl은 본 커밋) | aa1286d+ |
| std50k5 s1·s2 (부분) | s1 shapleyfl·flirds, s2 shapleyfl — 07-20 새벽 ~03:00 완주분 (잔여 7셀은 §4-F 파킹) | 〃 | 완주 후 커밋 |
| β0.3 재개 (부분) | 22셀 큐 중 **1B_silo5 {noisy,frrand} done[ok]** + {frzero,poison} 새벽 완주분. 드라이버 PY/PP env-오버라이드 패치 | `runs/phase2_matrix/rundirs*/` (canonical 덮어씀; **일괄 커밋은 캠페인 재개 후**) | 8ef417e(패치만) |

파일럿 silo5 GPU-h 실측: clean 2.02/noisy 2.29/frrand 1.73/frzero 1.74 (4셀 7.77).

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
5. **[F-세션 추가] Track G 서술** — overview 신규 절(게이팅 실효성; §0-F 수치) + paper-ko §6.5.
   소스 = `runs/track_g/analysis/README.md`(Δ+recovery·r2t·게이트 P/R·§2.1 예측표 자동대조) — std50k5 7셀 완주 후 최종 재생성해 인용.

## 2. 남은 것 B — 무GPU 부수분석 (PROMPT §3; 선택)

- **3.1 정본화**: loss-heur runtime이 acct 로그에만 있음(`flirds_batch/logs/cells/acct_fix_seed{0,1,2}.log`).
  R1/R2 재발방지 위해 CSV/rundir로 영속 권장 (또는 overview 각주에 seed별 명기).
- **oracle noisy AUROC 0.604 vs 0.660 불일치** — `runs/phase2_matrix/analysis/00_overview/master_metrics.csv` 확정, overview §3.4.2 각주.
- **bootstrap CI(B=1000)** — 기존 rundir 재집계로 헤드라인 표 CI(§5).
- **momentum 열화 수치(CNN 0.73 vs 0.81) 정본 rundir 위치** 확인(§5 🟣VERIFY).

## 3. 남은 것 C — β0.3 잔여 + 장기 이월 [F-세션 갱신]

- **β0.3 잔여 18셀 (일시정지)**: device100 sweep+poison 14 + 3B silo5 4.
  F-세션이 22셀 큐(`runs/rerun_beta03/logs/resume36h.txt`, gitignored) 중 4셀 완료 후
  라인 5-22를 주석-보류함(Yonghee 신규실험 우선 지시). **복원**:
  `sed -i 's/^#phase2/phase2/' runs/rerun_beta03/logs/resume36h.txt` 후 드라이버 재기동
  (RESUME_AFTER_MIGRATION.md §재개 명령에 **PY=<flirds_batch venv>·PP=<이 리포>/codes env 필수** — 8ef417e 패치가 오버라이드 허용;
  큐 유실 시 RESUME 문서 31줄 목록에서 완료분 {1B_silo5_noisy,frrand,frzero,poison} 빼고 재생성).
  완료분 4셀 rundir(canonical 덮어씀)은 **아직 미커밋** — 재개·완료 후 일괄 커밋 + overview §3.4 갱신.
- **β0.3 deferred 9셀**(불변): 7B_std20×3(70-90h) + device100-a0.5 anchor×3(63h) + 7B_anchor5×3(35-45h).
- **E5 seed1·2**: 셀당 33h(=(b) 2¹⁰) → 마감 초과로 seed0만 함(Yonghee 결정 2026-07-18). N=10 3-seed 원하면 다음 컨테이너.
- **CLAUDE.md 장기 대기**: lr·steps intervention 2차검증, 1B·CNN β-불변 canon 확인, probe A축 seeds 1-2(신호크기 §3.5 잔여).

## 4. Track G — Phase-B 실행 완료 [F-세션 갱신; 구 이관 항목 소화됨]

- ~~스모크 진단·그리드·파일럿·std50k5 이관~~ → **전부 F-세션 완료**(§0-F; PHASE_B_STATUS 문서는 흡수 후 삭제됨=69cb6bf).
- **잔여 실험 = std50k5 mixed 7셀 (일시정지)**: s1 {vanilla,oracle_excl,random_excl} + s2 {flirds_gate_v2,vanilla,oracle_excl,random_excl}.
  단일-arm 셀 방식(rundir 이름 multi-arm과 동일). **복원** — codes/에서 각 줄을 순차 실행(GPU당 1줄):
  `REGIME=std50k5 THREAT=mixed SEED=<s> ARMS=<arm> V3=0 PYTHONPATH=. CUDA_VISIBLE_DEVICES=<g> <venv python> -u experiments/track_g.py`
  (env: HOME/HF_HOME=flirds_batch 오프라인 셋 — README §실행 참조. arm당 3.1~4.6 GPU-h 실측.)
- **잔여 마무리(무GPU)**: ①7셀 완주 후 `python runs/track_g/make_analysis.py` 최종 재생성 ②미커밋 rundir
  (07-20 새벽 완주 5셀 + 잔여 7셀) 커밋 ③overview 신규 절 + paper-ko §6.5 서술(§1에 편입).
- 판정 확정치: V2w 불승격(CNN-only 정직 보고), frzero 회수 1.0(=천장), noisy 게이트 침묵(z/V2w·flirds_w만 회수축),
  frrand seed-의존 부분회수 — README §예측표와 전 셀 대조는 analysis/README.md가 자동 수행.

## 6. 현재 일시정지 상태 (2026-07-20 00:00; Yonghee 신규 실험 코드 우선 지시)

- **마지막 실행 5셀** (완주 후 해당 GPU 자동 유휴 — 새 셀 투입 없음, 킬 없음):
  GPU0 std50k5 flirds s1(~02:20) / GPU1 shapleyfl s1(~02:45) / GPU2 shapleyfl s2(~03:00) /
  GPU3 β0.3 1B_silo5_poison(~01:00-03:00) / GPU4 β0.3 1B_silo5_frzero(~00:30).
  → **~03:00부터 GPU 5장 전부 신규 실험용 프리** (첫 GPU는 ~00:30).
- 보류 목록: Track G std50k5 7셀(§4 복원법) + β0.3 18셀(§3 복원법). 신규 실험 종료 후 이 순서로 재개 권장.
- 07-20 새벽 완주 5셀 rundir 커밋 = track_g 3셀(닫히는 대로) + β0.3 4셀(캠페인 재개 시 일괄).

## 5. E-세션 실행물 위치 (참고)
- 실행 스크립트·체인 로그·재개 노트: `/NHNHOME/WORKSPACE/26msit001_A/flirds_batch/scripts/e_session_0717/`
  (`RESUME_NOTE.md` = 실험 재개용; done-마커 `flirds_batch/state/done_e0717/`).
- 이 파일과 RESUME_NOTE의 차이: RESUME_NOTE=실험 재발사, 이 파일=실험 후 문서화·이월 작업.
