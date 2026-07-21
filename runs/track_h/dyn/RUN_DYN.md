# Dyn 실험 — 매 라운드 오염 재추첨 (Slurm 서버 실행 지시서)

> **이 문서 = Dyn 실험의 실행 정본** (Yonghee 2026-07-21 지시: "오염 클라이언트가
> 고정이 아니라 매 번 바뀌는 경우"; 동학 = **매 라운드 재추첨**, 무대 = **R1 frac 0.1**
> 확정 — 세션 Q&A). §5 pre-flight → §6 sbatch 제출 → §7 보고·커밋 순서. 예측(§4)은
> 사전 등록이며 **실행 후 수정 금지**(MISS 그대로 보고).

## 1. 배경 — 왜 roundwise인가

Track H의 모든 오염 무대는 정적(오염 클라가 라운드 0~R 내내 오염)이라, 클라별
누적 통계(P1 cum / P5 t)가 원리상 유효한 레짐이었다. 본 실험은 그 전제를 제거한
극한: **오염이 클라 속성이 아니라 라운드 속성**(매 라운드 새 40명 추첨)이 되면 전
클라가 확률적으로 동질화되어 **클라-수준 구분 신호가 구조적으로 소멸**한다.
질문: 신호가 없을 때 ① P5s(pweight)는 설계 의도대로 무해한가(do-no-harm; Φ
공통인자 약분), ② P1(strict sign)은 경계선-과금 결함이 준-무작위 제외로
발현되어 해를 끼치는가. (블록-교체(전향 감지) 동학은 이번 스코프 아님 — Yonghee
선택 "매 라운드 재추첨만".)

## 2. 설계 — R1과 동일, 오염 동학만 roundwise

| 항목 | 값 |
|---|---|
| 무대 | **R1 그대로**: cifar10, FedSVCNN(w1), dir1, N=100, **frac 0.1**, R=120, E=5, lr 0.01(SGD mom=0), batch 64, val 2000/test 8000(split seed 0) |
| 오염 동학 | **매 라운드 m=40명 재추첨**(`make_roundwise_mask`; (seed,salt,r) 결정론). 정적 fr/gn의 고정 40과 동수. ⚠ 정적 lf만 Bernoulli(0.4)였으나 dyn은 3위협 모두 **40 고정**으로 통일(명시적 편차) |
| 위협 | label_flip(**dose 0.70 고정**) / free_rider(zero) / grad_noise(σ=0.1). clean 셀 없음(오염 0 → 정적과 동일) |
| arm 5종 | vanilla / **oracle_excl(per-round: 그 라운드 오염 40 제외)** / **random_excl(per-round: 독립 랜덤 40 제외, salt=1)** / flirds_gate_v2(P1) / flirds_pweight(P5s) |
| 게이트 | burn_in=10, tau=0, min_obs=2, probation=5(P1) · Φ(t) 가중(P5s) — **R1/P5 공통값, 튜닝 금지** |
| seeds | 0, 1, 2 → 9셀 |

- P5h(cgate)는 스코프 아님(Yonghee 지정 arm = P1·P5s 2종). T2 없음(**러너가 가드로
  금지** — 정적 corrupt가 없어 최종-통계 kept가 정의 불가).
- **클라-수준 검출 AUROC는 이 무대에서 원리상 정의 불가**(전 클라 동질; corrupt
  마스크 all-zero → 러너가 자동 스킵). 판정은 **성능(절대 test acc)만**.

## 3. 구현 (2026-07-21 이 세션; 전부 `C2_DYN=1` 뒤에 격리 — 정적 경로 무변)

`fl/intervene.py`: `make_roundwise_mask(n,m,seed,salt)`(정본 1곳 — 러너·분석 공유) +
`make_delta_transform`이 callable 마스크 수용(정적 iterable 경로 비트동일).
`experiments/track_c2.py`: fr/gn=동적 dtf, lf=클라별 clean/flipped 이중로더 +
`_RoundClock`(select seam이 로컬학습 전에 라운드 스탬프), per-round
oracle/random_excl select, config/metrics `dyn` 블록, T2 가드.
테스트 `tests/test_dyn.py` 6종 + 기존 32 회귀 green.

## 4. 사전 등록 예측 (DP-1~4; MISS 그대로 보고)

| # | 예측 | 근거 |
|---|---|---|
| DP-1 | **P5s = 전 셀 vanilla ±band(.006)** | 전 클라 교환가능 → t 동질 → Φ 공통인자가 재정규화로 약분 (P5-soft do-no-harm 설계의 극한 검증) |
| DP-2 | **P1은 gn·lf에서 vanilla 하회**(공통 음수 드리프트 → cum<0 배제 = 준-무작위 제외·fallback churn ≈ random_excl 방향). **fr은 소폭 하락에 그침**(오염 라운드 raw가 exact-0이라 드리프트 ≈0 = clean-레짐 경계선 오배제만) | 부호-게이트의 경계선 과금이 신호-부재 무대에서 순손실화 |
| DP-3 | per-round **oracle_excl > vanilla 전 셀**(gn 최대), **random_excl ≤ vanilla** | 완벽 라운드-검출은 유효; 무작위 제외는 오염률 불변+데이터만 감소 |
| DP-4 | 게이트 행동: P1의 배제가 "지금 오염인 클라"를 우연 수준(≈40%)으로만 맞힘 | cum은 라운드-수준 정체를 원리상 추적 불가 |

## 5. Pre-flight (서버; 실패 시 중단·보고)

fmnist 스모크 3종(각 ~2-3분, GPU 1장): `C2_DYN=1 C2_MODE=smoke` × {label_flip(+
`C2_FLIP_RATE=0.70`), free_rider, grad_noise} × arm 5종 전부 — 확인: 5 arm 라인 +
`TRACK-C2 RUN OK`(+ corrupt=0 출력 = 동적 마스크 정상). 사전에 `tests/test_dyn.py`
6종 + 기존 3파일 32종 green (2026-07-21 로컬 확인 완료; 서버 pre-flight가 재확인).

## 6. 실행

`sbatch_dyn.sh` — 9 task(3위협×3seed), task당 arm 5종 순차(~30분), rundir 루트
`runs/track_h/rundirs_cnn_dyn/`, RUN_NAME `cifar10_dir1_dyn_<ttag>_seed<n>` (기존
rundir과 충돌 없음). 총 **~5 GPU-h**. 부분 실패는 해당 index만 재제출(멱등).

## 7. 종료 후

1. `PYTHONPATH=codes <PY> runs/track_h/dyn/make_analysis.py` — 절대 acc 표 +
   parity 판정 + 게이트 행동(동적 마스크 대비).
2. 보고: acc 표(3-seed mean±sd) + DP-1~4 HIT/MISS + GPU-h 실측.
3. 커밋: rundir + analysis + (수치는 overview §3.2.6 계열 블록에만 — 이 문서 수정 금지).

## 8. 금지

- 게이트·신뢰 하이퍼 변경/셀별 튜닝, T2 추가, poison 위협.
- 이 문서·예측(§4)의 사후 수정. 결과 수치 기입(overview에만).
