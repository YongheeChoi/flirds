# T3 — P1w(크기-가중 게이트) LLM R4 구현·실행 (= L7)

> 정책 확정(Yonghee 07-23): P1의 크기-가중 일반화 — **w_i ∝ n_i·max(cum_i, 0), 참여자 합 1 재정규화**
> (합-유지; max=1 정규화 변형은 유효 스텝 수축=LR-감쇠 교란으로 **폐기**). R4는 등n이라 w ∝ max(cum,0).
> 이 정책은 Track H **P2(sign_weight)와 동일** — CNN dir1 실측 이미 존재(재사용 = T4의 W-A).
> P1 = 이 정책의 "양수 균등" 특수형 → "부호만 vs 부호+크기" ablation 구도.

## 1. 구현 (codes/, 로컬 세션)

- **T1 online arm** `flirds_p1w`: `fl/intervene.py`의 `make_gatedweight_weights_fn`(V2w, α=1) 재사용 —
  `experiments/track_g.py`의 R4(gsm50k5) ARMS에 등록만(과거 "V2w는 CNN 승격 후 LLM ARMS 추가" 보류를 지금 해제).
  게이트 하이퍼 = P1과 동일(burn_in 10·tau 0·min_obs 2·probation 5; 셀별 튜닝 금지).
- **T2 arm** `t2_w_flirds`: 기존 P5-soft `t2_pw_*` 블록(관찰자 최종값 → 고정가중 재학습)을 **가중함수 파라미터화**로
  일반화 — Φ(t) 대신 `max(cum,0)` 정규화 가중. kept 의미론: 가중 0 = 배제(φ≤0). kept=전원+가중 균등이면
  vanilla-동일 처리(기존 equals_vanilla 관례), 가중이 불균등하면 재학습 수행. dedupe: 가중 벡터 동일 시에만 공유.
- **테스트**(`tests/` 신규 4~6개): ① 가중 방향(양수 φ 클수록 큰 w)·합=1 ② frzero 클라 w=0(=P1 kept와 동일 집합)
  ③ clean 전원-양수 시 w=정규화 cum(vanilla와 다른 궤적 허용) ④ T2 가중 전달 e2e(tiny) ⑤ 기존 P1·P5s 경로 비트동일 회귀.
- **스모크**: `PERSIST=0 ROUNDS=3 VAL=20 MAX_STEPS=2 REGIME=gsm50k5 THREAT=noisy SEED=0` + P1w arm — 오류 0·가중 로그 확인.

## 2. 사전등록 H-14 (runs/track_h/README.md 예측표에 추가 후 **실행 전 커밋**)

> H-14 (R4 P1w: 크기-가중의 추가 가치; flirds-only) | frzero: kept=P1과 동일(exact-0→w=0) → EM ≈ P1(oracle-동값 예상).
> noisy **T2는 P1 ≥ P1w 예측**(CNN P2-retrain 열세·P5s-T2 −1.2pt 전례), **T1은 P1w ≥ P1 가능**(CNN P2-online 전례).
> clean: T1 오발화 비용 유사~악화, T2는 가중 차등만(±노이즈 바닥) 예측. MISS 그대로 보고.
> 수록 규칙(사전 고정): 전 범위(W-A·W-B·L7) 승 → 본문 / 동률 → "부호가 가치의 대부분" ablation 1문장 / 열세 → 미수록(P1만).

## 3. 실행 (B200; QUEUE 순서 = L1 → L2 → **L7** → L4(승인 시))

- 셀: {clean, noisy nr0.7, frzero} × seeds {0,1,2} × {T1 `flirds_p1w`, T2 `t2_w_flirds`} = 18런.
  관찰자·통제는 **L1 산출 재사용**(같은 seed 무대; 신규 런은 P1w arm뿐).
- 비용 ~80 GPU-h(런당 ~4.5). 완료 후 `runs/track_h/make_analysis.py` 확장(P1w 행) → H-14 자동 대조 → rundir+분석 커밋.
- W-D(비-flirds 점수원 P1w)는 **후순위 별도 승인**: L7 결과 보고 후 Yonghee 게이트(R4 T2 4소스 ≈ +27 GPU-h~).

## 4. 완료 조건

- [ ] 코드+테스트 green+스모크 로그, H-14 커밋(실행 전), QUEUE/REMAINING 상태 갱신
- [ ] 18런 rundir(fix-후 git_sha) + 분석 CSV + overview §3.2.4 이웃 절에 결과 기입
- [ ] 수록 규칙 판정 1줄(승/동률/미수록)을 00-INDEX §1에 기록
