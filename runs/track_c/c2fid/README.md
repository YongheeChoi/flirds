# Track C2-FID — C2 cross-device 무대 위 fidelity vs (b) oracle

> **⚠️ 게임 캐비엇(1행 필독)**: 이 leg 의 φ 는 10/100 참여(클라당 ~12/120 라운드) 부분게임
> 기여 합 — **`runs/track_c/c1`(N=10 전원참여)과 다른 게임이므로 두 표의 수치를 직접 비교
> 금지**. (b)가 이 게임의 exact 값이라 fidelity 정의는 자기완결 (계획 §4.5).

설계·결정 전문: 루트 `CNN_CAMPAIGN_PLAN_2026-07-22.md` §4. 러너:
`codes/experiments/track_c2_fid.py` (2026-07-23 구현; `tests/test_c2fid.py` 5개 green).

## 무대 (downstream 과 비트 동일)

- `track_c2.build()` verbatim — N=100, C=0.1, R=120, E=5, lr .01, split seed 0.
  같은 (dataset, partition, threat, seed)에서 **오염 집합·per-client rate·동결 vanilla
  궤적까지 Track G twin 과 동일 실현**(fedavg 재시드; on_round 는 관찰만 —
  `test_trajectory_join_logged_equals_unlogged`가 비트동일 강제).
- 정답 = **(b) in-run per-round oracle만**(`in_run_shapley_perround` 동치 증명 경로;
  (a) 2^100 재학습 불가). 셀당 Σ_r 2^{|P_r|} = 120×2^10 = 122,880 평가.
- 방법 9종 = C1 11종 − Banzhaf(2^N) − Ripple(자체궤적) + Fed-LOO.
  부호 전부 good→low (ComFedSV·ShapleyFL·FedIF 는 negate).
- 지표 = C1 세트(Spearman/Kendall/Pearson + 거리 3종 + wall-clock) + AUROC(오염 셀)
  + **spearman_vs_rate 양변형**(`spearman_vs_rate`=전클라·C1 호환 /
  `spearman_vs_rate_corrupt`=corrupt 40클라만; lf 셀에서 산출, 고정 dose 는 corrupt-only 가
  상수라 NaN — strmain 셀이 실질 ruler).

## 그리드 144셀 (= downstream twin 전수)

{cifar10 × [iid, dir1, shard, qskew], fmnist × [iid, dir1]} × {clean, free-rider, frrand,
grad-noise, lf@0.15, lf@0.35, lf@0.70, lf strmain} × seed {0,1,2}. 범위 144 확정
(교차검증 세션 07-23: 궤적 조인이 핵심 자산 — downstream 쌍 없는 fmnist×{shard,qskew}
+48 은 여유 시 **쌍으로 동반** 추가).

실행: **파일럿 1셀 먼저**(현 큐 종료 후) — `sbatch --array=11 sbatch_fid.sh`
(= cifar10_dir1_grad-noise_fid_seed0) → GPU-h 실측 보고 → GO 후 전체
`sbatch sbatch_fid.sh`. rundir 착지 = `rundirs/<ds>_<part>_<ttag>_fid_seed<N>/`.

## 사전등록 예측 F-1~F-4 (2026-07-23, 본런 실행 전 등록 — MISS 도 그대로 보고)

교차검증(실험계획) 세션 권고 4건을 본 세션이 코드 근거로 구체화. 대조는 본런 완료 후
`make_analysis`(작성 예정)가 자동 수행.

- **F-1 (qskew·비등n → P5c 실증)**: 부분집합을 **1/|S| 균등 합성**하는 계열의 ρ(b) 하락이
  qskew(24× 크기격차)에서 iid 대비 뚜렷. 코드 사실: 균등 합성 = **ShapleyFL**(shapleyfl.py
  `1/len(subset)`)·**ComFedSV**(comfedsv.py 동일), n-비례 재구성 = GTG·FedSV(gtg.py
  `nc/tot`). 예측: ShapleyFL·ComFedSV 하락 / GTG·FedSV·Taylor 계열(Flirds·1st·FedIF)·
  게임직접 계열(loss-heur·Fed-LOO) 상대 유지. *(계획 세션 릴레이 표기는 "uniform-subset
  계열(G4·G6)" — G-번호 매핑을 교환 안 했으므로 여기선 코드 근거 명단으로 등록;
  FedSV 포함 여부가 릴레이와 다를 수 있어 대조 시 FedSV 를 별도 행으로 함께 보고.)*
- **F-2 (free-rider·frrand → exact-0 계열 vs renorm 유령값)**: free-rider(Δ=0) 셀 φ
  exact-0 = Flirds·Flirds1st·(b)·loss-heur·Fed-LOO·FedIF, vs GTG·FedSV 는 재구성/renorm
  잔차 ≠0 유령값(N=5 canon 재현) → AUROC 1.0 vs 저하. frrand(benign-매칭 노이즈) 셀은
  exact-0 아님 — Taylor·게임직접 계열은 2차항 ½ΔᵀHΔ≥0 로 일관된 양-suspicion(순위 안정),
  renorm 계열은 부호 불안정 → AUROC 격차(Flirds 계열 ≥ GTG/FedSV) 예측.
- **F-3 (참여 10/100 → LLM std50k5 붕괴 패턴 재현)**: LLM 5/50 참여축 실측(ComFedSV·
  ShapleyFL ρ 음수 붕괴, FedSV +0.91, GTG +0.98, Flirds/1st +1.00)의 서열이 CNN 10/100
  에서 재현: ComFedSV·ShapleyFL 최하위(음수 가능) < GTG·FedSV 중간 감쇠 <
  Flirds·Flirds1st(전 방법 중 최상).
- **F-4 (strmain dose 해상도)**: spearman_vs_rate 양변형에서 **Flirds ≈ (b) 자기천장 >
  Flirds1st** (2차항의 dose 해상도 기여; corrupt-only 변형에서 격차 더 뚜렷).

해석 주의: clean 셀은 신호부재 레짐(IID-clean cross-seed ρ≈0,
`flirds-signal-size-diagnosis.md`) — ρ(b) 해석 시 non-IID clean 과 분리. qskew fidelity
착지 시 **F-L2(LLM 비등n silo5)와 cross-track 쌍**(P5c 양 트랙 실증) — 교차검증 세션 합의.

## (b) 라운드 샤딩 (채택; 파일럿에서 oracle wall 과대 시 사용)

라운드 독립(가법)이라 손실 0. 오라클 샤드 = `C2FID_B_ROUNDS=lo:hi`(방법·phi.parquet
생략, rundir `_b<lo>-<hi>` 접미) + 방법 런 = `C2FID_ORACLE_B=0`. 병합 = 샤드들의
`phi_b_rounds.parquet` groupby(client).sum — **커버리지 assert 1개**(라운드 합집합 =
0..119 정확히 1회; `test_round_shard_merge_and_coverage`가 규약 고정). 비샤드 런은
러너 내부 assert(Σφ = 독립 계산 U(N), eff-gap<1e-3; 스모크 실측 0.0)로 동일 성질 보증.

## 산출물 (rundir 당)

- `phi.parquet`: client, rate(실현 flip rate; 캡처 비트중립 테스트로 강제), corrupt,
  n_rounds, phi_<방법9+오라클> — **C1 phi.parquet 열-호환 + 확장열**.
- `phi_b_rounds.parquet`: round, client, phi_b (per-round (b) 분해; 병합·라운드 분석용).
- `metrics.json`: stage="c2fid", 방법별 지표, corrupt/rates/n_rounds, final_acc +
  acc_curve(**조인 검증용** — twin vanilla arm 과 비트 동일해야 함), traj_time.
- `timing.json`: client-training / oracle-b / valuation 페이즈(§15.1;
  aggregate_runs.py 흡수 가능).

분석 도구(작성 예정): rundir-only 재생성, **산출 스키마 = c1 fidelity.csv 열-호환 +
`stage` 컬럼**(LLM fidelity 표와 병합·논문 표 생성기 재사용 — 교차검증 세션 합의).
