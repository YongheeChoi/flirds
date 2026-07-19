# individual-utility(loss-heur) silo5 런타임 — C6 회계-교정 후 재측정 정본

- **수치**: seed 0/1/2 = **96.6 / 100.1 / 100.2 초** (평균 ~99.0초). `loss_heur_silo5_runtime.csv`.
- **배경**: `in_run_sv.py`의 singleton utility가 base U(P_r)를 클라마다 중복 평가하던 회계 버그(C6)를
  `in_run_singletons` base-캐시로 교정(라운드당 forward 2|P_r| → 1+|P_r|). φ는 비트동일(fidelity 무영향),
  runtime만 ~1.7× 과대였음(구 정본 ~170초 → ~99초).
- **측정**: 2026-07-18 E-세션, DGX B200 1장, 1B silo5 (N=5, R=10) frozen 궤적 위 valuation-only wall-clock.
- **원본 로그**(원격): `flirds_batch/logs/cells/acct_fix_seed{0,1,2}.log`
  (E-세션 커밋 5ed9b9e; 목록 문서 `REMAINING_after_e_session_2026-07-19.md` §0 3.1).
- 이 파일은 R1/R2류 "노트 전용 수치" 재발 방지를 위한 로컬 정본화다. N=5 full(anchor5) 교정치는
  `runs/track_d/rundirs_e4_fedloo/1B_anchor5_seed{0,1,2}/metrics.json`(runtime['loss-heur'] = 675/638/660,
  평균 657±15)이 rundir 정본으로 이미 존재한다.
