# Track G Phase-B 착수 상태 — 다음 세션 인수 (2026-07-19)

> 이 세션(E-세션 마감, 컨테이너 종료 ~07-19 13:30)에서 Track G **스모크만 시도**했고, assert 실패로
> 확정 진단·그리드는 **다음 컨테이너의 다른 세션으로 이관**(Yonghee 결정 2026-07-19). 스펙=`README.md`(유지).

## 0. 실행 환경 (이 컨테이너 기준 — 다음 컨테이너서 재확인)
- ⚠️ `README.md`/스크립트의 기본 `PY=/home/korea_bupj/miniconda3/envs/flirds/bin/python`은 **이 컨테이너에 없음**.
  대체: `PY=/NHNHOME/WORKSPACE/26msit001_A/flirds_batch/venv/bin/python`(torch 2.12+cu130, transformers 5.9, 정상).
- HF 오프라인: `HF_HOME=/NHNHOME/WORKSPACE/26msit001_A/flirds_batch/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HOME=…/flirds_batch/home`.
- GPU 0–3 (B200 4장; 원래 스크립트 5-GPU 가정 아님, CNN 그리드는 1-GPU 순차 설계라 샤딩 필요).

## 1. 스모크 결과 — **assert 1건 실패 (게이트 핵심기능은 정확)**
실행: `PY=<venv> HF_HOME=… bash runs/track_g/run_smoke.sh 0`

- **LLM 실행 자체는 완주**("TRACK G DONE" 출력). **CNN mini는 미도달**(`set -e`가 assert에서 중단).
- **실패 지점**: 인라인 assert `AssertionError: round 2: FR participated=False, probation=True`.
- **phi_rounds.parquet**(FR=client 1, burn_in=2, probation_every=3) 실측:

  | round | participated | raw | cum | 해석 |
  |---|---|---|---|---|
  | 0,1 | True | -0.0 | 0.0 | burn-in 관찰 |
  | 2,3,4 | **False** | NaN | 0.0 | 제외 (cum=0 ≤ τ=0) |
  | 5 | True | -0.0 | 0.0 | **첫 probation 복귀** |

- **판정**: 게이트의 **frzero 제외는 정확**(2·3·4 제외, 복귀는 5에서). 즉 탐지·실험 본목적은 무손상.
  문제는 **probation 위상(round 2 vs 5)의 불일치** 딱 하나.

## 2. 모순 — 다음 세션이 확정할 것
- `codes/flirds/fl/intervene.py:285`:
  `if probation_every and excluded and (r - burn_in) % probation_every == 0:`
  → r=2에서 `(2-2)%3==0` → **probation이 발화해 FR을 코호트에 복귀시켜야** 함.
- 그러나 실측은 r=2에서 **participated=False·raw=NaN**(코호트에 아예 없음) → **첫 probation이 round 5에서야** 일어남.
- 즉 **게이트 코드(285행)·스모크 assert 양쪽이 "round 2 probation"을 기대하는데 런타임은 안 함**. 둘 중:
  - (a) **assert off-by-one**: 설계 의도가 "첫 제외 라운드(2)는 probation 아님, 복귀는 burn+prob=5부터" 라면 assert의 `on_probation=(round-burn)%prob==0`이 round 2를 잘못 포함. → assert를 `(round-burn)>0 and …`로 수정.
  - (b) **게이트/로깅 버그**: 285행대로면 r=2 복귀해야 하는데 안 됨. `track_g.py`의 per-round 로깅에서 `participated`가 "코호트 선택" vs "V1 스크린 후 weight>0" 중 무엇을 기록하는지, `select_fn`의 probation append가 실제 반영되는지 추적 필요.
- **진단 시작점**: `codes/experiments/track_g.py`의 phi_rounds 로깅부 + `intervene.py:270-293 select_fn` + `test_signgate.py`. 
  round 2에서 `_gate_select_fn`이 실제로 무엇을 반환하는지 print로 확인 → assert냐 게이트냐 확정 → 해당 쪽 수정 → 스모크 green.

## 3. Green 후 실행 계획 (README §실행; 마감이 이관 사유)
1. `bash runs/track_g/run_smoke.sh` green 확인.
2. **CNN 그리드** — `run_cnn_grid.sh`. {iid,dir1}×{clean,lf@{.15,.35,.70},gn,fr}×3seed×9arm = **36 c2셀 + 12 V3 c1셀**.
   1-GPU 순차 설계 → **4-GPU 샤딩 권장**(셀 독립). 완료 후 `python runs/track_g/make_analysis.py` → **V2w 승격 판정**(오염서 V2w≥V2 + clean parity |Δacc|<0.006, 둘 다 충족 시만 LLM에 flirds_gatew_v2 추가).
3. **LLM silo5 파일럿** seed0(`run_llm_pilot.sh`) → GPU-h 보고 → 3-seed + iid5{clean,frzero} + (선택)noisy nr0.75 1셀.
4. **std50k5-mixed** 파일럿 seed0 → 비용 보고 → **Yonghee 승인 게이트** → 3-seed.
5. rundir → `make_analysis.py` → overview 신규 절 → paper-ko §6.5.

## 4. 참고
- 스모크 rundir `runs/track_g/_smoke/`(disposable) + `_smoke_run.log` 남겨둠(진단 근거). 다음 세션이 지워도 됨.
- Stage 0 감사(부호·예측표)는 `audit/SIGN_AUDIT.md`+`make_analysis.py`에 임베드(README §예측표). 커밋 6623fdf·7055f98.
