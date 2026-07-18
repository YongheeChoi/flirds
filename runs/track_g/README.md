# Track G — 기여도 부호-게이팅(φ-gated participation) 실효성 실험

> 스펙: 2026-07-19 Track G 프롬프트(구현 완료본; 이 디렉토리가 정본).
> 질문: **계산된 기여도를 참여/집계 결정에 실제로 쓰면 성능·수렴 이득이 나는가**
> (핵심 질문 위계 2차-①②). poison·Banzhaf·top-k 없음(§7 금지 목록).

## 부호 규약 (D-3 — 이 실험의 최대 리스크)

- 저장 φ(`phi.parquet`)는 **suspicion orientation**(도움=음수) — 리포 전체 관례.
- 게이트가 읽는 raw/cum(`phi_rounds.parquet`)은 **contribution orientation**(도움=양수,
  = −φ). 게이트 규칙 = **strict `cum > τ`(τ=0) 포함** → frzero의 bit-exact 0.0은 제외.
- 방향 고정 단위테스트: `codes/tests/test_signgate.py` (합성 이차게임에서 도움/해로움/영벡터
  3종의 raw 부호를 해석적으로 검증).

## 정책 arm

| arm | 기계 (codes/flirds/fl/intervene.py Track G 블록) |
|---|---|
| `vanilla` | 개입 없는 **flirds raw 관찰자**(bit-identical n-가중) + per-round 로깅 |
| `oracle_excl` / `random_excl` | 진짜 오염 / 동수 무작위 고정 제외 (상한/통제; clean 셀 생략) |
| `flirds_gate_v1` | 라운드 집계-게이트: 전원 학습, 該라운드 raw≤τ 델타 집계 제외 |
| `flirds_gate_v2` | 참여-게이트: 누적≤τ 학습 제외 (burn-in·min_obs·probation) + V1 스크린 |
| `flirds_zgate_v2` | 코호트-상대 z-게이트(cum z<−c 제외; noisy 회수용 보조 정책) |
| `flirds_gatew_v2` | **V2w**: V2 선택 + w∝n·max(cum,0)^α (α=1 고정) — **CNN 선행, 승격 후에만 LLM** |
| `flirds_gatew_v1` | (CNN 전용 ablation) per-round raw 크기 가중 |
| `flirds_w` | 기존 min-max EMA mult β0.5 (소프트 대조 — 0점이 상대적) |
| `lossheur_gate_v2` | loss-heur singleton raw 게이트 (C6 캐시 경로) |
| `oracleb_gate_v2` | (silo5만) 라운드당 2^5 exact (b) sub-game Shapley 게이트 = 정책 천장 |
| `shapleyfl_gate_v2` | (std50k5만) ShapleyFL un-normalized raw 게이트 — fidelity 붕괴 무대 대조 |
| `v3_sign` / `v3_z` / `v3_random` | 사후: vanilla 완주 후 kept={cum>0}(+z, 동수 random) 1회 재학습(위협 유지 = deployment 의미) |

게이트 기본값(셀별 튜닝 금지 — ablation 셀로만): burn_in=3(silo5·iid5)/10(std50k5·CNN),
τ=0, min_obs=2, probation_every=5, decay=1.0, z c=1.5, V2w α=1.0.

## 예측표 (§2.1 + Stage 0 감사 수정 — make_analysis가 자동 대조)

| 셀 | 예측 | 근거 |
|---|---|---|
| clean (silo5·iid5) | parity + **오배제 0** (canonical 누적 전원 양수) | audit P1 |
| frzero | 이득(수렴 위주 + ~+0.007급) — exact-0 규칙 | audit P2 |
| frrand | **[감사 수정]** 누적부호 ±코인플립(silo5 3+/8−, iid5 3+/0−) → 제외는 seed-의존; per-round 스크린·min_obs가 관건 | audit P2/권고3 |
| noisy@canon (sign-게이트) | **parity(게이트 침묵)** — nr∈(0,1]에 0-교차 없음(Flirds ~3.4 extrapolated=도달불가) | audit P3/권고1 |
| noisy (z-게이트·V2w) | 회수 후보(z=상대 기준, V2w=연속 하향가중은 0-교차 불요) | §2-3 보완 |
| std50k5-corrupt(mixed) | Flirds-V2 → oracle_excl 근접(FR분), ShapleyFL-V2 ≤ random_excl | §2-6 |
| CNN label_flip dose {0.15,0.35,0.70} | 교차 span ~0.13–0.55를 좌/중/우로 관통 — hard-gate vs soft-weight 신규 비교 | audit P4/권고4 |
| clean — V2w | **1차 관문**: 유일하게 clean 개입(크기 경사) — CNN parity(|Δacc|<0.006) 먼저 판정 | §2-1 |

참고: GTG/FedSV 게이트는 noisy@canon에서 발화할 것(교차 ~0.76/~0.65) — 단 그것은 (b)-진실(+)
대비 값-수준 오차의 부산물(audit 권고2). in-run(b) 게임 0점 ≠ 재학습(a) 게임 0점 주의.

## V2w LLM 승격 기준 (spec §5-2; make_analysis가 자동 판정)

① 오염 threat에서 V2w ≥ V2(특히 noisy에서 V2w > V2 여부가 핵심 관찰)
② clean parity 유지(|Δacc| < 0.006). **둘 다 충족 시에만** LLM ARMS에 `flirds_gatew_v2` 추가.
미충족 시 CNN 결과만 보고(정직한 결론).

## 산출물 스키마

- rundir: `rundirs/<REGIME>_<THREAT>[_nr<r>]_<arm>_seed<s>/`
  (CNN: `rundirs_cnn/`=track_c2 셀, `rundirs_cnn_v3/`=track_c1 C1_V3 셀 — env로 루트 지정,
  기존 트랙 경로 무수정)
- `phi_rounds.parquet`: **프로젝트 최초 per-round φ 영속** — {round, client, participated,
  raw, weight, cum, n_obs, fallback} × 전 클라 × 전 라운드 (contribution orientation).
  clean 오발화율·burn-in 캘리브레이션·오프라인 정책 분석의 근거.
- `phi.parquet`: 최종 누적(suspicion orientation) / `metrics.json`: 성능+게이트 P/R /
  `timing.json`: §15.1 (fl+online-scoring 단계에 HVP 피크 포함).

## 실행 (Phase B — 서버 이전 후; 순서·우선순위는 Yonghee 결정)

1. `bash run_smoke.sh` — gpt2 silo5-mini frzero: burn-in 후 FR 제외 assert +
   phi_rounds 생성 확인 + CNN 미니 1셀. (GPU 수 분)
2. `bash run_cnn_grid.sh <gpu>` — CNN 그리드(§4.4; {iid,dir1}×{clean,lf@3점,gn,fr}×3seed
   ×게이트 arm + C1 V3) → **V2w 승격 판정**(`make_analysis.py`).
3. LLM silo5 파일럿 seed0(4 threat × 핵심 arms; `run_llm_pilot.sh <gpu>`) → GPU-h 실측
   보고 → 3-seed 확장 + iid5 {clean,frzero} + (선택) noisy nr0.75 1셀(감사 권고1).
4. std50k5-corrupt(mixed) 파일럿 seed0 → **비용 보고 후 Yonghee 승인 게이트** → 3-seed.
5. 결과 경로: rundir → `make_analysis.py` → overview 신규 절 → paper-ko §6.5.

로컬(무네트워크) 와이어링 스모크: `SMOKE_MODEL=tiny-gpt2 SYNTH_DATA=1 ...` (track_g.py 헤더).

## Stage 0 감사

`phi_sign_audit.py` → `audit/sign_table.csv` + `audit/SIGN_AUDIT.md` (2026-07-19 완료·커밋).
핵심: canonical clean 전원 양수 / frzero exact-0 / frrand 코인플립(예측 수정) /
noisy 0-교차 부재(sign-게이트 작동영역 없음 확정) / CNN dose 3점 {0.15,0.35,0.70} 선정.
