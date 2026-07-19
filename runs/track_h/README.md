# Track H — 점수원 경쟁(score-source competition): 어느 기여도 정의가 학습을 가장 잘 만드는가

> 스펙: 2026-07-19 Yonghee 결정 세션. 상태 = **설계(승인 대기; 코드·실행 전)**.
> 질문: 서로 다른 기여도 정의(방법)들이 **같은 개입 정책** 아래 경쟁할 때, 어느
> 기여도가 다운스트림 학습을 가장 잘 만드는가.
> 동기: fidelity(§3.1)는 "우리가 정의한 게임((b) oracle)"을 기준으로 한 자기-일치라
> 타 정의 방법의 심판이 될 수 없음 — **다운스트림 성능이 게임-무관 중립 심판**
> (Yonghee 2026-07-19: "다른 baseline들이 계산한 기여도로 똑같은 실험을 했을 때
> 우리가 더 잘한다를 보여줘야 우리 정의가 의미 있음을 보인다").

## 0. 공정성 원칙 (Yonghee 결정 — sign-게이트 포함)

- **sign-게이트(cum>0) 경쟁은 불공정이 아니다**: 0의 의미론(zero-semantics)은 기여도
  측정치 품질의 일부 — "0을 다른 의미로 정의한다는 것부터 기여도로 사용되기에
  부적합하다는 것"(Yonghee). GTG/FedSV의 renorm ≠ exact-0(Stage 0 감사 판정 2)이
  게이트 오발화로 이어진다면 그것은 비교 오염이 아니라 **그 방법의 실측 감점**.
- 따라서 정책군은 **rank-기반**(순위만 사용)과 **sign-기반**(절대 0점 사용) 둘 다
  포함 — 전자는 순위 품질을, 후자는 값·0점 품질을 심판한다.
- Track G의 "GTG/FedSV 게이트 제외" 근거(audit 권고2)는 *게이트 정책 자체의 실효성*
  질문에는 유효했으나, *점수원 경쟁* 질문에는 적용되지 않음 — 여기선 그 오발화가
  곧 데이터다.

## 1. 설계 축 (4축 직교)

### S — 점수원 (경쟁자)

Flirds · Flirds-1st · loss-heur · GTG · FedSV · ComFedSV · ShapleyFL(β=0.3) · FedIF
(+ (b)oracle 천장 = coalition 가능 무대만. **Banzhaf 제외 — Yonghee 2026-07-19**)

- 전부 **per-round in-run 산출 가능**(기존 phase2_matrix/C2 per-round 채점기 재사용).
- 각 점수원의 raw는 **contribution orientation**(도움=양수)으로 통일 주입(track_g
  부호 규약 D-3; 방향 고정 단위테스트 필수).
- CNN N=100 무대: (b) 제외(2^cohort 초과). ShapleyFL per-round exact는
  cohort=10 → 2^10 가능(C2 기존 관례).

### P — 정책 (전 점수원 공통; 셀별 튜닝 금지)

| 코드 | 정책 | 사용 정보 | 기계 (intervene.py 기존) |
|---|---|---|---|
| `P1 sign_plain` | cum>τ=0 참여, n-가중 (0 이하 배제·양수는 균등 참가) | 값·0점 | `make_signgate_select_fn` (V2: burn-in·min_obs·probation) |
| `P2 sign_weight` | cum>0 배제 + w∝n·max(cum,0)^α, α=1 (양수는 크기 가중 참가) | 값·0점·크기 | `make_gatedweight_weights_fn` (V2w) |
| `P3 soft_mult` | w∝n·s (EMA min-max; 게이트 없음) | 순위·상대크기 | `make_weights_fn("multiplicative")` |
| `P4 zgate` | cohort-상대 z<−c=1.5 제외 | 순위(상대)만 | `make_zgate_select_fn` |

- P1 vs P2 = Yonghee 지정 비교: "0 초과의 기여도를 **그냥 참가** vs **기여 정도
  가중 참가**".
- P3/P4 = rank-기반 대조(2026-07-19 원제안) — sign 축과 분리된 순위-품질 심판.

### T — 계산 시점 (Yonghee: "이것도 전부 비교")

| 코드 | 시점 | 기계 |
|---|---|---|
| `T1 online` | 매 라운드 누적 cum으로 온라인 결정 | 기존 track_g/c2 arm 루프 |
| `T2 retrain` | **관찰자 런**의 최종 누적으로 1회 결정 → init부터 재학습 | V3 기계(`v3_sign`/`v3_z` 일반화) |

- T2는 P1/P2/P4에 적용(P2-retrain = kept + 고정 cum-가중 재학습). P3-retrain(고정
  soft 가중 재학습)은 2차 티어.
- **관찰자 런(신규 핵심)**: vanilla 궤적 1회에 **전 점수원을 동시 부착**해 per-round
  raw를 한꺼번에 로깅(`phi_rounds.parquet`에 `method` 열 추가) — fidelity와 같은
  "한 궤적, 전 방법 채점" 구조라 T2의 점수 비용은 셀당 1런.
- **T2 dedupe**: 관찰자 런에서 각 점수원의 kept-set(P1)·가중 벡터(P2)를 먼저 산출,
  **동일 결정이면 재학습 1회를 공유**하고 전 점수원에 귀속(순위 일치 무대에서 비용
  붕괴 방지). T1은 결정이 궤적에 피드백되므로 dedupe 불가(전 arm 실행).

### R — 무대 (기여도 차이가 큰 환경 우선 — Yonghee)

| 코드 | 무대 | 선택 근거 (차이가 갈리는 이유) |
|---|---|---|
| `R1` | CNN C2 cifar10 **dir1** × {grad-noise, label-flip fr0.70, free-rider, clean 통제} | 순위 스프레드 최대(fidelity 0.98↔0.30) + 최저가 |
| `R2` | LLM std50k5 **mixed** (N=50, 5/50) | 부분참여 붕괴 무대 — ShapleyFL/ComFedSV 음수 vs Flirds +1.00 |
| `R3` | LLM silo5 **noisy nr1.0** | **zero-semantics 판별 셀**: 감사상 GTG(~0.76)/FedSV(~0.65)만 nr∈(0,1]서 0-교차 → P1 발화 예측; Flirds/loss-heur/(b) 침묵. **판정은 성능으로**: 발화가 옳으면 final val-loss가 oracle_excl 쪽으로(+0.0015~0.0020 갭 회수), 틀리면(clean 오배제) vanilla보다 악화. 주의: 성능 갭 자체가 작은 셀 — 효과 크기 한계를 사전 명시 |

- **poison 무대는 전면 제외** — Yonghee 2026-07-19 ("모든 실험에서 poison 제외").
- 공통 통제 arm: `vanilla`(관찰자) / `oracle_excl` / `random_excl` (+T2엔 `v3_random`).
- silo5 frzero-류는 **불포함**: 순위·0점 전 방법 일치(강한 방법 전부 exact-0)라
  경쟁 무정보 — track_g 기존 결과가 이미 커버.

### 1.5 중복 방지 — 기존 rundir 재사용 (재실행 금지; 2026-07-19 대조 완료)

Track H의 Flirds-점수원 arm과 통제 arm은 **track_g Phase B가 이미 동일 무대·동일
기계·동일 seed로 실행**했다 — 아래는 재실행하지 않고 rundir를 그대로 귀속:

| 무대 | 기존 rundir | 재사용 arm (Track H 표기) | 신규 실행만 |
|---|---|---|---|
| R1 (CNN dir1 4셀×3seed) | `track_g/rundirs_cnn` | Flirds×P1(=gate_v2)·P2(=gatew_v2)·P3(=mult)·P4(=zgate) + vanilla/oracle_excl/random_excl | 비-Flirds 점수원 7종 × P1–P4 × T1 + **T2 전체**(C2 무대 retrain은 최초; 관찰자 런 포함) |
| R3 (silo5 noisy nr1.0 ×3seed) | `track_g/rundirs` | Flirds-P1(gate_v2)·loss-heur-P1(lossheur_gate_v2)·(b)-P1(oracleb_gate_v2)·Flirds-P4(zgate)·P3(flirds_w) + 통제 + v3_* | **GTG·FedSV·ComFedSV·ShapleyFL의 P1 게이트 4종만** |
| R2 (std50k5 mixed seed0) | `track_g/rundirs` | vanilla·oracle_excl·random_excl·Flirds-P1(gate_v2) | Flirds-P2 + 나머지 점수원 × {P1,P2} |

- **원격 진행/대기 작업과의 교차** (`REMAINING_after_e_session_2026-07-19.md` §3–4 대조, 07-19):
  - ❌ 불교차: β0.3 deferred 9셀(7B_std20·7B_anchor5·device100-a0.5 anchor — 무대·모델
    다름) · E5 seeds1-2(N=10 (b) oracle).
  - ⚠️ **교차 1건 — R2/Tier 3**: track_g std50k5-mixed는 seed0 **4/5 arm만 커밋**
    (`aa1286d`; `shapleyfl_gate_v2` + seeds1–2 미완 = 원격 진행분일 수 있음).
    `shapleyfl_gate_v2`@std50k5 = **Track H ShapleyFL-P1@R2와 동일 실험**, seeds1–2의
    vanilla/oracle/random/flirds_gate_v2 = Tier 3 재사용분의 3-seed 확장과 동일.
    → **Tier 3 착수 전 필수**: 원격 done-마커(`flirds_batch/state/`)·`track_g/rundirs`
    신착 여부 확인 — 완료분은 재사용 귀속, 미완·진행 중이면 **대기(재실행 금지)**.
    Tier 3 신규 arm 수는 그 시점에 재산정.
  - 참고: REMAINING §4(Track G Phase B "다음 컨테이너 이관·스모크 실패")는 **stale** —
    이후 Phase B 완주·커밋됨(CNN `5a1d2bb`·LLM `aa1286d`).
- track_c C2의 shapleyfl/fedif/sfedavg arm은 *자기-논문-정책*(§3.2.2)이라 Track H의
  같은-정책 경쟁과 **다른 실험** — 중복 아님(참조 비교만).

## 2. 예측표 (사전 등록 — make_analysis 자동 대조)

| # | 셀 | 예측 | 근거 |
|---|---|---|---|
| H-1 | R1 grad-noise | dAcc 순서 ≈ fidelity 순서(Flirds≈loss-heur > GTG > FedSV > ShapleyFL/ComFedSV); ShapleyFL·ComFedSV 점수원은 random_excl 수준 | §3.1.2 스프레드 |
| H-2 | R1 clean (P1) | Flirds/loss-heur **무발화**(누적 전원 양수); ComFedSV **간헐 오배제**(감사: 간헐 1클라 음수); GTG/FedSV 소폭 오발화(renorm 음수) = zero-semantics 실측 감점 | audit P1·판정 2 |
| H-3 | R2 | Flirds-P1 → oracle_excl 근접; **ShapleyFL-P1 ≤ random_excl**(track_g §2-6 예측 승계)·ComFedSV 동반 붕괴 | §3.6.1 붕괴 |
| H-4 | R3 (P1) | GTG/FedSV 게이트 발화 — noisy 클라를 맞게 잡으면 소폭 이득이나 renorm-교차의 부산물이라 **clean 오배제 동반** 예측; Flirds/loss-heur 침묵=parity | audit 권고1·2 |
| H-5 | P2 vs P1 | 오염 무대서 P2 소폭 우위(V2w 전례 +0.34~0.36 vs V2 +0.32~0.36), clean서 P2 오발화 악화(V2w 불승격 전례) — 이 trade-off가 점수원 무관하게 재현되는지 | §3.2.4 |
| H-6 | T1 vs T2 | 스텝-함수 위협(FR-류)은 동률(전례 v3_sign=gate_v2=1.000); 점진 위협(noisy·label-flip)은 T1 우위(학습 중 조기 배제 이득) | §3.2.3 |
| H-7 | 종합 | **fidelity가 다운스트림을 예측한다** — 예측 실패(fidelity 높은데 경쟁 패배, 또는 역) 자체가 1급 결과 | 경쟁 실험의 존재 이유 |

## 3. 판정 지표 — **우열 기준은 학습 성능만** (탐지 아님)

> Track H의 질문은 "오염 클라 검출을 잘하나"가 아니라 **"그 기여도로 학습하면 실제
> 성능이 오르나"**(Yonghee 2026-07-19 확인). 탐지 AUROC류 지표는 Track H에 **없음**.

1. **1차(경쟁 심판 — 유일한 우열 기준)**: 다운스트림 학습 성능 —
   R1 final test acc의 dAcc(+recovery=(arm−vanilla)/(oracle_excl−vanilla)),
   R2/R3 final val-loss(+recovery), rounds-to-target(수렴 보조).
2. **2차(설명용 진단 — 우열 판정에 불사용)**: per-round 배제 P/R·clean-클라 오배제
   기록 — "왜 성능이 움직였나"의 원인 해석 전용(오배제의 성능 비용은 어차피 1차
   지표의 clean 셀 dAcc/Δval-loss에 반영됨).
3. **3차**: 예측표 H-1~7 자동 HIT/MISS.
4. **종합 랭킹(경쟁 스코어) — 성능 단위로만**: 점수원별 = 오염 셀 recovery 평균 +
   clean 셀 Δ성능(오발화의 성능 비용; 음수면 감점). 산식 세부(셀 가중)는 구현 시
   확정하되 **실행 전 이 README에 고정**(사후 튜닝 금지).

## 4. 비용 산정 + 티어별 승인 게이트

> 러닝타임 기준: C2 arm ~5–8분(track_g CNN 그리드 실측), silo5 arm ~2 GPU-h,
> std50k5 arm ~4.4 GPU-h(§3.2.3 비용). 각 티어 종료 시 GPU-h 실측 보고 후 다음 티어
> 진행(Yonghee 게이트).

| 티어 | 내용 (**§1.5 재사용 차감 후 신규만**) | 신규 arm (셀당) | 총 신규 run | 추정 |
|---|---|---|---|---|
| **1** | R1 CNN: T1×{P1..P4}×**S7**(비-Flirds) = 28 + T2×{P1,P2}×S8(dedupe 전 16; C2 무대 retrain 최초) + 관찰자 1 | ~45 | 4셀×3seed ≈ **~540** (재사용 ~84 차감) | **~50–80 GPU-h** (dedupe로 실효↓; 관찰자 런 후 재산정) |
| **2** | R3 silo5 noisy: T1×P1 × **신규 4종만**(GTG·FedSV·ComFedSV·ShapleyFL — Flirds·loss-heur·(b)는 §1.5 재사용) | 4 | 3seed = **12** | **~24 GPU-h** |
| **3** | R2 std50k5: seed0 파일럿 — Flirds-P2 + {Flirds-1st·loss-heur·GTG·FedSV·ComFedSV·ShapleyFL}×{P1,P2} (통제·Flirds-P1은 §1.5 재사용; **ShapleyFL-P1은 원격 track_g 진행분과 동일 — 착수 전 §1.5 교차 확인 필수, 완료 시 재사용**) | ~12–13 | **~12–13** | **~53–57 GPU-h** → 결과 보고 후 3-seed 승인 |
| **4**(선택) | P3-retrain 확장 · (b) 천장 확장(N=5 무대만) · R1 iid 파티션 확장 | – | – | 별도 승인 |

- 실행 순서 = Tier 1 → 분석·예측 대조 → 2 → 3. Tier 1 결과가 H-1/H-5를 기각하면
  상위 티어 설계 재검토.

## 5. 구현 (완료 2026-07-19) + 실행 인수인계

구현 완료·커밋됨 — `codes/flirds/fl/score_providers.py`(신규) + track_g/track_c2 확장
(additive-only; 레거시 arm 분기 무변경) + `tests/test_track_h.py`(7 green, 기존
test_signgate 15 green 회귀) + `make_analysis.py`(재사용 rundir 검증 완료).
**실행 절차·명령·보고 프로토콜 = `HANDOFF_GPU_SERVER.md`** (GPU 서버 세션용).

## 6. 금지·주의

- 셀별 게이트 하이퍼 튜닝 금지(ablation 셀로만 — track_g 관례).
- ShapleyFL surrogate β=0.3 통일(overview §4.2-9).
- 결과 수치는 rundir 영속 + overview §3.2에만 기입(문서 관례); 이 README는 스펙·예측
  정본.
- top-k arm 없음 유지(track_g §7) — top-k 상한은 oracle_excl 대역. 단 Yonghee 요청 시
  P5로 추가 가능(자리만 예약).
