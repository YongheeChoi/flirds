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

## 확장 ② — skew-축 분해 + fmnist + frrand (사전등록 2026-07-22, 실행 전)

> 지시: Yonghee 2026-07-22. 기존 12셀/36런(cifar10 {iid,dir1})은 **read-only**,
> 세팅 verbatim(N=100·10/100·R=120·E=5·lr0.01·burn_in10·τ0·min_obs2·probation5·α1)로
> 조합만 추가. 이 절은 **실행 전에 등록**되며 완료 후 MISS 포함 그대로 대조한다.

### 무대 구성 (신규 30셀/90런)

| 축 | 값 |
|---|---|
| 파티션(2×2 완성) | iid(skew 없음) · **shard**(label만) · **qskew**(size만) · dir1(label+size) |
| 데이터셋 | cifar10(기존) + **fmnist**(신규; N=100 개입 무대의 C2 canon 짝) |
| 위협(7) | clean · free_rider(Δw=0) · **frrand**(Δw~U(−s,s), 신규) · grad_noise(Δw+N(0,0.1)) · label_flip@{0.15,0.35,0.70} |
| arm | 기존과 동일 9종(clean 셀은 excl 2종 제외한 7종) |
| 신규 셀 | cifar10×{shard,qskew} 42런 + fmnist×{iid,dir1} 42런 + cifar10×{iid,dir1}×frrand 백필 6런 |

**frrand**(2026-07-22 신설) = LLM leg에 이미 있던 `frrand`의 CNN 이식. Lin et al. 2019
free-rider 택소노미 "random" 모드를 `make_delta_transform`에 배선하고, 진폭은 **benign
per-entry std 정합**(s=√3·std(정직 Δw), LLM leg의 `_benign_std(warm)·√3·DOSE_MULT`와 동일
규칙을 warm-up 평균 대신 해당 클라의 would-be 업데이트에서 온라인 측정). 세 update-level
위협이 **신호/노이즈 사다리**를 이룬다: free_rider=신호0·노이즈0 / frrand=신호0·노이즈全 /
grad_noise=신호+노이즈. 크기로는 정직 업데이트와 구분 불가(norm 필터 무력).

### 파티션 축의 실측 특성 (cifar10 N=100, seed0) — 해석의 전제

| 파티션 | 크기 min/med/max | 크기 span | 클래스/클라 | 순수 축 |
|---|---|---|---|---|
| iid | 500/500/500 | 1.0× | 10.0 | — |
| shard | 500/500/500 | 1.0× | **1.95** | label |
| qskew | 40/510/960 | **24×** | 10.0 | size |
| dir1 | 187/488/1168 | 6.2× | 9.87 | label(비율)+size |

⚠️ **2×2는 가법 분해가 아니다**: shard의 label-skew는 dir1보다 훨씬 세고(1.95 vs 9.87
클래스/클라), qskew의 size-skew도 dir1보다 세다(24× vs 6.2×). 따라서 "dir1 = shard +
qskew"를 수치로 검산하는 것이 아니라 **어느 축이 어느 현상을 만드는지의 귀속(attribution)**
만 읽는다. 서술에서 가법성을 주장하지 않는다.

### recovery 분모 가드 (기존 그리드에서 확인된 함정 — 신규 규칙)

recovery=(arm−van)/(oracle_excl−van)은 **위협이 약하면 분모가 붕괴**한다. 기존 실측
분모: lf@0.15 iid 0.0033 / dir1 0.0064 (vs grad_noise 0.379) → 같은 셀의 seed-std가
3.14·1.58까지 튐. 따라서 **분모 |oracle_excl−vanilla| < 0.02인 셀은 recovery를 표에서
공란 처리하고 절대 Δacc만 보고**한다(고정 결정 "절대 accuracy 표"와 정합).

### 예측 (make_analysis가 자동 대조; MISS 그대로 보고)

| id | 예측 | 근거 | crisp check |
|---|---|---|---|
| **H-K1** | free_rider 회복은 **파티션-불변** — shard·qskew에서도 V2 recovery ≥ 0.6, 4파티션 spread < 0.35 | φ=exact 0 → strict cum>0는 데이터 분포와 무관한 구조적 발화(audit P2). 기존 실측 iid 0.808 / dir1 0.838 | `recovery(V2) ≥ 0.6` (shard·qskew) |
| **H-K2** | **frrand도 체계적으로 잡힌다** — V2 recovery ≥ 0.7, frzero와 같은 급 | 1차항 ⟨∇ℓ,Δw⟩는 랜덤 방향이라 평균0(=LLM 감사의 ±코인플립)이지만, **2차항 ½Δwᵀ**H**Δw는 H⪰0이므로 항상 손실증가=음의 기여**이고 d~10⁵에서 1차항 변동을 압도. grad_noise가 최고 회복(0.944/0.855)인 것이 같은 기전 | `recovery(V2,frrand) ≥ 0.7` |
| | ↳ **반증 시 해석**: recovery ≈ 0.5×frzero면 CNN 스케일에서도 2차항이 약하다는 뜻이며 **LLM leg 감사(frrand 부호=코인플립)와 일치**하는 결과로 보고 | — | — |
| **H-K3** | **clean 오발화는 shard에서 최대** — 오발화 제외 클라 수 shard > dir1 > qskew ≈ iid, 그리고 shard clean에서 V2 Δacc < −0.006(parity 밴드 위반), qskew clean은 밴드 유지 | 라벨 이질성 → 정직 클라의 φ 분산↑(2개 클래스만 개선, 글로벌 val엔 손해 가능) → cum≤0 정직 클라 발생. 기존 clean Δacc는 이미 경계값(iid −0.0060, dir1 −0.0074) | `shard: Δacc < −0.006` / `qskew: |Δacc| < 0.006` |
| **H-K4** | **qskew에서 seed 분산 최대** — V2 recovery의 seed-std가 iid의 1.5배 초과(free_rider·grad_noise) | 오염 클라 추첨은 크기와 무관 → 제외되는 데이터 mass가 seed마다 24× 범위에서 요동. iid는 항상 정확히 40% | `sd(qskew) > 1.5·sd(iid)` |
| **H-K5** | **lf@0.15는 4파티션 모두 분모 < 0.02**(위협이 약해 회복 여지 자체가 없음) → recovery 공란·Δacc만. 단 shard에서는 flip 해악이 iid보다 커서 분모가 iid(0.0033)의 2배 이상 | 저용량 flip은 오염 클라도 순기여 양수 → 제외가 손해. shard는 클라당 클래스가 2개라 flip이 해당 클래스 학습을 통째로 파괴 | `gap(lf0.15) < 0.02` (4/4) |
| **H-K6** | **fmnist는 효과 크기만 축소, 상대 유효성은 보존** — 같은 (part,threat)에서 분모 gap은 fmnist < cifar10, 그러나 recovery는 cifar10 대비 ±0.15 이내 | 쉬운 무대(fmnist vanilla 0.57–0.85)는 위협의 해악·회복 여지를 함께 줄이므로 비율은 보존 | `|rec(fmnist) − rec(cifar10)| ≤ 0.15` (분모 ≥ 0.02 셀만) |

### 대조표 계획

- **C2 소프트-arm 같은-셀 대조**(§3.2.2 30셀): 대응 존재 = `cifar10_shard` · `fmnist_{iid,dir1,shard}`
  의 {clean, free-rider, grad-noise} strmain 셀. **qskew·frrand는 C2 대응 없음**(표에 명시).
  ⚠️ label_flip은 C2가 strmain(=FedCorr rate~U(0.5,1), 평균 0.75)이고 Track G는 고정 dose라
  **같은 셀이 아니다** — fr0.70만 근사 대조로 쓰고 캐비엇을 단다.
- **스택 경계**(감사 M1): 기존 36런 = torch 2.12.0 / B200 / git 69cb6bf, 신규 = torch 2.11.0 /
  RTX3090. 2×2 표의 iid·dir1(기존)과 shard·qskew(신규)가 하드웨어+스택 경계를 가로지른다.
  셀 내부 비교(Δacc·recovery)는 각 셀이 자기 vanilla/oracle 앵커를 갖고 있어 무관하지만
  **절대 acc의 파티션 간 비교에는 캐비엇 필수**. cifar10 {iid,dir1} 12셀 동일-스택 재실행
  (+19.5 GPU-h)은 Yonghee 결정 사항.

## Stage 0 감사

`phi_sign_audit.py` → `audit/sign_table.csv` + `audit/SIGN_AUDIT.md` (2026-07-19 완료·커밋).
핵심: canonical clean 전원 양수 / frzero exact-0 / frrand 코인플립(예측 수정) /
noisy 0-교차 부재(sign-게이트 작동영역 없음 확정) / CNN dose 3점 {0.15,0.35,0.70} 선정.
