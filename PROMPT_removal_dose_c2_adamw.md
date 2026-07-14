# Prompt: C-1~C-5 완화 실험 4종 (removal-curve · dose-response · target-stability · AdamW-fidelity)

> 작성 배경: `research-wiki/survey/review/review-claude.md`의 리뷰어 공격 C-1~C-5에 대한
> 완화 실험을 **기존 실험 코드에 추가하는 형식**으로 구현한다. 오라클 (b)와의 fidelity 표는
> 건드리지 않는다 — 이 실험들은 그것과 별개의 상호보완적 비교 축이다.
> Yonghee 승인 결정(2026-07-14 대화)이 아래 "Locked decisions"에 전부 반영돼 있다.

---

## 0. 이 실험들이 무엇을 방어하고, 무엇은 방어하지 못하나 (구현 전 필독)

review-claude.md의 약점 코드에 대응:

- **C-1 (self-referential GT)**: removal-curve/dose-response는 "게임-무관 downstream 척도"
  갈래로 C-1을 **부분 방어**한다(평가를 논쟁 대상인 (b) 게임에서 실제 재학습 결과로 이동).
  단 review가 C-1의 결정타로 지목한 **own-game 대조(G-1)는 별개 실험이며 여기 미포함** —
  후속 프롬프트로 남긴다.
- **C-2 (매칭 대상 (b)의 seed-안정성 미검증)**: Experiment C가 정면 대응. **재실행 0**,
  기존 phi.parquet만 읽어 (b) xseed를 정본화(persist)한다.
- **C-3 (LLM downstream 실효성 부재)**: removal-curve가 오염 레짐에서 "신호 있는 곳에선
  이득"의 조건부 법칙을 실증 → **절반 해결**. clean-IID parity(설계상 null)와 인센티브·
  정산 실증(G-6)은 이 실험 밖 → 별도.
- **C-4 (baseline 공정성)**: removal-curve는 게임 정의와 무관한 공통 자(尺)라 "strawman"
  공격을 구조적으로 무력화 + Fed-LOO 수치 보강 → **상당 부분 해결**. β0.3 재실행(R11)과
  own-game 대조(G-1)는 별도.
- **C-5 (레짐이 실무 FL-LLM과 괴리; plain SGD 등)**: Experiment D(AdamW-fidelity)가
  external-validity 구멍의 최대치를 저비용으로 겨냥.

**핵심 재현성 규율**: review-claude.md가 R1/R2/R9에서 반복 지적한 "노트-only 수치 → 나중에
오귀속"을 새 실험에서 재현하지 말 것. **모든 신규 산출물은 기존 `RunLogger`/rundir 관례
(config + meta(git/env) + parquet/csv)로 반드시 영속화**한다.

---

## 1. Locked decisions (Yonghee 승인, 2026-07-14)

| 항목 | 결정 |
|---|---|
| 오염 축 범위 | 현재 정의된 4 threat 전부: `noisy` / `freerider_random` / `freerider_zero` / `poison` |
| 방법 범위 | fidelity 표의 **전체 방법** (Flirds/Flirds1st/GTG/FedSV/ComFedSV/ShapleyFL/FedIF/Banzhaf/loss-heur/Fed-LOO) |
| Runner | **둘 다**: `track_d.py`(anchor5) + `phase2_matrix.py`(silo5) |
| 착수 규모 | **파일럿 먼저**(1 threat × N=5 × seed0 × 축소 방법셋) → 승인 후 full sweep |
| AdamW(C-5) | **포함**, 단 별도 저우선순위(removal-curve 파일럿 이후 마지막). AdamW lr = "브리지 설정"(현 constant lr 유지 + 논문 5e-5 cosine은 caveat) |
| freerider_zero dose | dose-response에서 **제외**(Δw=0은 이미 극값). removal-curve엔 포함 |
| poison removal 범위 | **k=1 단일 지점**(φ 최하위 1명 제거 후 재학습 → 배포 ASR 하락 확인) |

---

## 2. Experiment A — Removal / selection curve

개념 원형: `research-wiki/wiki/sources/ghorbani-zou-data-shapley.md` ("value-based filtering:
drop low-value points → retrain → comparable or better performance").
각 방법의 φ로 클라 순위를 매기고, **worst-first**(전체→최하위 제거→…)와 **best-first**
(전체→최상위 제거→…) 두 방향 nested subset을 만들어 각 단계 utility를 측정. 두 방향 **둘 다**
필수(하나만 보면 진단력 없음).

### A1. `track_d.py` / anchor5 — "free" 재분석 (신규 재학습 0)

`exact_shapley()`가 이미 $2^5=32$개 전체 subset utility를 계산하지만 phi만 반환하고 버린다.
이 캐시를 재사용하면 재학습 없이 removal curve를 조회로 뽑을 수 있다.

1. `flirds/oracle/exact_sv.py:74` `exact_shapley(n_clients, utility_fn)` 확장 —
   **하위호환 유지**하며 내부 `u` dict(전체 subset→utility)를 옵션 반환
   (예: `return_u=True` → `(phi, u)`). 호출부는 `track_d.py:187`
   (`exact_shapley(n, util)`) 한 곳만 갱신.
2. `compute_fidelity()`가 반환하는 모든 방법의 φ 벡터(`track_d.py:174-223`)에 대해:
   φ 내림차순 정렬 → worst-first / best-first nested subset의 각 단계 utility를 `u`에서
   조회(재학습 없음). k=0(전체)은 기존 vanilla utility 재사용.
3. 결과를 rundir에 영속화(예: `removal_curve: {method: {direction: [(k, util), …]}}`를
   `metrics.json`에 추가하거나 별도 parquet). `_persist()`(`track_d.py:322`)에 얹는다.
4. 이 A1 경로는 **(a)-retrain 기준** removal curve = "게임과 무관하게 실제 재학습 성능이
   어떻게 변하나"의 가장 중립적 버전. C-1 방어 자료로 직접 사용.

### A2. `phase2_matrix.py` / silo5 — 실제 재학습 (메인 실험, GPU 비용 발생처)

1. `compute_methods()`(`phase2_matrix.py:290-348`)가 계산하는 각 방법 φ로 순위 → A1과 동일
   worst/best-first nested subset(k=1..N-1).
2. **noisy / freerider_random / freerider_zero**: 각 truncation point에서
   `_fl(model, tok, [clients[c] for c in kept], init, seed)` — 필터링된 클라 리스트로 호출
   (`_fl`은 그대로 재사용; `sample_frac=RCFG["k_frac"]`이 남은 리스트 기준으로 재해석됨).
   측정 = 최종 val-loss(기존 `loss_fn` 재사용). **MMLU/ROUGE는 이 실험에서 생략**
   (review K-2-2: MMLU는 이 SFT에서 검출력≈0). val-loss가 1순위 지표.
3. **poison** (k=1 단일 지점, locked): `build_trajectory`의 poison 분기
   (`phase2_matrix.py:247-272`)는 attacker 설치 라운드가 있어 단순 필터링이 안 통함.
   → φ 최하위(최혐의) 클라 1명을 제외하고 재학습했을 때 **배포 모델의 backdoor ASR이
   떨어지는가**만 측정. `backdoor_asr`(`phase2_matrix.py:78`에 이미 import) 재사용.
   `build_trajectory`를 `exclude` 파라미터 받도록 최소 확장.
4. **방법 범위 "전체"를 위해 `compute_methods()`에 2건 추가**:
   - **Fed-LOO 추가**(현재 phase2_matrix엔 없음, track_d에만 있음): `in_run_loo` import,
     loss-heur 블록(`phase2_matrix.py:334-336`) 옆에 동일 패턴으로.
   - **ComFedSV을 silo에서도 계산**: 현재 `if not silo:` 게이트(`phase2_matrix.py:329`)로
     device100 전용. silo5에서도 `partial=False`로 부르되, **결과 표 캡션에 "silo5는
     full-participation이라 ComFedSV의 low-rank 근사가 자명하게 성립하는 축퇴 레짐"이라는
     caveat 병기**(review R9/§C-4 — 새 발견 아님, 알려진 한계 명시).
5. 3-seed(파일럿은 seed0 하나).

---

## 3. Experiment B — Contamination-ratio dose-response

Threat별 dose 축이 다르다. 아래 표를 구현 스펙으로 사용. **freerider_zero는 dose 대상 제외**
(locked). 각 dose 레벨에서 전체 방법(A와 동일 커버리지, Fed-LOO/ComFedSV 확장 포함) φ 계산·저장.
핵심 산출물 = "dose ↑ → φ가 단조적으로 나빠지는가"(dose 레벨 vs φ Spearman, 또는 곡선).

| Threat | Dose 파라미터 | 상태 | 필요 작업 |
|---|---|---|---|
| `poison` | `POISON_FRAC` | **이미 연속값** (0,1] | 코드 변경 없음 — `{0.1,…,1.0}` 스윕만. review G-7 / Pass3 P3.2-4의 "γ 공격강도 스윕"과 동일 실험이므로 결과를 그 절에 직결. **poison만 추가로** 2차 Taylor 잔차·‖Δ‖ 진단(K-5 심화1 trust-region 경계) 동시 로그 |
| `freerider_random` | `free_rider_scale` | float, 현재 `_benign_std(warm)*sqrt(3)` 고정 | `build_trajectory` freerider 분기(`phase2_matrix.py:240-244`)에 배수 `DOSE_MULT` 곱해 `{0.25,0.5,1.0,2.0,4.0}` 스윕 |
| `noisy` (answer_swap) | 없음(전부-아니면-전무) | **신규 코드 필요** | `flirds/data/corruptors.py`에 `answer_swap_graded(records, client_id, rate, seed_base=100)` 추가 — CNN `label_flip(rate)`(`corruptors.py:24-42`) 패턴 그대로, `rate` 비율만 completion 스왑. `data/llm.py`의 `build()`/`build_alpaca_iid()`/`build_crossdevice()`(각 `noisy=` 인자 자리)에 `noisy_rate` 관통(기본 1.0 = 현행 동작, 하위호환). `{0,0.1,0.25,0.5,0.75,1.0}` 스윕 |
| `freerider_zero` | — | dose 제외(locked) | dose-response 미수행. removal-curve(Exp A)엔 포함 |

---

## 4. Experiment C — (b) target-stability (C-2 정면 대응, 재실행 0)

review C-2: 헤드라인 +1.000의 매칭 대상 (b) 자신이 seed 간 재현되는지 LLM에서 미검증.
저자가 이미 −0.37~−0.11을 쟀으나 "노트-only"(K-1/K-6) → **코드로 정본화**가 실제 작업.

기존 확인 사실: `runs/track_d/rundirs/{scale}_{regime}_seed{0,1,2}/phi.parquet` 각각에
`(b)oracle`의 per-client φ가 저장돼 있음(컬럼 `seed, method, client, phi`). 재학습·재실행 불필요.

1. `runs/track_d/make_fidelity.py` 옆에 새 분석 스크립트(예: `make_target_stability.py`):
   셀별로 3개 seed-rundir의 `(b)oracle` φ를 모아 client×seed pivot → **seed쌍 Spearman**
   (target self-stability) 계산 → `runs/track_d/target_stability.csv`로 영속화(gitignore 관례).
2. `make_fidelity.py`가 출력하는 fidelity 표에 **"target 안정성(xseed ρ)" 열 병기** —
   review §4/§5.1이 요구한 "fidelity와 함께 target 안정성 보고"를 프로토콜의 일부로 격상.
3. phase2_matrix silo5도 동일 계산(대조군: 오염·비IID면 xseed 높음 +0.93~1.00, IID면 ≈0/음수).
4. (선택·후속) K-5 심화3 val-bootstrap 안정성: per-item val loss가 캐시돼 있지 않아 계측
   추가가 필요 → 이번 스코프에서는 제외, "deferred"로 문서에 명시.

---

## 5. Experiment D — AdamW-fidelity (C-5 대응, 별도 저우선순위)

**중요 사실**(구현자 오해 방지): 기존 rundir엔 로그(델타)가 없고(저장물 = config/meta/metrics/
phi.parquet뿐) 있더라도 전부 SGD 로그다. 따라서 **기존 로그로 AdamW fidelity 계산 불가**.
→ AdamW로 vanilla FL을 **딱 한 번 새로 돌려** 로그를 생성하면, 그 위에서 (b)+estimator
fidelity를 **(a) 재학습 오라클 없이** 계산 가능(= G-4가 말한 저비용 경로). (b)/estimator는
동결 궤적 {w_r}과 델타 {Δw_c}만 입력받으므로 옵티마이저 불문.

1. **옵티마이저 파라미터화**: `flirds/fl/llm_server.py:52`의 하드코딩
   `optimizer_cls_and_kwargs=(torch.optim.SGD, {"lr": lr, "momentum": 0.0})`를 env/인자로
   선택 가능하게(예: `CLIENT_OPT=adamw` → `torch.optim.AdamW`). phase2_matrix `_train_delta`
   (`phase2_matrix.py:199-211`)도 AdamW 셀을 원하면 동일 처리.
2. **AdamW lr = 브리지 설정**(locked): 현 constant lr 유지(스케줄러 미변경). 논문 레시피
   AdamW 5e-5 cosine과의 갭은 deviation caveat로 문서화(현 track_d.py docstring §Deviations
   관례를 따름).
3. track_d anchor5(또는 std20) 1개 셀에서 AdamW vanilla FL 1회 → (b)+estimator+전 방법
   fidelity를 기존 `compute_fidelity`/`report_fidelity` 경로 그대로 재사용해 산출. rundir로
   영속화(RUN_NAME에 `_adamw` 접미사 권장).
4. **연구 caveat/기대**: AdamW 델타는 더 크고 적응적 → poison처럼 2차 Taylor trust-region을
   밀어붙임(C-8 축). fidelity가 살아남으면 external-validity 구멍 저비용 봉합, 깨지면 정직한
   경계 + trust-region 진단(Exp B poison)과 연결. **형식 실험 아님**.

---

## 6. 공통 인프라

- 신규 산출물 전부 `RunLogger`/rundir 관례(config + meta(git/env) + parquet/csv). rundir root는
  기존 관례대로(예: `runs/removal_dose/rundirs/…`, `runs/track_d/…`) env(`RUNDIR_ROOT`) 활용.
- 새 env 파라미터(`DOSE_MULT`, `NOISY_RATE`, `CLIENT_OPT` 등)는 기존 스타일대로
  `os.environ.get(...)` 기본값 방식, 하위호환(기본값 = 현행 동작) 엄수.
- **정본화 규율 재확인**: 어떤 수치도 .log/노트에만 남기지 말 것. review R1/R2/R9 재발 금지.

---

## 7. 진행 순서

1. **파일럿**: `threat=noisy` × N=5(silo5) × seed0 × 축소 방법셋(Flirds, Flirds1st, GTG,
   ShapleyFL, loss-heur)으로 **Exp A2 + Exp B(noisy graded)** 코드 경로를 끝까지 1회 실행.
   확인/보고 항목: (a) 코드 동작, (b) **phase2_matrix silo5 단일 FL 재학습 실측 시간**
   (현재 실측치 없음 — 이게 나와야 full sweep 총 GPU 시간 추산 가능).
   - Exp A1(track_d/anchor5 free 재분석)과 Exp C(target-stability, 재실행 0)는 GPU 비용이
     거의 없으므로 파일럿과 병행해도 무방.
2. 파일럿 결과 **Yonghee 승인** 후 full sweep: 4 threat × 전체 방법 × 3 seed × 두 runner.
3. Exp D(AdamW)는 full sweep 이후 마지막.
4. 실측치가 나오면 review-claude.md §5.4(경계 절)·G-1/G-4/G-7 대응 문단에 넣을 표/문장을
   초안으로 제안.

## 8. 환경

- python: `/home/korea_bupj/miniconda3/envs/flirds/bin/python`
- codes/ 에서 `PYTHONPATH=.`
- GPU 0-3 (0 비어있음)
- 스모크(코드 경로만): `SMOKE_MODEL=gpt2`로 각 러너 docstring의 smoke 예시 참조
  (`track_d.py:41-43`, `phase2_matrix.py:38-50`).
