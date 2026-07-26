# REMAINING (Slurm · HJ) — A6000 48GB: R4 online 점수원 경쟁 (진행 중)

> 배분 정본 = **`REMAINING-00-INDEX.md`** · 수록목록 정본 = `research-wiki/survey/flirds-paper-experiment-plan.md`.
> **역할 = R4 §5.3 online 표의 7 비-flirds 행 중 seed0·1**(42셀; seed2 = JB).
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 3-seed(seed-major). push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만.
>
> ## ★ 즉시 조치 — **flirds1st 3셀만 완주, 나머지 L11 취소**
>
> 07-25 19:50 진행 상황 기준. 실행 중 8셀의 정체는 `SRC_I=(IDX%21)/3` 로 갈린다:
>
> | idx | 정체 | 스코프 | 조치 |
> |---|---|---|---|
> | **0·1·2** | **flirds1st seed0** × clean/noisy/frzero (58%, ~23:00 완주) | ✅ 안 | **완주** |
> | 3·4·5 | lossheur seed0 × 3 (46%) | ❌ 밖 | `scancel` |
> | 6·7 | fedif seed0 × clean·noisy (44%, frzero=idx 8 미착수라 2/3) | ❌ 밖 | `scancel` |
>
> ```
> scancel <jobid>_[3-20]       # lossheur·fedif·renorm-4 = 스코프 밖 (실행 중 포함)
> squeue -u $USER              # 0·1·2 만 남았는지 확인
> ```
> **✅ 실행 완료 07-25 20:20 — 단 범위는 `[3-41]`.** 문서의 `[3-20]` 은 seed0(0–20)까지만 끊어 **seed1 21셀(idx 21–41)이 대기로 남는다** — `21·22·23` = flirds1st seed1(= B200 c4 몫), `24–29` = lossheur·fedif seed1, `30–41` = renorm-4 seed1 로 **전부 스코프 밖**이다. 판정 기준이 "`squeue` 에 0·1·2 만"이라 그쪽을 따랐다. 결과 = `1876764_0,1,2` 만 잔존(flirds1st seed0 clean/noisy/frzero, 확인 완료).
> - **flirds1st seed0 3셀은 B200 c4 의 해당 3줄을 지운다** — c4 는 seed1·2 6셀만 맡는다(폴백용 주석은 c4 하단에 있다).
> - **lossheur·fedif 를 끊는 이유**: LLM downstream 스코프 컷으로 표에서 빠졌고, **1-seed 는 3-seed 규칙상 논문에 못 넣는다**. 3-seed 로 승격하려면 seeds1·2 에 +50 GPU-h(B200)가 더 드는데 c4 에 그 여유가 없다(G5 를 밀어내야 한다).
> - **이미 착지한 rundir 은 지우지 않는다** — 집계는 canonical `rundirs_llm` 이 dup-win 으로 이긴다.
> - **그 뒤 HJ 는 A6000 을 그대로 이어 G12(§3) → 3090 으로 G10(§3b)** 순으로 간다.
>
> ### ✅ 착지 완료 — **3/3, 실패 0** (07-25 23:36) · **B200 확정통보 발송됨**
>
> | 셀 (`runs/track_h/rundirs_llm_hj/`) | 착지 | GPU-h | gate P / R | 오배제 | fallback | val_loss | gsm8k_em | rouge_l |
> |---|---|---|---|---|---|---|---|---|
> | `gsm50k5_clean_flirds1st_gate_v2_seed0` | 23:30 | 7.90 | 0 / None | 103 (1.03%) | 1 | 0.6021 | .3655 | .3468 |
> | `gsm50k5_noisy_nr0.7_flirds1st_gate_v2_seed0` | 23:36 | 8.00 | .859 / **.0447** | 28 | 0 | 0.6052 | .3387 | .3409 |
> | `gsm50k5_frzero_flirds1st_gate_v2_seed0` | 23:08 | 7.53 | .963 / .945 | 138 | 0 | 0.6024 | .3691 | .3452 |
>
> 전부 `EXIT=0` · R=200 완주 · `phi_rounds` 10,000행(50클라×200라운드). 커밋 `ae4f212`·`17315c3`·`381a5fd`.
> **→ B200 에 "3/3 완주 = 큐 그대로, c4 폴백 3줄 해제 불필요" 확정 통보 완료.**
>
> **집계 검증**: `make_analysis.py:95` 가 `rundirs_llm_hj` 를 로드하고 canonical `rundirs_llm` 을 **마지막에** 실어 dup-win 시킨다. 세 셀 모두 `_load()` 로 파싱 성공 확인(read-only 호출; `analysis/` 는 캠페인 중이라 재생성하지 않았다 — 부분 데이터로 tracked 산출물을 덮어쓰지 않기 위해).
>
> **읽을 때 주의 3가지** (전부 정상값이며 오독하기 쉬운 자리):
> 1. **noisy 셀 이름에 `nr0.7` 이 들어간다** — `rundirs_llm` canonical 명명과 같은 규칙이다. `gsm50k5_noisy_flirds1st_*` 로 글롭을 짜면 놓친다(실제로 이 세션 감시자가 이걸로 **오탐 FAIL** 을 냈다; 실체는 `EXIT=0`).
> 2. **clean 의 `precision=0 / recall=None` 은 축퇴값**이다 — 오염 클라가 0이라 참양성이 정의되지 않고 배제 103건이 정의상 전부 오배제가 된다. 읽을 값은 **오발화율 1.03%**.
> 3. **noisy recall 4.5% 는 미스가 아니라 §2.1 히트**다 — `runs/track_g/README.md:42` 에 "noisy@canon 부호-게이트 = parity(게이트 침묵); nr∈(0,1] 에 0-교차 없음(Flirds ~3.4 extrapolated=도달불가)" 로 **사전 등록된 예측**과 방향이 같다. 회수는 z-게이트/V2w 몫. (등록 예측은 Flirds 기준, 이 셀은 flirds1st 라 그 차이는 명시할 것.)

## 0. 환경 (실제 셋업 결과)

- **공유 `lora4cl` 사용 불가**: `/home/chyoyhr` 가 `drwx------`(700) → env 읽기 불가. **동일 스펙 재생성**으로 해결: `torch 2.11.0+cu128 / transformers 5.5.4 / trl 1.2.0 / peft 0.19.1`(= `requirements-llm.txt` 의 torch-2.11 스택). 빌드 = `runs/_probe/build_env.sh`.
- **공유 HF 캐시 직접 사용 불가**: `/scratch/chyoyhr/hf_home/token` 이 600 → 오프라인에서도 HF 가 읽어 `PermissionError`. **blob 자체는 읽힘** → 자체 `HF_HOME` 구성(`hub` 하위는 공유 캐시로 **심링크** = 재다운로드 0, `datasets/` 만 복사 = `.shuffle()` 캐시 쓰기용, **token 파일 없음**).
- 파티션 `suma_a6000,gigabyte_a6000` + `--qos=base_qos` · 동시 **8-GPU/user** 확인.
- 공통: `codes/` 에서 `PYTHONPATH=.`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### ⚠ `VAL_CHUNK` — 이제 **소스별로 자동**(sbatch 배선 완료)

- **증상(원래 문제)**: 기본 `VAL_CHUNK=10` 으로 flirds1st 를 돌리면 **~82 라운드에서 CUDA OOM**(48GB 소진; `flirds_values → _chunked → grad`). B200(180GB) 기준 설계라 생기는 **A6000 고유 제약**.
- **조치**: `VAL_CHUNK=3` 이면 정상(168 라운드 통과 확인). **φ 불변** — `_chunked` 는 청크별 grad 의 가중합이 전체 val-grad 와 **정확히 동일**(근사 아님). **`VAL_MAXLEN` 은 φ를 바꾸므로 건드리지 말 것.**
- **개선(07-25)**: 이 제약은 **grad 경로에만** 필요한데 7개 소스 전부에 걸려 있었다. 이제 sbatch 가 소스별로 준다 — **flirds1st·fedif = 3**(functorch val-grad), **lossheur·renorm-4 = 10**(forward-only @no_grad, 메모리 여유). 청크 합산이 exact 라 **어느 값이든 φ 동일 = 무위험**.
  속도 이득은 **미측정**이다: `make_llm_loss` docstring 은 (b) 오라클에서 청크를 키웠을 때 **~1.0×**(FLOP-bound)로 프로파일됐다고 적고 있고, 위 재측정상 same-game 셀은 학습이 지배한다. renorm 셀(스코어링 ~92%)에서만 여지가 있다.
- 제출은 `sbatch --array=0-41%8 runs/track_h/sbatch_l11_online.sh` 로 충분(`--export` 불요).

## 1. 완료 — silo5 (a)-leg (9/9)

| threat | 실측 범위 | 평균 | peak |
|---|---|---|---|
| clean | 8.56–8.87 h | 8.72 h | 26.3 GiB |
| noisy | 8.35–8.94 h | 8.65 h | 26.3 GiB |
| frzero | 5.73–6.13 h | 5.91 h | 26.3 GiB |

- **총 69.9 GPU-h**(사전 추정 26 은 2.7× 과소평가 — 실제 7.8 GPU-h/leg). peak 26.3 GiB > 24 → **48GB 필수**가 실측으로 재확인됐다.
- 산출 = `runs/phase2_matrix/rundirs/1B_silo5_{threat}_aonly_s{seed}` · 조인 = `PYTHONPATH=. $PY runs/phase2_matrix/merge_silo5_a.py`.
- **✅ 수록 예정**(Yonghee 확인 07-25). 앞선 "수록 위치 미정" 메모는 오독이었다 — anchor5 듀얼오라클과 **별개 항목으로 논문에 들어간다**. 추가 실행 0, 조인만 하면 된다(`merge_silo5_a.py`).
- **✅ 조인 실행 완료(07-25 21:51)**: 87 (threat,seed,method) 행 → `runs/phase2_matrix/silo5_a_fidelity_1B.csv`. 재생성분이 커밋본과 **바이트 동일**(워킹트리 클린) = 재현성 확인.
  - 3위협 전부 `Flirds / Flirds1st / Banzhaf / loss-heur = ρ_b, ρ_a 모두 +1.000`. 하위는 `FedIF +0.867/+0.933/+0.900`, `ComFedSV +0.833/+0.867`(clean 은 행 자체가 없다), `FedSV +0.933`(clean·frzero).
  - **✅ 해소(07-26 00:40) — 데이터 불일치가 아니라 스크립트 캡션 오류였다. Yonghee 판정 불요.**
    앞서 "스크립트 headline(clean +0.87 / noisy +0.93)과 실측(+1.000)이 어긋난다"고 올렸는데, **어긋난 게 아니라 서로 다른 두 양을 같은 이름으로 부른 것**이다:
    | 양 | 정의 | 값 | 출처 |
    |---|---|---|---|
    | **듀얼오라클 일치도** = 이 스크립트의 `(b)oracle` 행 `ρ_a` | (a) vs (b), **같은 seed 안에서** | **+1.000** (3위협) | 본 실행 |
    | **(b) 타깃 자기안정성** = `xseed ρ` | (b) vs (b), **seed 간** | **+0.87 / +0.93** | `flirds-results-fidelity.md` §1C 표(L346–362) |
    위키는 처음부터 **+1.000 로 적고 있었다** — `flirds-results-fidelity.md:300` "**두 오라클이 silo5에선 완전히 같은 순위를 준다**… (b) 타깃이 seed-안정한 무대(silo5 xseed +0.87~+0.93)에선 (a)와 정확히 일치", `flirds-paper-experiment-plan.md:106` "(b) 타깃 seed-안정(xseed +0.87~+0.93) · **듀얼오라클 일치도 1.000**". **즉 실측이 스펙 그대로이며 본문과도 일치한다.**
    - 위키 L300 이 예고한 **부수 귀결도 그대로 재현**됐다 — "(b)↔(a) 순위가 같으므로 모든 방법의 vs-(b) Spearman = vs-(a) Spearman". 출력에서 전 방법 `ρ_b == ρ_a` 확인(값까지 동일).
    - **조치**: `merge_silo5_a.py` 의 docstring·말미 print 를 고쳤다(기대값을 `+1.000` 으로, +0.87/+0.93 은 **다른 양**이라는 경고 병기). **숫자 산출 코드는 무수정** — 재실행 후 `silo5_a_fidelity_1B.csv` **바이트 동일** 확인. 이 캡션을 그대로 두면 다음 사람이 정상 실행을 −0.07/−0.13 회귀로 오독한다.

## 3. 담당 ① — G12: A축 lever probe seed 보강 — ⛔ **전량 취소 (07-26 04:57, Yonghee 지시)**

> **결과: 15셀 중 착지 0. 산출물 없음.** `scancel 1878707` (실행 6 + 홀드 9). 기존 probe rundir 21개는 **무손상**, 다른 워크스트림 영향 0. 취소 결정의 근거는 아래 §「valuation 병목」 — 요약하면 **48GB 카드에서 이 실험은 계획 단가의 3배가 든다**.
>
> **재개하려면 반드시 바꿔야 할 2가지** (그대로 다시 올리면 같은 결과가 난다):
> 1. **`ARMS=0`** — arm 은 이 하드웨어에서 24h 안에 못 끝난다(st30: FL+val 12.5h + arm 12.5h = 25h). 게다가 이 probe 의 기존 seed1·2 선례가 fidelity-only 이고, 등록된 질문(φ cross-seed)은 arm 이 필요 없다.
> 2. **`VAL_CHUNK`** 재검토 — 2 는 메모리상 불가피했지만(아래) 시간이 5배 든다. 3 이면 peak ~30 GiB 로 48GB 에 들어가면서 청크수는 1.5배 준다. **φ 는 어느 값이든 불변**(청크 합산 exact).
>
> 그 둘을 적용하면 셀당 ~9h, 15셀 ≈ 135 GPU-h. 여전히 계획 64 를 넘으므로 **재개 자체가 판정 사항**이다.

### ⚠ 취소 사유 — valuation 병목 (07-26 04:48 진단)

6셀 전부 `150/600` 에서 **8시간 이상 정체**했다. phase-1 FL 은 끝났고 `track_d.py:426` 의 `with pt.phase("valuation")`(11방법 + (b) exact 2⁵)이 안 끝난 것이다. **φ 체크포인트(`:437`)는 valuation 뒤에 있으므로 그때까지 rundir 이 하나도 안 생긴다.**

| | 참조셀(B200) | HJ 셀(48GB) |
|---|---|---|
| `val_chunk` | **10** | **2** |
| valuation 실측 | 4,326 s (1.20 h) | idx0 **6 h 08 m 경과·미완** |
| valuation peak | **98.9 GiB** | ~20 GiB |

**핵심**: 기존 1B anchor5 셀 21개는 **전부 `val_chunk=10`** 이고 그때 **peak 이 98.9 GiB** 다(`val_chunk=2` 는 원래 7B 용). 즉 48GB 카드에서 chunk=10 은 **애초에 안 들어간다** → 축소는 불가피했고 φ 도 불변이지만, **val 배치가 5배로 쪼개져 valuation 이 1.2h → 8h 대가 됐다**. 이 시간 비용이 계획 `4 h/셀` 에 빠져 있었다(계획치는 B200-class 메모리 = chunk 10 전제).

- 이 세션이 앞서 낸 두 진단은 **둘 다 틀렸다**: ①"계획 4h 는 B200 기준"(→ 아니다, `ARMS=1` 이 주범이었다) ②"phase-1 만 끝나면 φ 는 반드시 착지"(→ 아니다, valuation 이 벽보다 길면 **아무것도 안 남는다**). 최종 원인은 **`ARMS=1` + `VAL_CHUNK=2` 두 개가 겹친 것**이다.
- **취소로 잃은 것 = 계산시간뿐**(약 50 GPU-h, 산출 0). 취소로 얻은 것 = G10 이 즉시 **2슬롯 → 7슬롯**(8번째는 `Resources` 대기 = QOS 캡이 더는 병목이 아님).

## 3-구. (참고) 종전 G12 제출 기록 — `1878707`

> **B200 c4 에서 넘친 유일한 물량**이다. c4 = L1 clean(18.6) + flirds1st s1·2(25.2) + G5(20) + G12 앞 3셀(9) ≈ 72.8h 로 74h 창을 채우고, **G12 나머지 16셀만 흘러넘친다**.
> HJ 가 맡는 이유 = **A6000·HF 캐시 셋업이 이미 있고 지금도 거기 있다** → 전환 마찰 0. 끝나면 3090 으로 넘어간다.

- **셀**: anchor5 `lr{1,2,3}e-3 × st{20,30}` seed1·2 중 잔여 + anchor5 `r{32,64}` seed1·2 + `noise_1B_r64` seed1·2 = **15**(c4 가 `lr2e-3_st20` s1·s2 와 `lr3e-3_st20` s1 을 가져감).
  **16 → 15 (07-25 HJ 확인, 코드 대조 완료)**: `lr1e-3_st10 seed0` 은 **추가 실행이 필요 없다** —
  `track_d.py` 의 `ANCHOR5` 기본값이 `lr=1e-3 · max_steps=10 · r=16` 이라 기존
  `runs/track_d/rundirs/1B_anchor5_seed0` 이 곧 그 셀이고, `runs/probe_signal/make_figures.py:120`
  이 이미 그것을 `("lrsteps", 1e-3, 10)` / `("rank_anchor", 16)` baseline 으로 읽는다.
- **제출 = `--array=0-14%8`** (`1878707`). 배열 순서를 우선순위로 깔았다:
  `0-4` lr3e-3·lr2e-3 → `5-8` lr1e-3 → `9-12` r32/r64 → `13-14` noise.
  **마감에 걸리면 뒤 인덱스부터 자르면 핵심 질문은 지켜진다.** config 는 대응 seed0 셀과 일치
  (lr 격자 `MMLU_LIMIT=40` · rank 셀 MMLU 전체) — cross-seed 비교는 같은 셀의 seed 간 비교라
  **이 일치가 축의 유효성 자체**다. 현재 `0-4` 실행 중 · OOM 0.
- **⚠ 다른 계정에 파급되는 발견 — track_d 의 HF 캐시 공백**: `vicgalle/alpaca-gpt4` 와 `cais/mmlu`
  가 공유 캐시에도 HJ 캐시에도 없었다(gsm8k 계열 6종 + Llama-1B 뿐). track_d 는 이 둘이 없으면
  **오프라인에서 시작조차 못 한다**. `hf_pin.py` 의 `REVISIONS` 가 비어 있어 최신 커밋으로 받으면
  되고 둘 다 public → HJ 의 `HF_HOME` 에만 추가(공유 캐시 무수정, +238MB) · 오프라인 재로딩 검증 완료.
  → **B200 c4 의 G5·G12 도 track_d 계열이므로 같은 공백을 먼저 확인**해야 한다
  (B200 은 과거 anchor5·probe_signal 을 돌렸으니 있을 가능성이 높지만, 확인 전 제출 금지).
- 러너 `track_d.py` / `probe_val_noise.py`, 착지 `runs/probe_signal/rundirs`(noise 는 `noise_probe/`).
- **R4 무대가 아니다 → `ROUNDS` 를 주지 않는다.** anchor5 는 N=5·R=30 이라 작지만 flirds HVP 경로이므로 **`VAL_CHUNK=2`** 권장(청크 합산 exact → **φ 동일**).
```
# 예: 한 셀
REGIME=anchor5 LR=3e-3 MAX_STEPS=20 ORACLE_A=0 FIDELITY=1 ARMS=1 MMLU_LIMIT=40 \
  SEED=2 LORA_R=16 RUN_NAME=1B_anchor5_lr3e-3_st20_seed2 VAL_CHUNK=2 \
  RUNDIR_ROOT=$REPO/runs/probe_signal/rundirs PYTHONPATH=. $PY -u experiments/track_d.py
```
- **부록·최저 우선**이라 마감에 걸리면 꼬리부터 버려도 된다. 다만 핵심 질문("lr 로 커진 φ가 cross-seed 실재 신호인가", 예측 ρ≈0)에 필요한 **`lr{2,3}e-3` 계열을 먼저** 돌린다.

#### 제출 완료 — job `1878707` (07-25 20:32, `--array=0-14%8`)

- 스크립트 `runs/probe_signal/sbatch_g12_lever_seeds.sh`(신규). **16 이 아니라 15셀**이다 — `lr1e-3_st10 seed0` 은 **별칭이 유효**해서 뺐다: anchor5 기본값이 `lr=1e-3 / max_steps=10 / r=16`(`track_d.py:89`)이라 `runs/track_d/rundirs/1B_anchor5_seed0` 가 곧 그 셀이고, **`make_figures.py:120` 이 이미 그것을 baseline 으로 읽고 있다**(README 도 동일 명시). 추가 실행 0.
- **배열 순서 = 우선순위**: `0-4` lr3e-3·lr2e-3 → `5-8` lr1e-3 → `9-12` rank r32/r64 → `13-14` val-noise r64. 마감에 걸리면 **뒤 인덱스부터 `scancel`** 하면 핵심 질문은 지켜진다.
- **config 는 대응 seed0 셀과 정확히 일치**시켰다 — lr×steps 셀은 `MMLU_LIMIT=40`, rank 셀은 MMLU 전체(=0). cross-seed 비교는 같은 셀의 seed 간 비교라 이 일치가 축의 유효성 그 자체다. `VAL_CHUNK=2` 만 다른데 이건 메모리 knob 이고 청크 합산이 exact → **같은 게임**이다(참조셀 B200 peak 98.9 GiB → 48GB 에선 축소 필수).

##### ⚠ 실측 단가 — 계획 `4 h/셀` 이 아니라 **A6000 에서 9–17 h/셀** (07-25 22:00)

`train_runtime` 호출수로 5셀 동시 측정. **셀당 학습 호출 = 600 회**로 고정이다: phase-1 공유 FL(30라운드×5클라=150) + arm 3종 재학습(3×150=450). anchor5 는 `k_abs==n` 이라 `flirds_sel` 이 빠져(`track_d.py:319`) arm 이 4 가 아니라 **3**(`flirds_w·shapleyfl_w·fedif_w`; `base`·`vanilla` 는 재학습 없음).

| 셀 | 하드웨어 | 실측 | 600회 환산 |
|---|---|---|---|
| st20 (idx0) | RTX6000Ada | 51.8 s/call | **~8.6 h** |
| st30 (idx1–4) | A6000(노드 경합) | 104 s/call | **~17 h** |

- ~~**계획치 4 h/셀은 B200 기준이었다**~~ → **틀렸다. 아래 §「원인 재특정」 참조 — 진짜 원인은 하드웨어가 아니라 내가 켠 `ARMS=1` 이다.**
- **`--time` 조치**: 대기 10셀은 `scontrol update TimeLimit=24:00:00` 완료. **실행 중 5셀은 증액 불가**(`Access/permission denied` — 실행 잡의 TimeLimit 증액은 operator 권한) → **12 h 유지**.
- **다만 손실은 제한적이다** — `track_d.py:437` 이 fidelity 직후 `_persist(...)  # CHECKPOINT` 로 **φ 를 먼저 영속화**하고 그 다음에 arm 단계를 `try/except` 로 돈다. phase-1 은 150 호출 = st30 기준 ~4.3 h 이므로 **12 h 안에 φ 는 반드시 착지**한다. 12 h 벽에 걸려도 잃는 건 arm(MMLU·ROUGE·r2t)뿐이고, **G12 의 핵심 질문("lr 로 커진 φ가 cross-seed 실재 신호인가")은 φ 축이라 온전하다**. arm 까지 필요하면 그 셀만 24 h 로 재제출하면 된다(φ 는 이미 은행에 있다).

##### ⚠⚠ 정밀 재측정 + **st30 4셀 조기 회수 결정** (07-25 23:17)

150 호출(=phase-1 완주) 시점에서 `train_runtime` 합·경과시각으로 셀별 재계산:

| idx | 셀 | wall/call | 600회 환산 | φ 체크포인트 | arm 포함 완주 | 12 h 벽 |
|---|---|---|---|---|---|---|
| 0 | `lr3e-3_st20_seed2` | 65.6 s | **10.94 h** | 23:17 ✅ | 07-26 07:29 | 08:32 — **여유 1.1 h** |
| 1 | `lr3e-3_st30_seed1` | 103.2 s | **17.20 h** | 00:51 | 07-26 13:45 | 08:33 — **초과 5.2 h** |
| 2 | `lr3e-3_st30_seed2` | 103.2 s | 17.20 h | 00:51 | 13:45 | 초과 |
| 3 | `lr2e-3_st30_seed1` | 104.3 s | 17.38 h | 00:54 | 13:56 | 초과 |
| 4 | `lr2e-3_st30_seed2` | 105.4 s | 17.57 h | 00:57 | 14:07 | 초과 |

- **arm 은 부분 저장이 안 된다** — `track_d.py:446-454` 가 3 arm 을 **전부 재학습해 `rows` 를 채운 뒤** `:455` 에서야 downstream 평가·`res["arms"]` 기입을 한다. 중간 persist 지점이 없다. 즉 12 h 에 잘리면 **arm 2개를 끝냈든 0개든 결과는 동일하게 0**이다.
- **→ 결정: idx1–4 는 φ 체크포인트 착지 직후(≈01:00) `scancel`.** 그대로 두면 01:00–08:33 의 **~30 GPU-h 가 확정적으로 0 을 낳는다**. 반면 QOS 캡은 8 GPU 로 **파티션을 가로지르므로**(§3b′ 참조) 그 4 슬롯은 지금 `QOSMaxGRESPerUser` 로 묶여 있는 G10/G12-대기분이 즉시 가져간다. **확정 0-수익 시간을 확정 수익 시간과 맞바꾸는 것**이라 트레이드오프가 아니다.
- idx0(st20)은 **살린다** — 12 h 안에 arm 까지 들어온다(다만 valuation 단계가 위 환산에 안 잡혀 있어 여유 1.1 h 는 빠듯; 잘려도 φ 는 이미 착지).
- st30 4셀의 **arm 이 논문에 필요해지면** 24 h 로 재제출한다. 다만 G12 는 부록·최저이고 seed0 셀엔 arm 이 이미 있으므로, seed1·2 의 arm 은 **있으면 좋은 축**이지 핵심이 아니다.

##### ⚠⚠⚠ 원인 재특정 + 회수 시각 정정 (07-26 00:05) — **앞의 두 진단이 둘 다 틀렸다**

**(1) φ 체크포인트는 phase-1 FL 직후가 아니다 — 그 뒤 `valuation` 이 통째로 더 붙는다.**
`track_d.py:426` 의 `with pt.phase("valuation")` 이 11개 방법 + (b) exact 오라클을 다 계산한 **뒤에야** `:437` 체크포인트가 찍힌다. 기존 참조셀 `1B_anchor5_lr1e-3_st10_seed1/timing.json` 이 그 값을 준다 — **client-training 1,820 s / valuation 4,326 s** (B200). valuation 은 `max_steps` 와 무관(로그+HVP 연산)하므로 셀마다 같고, 하드웨어비만 곱하면 된다.

| idx | phase-1 FL 종료 | + valuation(환산) | **φ 착지** | 회수 시점 |
|---|---|---|---|---|
| 0 (Ada, st20) | 22:40 | ~2.5 h | **~01:10** | arm 가능(08:32 벽) → **살림** |
| 1–4 (A6000, st30) | ~00:43 | ~3.3 h | **~04:00** | arm 불가 → **04:00 회수** |
| 5 (Ada, st20, 24h) | ~01:43 | ~2.5 h | ~04:15 | arm 가능 → 살림 |

→ **앞서 적은 "~01:00 회수" 는 valuation 을 빼먹은 값이다. 실제 회수는 ~04:00.** 회수로 버는 시간도 ~30 이 아니라 **~18 GPU-h**(04:00→08:33 × 4셀).

**(2) 단가 폭증의 원인은 하드웨어 환산이 아니라 내가 켠 `ARMS=1` 이다.**
기존 seed1·2 셀 6개(`lr{1,2,3}e-3_st10_seed{1,2}`)의 `metrics.json` 에는 **`arms` 키가 아예 없다** — 이 probe 의 seed1·2 선례는 **fidelity-only** 다(seed0 셀에는 arm 5종이 있다). 그리고 fidelity-only 단가는 B200 1.71 h → A6000 환산 **~4.7 h/셀**로, **계획치 `4 h/셀`·총 64 GPU-h 와 사실상 일치**한다. 즉 계획치는 처음부터 A6000-fidelity-only 값이었고, 총량을 64→180 으로 3배 불린 것은 **내가 `ARMS=1` 을 넣은 것**이다(arm 재학습이 셀 학습시간의 450/600 = 75%).

**Yonghee 판정 1건 — 남은 대기 9셀(idx6–14)의 `ARMS`:**
- **그대로 둔다(현 상태·무조치)**: 24 h 한도라 arm 까지 완주한다. seed0 에 arm 이 있으므로 **"arm 효과가 seed 간 재현되는가"가 lr×steps 격자에서 답 가능해진다**. 대신 ~70 GPU-h 를 더 쓴다.
- **`ARMS=0` 으로 재제출**: 셀당 ~4.7 h, 총 ~42 GPU-h. **등록된 핵심 질문(φ cross-seed)은 완전히 동일하게 답해지고**, 기존 seed1·2 선례와도 일치한다. 잃는 건 위의 보너스 축뿐.
- **HJ 기본 동작 = 그대로 둔다**(비파괴·되돌릴 수 있는 쪽). 두 경로 모두 마감 07-28 24:00 안에 들어오므로 어느 쪽도 위험하지 않다. 잘라도 되면 알려주면 `scancel` 후 `ARMS=0` 으로 재제출한다.

##### ⚠ 선행 조치 — HF 캐시에 **alpaca-gpt4·cais/mmlu 가 없었다**

- 공유 캐시(`/scratch/chyoyhr/hf_home`)에도 **없다** — 거기엔 gsm8k 계열 6종 + Llama-3.2-1B 뿐이다. track_d(anchor5)는 `vicgalle/alpaca-gpt4`(학습 데이터)와 `cais/mmlu`(downstream)를 쓰므로 **오프라인으로는 시작 자체가 불가**했다.
- `flirds/hf_pin.py` 의 `REVISIONS` 가 전부 비어 있어(`rev()`→None) 최신 커밋을 받으면 되고, 둘 다 public(토큰 불요). **HJ 자체 `HF_HOME` 에만 추가**했다(공유 캐시 무수정, +238 MB → 419 MB). 오프라인 재로딩 확인 완료(alpaca 52,002 / mmlu-test 14,042 / 클라 샤드 4,000×5).
- **B200·YH 쪽에서 track_d 계열을 새로 돌린다면 같은 공백을 먼저 확인해야 한다.**

## 3b′. G10 착수 게이트 — **열렸다 · 제출 완료** (07-25 21:55)

- C-a 가 `d09e528` 로 착지(`track_c2.py:157` 에 `"mnist": LeNet5`) → HEAD 에 포함 확인. 파티션도 C-b 로 `base_suma_rtx3090,dell_rtx3090` 확장 반영됨.
- **제출**: `1878912`(`--array=0-71%8`, seed0) · `1878913`(`--array=72-215%8`, seeds1·2).
- **✅ 첫 셀 착지로 배선 검증 (07-25 23:28)** — `mnist_iid_clean_flirds_seed0` EXIT=0, arm 4종 전부 기입(`flirds_gate_v2 .9794 / gatew .9819 / mult .9830 / zgate .9821`, r2t 전부 7). `AUROC=nan` 은 clean 이라 오염 클라가 없어 **정의되지 않는 정상값**이다. override 3종(`--output`·`REPO`·`PY`)과 C-a(`"mnist": LeNet5`) 모두 살아 있음이 실행으로 확인됐다.
- ~~실측 단가 = 20.05 분/셀 → 216셀 ≈ 72 GPU-h~~ **← 표본 1개(가장 싼 clean·flirds 셀)로 낸 값이라 틀렸다.**
- **실측 단가(18셀 기준, 07-26 04:50)**: 중앙값 **25.2 분** · 평균 **33.3 분** · 범위 19.9–109.8 분 → **216셀 ≈ 120 GPU-h**. 계획치 135 와 사실상 일치한다. 편차가 큰 이유는 소스별 비용차(GTG·observer 계열이 비쌈)다.

| G10 슬롯 | 종료 예상 |
|---|---|
| 2 (G12 가 6슬롯 점유하던 상태) | 07-28 16:00 — **마감 8시간 전** |
| **7–8 (G12 취소 후, 현재)** | **07-26 19:00 ~ 21:00** |

- **건강도**: 18셀 전부 `EXIT=0`, 실패 0. 위협별 AUROC 이 의미 있게 나온다 — `label-flip` 0.987–1.000(32 arm) · `free-rider` 0.394–0.967 · `clean` nan(오염 클라 0 = 정의상 정상). 즉 clean 뿐 아니라 **오염 무대 배선도 실행으로 검증**됐다.

### ⚠ HJ 계정에서 제출할 때 필요한 3개 override (스크립트 파일은 **무수정**)

`sbatch_cnn_mnist_comp.sh` 는 YH 홈 기준이라 그대로 쓰면 **216런 전량이 기동 실패**한다 — `--output` 이 `/home/chyoyhr/projects/flirds/...`(700, 쓰기 불가)라 Slurm 이 출력 파일을 못 만든다. `REPO`·`PY` 도 같은 홈을 가리킨다. 제출 시 덮어썼다:

```
sbatch --qos=base_qos --output="$REPO/runs/track_h/_logs/%x_%A_%a.out" \
       --export=ALL,REPO=/home/rlaguswls186790/flirds,PY=<HJ python> \
       --array=0-71%8 runs/track_h/sbatch_cnn_mnist_comp.sh
```

### 선행 조치 2건

- **torchvision 부재** → `0.26.0+cu128` 설치. **`--no-deps` 로 넣어 `torch 2.11.0+cu128` 을 보호**했다(그냥 설치하면 torch 를 갈아끼워 실행 중인 LLM 잡의 스택이 바뀐다). `pillow` 도 없어 같이 설치. 검증: `torch 2.11.0+cu128 / torchvision 0.26.0+cu128`.
- **MNIST 미캐시** → `~/data` 에 선다운로드(60k/10k). `flirds/data/cnn.py:25` 가 `download=True` 라 **216런이 동시에 같은 경로로 내려받으면 레이스**가 난다(계산노드 인터넷 여부도 불확실).
- **CPU 스모크 통과**: mnist/dir1/label_flip@0.70, flirds 4-arm → `AUROC 1.000/0.978/0.989/1.000`, `TRACK-C2 RUN OK`. 216런 투입 전 C-a 배선을 실증했다.

### ⚠ QOS 캡은 **파티션을 가로지른다**

G10 은 3090 인데도 대기 사유가 `QOSMaxGRESPerUser` 다 — A6000 잡 8장이 3090 잡을 막는다. 즉 **파티션을 늘려도 총 8슬롯은 그대로**이고, L11 잔여 + G12 + G10 이 **같은 8슬롯을 나눠 쓴다**(총 ~200 GPU-h → ~25 wall-h → 07-26 밤~07-27 아침). 제출 순서대로 G12 → G10 으로 흘러간다.

## 3b. 담당 ② — G10: mnist downstream (216런 · ~135 GPU-h · 3090)

- **⚠ 착수 게이트 = 코드 C-a**(`track_c2.py:157` MODEL_FN 에 `"mnist": LeNet5` 1줄; YH 담당).
```
cd $REPO && mkdir -p runs/track_h/_logs
sbatch --array=0-71%8    runs/track_h/sbatch_cnn_mnist_comp.sh   # seed0 (72)
sbatch --array=72-215%8  runs/track_h/sbatch_cnn_mnist_comp.sh   # seeds 1-2
```
- 2파티션 × 4위협 × (**8**소스 + 관측자) × 3seed = 216. mnist 는 track_g 그리드가 없어 **flirds 소스도 여기서 생성**(7이 아니라 8인 이유). **P1w 는 같은 rundir 에서 동반 산출**(추가 런 0).
- CNN 은 conda `lora4cl`(torch 2.11) + torchvision 데이터만 필요 — HF 캐시 불요.
- **파티션 = `base_suma_rtx3090,dell_rtx3090`**(sbatch 내장 · 07-25 확장). base_suma 단독은 여유 0
  (총 71장, 빈 6장은 draining node01)이고 dell 에 9장이 놀고 있었다 — JW 실측. 같은 RTX3090 이라
  스택 캐비엇 동일. `sinfo -o "%P %G %a"` 로 다른 3090 파티션이 보이면 더 붙여도 된다.
- **깨끗한 상태에서 시작한다**: YH 가 이전 판 근거로 큐에 넣어 둔 216셀을 **취소했다**(전량 PD ·
  mnist rundir 0개). 중복·팬텀 없음.
- **torchvision 미설치 가능성**: JW 계정은 없어서 `0.26.0+cu128` 을 별도 설치했다(torch 2.11.0+cu128
  불변). HJ 도 CNN 첫 제출 전 `python -c "import torchvision"` 로 확인할 것.

## 3d. 담당 ③ — G9 seed0 잔여 (JB 이관): mnist dir1 (a)-오라클 2셀

> **JB 가 seed0 pending 2셀을 이관**(2026-07-26). JB 는 3090 큐에서 `scancel 1878820` 후 seed1·2(16셀)만 유지. HJ 가 G10 완주 뒤 이 2셀을 흡수한다.

- **셀**: `--array=14-15` on `runs/track_c/c1/sbatch_c1_axis.sh` = **mnist dir1 seed0 {free_rider(14) · grad_noise(15)}**. N=10 전원참여 **(a) 2¹⁰ 재학습 오라클 + (b) 2¹⁰ + 9방법 φ**.
- **단가 ~12–18 h/셀**(mnist (a) 2¹⁰ 지배; JB 실측 `oracle_a.time` = 63,327 s = 17.6 h @노드경합, 구 `c1_oracle` mnist = 41,168 s = 11.4 h @무경합). 2셀 병렬 = ~1셀 wall. **G10 완주(~07-26 21:00) 후 착수 → ~07-27 15:00 완료**, 마감 07-29 00:00 여유.
- **게이트 없음** — 축 분리 env(C-b)가 `d09e528` 착지. **셋업도 이미 됨**(§3b: torchvision 0.26.0+cu128 · `~/data` mnist 선다운로드) → 추가 설치 0.
- **제출**(스크립트 무수정 · override 3종 = §3b G10 과 동일 패턴; `sbatch_c1_axis.sh` 도 `--output` 이 chyoyhr 하드코딩):
```
cd $REPO && mkdir -p runs/track_c/c1/_logs
sbatch --qos=base_qos --output="$REPO/runs/track_c/c1/_logs/%x_%A_%a.out" \
       --export=ALL,REPO=/home/rlaguswls186790/flirds,PY=<HJ torch2.11 python> \
       --array=14-15%8 runs/track_c/c1/sbatch_c1_axis.sh
```
- 인덱스 규약: `SEED=IDX/16`=0 · `8-15`=mnist · `PART=dir1` · `T=2 free_rider(14) / 3 grad_noise(15)`. **corrupt(seed0) = 고정 `[1,4,6,7]`**(전위협 공통·seed-only, `default_rng(1000+seed)` 첫 소비). `C1_ORACLE_A=1`(스크립트 기본) = 2¹⁰ 재학습.
- **⚠ 중복 금지**: seed1·2(`24-31,40-47`)는 JB 몫. HJ 는 **seed0 `14-15` 만**. 같은 rundir 명 = last-writer-wins.
- **채우는 것**: 계획서 §2.1/§3.1 (a) 무대 · §3.4 φ 부호 감사 CNN 레그의 **seed0 leg**(frzero·grad-noise).
- **완료 후**: rundir 커밋(push=Yonghee) → `runs/track_c/make_figures.py load_c1()` 집계. **phi_a = `metrics.json['oracle_a']['phi']`**(rundir 내장; JB 확인, 별도 `c1_oracle/*_aonly_*` 신규축 없음). 완료마커 = `metrics.json`(run 끝에 `phi.parquet` 와 동시 기록).

## 3c. 완주 후

- **JW·JB 의 `sbatch_c1_axis.sh` 잔여를 work-steal**(남은 `--array` 범위만; 같은 rundir 이름 = last-writer-wins 라 중복 = GPU 낭비).

---

## (이하 참고) 종전 L11 63셀 계획 — **스코프 컷으로 폐기**

> R4 §5.3 online 표는 CNN 처럼 8방법인데 현재 online 은 **flirds 만**(B200 L1) → 나머지 **7방법 T1 부호-게이트**를 채운다.
> **손대지 않는다** — 재제출·재정렬 금지. 완료분만 커밋.

- **셀**: 7 비-flirds(flirds1st·lossheur·fedif·gtg·fedsv·comfedsv·shapleyfl) × {clean, noisy, frzero} × seed{0,1,2} = **63**.
  - **G4a = same-game+FedIF 3종**(flirds1st·lossheur·fedif) 27셀 — **싸다**.
  - **G4b = renorm-4**(gtg·fedsv·comfedsv·shapleyfl) 36셀 — **비싸다**.
- 비-flirds 는 online 스코어링에 **HVP 불요**(값·1차) → retrain-scoring ~32 GiB → 48GB, **B200 cum 불요·자체완결**.
- 착지 root `rundirs_llm_hj`(canonical `rundirs_llm` 무수정) → `make_analysis` dup-win 병합.

### 실측 단가 (200 라운드 학습부 · `VAL_CHUNK=3`)

> **⚠ 종전 표(same-game 2.5 h / lossheur 3.2 h)는 HJ 오측정이었다 — 실측은 그 ~3×.** 07-25 19:00 8셀 동시 가동분에서 `train_runtime` 호출수(=라운드×K=5)로 재측정. 라운드당 132 s(same-game) / 166 s(lossheur) — 그중 학습만 110 s 라 `VAL_CHUNK` 영향은 미미하고, **비용은 200라운드 학습 자체**다. 아래 값은 8셀 전부 ±2% 내로 일치.

| source | R=200 실측(재측정) | **R=100** | 셀 수 | 소계 @R=100 |
|---|---|---|---|---|
| flirds1st | 7.33 / 7.37 / 7.51 h → ~7.4 h | **~3.7 h** | 9 | ~33 |
| fedif | 7.41 / 7.23 h → ~7.4 h | **~3.7 h** | 9 | ~33 |
| lossheur | 9.23 / 9.15 / 9.59 h → ~9.3 h | **~4.65 h** | 9 | ~42 |
| **renorm-4**(gtg·fedsv·comfedsv·shapleyfl) | ~23–28 h(JB 실측, 미검증) | **~12.5 h** | 36 | **~450** |
| | | | **63** | **~559 GPU-h** |

> **R=100 환산이 거의 정확히 절반인 이유**: 비용이 라운드에 선형인 두 항(200라운드 학습 110 s/round + 라운드당 스코어링)으로만 이뤄져 있다. 고정비(모델 로드·말미 downstream 평가)는 소수라 셀당 약간의 하방 오차만 남는다.

- 값은 진행률 외삽(200라운드 학습부)이며 **모델 로드 + 말미 downstream 평가는 미포함** — 실제 end-to-end 는 이보다 다소 길다. 첫 셀 착지 시 확정치로 교체할 것.
- **`--time` 조치 완료(07-25 19:00)**: 대기 셀 34개 중 **renorm-4 24셀만 `scontrol update TimeLimit=42:00:00`** 으로 증액(같은 근거 = 실측 23–28 h 가 24 h 벽에 걸림; rundir 은 말미에 한 번에 써지므로 timeout = 셀당 24 GPU-h 전손). **same-game 10셀은 24 h 유지**(7–10 h 면 충분 + 긴 `--time` 은 backfill 불리). **실행 중 잡은 건드리지 않았고**(INDEX §0 운용 원칙), `scancel`·재제출 0 — 증액은 비파괴 갱신이다. 파티션 `MaxTime=14-00:00:00` 이라 42 h 는 한도 내.
- **renorm 이 느린 이유**: 라운드마다 coalition/submodel 평가(`shapleyfl_round_raw` = val forward)가 들어간다 — forward 만 하는 lossheur 대비 2.4×, same-game 대비 라운드당 ~13×. **이건 버그가 아니라 논문이 주장하는 지수 비용 그 자체**(op-count §7.3: (b)·ShapleyFL 은 K 에 지수).

### 예상 종료 · 리스크

- **HJ 몫(seed0·1 42셀) = ~372 GPU-h @R=100**(same-game 18셀 ~72 + renorm-4 24셀 ~300). 8슬롯 → **~47 wall-h → 07-27 후반**. R=200 이었으면 88–103h(07-29~30)로 마감을 못 맞췄다 — **R=100 전환이 이 레그를 마감 안으로 들여놓은 것**이다.
- **JB 가 seed2(21셀)를 병렬로 맡는다** → 63셀 전체가 07-27 후반 착지. 단 **A6000 클러스터 여유가 10장**이라 HJ+JB 가 상시 16슬롯을 채우지는 못한다 → `asus_6000ada`(RTX6000Ada 48GB 8장)를 파티션 목록에 추가해 뒀다.
- **착지 순서가 방어선**: seed-major + 소스-major(21셀/seed = flirds1st→lossheur→fedif→gtg→fedsv→comfedsv→shapleyfl)라 **싼 same-game 이 먼저 착지**한다. 07-26 오전이면 seed0 same-game 9셀이 다 들어와 §5.3 online 표의 G4a 3행은 **1-seed 로 조기 확보**된다. 물량이 마감에 걸리면 잘리는 쪽은 항상 renorm 꼬리(seed1)다.
- **⚠ clean 은 컷하지 않는다 (2026-07-25 Yonghee: "clean 은 필수").** renorm-4 의 clean 12셀을 빼면 ~300 GPU-h 를 아낄 수 있지만, clean 열은 **오발화(false-firing) 판정의 근거**라 유지한다 — renorm 은 flirds 와 달리 clean 에서도 음수 φ 로 발화해 `equals_vanilla` 스킵이 안 되고 실제 개입이 일어난다. 즉 renorm 의 clean 칸은 "비어도 되는 대조"가 아니라 **결과 그 자체**다.
- **판정 시점 = 07-27 아침**: seed0 착지 + seeds1·2 진척으로 완주 여부를 본다.

### ⚠⚠ 최대 변수 = A6000 노드 drain (07-25 19:50 관측, 신규)

- **12노드 중 7노드가 `drng`/`drain`**(`Reason=Kill task failed`, root@07-25T17:19 및 T19:35). 클러스터 총 96 GPU 중 **할당 70 / 유휴 26 인데 유휴 26장이 전부 drain 노드 위**에 있다 → **스케줄 가능한 유휴 GPU = 0**.
- `PrivateData=jobs` 라 `squeue` 로는 내 잡만 보인다. **"클러스터가 비어 있다"는 오독** — 실제로는 73% 점유 중이다.
- **`ArrayTaskThrottle` 증액은 무효 — 상한은 QOS 다.** 20:35 에 두 잡(L11 3 + G12 5)이 8 GPU 를 채우자 대기 사유가 **`QOSMaxGRESPerUser`** 로 바뀌었다. 즉 §0 의 "동시 8-GPU/user" 는 **QOS 하드 캡**이고, 배열 `%8` 은 그 캡을 넘지 않는 표현일 뿐이다(단일 배열만 보면 사유가 `JobArrayTaskLimit` 로 가려져 이 캡이 안 보인다). **8슬롯은 영구 천장이다.**
- **실행 중 8셀 가운데 5셀이 drain 노드 위**(node44×3, node51, node27). drng 는 **실행 중인 잡을 죽이지 않으므로 이 5셀은 안전**하다. 문제는 **끝난 뒤**다 — drain 노드는 새 잡을 받지 않으므로 그 슬롯이 나에게 되돌아오지 않는다. 재활용되는 건 mix 노드 3셀(node26·45·47)뿐이고 그마저 타 사용자와 경쟁한다.
- **아래 감쇠 시나리오는 취소된 42셀 계획 기준이라 지금은 무효**(잔여가 L11 3 + G12 15 로 줄었다). 20:35 현재 drain 에도 불구하고 **8/8 슬롯을 다 받고 있다** — mix 노드에서 GPU 가 풀렸다. 기록으로만 남긴다:

  | 유효 슬롯 | wall-h | 완주 |
  |---|---|---|
  | 8 (drain 해소) | 83–98 | **07-29 07:00 ~ 22:00** |
  | 5 | 133–157 | 07-31 ~ 08-01 |
  | 3 (drain 지속) | 222–262 | **08-04 ~ 08-05** |

- **`Kill task failed` 은 통상 stale 프로세스 정리로 관리자가 수 시간 내 해제**하는 상태다. 내 8셀은 전부 정상 진행 중(19:50 기준 실패 시그니처 0)이라 원인은 내 잡이 아닐 가능성이 높다. **관리자에게 undrain 요청이 필요한지 Yonghee 판단 요망** — 이 한 건이 완주일을 07-29 ↔ 08-05 로 가르는 최대 변수다.

## 3. 완료 후

1. rundir 커밋(push는 Yonghee) → `runs/track_h/make_analysis.py`(LLM 로더가 `rundirs_llm_hj` 를 이미 읽는다) → `flirds-results-downstream` §5.3 R4 online 표 7행.
2. **스택 캐비엇**: A6000(torch 2.11) vs canonical(B200 torch 2.12) — fidelity·recovery 는 stack-robust(mean|Δ|≤0.006)라 recovery 정규화로 병치. **`timing.json` 은 §5.5 cost 표에 쓰지 않는다.**
3. **완료 판정**: 로그 `TRACK G DONE` + rundir mtime.
