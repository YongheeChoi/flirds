# REMAINING (Slurm · HJ) — A6000 48GB: R4 online 점수원 경쟁 (진행 중)

> 배분 정본 = **`REMAINING-00-INDEX.md`** · 수록목록 정본 = `research-wiki/survey/flirds-paper-experiment-plan.md`.
> **역할 = R4 §5.3 online 표의 7 비-flirds 행 중 seed0·1**(42셀; seed2 = JB).
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 3-seed(seed-major). push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만.
>
> ## ★ 즉시 조치 — L11 을 **flirds1st 9셀로 축소**
>
> **LLM downstream 스코프 컷(2026-07-25 Yonghee)**: §5.3 비교 대상 = {vanilla·oracle_excl·random_excl·flirds류}.
> **loss-heur·FedIF·renorm-4 의 online 셀은 전부 스코프 밖**이 됐다(근거·보전 = `REMAINING-00-INDEX.md` §0·§1).
> **R4 는 R=200 유지** — R=100 전환은 함께 철회됐다(이미 완결된 L1 3-seed 를 재실행할 이유가 없어짐).
> ```
> scancel <l11 jobid>                       # 63셀 배열 전량 중단
> ls runs/track_h/rundirs_llm_hj | grep flirds1st     # ← 먼저 착지분 확인
> cd $REPO && mkdir -p runs/track_h/_logs
> sbatch --array=0-2,21-23,42-44%8 runs/track_h/sbatch_l11_online.sh   # flirds1st 9셀
> ```
> - **이미 착지한 flirds1st 셀은 그대로 유효하다**(같은 R=200 무대). **빠진 인덱스만** 제출할 것 — 싼 소스가 먼저 착지하므로 seed0 은 이미 있을 가능성이 높다.
> - `RUNDIR_REPLACE=1` · 소스별 `VAL_CHUNK`(flirds1st=3) 는 **sbatch 배선 완료** → `--export` 불요. `--time` 기본 24h 로 충분(셀 ~7.4h).
> - **왜 flirds1st 만 살리나**: online 표가 flirds 단독이면 "vanilla·random 보다 낫다"까지만 말할 수 있다. Flirds-1st 를 넣으면 **§5.6①(2차항) 주장이 online 에서도 성립**한다 — retrain 표엔 이미 flirds·flirds1st·loss-heur·FedIF 가 3-seed 로 있다.
> - **완주 후(~8 wall-h) 3090 으로 넘어가 CNN work-steal** — 그쪽이 남은 물량의 대부분이다(§4).

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
- **⚠ 수록 위치 미정**: 07-25 확정 계획서의 본문·부록 목록에 **silo5 (a)-leg 항목이 없다**(LLM (a) 역할은 "1B-LLM 소형 앵커 듀얼오라클 vs (a)" = anchor5 가 담당, ● 완료). 이미 디스크에 있는 완성품이므로 **버리지 말고 Yonghee 판정 대기** — 되살릴 경우 추가 실행 0.

## 2. L11′ — online Flirds-1st (9셀 · ~67 GPU-h · ~8 wall-h)

- **셀**: `flirds1st_gate_v2` × {clean, noisy, frzero} × seed{0,1,2} = **9**(array `0-2,21-23,42-44`).
- 단가 **~7.4h**(HJ 재측정, R=200). 착지 root = `rundirs_llm_hj`(seed2 는 자동으로 `rundirs_llm_yh` — 집계는 둘 다 읽는다).
- **분모 의존(분석 시)**: recovery 분모(vanilla/oracle_excl/random_excl)는 B200 L1 이 **이미 3-seed 로 산출해 뒀다**(noisy·frzero). clean 분모는 B200 c4 의 clean seed1·2 로 닫힌다.

## 4. CNN work-steal (L11′ 완주 후 · 3090)

- 남은 물량의 대부분이 CNN(760 GPU-h)이고 **CNN 은 어느 파티션에서도 돈다**. HJ 는 ~8h 만에 비므로 3090 으로 넘어간다.
- 흡수 대상(먼저 비는 순): **JW 의 `sbatch_c1_axis.sh` 잔여**(최대 물량) → YH 의 G10. 제출 시 `--array` 로 남은 범위만 지정(중복 = GPU 낭비).
- CNN 은 conda `lora4cl`(torch 2.11) + torchvision 데이터만 있으면 된다 — HF 캐시 불요.

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

## 3. 완료 후

1. rundir 커밋(push는 Yonghee) → `runs/track_h/make_analysis.py`(LLM 로더가 `rundirs_llm_hj` 를 이미 읽는다) → `flirds-results-downstream` §5.3 R4 online 표 7행.
2. **스택 캐비엇**: A6000(torch 2.11) vs canonical(B200 torch 2.12) — fidelity·recovery 는 stack-robust(mean|Δ|≤0.006)라 recovery 정규화로 병치. **`timing.json` 은 §5.5 cost 표에 쓰지 않는다.**
3. **완료 판정**: 로그 `TRACK G DONE` + rundir mtime.
