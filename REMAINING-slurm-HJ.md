# REMAINING (Slurm · HJ) — A6000 48GB: R4 online 점수원 경쟁 (진행 중)

> 배분 정본 = **`REMAINING-00-INDEX.md`** · 수록목록 정본 = `research-wiki/survey/flirds-paper-experiment-plan.md`.
> **역할 = 계획서 G4 의 online 레그 전량**(R4 §5.3 표의 7 비-flirds 행). **63셀 전량 제출 완료 = 진행 중 → 손대지 않는다.**
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 3-seed(seed-major). push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만. 기존 rundir read-only.

## 0. 환경 (실제 셋업 결과)

- **공유 `lora4cl` 사용 불가**: `/home/chyoyhr` 가 `drwx------`(700) → env 읽기 불가. **동일 스펙 재생성**으로 해결: `torch 2.11.0+cu128 / transformers 5.5.4 / trl 1.2.0 / peft 0.19.1`(= `requirements-llm.txt` 의 torch-2.11 스택). 빌드 = `runs/_probe/build_env.sh`.
- **공유 HF 캐시 직접 사용 불가**: `/scratch/chyoyhr/hf_home/token` 이 600 → 오프라인에서도 HF 가 읽어 `PermissionError`. **blob 자체는 읽힘** → 자체 `HF_HOME` 구성(`hub` 하위는 공유 캐시로 **심링크** = 재다운로드 0, `datasets/` 만 복사 = `.shuffle()` 캐시 쓰기용, **token 파일 없음**).
- 파티션 `suma_a6000,gigabyte_a6000` + `--qos=base_qos` · 동시 **8-GPU/user** 확인.
- 공통: `codes/` 에서 `PYTHONPATH=.`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### ⚠ 필수 knob — `VAL_CHUNK=3` (OOM 회피)

- **증상**: 기본 `VAL_CHUNK=10` 으로 L11(flirds1st)을 돌리면 **~82 라운드에서 CUDA OOM**(48GB 소진; `flirds_values → _chunked → grad`).
- **원인**: 이 경로는 B200(180GB) 기준 설계 — 라운드마다 누적되는 궤적 + val-grad 피크가 48GB 초과. **A6000 고유 제약.**
- **조치**: `VAL_CHUNK=3` 이면 정상(동일 셀 168 라운드 통과 확인). **φ 불변** — `_chunked` 는 청크별 grad 의 가중합이 전체 val-grad 와 **정확히 동일**(근사 아님) → 논문 수치 무영향. **`VAL_MAXLEN` 은 φ를 바꾸므로 건드리지 말 것.**
- 제출 예: `sbatch --export=ALL,VAL_CHUNK=3 --time=24:00:00 --array=0-41%8 runs/track_h/sbatch_l11_online.sh`

## 1. 완료 — silo5 (a)-leg (9/9)

| threat | 실측 범위 | 평균 | peak |
|---|---|---|---|
| clean | 8.56–8.87 h | 8.72 h | 26.3 GiB |
| noisy | 8.35–8.94 h | 8.65 h | 26.3 GiB |
| frzero | 5.73–6.13 h | 5.91 h | 26.3 GiB |

- **총 69.9 GPU-h**(사전 추정 26 은 2.7× 과소평가 — 실제 7.8 GPU-h/leg). peak 26.3 GiB > 24 → **48GB 필수**가 실측으로 재확인됐다.
- 산출 = `runs/phase2_matrix/rundirs/1B_silo5_{threat}_aonly_s{seed}` · 조인 = `PYTHONPATH=. $PY runs/phase2_matrix/merge_silo5_a.py`.
- **⚠ 수록 위치 미정**: 07-25 확정 계획서의 본문·부록 목록에 **silo5 (a)-leg 항목이 없다**(LLM (a) 역할은 "1B-LLM 소형 앵커 듀얼오라클 vs (a)" = anchor5 가 담당, ● 완료). 이미 디스크에 있는 완성품이므로 **버리지 말고 Yonghee 판정 대기** — 되살릴 경우 추가 실행 0.

## 2. 진행 중 — L11 = G4 online 레그 (63셀 전량 제출됨)

> R4 §5.3 online 표는 CNN 처럼 8방법인데 현재 online 은 **flirds 만**(B200 L1) → 나머지 **7방법 T1 부호-게이트**를 채운다.
> **손대지 않는다** — 재제출·재정렬 금지. 완료분만 커밋.

- **셀**: 7 비-flirds(flirds1st·lossheur·fedif·gtg·fedsv·comfedsv·shapleyfl) × {clean, noisy, frzero} × seed{0,1,2} = **63**.
  - **G4a = same-game+FedIF 3종**(flirds1st·lossheur·fedif) 27셀 — **싸다**.
  - **G4b = renorm-4**(gtg·fedsv·comfedsv·shapleyfl) 36셀 — **비싸다**.
- 비-flirds 는 online 스코어링에 **HVP 불요**(값·1차) → retrain-scoring ~32 GiB → 48GB, **B200 cum 불요·자체완결**.
- 착지 root `rundirs_llm_hj`(canonical `rundirs_llm` 무수정) → `make_analysis` dup-win 병합.

### 실측 단가 (200 라운드 학습부 · `VAL_CHUNK=3`)

> **⚠ 종전 표(same-game 2.5 h / lossheur 3.2 h)는 HJ 오측정이었다 — 실측은 그 ~3×.** 07-25 19:00 8셀 동시 가동분에서 `train_runtime` 호출수(=라운드×K=5)로 재측정. 라운드당 132 s(same-game) / 166 s(lossheur) — 그중 학습만 110 s 라 `VAL_CHUNK` 영향은 미미하고, **비용은 200라운드 학습 자체**다. 아래 값은 8셀 전부 ±2% 내로 일치.

| source | 실측(재측정) | 셀 수 | 소계 |
|---|---|---|---|
| flirds1st | 7.33 / 7.37 / 7.51 h (clean/noisy/frzero) → **~7.4 h** | 9 | ~67 |
| fedif | 7.41 / 7.23 h (clean/noisy) → **~7.4 h** | 9 | ~67 |
| lossheur | 9.23 / 9.15 / 9.59 h → **~9.3 h** | 9 | ~84 |
| **renorm-4**(gtg·fedsv·comfedsv·shapleyfl) | **~23–28 h**(JB 실측, 미검증) | 36 | **~900** |
| | | **63** | **~1,120 GPU-h** |

- 값은 진행률 외삽(200라운드 학습부)이며 **모델 로드 + 말미 downstream 평가는 미포함** — 실제 end-to-end 는 이보다 다소 길다. 첫 셀 착지 시 확정치로 교체할 것.
- **`--time` 조치 완료(07-25 19:00)**: 대기 셀 34개 중 **renorm-4 24셀만 `scontrol update TimeLimit=42:00:00`** 으로 증액(같은 근거 = 실측 23–28 h 가 24 h 벽에 걸림; rundir 은 말미에 한 번에 써지므로 timeout = 셀당 24 GPU-h 전손). **same-game 10셀은 24 h 유지**(7–10 h 면 충분 + 긴 `--time` 은 backfill 불리). **실행 중 잡은 건드리지 않았고**(INDEX §0 운용 원칙), `scancel`·재제출 0 — 증액은 비파괴 갱신이다. 파티션 `MaxTime=14-00:00:00` 이라 42 h 는 한도 내.
- **renorm 이 느린 이유**: 라운드마다 coalition/submodel 평가(`shapleyfl_round_raw` = val forward)가 들어간다 — forward 만 하는 lossheur 대비 2.4×, same-game 대비 라운드당 ~13×. **이건 버그가 아니라 논문이 주장하는 지수 비용 그 자체**(op-count §7.3: (b)·ShapleyFL 은 K 에 지수).

### 예상 종료 · 리스크

- **HJ 몫(seed0·1 42셀) = ~700–820 GPU-h**(재측정 반영; 종전 ~654 는 same-game 오측정 기반). 내역 = same-game 18셀 **~145**(flirds1st 44 + fedif 44 + lossheur 56) + renorm-4 24셀 **~552–672**. 8슬롯 → **~88–103 wall-h → 07-29 오전 ~ 07-30 새벽**. **마감(실험 07-28 24:00)을 HJ 단독으로는 못 맞춘다.**
- **JB 가 L9 를 전량 중단하고 seed2 를 먼저 끝낸 뒤(07-27) 꼬리를 work-steal** 하면 당겨지지만, 재측정치로는 **07-28 내 63셀 완주가 빠듯**하다. 단 **A6000 클러스터 여유가 10장**이라 HJ+JB 가 상시 16슬롯을 채우지는 못한다.
- **착지 순서가 방어선**: seed-major + 소스-major(21셀/seed = flirds1st→lossheur→fedif→gtg→fedsv→comfedsv→shapleyfl)라 **싼 same-game 이 먼저 착지**한다. 07-26 오전이면 seed0 same-game 9셀이 다 들어와 §5.3 online 표의 G4a 3행은 **1-seed 로 조기 확보**된다. 물량이 마감에 걸리면 잘리는 쪽은 항상 renorm 꼬리(seed1)다.
- **⚠ clean 은 컷하지 않는다 (2026-07-25 Yonghee: "clean 은 필수").** renorm-4 의 clean 12셀을 빼면 ~300 GPU-h 를 아낄 수 있지만, clean 열은 **오발화(false-firing) 판정의 근거**라 유지한다 — renorm 은 flirds 와 달리 clean 에서도 음수 φ 로 발화해 `equals_vanilla` 스킵이 안 되고 실제 개입이 일어난다. 즉 renorm 의 clean 칸은 "비어도 되는 대조"가 아니라 **결과 그 자체**다.
- **판정 시점 = 07-27 아침**: seed0 착지 + seeds1·2 진척으로 완주 여부를 본다.

### ⚠⚠ 최대 변수 = A6000 노드 drain (07-25 19:50 관측, 신규)

- **12노드 중 7노드가 `drng`/`drain`**(`Reason=Kill task failed`, root@07-25T17:19 및 T19:35). 클러스터 총 96 GPU 중 **할당 70 / 유휴 26 인데 유휴 26장이 전부 drain 노드 위**에 있다 → **스케줄 가능한 유휴 GPU = 0**.
- `PrivateData=jobs` 라 `squeue` 로는 내 잡 8개만 보인다. **"클러스터가 비어 있다"는 오독** — 실제로는 73% 점유 중이다. 따라서 `ArrayTaskThrottle=8` 을 올려도 **지금은 아무 효과가 없다**(대기 사유가 `JobArrayTaskLimit`→`Resources` 로 바뀔 뿐).
- **실행 중 8셀 가운데 5셀이 drain 노드 위**(node44×3, node51, node27). drng 는 **실행 중인 잡을 죽이지 않으므로 이 5셀은 안전**하다. 문제는 **끝난 뒤**다 — drain 노드는 새 잡을 받지 않으므로 그 슬롯이 나에게 되돌아오지 않는다. 재활용되는 건 mix 노드 3셀(node26·45·47)뿐이고 그마저 타 사용자와 경쟁한다.
- **동시성 감쇠 시나리오**(잔여 666–786 GPU-h 기준):

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
