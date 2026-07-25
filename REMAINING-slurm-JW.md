# REMAINING (Slurm · JW 계정) — A6000 48GB: L4 renorm-4 T2

> 실행처별 인수인계 **5-서버 분할** 중 **JW**(신규 계정) 몫. 짝 = `REMAINING-b200.md`(HVP 전용)·`REMAINING-slurm-YH.md`(CNN)·`REMAINING-slurm-HJ.md`(silo5-a·L11)·`REMAINING-slurm-JB.md`(L9 비-flirds arms).
> **역할 = A6000 48GB에서 L4 = renorm-4 T2 재학습.** **자체완결 = B200 cum 독립**(즉시 가동). flirds 가중 재학습(L7·L1 flirds T2)은 관찰자 HVP+cum이 필요해 **B200 잔류**(여기 아님).
> **마감: 실험 07-28 / 논문 07-29 21:00** — seed0 우선. push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만.
> 논문·문서 정본 = `paper/workplan/00-INDEX.md`. 기존 rundir은 read-only.

## 0. 신규 계정 셋업 (최초 1회 · HJ/JB와 동일 체크리스트)

1. **repo**: `/home/<JW>/projects/flirds/`(clone 또는 공유 `/home`).
2. **conda env**: `lora4cl`(torch 2.11) — 공유 재사용 or 재생성.
3. **HF 캐시**(offline): model(Llama-3.2-1B-Instruct)+gsm8k. `HF_HOME` = YH `/scratch/chyoyhr/hf_home` 공유-읽기 or 복제. `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.
4. **QOS/파티션**: `base_qos` → A6000 48GB(`suma_a6000`/`gigabyte_a6000`). 동시 **8-GPU/user**.
5. 공통: `codes/`에서 `PYTHONPATH=.`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

> **왜 48GB**: L4 = **retrain-scoring 클래스 ~32 GiB**(renorm 값 산출 + T2 재학습; `REMAINING-b200.md` §1 실측 `*_gate_v2` 32.6·`oracle_excl` 31.0) → **24GB 불가·48GB 필요**. HVP는 없음(renorm은 값 스코어).

## 1. L4 — R4 Tier B T2-only renorm-4 (자체완결·B200 독립)

> `REMAINING-b200.md` §2 L4의 실행처 = **JW 48GB**. renorm 4점수원의 **재학습(T2) 개입** — clean 오발화 열 포함.

- **무엇**: renorm 4점수원(gtg·fedsv·comfedsv·shapleyfl) T2 재학습 × {clean,noisy,frzero} × seed{0,1,2}. renorm은 **value-only·HVP 의존 0·자체완결**(자체 renorm 값 산출 후 그 가중으로 재학습; `track_g build_arm`이 매 실행 fresh 누적기로 인라인 스코어) → B200 cum 불요.
- **clean 열의 의미(2026-07-24 Yonghee)**: flirds와 달리 renorm은 clean에서도 음수 φ로 **오발화(false-firing)** 가능 → clean T2가 `equals_vanilla` 스킵 안 되고 실제 재학습 발생 → §5.3 clean 열의 renorm-4 칸을 채운다.
- **실행(sbatch — REMAINING만 보고 실행)**: `runs/track_h/sbatch_l4_renorm_t2.sh`(A6000 48GB·9셀 seed-major·root `rundirs_llm_jw`). 각 셀 = `observer OBS_SOURCES=gtg,fedsv,comfedsv,shapleyfl T2=1 T2_LEGACY=1`(관찰자 궤적 + renorm-4 t2_sign 재학습; 셀당 ~20h, arm-단위 영속). §0 셋업 후:
  ```
  cd $REPO && mkdir -p runs/track_h/_logs
  sbatch --array=0-2%8 runs/track_h/sbatch_l4_renorm_t2.sh     # seed0 파일럿 3셀 → GPU-h 보고
  sbatch --array=3-8%8 runs/track_h/sbatch_l4_renorm_t2.sh     # seeds 1-2
  ```
  (env·모델=1B 전부 sbatch 내장. clean 셀 포함 = renorm 오발화로 실제 재학습.)
- **비용**: ≈ **~200–230 GPU-h**(9셀 내부 renorm-4 재학습; 8슬롯 ~28 wall-h; seed0 3셀 우선 ~10 wall-h).
- **분석**: `make_analysis.py` LLM 로더에 `rundirs_llm_jw` root 추가(dup-win) → 셀키 병합. **채우는 overview ⬚**: §5.3 R4 retrain 표 renorm-4 칸.

## 2. 우선순위·큐 운용

- **seed0 우선**: L4 seed0(3셀) 먼저 → 논문 착수선. 이후 seeds 1-2.
- **가동**: 셋업 직후 **L4 seed0**(독립·즉시) → seeds 1-2.
- **work-stealing**: L4는 셀-단위 독립·idempotent → JW 큐가 비면 HJ(L11)·JB(L9-arms)의 잔여 물량을 가져와도 무방(착지 root만 계정별 분리 후 병합). L4가 3셀×3seed로 상대적으로 가벼우니 **완주 후 HJ L11 tail 흡수 권장**(L11이 최대 물량).
- **스택 캐비엇**: A6000(torch2.11) vs canonical(B200 torch2.12) — fidelity/개입은 recovery 정규화로 읽음(W-A mean|Δ|≤0.006). **timing.json은 §5.5 cost에 사용 금지**(B200 실측만).
- **완료 판정**: `TRACK G DONE`+rundir mtime. 완료분 커밋(push는 Yonghee).

## 3. ⚠ L4 실측 비용·예상 종료 시각 (2026-07-25, JW A6000 실측)

> **§1의 "셀당 ~20h / 총 200–230 GPU-h" 추정은 과소평가.** 실측 ≈ **셀당 ~100h / 9셀 ~900 GPU-h (~4배)**.
> **`sbatch_l4_renorm_t2.sh` 의 `--time=24:00:00` 으로는 9셀 전부 산출물 0으로 소멸** — `persist()` 는 arm 완료 후에만
> 호출(`track_g.py:704`)인데 observer arm 하나가 24h를 넘김. 2026-07-25 1차 가동(6셀 실행/3셀 대기)은 이 사유로 취소함.

**실측 근거** (2026-07-25, 6셀 · node26/node45 · 최장 3.7h · 라운드당 train-block 5개 기준)

| 항목 | 실측/근거 | R=200 환산 |
|---|---|---|
| observer arm 라운드당 (renorm-4 스코어링) | clean 21.7분 · noisy 19.7분 · frzero 29.4분 | **66–98 h** |
| T2 재학습 라운드당 (스코어링 없음) | 클라 5 × ~22초 ≈ 1.8분 | ~6.1 h/arm × 4 = **~24 h** |
| downstream (gsm8k EM 생성) | test = **1,119문항** 고정, **arm마다** 실행(셀당 5회) | 측정중 |
| val-curve | B200 589s → A6000 미측정 | 측정중 |
| **셀 합계** | | **≈ 100 h (~4.2일)** |

**과소평가 원인**: canonical B200 observer(`rundirs_llm/gsm50k5_clean_observer_seed0`)는 `obs_sources=[flirds,
flirds1st, lossheur, fedif]`(같은-게임 계열, 저렴 = fl+online-scoring 19,001s ≈ 5.28h/200R = 1.58분/R)로 측정된 값.
**L4는 renorm 4소스**라 라운드마다 부분집합 모델 평가(GTG 샘플링·FedSV 2⁵ 서브셋·ComFedSV·ShapleyFL)가 필요해
본질적으로 훨씬 비쌈 → A6000 실측은 B200 flirds-계열 대비 **라운드당 ~13배**. 하드웨어 차이만이 아니라 **워크로드 차이**가 주원인.

**`--time` 배당 권고**

| 대상 | 권고 `--time` | 비고 |
|---|---|---|
| L4 1셀 (observer + T2×4) | **`5-00:00:00` (5일)** | 실측 ~100h=4.2일 + 여유. QOS `base_qos` 7일까지 수락 확인(파티션 MaxTime 14일) |
| 안전 상한 | `7-00:00:00` | frzero 계열이 최장(29.4분/R) |
| 측정 전용 소형런 | `10:00:00` | ROUNDS=5 실측용 |

**예상 종료 시각** (A6000 8-GPU/user 상한 · 9셀 · 큐 대기 제외 순수 계산)

- 8셀 동시 가동 시: 제출 + **~4.2일** → 8셀 완료, 9번째 셀은 다시 +4.2일 → **총 ~8.4일**
- 즉 **07-25 재제출 기준 8셀 ≈ 07-29~30, 9셀 전부 ≈ 08-03** → **실험 마감 07-28 준수 불가**
- 큐 대기 추가 변수: A6000 98개 중 여유 10개 내외(가동률 90%) — 1차 가동 때 6셀 즉시·3셀 대기 발생

**미해결 결정 사항(Yonghee)**: ① `--time` 늘려 마감 초과 감수 ② 스코프 축소(seed0만 / R 축소=canonical 비교 불가)
③ 실행처 재배치(B200 등) — 셋 중 택일 필요. **A6000 유지 시 07-28 내 완주는 불가.**

> 참고 실측(2026-07-25 JW 계정): A6000 = 49,140 MiB · 드라이버 580.126.09. `base_qos` 접근 가능 GPU 288/435개
> (A6000 98 · RTX4090 66 · RTX3090 95 · A5000 21 · RTX6000ADA 8). **불가**: RTXPRO6000 32 · a100 24 · RTX3090 91(big_suma).
