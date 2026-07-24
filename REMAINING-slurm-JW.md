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
- **러너·명령**(A6000 1장/셀; **착지 root = JW 전용**, canonical `rundirs_llm` 무수정):
  ```
  RUNDIR_ROOT=$REPO/runs/track_h/rundirs_llm_jw \
  REGIME=gsm50k5 THREAT=<clean|noisy|frzero> SEED=<0|1|2> \
    ARMS=<renorm T2 arm 세트> T2=1 T2_LEGACY=0 T2_P5=0 \
    PYTHONPATH=. $PY -u experiments/track_g.py
  ```
  (정확한 renorm-4 T2 arm 라벨·OBS_SOURCES = B200 L4 큐와 동일 — Yonghee 확정 후 sbatch array 전개. **seed0 3셀 우선**.)
- **비용**: ≈ **~200–230 GPU-h**(9셀 내부 renorm-4 재학습; 8슬롯 ~28 wall-h; seed0 3셀 우선 ~10 wall-h).
- **분석**: `make_analysis.py` LLM 로더에 `rundirs_llm_jw` root 추가(dup-win) → 셀키 병합. **채우는 overview ⬚**: §5.3 R4 retrain 표 renorm-4 칸.

## 2. 우선순위·큐 운용

- **seed0 우선**: L4 seed0(3셀) 먼저 → 논문 착수선. 이후 seeds 1-2.
- **가동**: 셋업 직후 **L4 seed0**(독립·즉시) → seeds 1-2.
- **work-stealing**: L4는 셀-단위 독립·idempotent → JW 큐가 비면 HJ(L11)·JB(L9-arms)의 잔여 물량을 가져와도 무방(착지 root만 계정별 분리 후 병합). L4가 3셀×3seed로 상대적으로 가벼우니 **완주 후 HJ L11 tail 흡수 권장**(L11이 최대 물량).
- **스택 캐비엇**: A6000(torch2.11) vs canonical(B200 torch2.12) — fidelity/개입은 recovery 정규화로 읽음(W-A mean|Δ|≤0.006). **timing.json은 §5.5 cost에 사용 금지**(B200 실측만).
- **완료 판정**: `TRACK G DONE`+rundir mtime. 완료분 커밋(push는 Yonghee).
