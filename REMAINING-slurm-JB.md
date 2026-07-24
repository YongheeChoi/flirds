# REMAINING (Slurm · JB 계정) — A6000 48GB: L9 frrand 비-flirds arms

> 실행처별 인수인계 **5-서버 분할** 중 **JB**(신규 계정) 몫. 짝 = `REMAINING-b200.md`(HVP 전용)·`REMAINING-slurm-YH.md`(CNN)·`REMAINING-slurm-HJ.md`(silo5-a·L11)·`REMAINING-slurm-JW.md`(L4 renorm T2).
> **역할 = A6000 48GB에서 L9 frrand 비-flirds arms**(7방법 × {T1,T2} × 3seed). **자체완결 = B200 cum 독립**(즉시 가동). flirds arm은 B200 관찰자와 묶여 B200; 나머지 7방법 arm = 여기.
> **마감: 실험 07-28 / 논문 07-29 21:00** — seed0 우선. push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만.
> 논문·문서 정본 = `paper/workplan/00-INDEX.md`. 기존 rundir은 read-only.

## 0. 신규 계정 셋업 (최초 1회 · HJ/JW와 동일 체크리스트)

1. **repo**: `/home/<JB>/projects/flirds/`(clone 또는 공유 `/home`).
2. **conda env**: `lora4cl`(torch 2.11) — 공유 재사용 or 재생성.
3. **HF 캐시**(offline): model(Llama-3.2-1B-Instruct)+gsm8k. `HF_HOME` = YH `/scratch/chyoyhr/hf_home` 공유-읽기 or 복제. `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.
4. **QOS/파티션**: `base_qos` → A6000 48GB(`suma_a6000`/`gigabyte_a6000`). 동시 **8-GPU/user**.
5. 공통: `codes/`에서 `PYTHONPATH=.`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

> **왜 48GB**: L9 비-flirds arms = **retrain-scoring 클래스 ~32 GiB**(값 산출 + T2 재학습; `REMAINING-b200.md` §1 실측) → **24GB 불가·48GB 필요**. HVP는 없음(비-flirds는 값·1차 스코어).

## 1. L9 arms — R4 frrand 비-flirds (자체완결·B200 독립)

> `REMAINING-b200.md` §3 L9(frrand full-8)의 **비-flirds arm 몫**. R4(gsm50k5)의 free-rider 축은 frzero만 있고 frrand 전무 → full-method로 완성해 "exact-0 생존 vs renorm 붕괴"를 random free-rider에서도 시연(frzero 대칭화).

- **무엇**: frrand(zero 대신 무작위 free-rider) × 7 비-flirds(same-game flirds1st·lossheur + FedIF + renorm-4 gtg·fedsv·comfedsv·shapleyfl) × {T1 online, T2 retrain} × seed{0,1,2}. `track_g build_arm`이 매 실행 fresh 누적기로 **자체 인라인 스코어**(HVP 없음·저장 cum 미로드) → B200 독립.
- **러너·명령**(A6000 1장/셀; **착지 root = JB 전용**, canonical `rundirs_llm` 무수정):
  ```
  RUNDIR_ROOT=$REPO/runs/track_h/rundirs_llm_jb \
  REGIME=gsm50k5 THREAT=frrand SEED=<0|1|2> \
    ARMS=<비-flirds gate/T2 arm 세트> T2=<0|1> T2_LEGACY=0 T2_P5=0 \
    PYTHONPATH=. $PY -u experiments/track_g.py
  ```
  (arm 세트 = HJ L11과 동형·threat만 frrand — Yonghee 확정 후 sbatch array 전개. **seed0 우선**. B200 L9 flirds 관찰자와 make_analysis에서 셀키 병합.)
- **비용**: ≈ **~150–170 GPU-h**(7방법×{T1,T2}×3seed; 8슬롯 ~20 wall-h; seed0 ~7 wall-h).
- **분석**: `make_analysis.py` LLM 로더에 `rundirs_llm_jb` root 추가(dup-win) → 셀키 병합. **채우는 overview ⬚**: §5.3 R4 frrand 열(비-flirds+retrain) + §5.4 frrand 탐지 AUROC.

## 2. 우선순위·큐 운용

- **seed0 우선**: L9-arms seed0 먼저 → 논문 착수선. 이후 seeds 1-2.
- **가동**: 셋업 직후 **L9-arms seed0**(독립·즉시) → seeds 1-2.
- **work-stealing**: L9-arms는 arm-level 독립·idempotent → JB 큐가 비면 HJ(L11)의 잔여 물량을 가져와도 무방(착지 root만 계정별 분리 후 병합). L9-arms가 상대적으로 가벼우니 **완주 후 HJ L11 tail 흡수 권장**(L11이 최대 물량).
- **스택 캐비엇**: A6000(torch2.11) vs canonical(B200 torch2.12) — fidelity/개입은 recovery 정규화로 읽음(W-A mean|Δ|≤0.006). **timing.json은 §5.5 cost에 사용 금지**(B200 실측만).
- **완료 판정**: `TRACK G DONE`+rundir mtime. 완료분 커밋(push는 Yonghee).
