# REMAINING (Slurm · HJ 계정) — A6000 48GB: silo5-a + L11 online

> 실행처별 인수인계 **5-서버 분할** 중 **HJ**(신규 계정) 몫. 짝 = `REMAINING-b200.md`(HVP 전용)·`REMAINING-slurm-YH.md`(CNN)·`REMAINING-slurm-JW.md`(L4 renorm T2)·`REMAINING-slurm-JB.md`(L9 비-flirds arms).
> **역할 = A6000 48GB에서 (1) silo5-a (a)-leg**(24GB OOM → 48GB 해결) **+ (2) L11 seed0·1**(R4 §5.3 online 7 비-flirds, 42런; **seed2=YH로 분할 = 부하균형**). 둘 다 **B200 독립**(cum 대기 없이 즉시 가동).
> **마감: 실험 07-28 / 논문 07-29 21:00** — seed0 우선(`REMAINING-b200.md` §1a). push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만.
> 논문·문서 정본 = `paper/workplan/00-INDEX.md`. 기존 rundir은 read-only.

## 0. 신규 계정 셋업 (최초 1회 · 실행 전 체크리스트)

HJ는 지금까지 안 쓰던 계정 → 아래 4개가 준비돼야 잡이 돈다. (YH 계정 자산 재사용이 최단 경로.)

1. **repo**: `/home/<HJ>/projects/flirds/` (git clone 또는 공유 `/home` 마운트). 이 3개 REMAINING + `codes/` 접근.
2. **conda env**: `lora4cl`(torch 2.11) — 공유 `/home/chyoyhr/anaconda3/envs/lora4cl`가 읽히면 그대로 `$PY` 지정, 아니면 동일 스펙 재생성.
3. **HF 캐시**(offline): model(Llama-3.2-1B-Instruct)+**5도메인 데이터셋**(silo5)+gsm8k. YH의 `/scratch/chyoyhr/hf_home` 공유-읽기 가능하면 `HF_HOME`으로 지정, 아니면 그 캐시를 HJ scratch로 **복제**. 런타임 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.
4. **QOS/파티션**: `base_qos`로 A6000 접근(모든 파티션 AllowAccounts=ALL). 파티션 = `suma_a6000` 또는 `gigabyte_a6000`(48GB). 동시 상한 **8-GPU/user**(QOSMaxGRESPerUser) → HJ 단독 8슬롯.
5. 공통: `codes/`에서 `PYTHONPATH=.`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

> ⚠ silo5-a가 24GB(3090)서 OOM 난 이유 = **1B full SFT 재학습이 batch16서 ~25GB peak**(trl `compute_loss` 활성서 ~1GB 초과). **batch는 줄이면 안 됨**(재학습 궤적이 바뀌어 (a) 게임이 달라짐 = canonical 프로토콜 불일치) → **48GB 필수**. 48GB선 기본 knob으로 여유.

## 1. silo5-a (a)-leg — ★활성 (유일 (a)-무대; B200 독립·즉시 가동)

> 스펙 = `paper/workplan/T5-retrain-a-suite.md` §2. **gsm5·anchor5는 보류**(silo5 단독 (a)-오라클; 근거 = 그 문서 §1 배너·`REMAINING-b200.md` §4). 코드 커밋 완료·TRACKED.

- **무엇**: 기존 canonical `1B_silo5_{clean,noisy,frzero}` rundir과 **동일 split·seed** 재현 → (a) 32-retrain(R=10) **오라클만 신규**. estimator HVP 없음(재학습 + no_grad val-loss).
- **셀**: {clean, noisy, frzero} × seed{0,1,2} = **9 leg**. seed-major(array 0-2 = seed0 우선).
- **전제**: canonical `runs/phase2_matrix/rundirs/1B_silo5_{clean,noisy,frzero}` 3개 존재 확인됨(read-only) + 5도메인 HF캐시(§0-3).
- **실행(sbatch — REMAINING만 보고 실행)**: `runs/phase2_matrix/sbatch_silo5_a.sh`(A6000 48GB·qos base_qos·9셀 seed-major·%8). §0 셋업(repo·lora4cl·HF캐시) 후:
  ```
  cd $REPO && mkdir -p runs/phase2_matrix/_logs
  sbatch runs/phase2_matrix/sbatch_silo5_a.sh                 # 9 leg 전부 (seed0 = array 0-2 먼저)
  ```
  - env는 sbatch에 내장: `HF_HUB_OFFLINE=1`·`HF_HOME`(기본 YH scratch, 안 읽히면 `HF_HOME=<내 캐시>` 오버라이드)·`expandable_segments`·모델=1B(SMOKE_MODEL 미설정). `PY`/`REPO`도 `${VAR:-기본}`이라 다르면 `PY=… sbatch …`.
  - 48GB선 기본 knob(batch16·val_chunk10·maxlen768) 그대로. 완료 = 로그 `SILO5 (a)-LEG DONE`.
  - 만약 48GB도 빠듯하면(예상 안 됨) `VAL_CHUNK=3 VAL_MAXLEN=64`만 낮춤(val-loss 전용·**φ 불변**; batch는 불변 유지).
- **산출**: `runs/phase2_matrix/rundirs/1B_silo5_{threat}_aonly_s{seed}`.
- **비용**: ≈ 2.9 GPU-h/leg(anchor5 단가) → 9 leg ≈ **~26 GPU-h**(8슬롯 ~4 wall-h).

### 조인·분석 (silo5-a 완료 후)
```
PYTHONPATH=. $PY runs/phase2_matrix/merge_silo5_a.py     # canonical ⋈ *_aonly → spearman_a/pearson_a + silo5_a_fidelity_1B.csv
```
- **헤드라인(T5 §2)**: silo5 `(b)oracle` 행 `rho_a` = 두 오라클(a↔b)의 **실재 신호 일치**(목표 clean +0.87 / noisy +0.93; overview §5.4).
- **방법 범위**: (a) retrain 표 = 전 방법(renorm-4+FedIF 포함) 산출(무해) · **표기는 vs(b) 열 = same-game 3만**(flirds·flirds1st·lossheur).
- **스택 캐비엇**: A6000(torch2.11) vs canonical(B200 torch2.12) — fidelity(Spearman·recovery)는 stack-robust(W-A mean|Δ|≤0.006). (a)-leg는 재학습 오라클이라 하드웨어 독립 = φ·(a)-vs-(b) 유효. **timing.json은 §5.5 cost에 사용 금지**.
- **채우는 overview ⬚**: §5.2 sub silo5 (a)-leg → paper §5.2 sub(T1)·T2 F2.

## 2. L11 — R4 §5.3 online 완성 (7 비-flirds; **HJ=seed0·1, YH=seed2**)

> `REMAINING-b200.md` §2 L11의 실행처 = **HJ 48GB(seed0·1) + YH(seed2)**. 최대 물량(63런)을 seed로 분할해 HJ 과부하 방지 — HJ가 우선 seed0·1(42런), YH가 CNN 완주 후 seed2(21런·`REMAINING-slurm-YH.md` §5). R4 §5.3 online 표는 CNN처럼 8방법인데 현재 online=flirds만(B200 L1) → 나머지 **7방법 T1 online**을 채운다. (retrain 8방법 = L1 4 exact-0 + L4 renorm-4로 별도 완성.)

- **무엇(HJ 몫)**: 7 비-flirds(flirds1st·lossheur·fedif·gtg·fedsv·comfedsv·shapleyfl) T1 부호-게이트 online × {clean,noisy,frzero} × **seed{0,1}** = **42 run**(seed2 21런 = YH §5). 비-flirds는 online 스코어링에 **HVP 불요**(값·1차) → **retrain-scoring ~32 GiB → 48GB**, B200 cum 불요·자체완결.
- **실행(sbatch — REMAINING만 보고 실행)**: `runs/track_h/sbatch_l11_online.sh`(A6000 48GB·63셀 seed-major·root는 seed로 자동 라우팅 seed2→YH·그외→HJ). 7소스={flirds1st,lossheur,fedif,gtg,fedsv,comfedsv,shapleyfl} 각 `<src>_gate_v2`(자체 인라인 스코어). §0 셋업 후:
  ```
  cd $REPO && mkdir -p runs/track_h/_logs
  sbatch --array=0-41%8 runs/track_h/sbatch_l11_online.sh      # HJ = seed0·1 (42런; seed0 = 0-20 먼저)
  ```
  (seed2 21런 = YH가 `--array=42-62%8` 제출 = `REMAINING-slurm-YH.md` §5. env·모델=1B 전부 sbatch 내장.)
- **분모 의존(런타임 아님·분석 시)**: 각 셀 recovery 분모(vanilla/oracle_excl/random_excl)는 B200 L1(clean·noisy·frzero 관찰자+online)이 같은 셀키로 산출 → make_analysis 병합. 잡 자체는 B200 대기 없이 즉시 실행.
- **비용(HJ 몫)**: ≈ 4–4.8 GPU-h/run → 42런(s0·1) ≈ **~184 GPU-h**(8슬롯 ~23 wall-h; seed0 21런 ~13 wall-h). +silo5-a 26 → **HJ 계 ~210**.
- **분석**: `runs/track_h/make_analysis.py` LLM 로더에 `rundirs_llm_hj` root 추가(track_h dup-win) → B200 산출과 셀키 병합. **스택 캐비엇 = recovery 정규화**(위 §1과 동일). **timing cost 금지**.
- **채우는 overview ⬚**: §5.3 R4 online 표의 7 비-flirds 행.

## 3. 우선순위·큐 운용

- **seed0 우선**: silo5-a seed0(3 leg) + L11 seed0(21런)을 먼저 → 논문 착수선. 이후 seeds 1-2.
- **가동 순서 제안**: 셋업(§0) 직후 **silo5-a 9 leg**(~4 wall-h, 즉시·독립) → 빈 슬롯에 **L11 seed0 21런** 착수(독립) → **L11 seed1 21런**(seed2는 YH).
- **work-stealing**: L11·silo5-a는 arm-level 독립·idempotent → HJ 큐가 비면 JW/JB/YH의 독립 물량을 가져와도 무방(착지 root만 계정별 분리 후 make_analysis에서 병합). **역으로 L11이 최대 물량이라, JW/JB/YH가 먼저 비면 HJ L11 tail을 흡수**해 균형.
- **완료 판정**: silo5-a=`SILO5 (a)-LEG DONE` / L11=`TRACK G DONE`+rundir mtime. 완료분 커밋(push는 Yonghee).
