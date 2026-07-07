# β=0.3 통일 재실행 — 서버 이전 후 재개 노트  (PAUSED 2026-07-07)

ShapleyFL(surrogate FSV)의 cross-round EMA 계수 β를 논문값 **0.3**(Sun et al. KDD'23, Def 4.3;
과거 임시 기본값 0.5→0.3)으로 프로젝트 전역 통일하는 재실행. 서버 환경 이전으로 **중단**했다.

## 상태
- **완료(β=0.3, 커밋됨)**: `track_d` 1B·3B (std20·anchor5, 각 3-seed) + `track_c` CNN (c1 30 + c2 90).
  - 3B 반영 커밋 = `b1b95d0`. 1B·CNN은 이전 커밋들에서 β=0.3(값 동일/β-불변).
- **대기(아직 β=0.5)**: **7B `track_d` ×6 + `phase2_matrix` ×25 = 31셀** (아래 목록).
- **결과 문서** `research-wiki/survey/flirds-experiment-results-overview-2026-06-25.md`:
  1B/3B/CNN은 반영됨. **7B 열(§3.1.1 fidelity·§3.5 runtime)과 phase2 ShapleyFL 행(§3.4 AUROC/Spearman)은 아직 β0.5** → 이 31셀 완료 후 갱신할 것.

## 재개 방법 (이전 후)
- 환경: `python=/home/korea_bupj/miniconda3/envs/flirds/bin/python`, `codes/`에서 `PYTHONPATH=.`, GPU는 이전 후 재확인.
- 드라이버(tracked, 이전 후에도 존재): `runs/rerun_beta03/run_multi_driver.sh` = GPU 슬롯 스케줄러
  (큐 각 줄 `script|run_name|envs`; done-marker `MATRIX DONE|TRACK D DONE|[persist]`로 완료 감지;
  `GPUS_FILE` 파일 편집으로 재시작 없이 GPU 추가/제거).
- 큐 파일은 gitignored(`logs/`)라 이전 시 소실 가능 → **아래 31줄로 재생성**:
  ```bash
  cd <repo>/runs/rerun_beta03
  { echo "# beta0.3 resume: 31 pending cells"; sed -n '/^phase2\|^track_d/p' RESUME_AFTER_MIGRATION.md; } > logs/resume.txt   # 또는 아래 목록을 직접 붙여넣기
  echo "0 1 2" > logs/gpus.txt                 # 쓸 GPU
  QUEUE=logs/resume.txt GPUS_FILE=logs/gpus.txt LOGDIR=logs GPUS="0 1 2" \
    bash run_multi_driver.sh >> logs/_driver.log 2>&1 &
  ```
- 대안(목록 유실 시): `python build_queue.py`로 전체 163셀 큐 재생성 → `7B_*`·`phase2_matrix.py` 줄만 추림
  (1B/3B/CNN은 이미 완료이니 재실행 불필요; RunLogger는 같은 이름 rundir을 덮어씀).

## 31 대기 셀  (cheap→expensive; 그대로 `script|run_name|envs`)
```
phase2_matrix.py|1B_silo5_noisy|REGIME=silo5 THREAT=noisy
phase2_matrix.py|1B_silo5_frrand|REGIME=silo5 THREAT=freerider_random
phase2_matrix.py|1B_silo5_frzero|REGIME=silo5 THREAT=freerider_zero
phase2_matrix.py|1B_silo5_poison|REGIME=silo5 THREAT=poison LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=0.8
phase2_matrix.py|1B_device100-a0.1_noisy|REGIME=device100 ALPHA=0.1 THREAT=noisy POOL=12000
phase2_matrix.py|1B_device100-a0.1_frrand|REGIME=device100 ALPHA=0.1 THREAT=freerider_random POOL=12000
phase2_matrix.py|1B_device100-a0.1_frzero|REGIME=device100 ALPHA=0.1 THREAT=freerider_zero POOL=12000
phase2_matrix.py|1B_device100-a0.01_noisy|REGIME=device100 ALPHA=0.01 THREAT=noisy POOL=12000
phase2_matrix.py|1B_device100-a0.01_frrand|REGIME=device100 ALPHA=0.01 THREAT=freerider_random POOL=12000
phase2_matrix.py|1B_device100-a0.01_frzero|REGIME=device100 ALPHA=0.01 THREAT=freerider_zero POOL=12000
phase2_matrix.py|1B_device100-a0.0_noisy|REGIME=device100 ALPHA=0.0 THREAT=noisy POOL=12000
phase2_matrix.py|1B_device100-a0.0_frrand|REGIME=device100 ALPHA=0.0 THREAT=freerider_random POOL=12000
phase2_matrix.py|1B_device100-a0.0_frzero|REGIME=device100 ALPHA=0.0 THREAT=freerider_zero POOL=12000
phase2_matrix.py|1B_device100-a5.0_noisy|REGIME=device100 ALPHA=5.0 THREAT=noisy POOL=12000
phase2_matrix.py|1B_device100-a5.0_frrand|REGIME=device100 ALPHA=5.0 THREAT=freerider_random POOL=12000
phase2_matrix.py|1B_device100-a5.0_frzero|REGIME=device100 ALPHA=5.0 THREAT=freerider_zero POOL=12000
phase2_matrix.py|1B_device100-a0.5_poison|REGIME=device100 ALPHA=0.5 THREAT=poison POOL=12000 LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=0.8 ROUNDS=60 MAX_STEPS=10
phase2_matrix.py|1B_device100-a0.0_poison|REGIME=device100 ALPHA=0.0 THREAT=poison POOL=12000 LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=0.8 ROUNDS=60 MAX_STEPS=10
phase2_matrix.py|3B_silo5_noisy|SMOKE_MODEL=meta-llama/Llama-3.2-3B-Instruct REGIME=silo5 THREAT=noisy SEED=0 ORACLE_B=1 COALITION=0
phase2_matrix.py|3B_silo5_frrand|SMOKE_MODEL=meta-llama/Llama-3.2-3B-Instruct REGIME=silo5 THREAT=freerider_random SEED=0 ORACLE_B=1 COALITION=0
phase2_matrix.py|3B_silo5_frzero|SMOKE_MODEL=meta-llama/Llama-3.2-3B-Instruct REGIME=silo5 THREAT=freerider_zero SEED=0 ORACLE_B=1 COALITION=0
phase2_matrix.py|3B_silo5_poison|SMOKE_MODEL=meta-llama/Llama-3.2-3B-Instruct REGIME=silo5 THREAT=poison SEED=0 ORACLE_B=1 COALITION=0 LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=0.8
phase2_matrix.py|1B_device100-a0.5_noisy_anchor|REGIME=device100 ALPHA=0.5 THREAT=noisy POOL=12000 ORACLE_B=1 COALITION=1
phase2_matrix.py|1B_device100-a0.5_frrand_anchor|REGIME=device100 ALPHA=0.5 THREAT=freerider_random POOL=12000 ORACLE_B=1 COALITION=1
phase2_matrix.py|1B_device100-a0.5_frzero_anchor|REGIME=device100 ALPHA=0.5 THREAT=freerider_zero POOL=12000 ORACLE_B=1 COALITION=1
track_d.py|7B_std20_seed0|SMOKE_MODEL=meta-llama/Llama-2-7b-hf REGIME=std20 SEED=0 ORACLE_A=0
track_d.py|7B_std20_seed1|SMOKE_MODEL=meta-llama/Llama-2-7b-hf REGIME=std20 SEED=1 ORACLE_A=0
track_d.py|7B_std20_seed2|SMOKE_MODEL=meta-llama/Llama-2-7b-hf REGIME=std20 SEED=2 ORACLE_A=0
track_d.py|7B_anchor5_seed0|SMOKE_MODEL=meta-llama/Llama-2-7b-hf REGIME=anchor5 SEED=0 ORACLE_A=0
track_d.py|7B_anchor5_seed1|SMOKE_MODEL=meta-llama/Llama-2-7b-hf REGIME=anchor5 SEED=1 ORACLE_A=0
track_d.py|7B_anchor5_seed2|SMOKE_MODEL=meta-llama/Llama-2-7b-hf REGIME=anchor5 SEED=2 ORACLE_A=0
```

## 주의
- **poison 셀**: `LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=0.8`(백도어 설치 조건 — 빼면 ASR=0). device100 poison은 추가 `ROUNDS=60 MAX_STEPS=10`.
- **7B** = `meta-llama/Llama-2-7b-hf`, `ORACLE_A=0`(7B는 (a)-retrain oracle 없음). batch 등은 track_d 내부 7B 설정 사용.
- 순서: 1B silo5(4, 저비용) → 1B device100 sweep(14) → device100 poison(2) → 3B silo5(4) → device100 anchor(3, 2ᴺ per-round 오라클=비쌈) → **7B track_d(6, 가장 무거운 꼬리)**. 2-GPU ~5–7일, 3-GPU ~4–5일(러프).

## 완료 후 (문서 갱신)
1. 31셀 rundir 커밋.
2. overview 문서 β0.3 반영: **§3.1.1 7B fidelity 열(순위+거리)** + **§3.5.1 7B runtime** + **§3.4 phase2 ShapleyFL AUROC/Spearman 행**.
   재집계: `python runs/track_d/make_fidelity.py`(7B) / `python runs/phase2_matrix/make_analysis.py`(phase2).
