# 항목 1 Taylor 잔차 실측 — 1B 본실행 커맨드 (준비 완료; 실행은 GPU 비는 시점에 별도 주체가)

- 스크립트: `research-wiki/survey/irds-fl-math-rigor-2026-07/measure_taylor_residual.py`
  (원격 사본: `/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds_verify_scratch/taylor/measure_taylor_residual.py`)
- 무대: 2026-06-06 valuation-baseline과 동일 — Llama-3.2-1B-Instruct fp32+eager, LoRA r16 α32,
  silo5 N=5(1 domain/client), R=10, per-domain train=200, val=100(=20/domain), lr=1e-3,
  max_steps=10, batch=16, maxlen=768, val_maxlen=384, val_chunk=10 — 전부 스크립트 기본값.
- gpt2 CPU 스모크 검증: 본 문서와 같은 폴더의 `gpt2_smoke_weakdelta_summary.json` 참조.

## 실행 커맨드 (원격, GPU 1장 — `<X>`를 빈 GPU 번호로)

```bash
ssh "[tmp]korea_bupj" 'cd /NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/codes && CUDA_VISIBLE_DEVICES=<X> OMP_NUM_THREADS=16 PYTHONPATH=. nohup nice -n 10 /home/korea_bupj/miniconda3/envs/flirds/bin/python -u /NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds_verify_scratch/taylor/measure_taylor_residual.py --model meta-llama/Llama-3.2-1B-Instruct --device cuda --rounds 10 --val_size 100 --seed 0 --renorm --check_inrun --outdir /NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds_verify_scratch/taylor/llama1b_r10_seed0 > /NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds_verify_scratch/taylor/llama1b_r10_seed0.log 2>&1 &'
```

- `--renorm` = P5 재정규화 게임 ũ 부가 측정(라운드당 forward 2배). `--check_inrun` = P1 수치
  확인(in_run_shapley 2^N 직접 호출 대조; forward 추가 ~2^5×R회). 시간 아끼려면 이 둘을 빼면
  forward 비용이 약 1/3로 줆 (핵심 잔차 측정에는 불필요하지 않음 — P5/P1 실측이 목적이면 유지).
- 산출물: `llama1b_r10_seed0/{coalitions.csv, coalitions.parquet, phi.csv, summary.json}` + 로그.

## 예상 소요 (B200 1장, fp32)

06-06 baseline 실측치( (b)oracle 2^5 exact ≈531s, Flirds estimator ≈107s, Flirds-1st ≈35s )로 외삽:

| 단계 | 비용 | 추정 |
|---|---|---|
| FL 궤적 (R=10, 5클라×10스텝×batch16) | 학습 500 스텝 | ~15–25분 |
| u_true forward (31 S×10 라운드) | ≈ (b)oracle 1회분 | ~9분 |
| renorm forward (31×10) | 동일 | ~9분 |
| 클라별 HVP (5/라운드×10 = 50회) | ≈ estimator 5배 | ~9분 |
| flirds_values 직접 호출 (2차+1차) | 107+35s | ~2.5분 |
| check_inrun (2^5 oracle 재계산) | ≈531s | ~9분 |
| **합계** | | **~55–75분** |

- 메모리: fp32 1B(~4.5GB) + LoRA + eager-HVP 청크(val_chunk=10, val_maxlen=384) — phase2 1B
  estimator와 동일 프로파일, **peak ~20–30GB** (B200 180GB 여유). logs는 LoRA-only라 RAM 부담 미미.
- 주의: 다른 캠페인이 GPU0-3 점유 중이면 실행 금지 (이 실측은 fresh FL run이 필요 — logs 미영속,
  감사노트 §9). 산출물은 flirds_verify_scratch/ 아래에만 쓰며 저장소를 건드리지 않음.
