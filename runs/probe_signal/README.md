# probe_signal — 신호 크기 probe (rank / 참여수 lever)

계획·근거: `research-wiki/wiki/flirds-signal-size-diagnosis.md` §2 (Yonghee 승인 2026-07-02:
seed0 파일럿 먼저, **full 11종 스위트**, ORACLE_A=0, CNN C1+C2).

## 레이아웃
- `run_pilot.sh` — LLM 1B seed0 파일럿 드라이버(nohup). GPU3: anchor r32→r64→noise-probe
  r16→r64; GPU0/1/2: std N=50/5 r16/32/64 (해당 GPU가 idle 되면 자동 시작).
- `rundirs/` — track_d 산출(1B_anchor5_r{32,64}_seed0, 1B_std50k5_r{16,32,64}_seed0, …).
  rank16 anchor는 기존 `runs/track_d/rundirs/1B_anchor5_seed*` 재사용(재실행 금지).
- `noise_probe/` — `experiments/probe_val_noise.py` 산출(val-chunk bootstrap SE, 4-i).
- `cnn_c1/`, `cnn_c2/` — yonsei SLURM 결과가 push되어 들어올 자리.
- `_logs/` — 셀 로그 + `_driver.log`.

## CNN 제출 (yonsei SLURM에서, Yonghee)
```bash
cd /home/chyoyhr/projects/flirds/slurm
sbatch -J probe_c1 -t 3:00:00 --array=0-65 scripts/run_array.sbatch grids/probe_c1.txt experiments/track_c1.py
sbatch -J probe_c2 -t 8:00:00 --array=0-29 scripts/run_array.sbatch grids/probe_c2.txt experiments/track_c2.py
```
- probe_c1 66셀 = cifar10 × {iid,label_flip} × w{0.5,1,2,4} × k{0.2,0.5,1.0} × 3seed에서
  기존 c1과 중복인 (w=1,k=1.0) 6셀 제외. Ripple 제외, (a)오라클 제외.
- probe_c2 30셀 = iid × {clean,label_flip} × [w{0.5,2,4}@f0.1 + f{0.05,0.2}@w=1] × 3seed
  ((w=1,f=0.1)은 기존 c2 strmain 재사용).
- w=4 C1 셀이 가장 무거움(연산 ~16×) → `-t 3:00:00`; C2는 T=120이라 `-t 8:00:00`.

## env 노브 (전부 기본값=현행; 이번 probe로 추가)
- `track_d.py`: `LORA_R`(16)/`LORA_ALPHA`(2r), `N_CLIENTS`/`K_ABS` override.
- `track_c1.py`: `C1_WIDTH`/`C1_KFRAC`/`C1_RIPPLE`/`C1_RUN_NAME`; ComFedSV는 kfrac<1이면 partial.
- `track_c2.py`: `C2_WIDTH`/`C2_FRAC`/`C2_RUN_NAME`.
- `models/cnn.py`: LeNet5/FedSVCNN `width=1.0` (기본 치수 동일 → bit-identical 가드 green).
