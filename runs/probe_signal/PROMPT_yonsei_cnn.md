# yonsei 세션용 프롬프트 (복사해서 새 Claude 세션에 붙여넣기)

아래 블록을 그대로 전달:

---

flirds 프로젝트의 "신호 크기 probe" CNN 실험을 이 SLURM 서버에서 제출·완주시켜줘.
배경 문서는 `research-wiki/wiki/flirds-signal-size-diagnosis.md` §2.2 (필요할 때만 참조).

## 환경
- 저장소: `/home/chyoyhr/projects/flirds` (conda `lora4cl`, 파티션 `base_suma_rtx3090`)
- 실행 관례: `slurm/scripts/run_array.sbatch` (제네릭 어레이 러너; grid 라인 = `RUN_NAME|ENVS`)
- 이 실험은 이미 B200 서버에서 코드 스모크+bit-identical 가드까지 green — 코드 수정 금지,
  제출·모니터링·검증만.

## 선행 확인
1. `git pull` 후 아래가 존재하는지 확인 (커밋 "signal-size probe" 계열, 2026-07-02):
   - `slurm/grids/probe_c1.txt` (66줄) / `slurm/grids/probe_c2.txt` (30줄)
   - `codes/experiments/track_c1.py`에 `C1_WIDTH`/`C1_KFRAC`/`C1_RIPPLE`,
     `codes/flirds/models/cnn.py`에 `width` 인자
2. 없으면 중단하고 보고 (main이 아직 push 안 된 것).

## 제출
```bash
cd /home/chyoyhr/projects/flirds/slurm
sbatch -J probe_c1 -t 3:00:00 --array=0-65 scripts/run_array.sbatch grids/probe_c1.txt experiments/track_c1.py
sbatch -J probe_c2 -t 8:00:00 --array=0-29 scripts/run_array.sbatch grids/probe_c2.txt experiments/track_c2.py
```

## 모니터링·재제출
- `squeue -u chyoyhr`, 로그 `slurm/logs/probe_c{1,2}-<jobid>_<idx>.out`
- 성공 판정: 로그 끝 `TRACK-C1 RUN OK` / `TRACK-C2 RUN OK` + `rc=0`
- 실패/TIMEOUT 인덱스만 재제출: 같은 커맨드에 `--array=<idx>`;
  c1의 w=4 셀이 3h를 넘기면 그 인덱스만 `-t 5:00:00`으로.

## 완료 검증·마무리
- `runs/probe_signal/cnn_c1/`에 66개, `runs/probe_signal/cnn_c2/`에 30개 rundir
  (각각 config.yaml+meta.json+metrics.json, c1은 +phi.parquet).
- 기존 `runs/track_c/{c1,c2}/`는 절대 건드리지 않기 (w=1 기준점은 기존 셀 재사용).
- 전부 green이면 `runs/probe_signal/cnn_c{1,2}/` rundir만 커밋
  (메시지: "probe_signal: CNN C1/C2 width·참여 sweep 결과 (66+30셀)"). push는 내가 한다.
- 요약 보고: 셀 수/실패·재제출 내역/총 소요.

---
