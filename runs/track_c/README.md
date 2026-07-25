# Track C — CNN 표준-세팅 비교 실험 결과 (plan §3.11)

2026-06-12 제출 → 2026-06-15 완료. 150 셀, 실패 0. 3 seeds (0,1,2).
환경: conda `lora4cl` (torch 2.11+cu130), RTX 3090 (yonsei Slurm). SGD mom=0.

이 디렉토리는 **불변 결과 보존**(phase2_matrix/rundirs 관례와 동일; 명명·배치 규칙은 `runs/README.md`).
러너(`track_c{1,2}.py`)는 기본값으로 `runs/track_c/{c1,c2}`에 바로 쓴다(repo-루트 상대;
`C1_RUN_ROOT`/`C2_RUN_ROOT`로 변경 가능).

## 레이아웃

| 경로 | 셀 | 내용 |
|---|---|---|
| `c1/` | 30 (2 ds × 5 scen × 3 seed) | C1 fidelity: frozen-traj + 11 method φ (config/meta/metrics/phi.parquet) |
| `c1/` (축 그리드) | 48 (2 ds × 2 part × 4 threat × 3 seed) | **2026-07-25 신규**: 아래 "축 그리드" 참조 |
| `c1_oracle/` | 30 | C1 (a) 2¹⁰ retrain oracle (`*_aonly_seed*`, phi_a만; efficiency-gap ≤1e-15) |
| `c2/` | 90 | C2 개입: 8/6 arm + dismissal q-sweep + strength sweep (config/meta/metrics) |
| `RESULTS.txt` | — | C3 c1 stability + (a)-oracle fidelity + C3 c2 outcome 합본 |

### 축 그리드 — `c1/` 안에 레거시와 **공존**한다 (2026-07-25)

레거시 5-시나리오는 논문 확정 오염축 3종(lf@0.70·free-rider-zero·grad-noise)과 한 칸도
겹치지 않는다 → `{cifar10(본문 G2), mnist(부록 G9)} × {iid, dir1} × 4위협 × 3seed = 48셀`을
같은 루트에 추가했다 (제출 = `sbatch_c1_axis.sh`, 러너 env = `C1_PARTITION`/`C1_THREAT`).

- **이름 충돌 없음**: 레거시 `{ds}_{scen}_seed{k}` vs 축 `{ds}_{part}_{ttag}_seed{k}`.
  기존 rundir 은 하나도 덮이지 않는다(read-only 규칙 준수).
- **(a) 오라클이 셀 안에 있다**: 축 셀은 `C1_MODE=full C1_ORACLE_A=1` 이라 `metrics.json` 의
  `oracle_a.phi` 에 (a) 가, `methods.*` 에 (b)+9방법이 함께 들어간다 — 레거시처럼
  `c1_oracle/` 로 분리돼 있지 않다. **집계 경로가 갈리는 이유가 이것이다.**
- **기존 툴 영향 없음**:
  `make_figures.py` 는 레거시 시나리오명이 하드코딩돼 있어 축 셀을 아예 읽지 않는다.
  `track_c3.py` 는 `_seed<k>` 만 떼어 그룹을 만들므로 축 셀은 **새 그룹**(`cifar10_iid_clean`
  등)으로 붙고 레거시 그룹 수치는 움직이지 않는다.
  `runs/track_g/phi_sign_audit.py` 는 `phi.*` 를 재귀 glob 하므로 자동 흡수된다(§3.4 CNN 레그).
- **위협 주입 규약**: label_flip 은 데이터 레벨(`client_loaders`), free_rider(zero)·grad_noise 는
  업데이트 레벨(`delta_transform`). 후자는 **(a) 재학습에도 같은 seam 으로 주입**된다 —
  안 그러면 (a) 가 위협을 학습으로 지워 버려 추정기와 **다른 게임**의 오라클이 된다.
  Ripple 은 자기 궤적을 다시 돌려 drop-utility 를 만들 뿐 주입 seam 이 없어
  업데이트-레벨 위협을 관측할 수 없다 → 해당 셀에서 **스킵**(`ripple_skipped: true`).
- 오염 클라 = **고정 개수** `round(C1_MAL_FRAC × N)` = 4/10 (기본 0.4, seed 별 고정 추출).
  track_c2 의 FedCorr Bernoulli 추출을 쓰지 않은 이유: 도즈가 0.70 으로 고정돼 FedCorr 재현이
  아니고, N=10 에서 Bernoulli 는 seed 마다 2–6 으로 흔들려 위협 간 비교가 깨진다.

C1=MNIST+LeNet5 / CIFAR-10+FedSVCNN, N=10 full. C2=CIFAR-10+FMNIST, N=100 C=0.1 T=120.

## 재현 (codes/ 에서)

```bash
PYTHONPATH=. python experiments/track_c3.py c1     # (b) fidelity cross-seed stability (기본 root=runs/track_c/c1)
PYTHONPATH=. python experiments/track_c3.py c2     # 개입 outcome mean/std (기본 root=runs/track_c/c2)
python slurm/scripts/merge_oracle_a.py             # C1 fidelity (a)+(b) × Spearman+Pearson → runs/track_c/fidelity.csv (standalone)
```

축 그리드(레포 루트에서):

```bash
python runs/track_c/c1/make_analysis.py            # 축 48셀 → runs/track_c/c1/analysis/{README.md,*.csv}
```

주의: `track_c3.py`는 `_aonly` 디렉토리(methods 키 없음)와 섞이면 KeyError → C1 traj와 oracle은
별도 루트로 분리 필수(여기선 `c1/` vs `c1_oracle/`).

## 실행 인프라

`slurm/` (루트): `scripts/run_array.sbatch` (제네릭 array 러너) + `grids/*.txt` (셀→env 매핑,
실제 실행 provenance) + `scripts/{probe_oracle_a,merge_oracle_a}.py` + `status.sh`. 로그는 gitignore.

타임아웃 메모: CIFAR C1 traj는 Ripple이 1.8–4.0h로 변동 → `-t ≥ 9h` 필요
(label_skew_seed2가 4.5h 한도 초과 1회 → 재제출 통과).
