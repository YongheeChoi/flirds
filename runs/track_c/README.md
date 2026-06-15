# Track C — CNN 표준-세팅 비교 실험 결과 (plan §3.11)

2026-06-12 제출 → 2026-06-15 완료. 150 셀, 실패 0. 3 seeds (0,1,2).
환경: conda `lora4cl` (torch 2.11+cu130), RTX 3090 (yonsei Slurm). SGD mom=0.

이 디렉토리는 **불변 결과 보존**(phase2_matrix/rundirs 관례와 동일). 실행은
`codes/runs/track_c{1,2}` scratch(gitignored)에서 돌고, 여기로 승격해 커밋한다.

## 레이아웃

| 경로 | 셀 | 내용 |
|---|---|---|
| `c1/` | 30 (2 ds × 5 scen × 3 seed) | C1 fidelity: frozen-traj + 11 method φ (config/meta/metrics/phi.parquet) |
| `c1_oracle/` | 30 | C1 (a) 2¹⁰ retrain oracle (`*_aonly`, phi_a만; efficiency-gap ≤1e-15) |
| `c2/` | 90 | C2 개입: 8/6 arm + dismissal q-sweep + strength sweep (config/meta/metrics) |
| `RESULTS.txt` | — | C3 c1 stability + (a)-oracle fidelity + C3 c2 outcome 합본 |

C1=MNIST+LeNet5 / CIFAR-10+FedSVCNN, N=10 full. C2=CIFAR-10+FMNIST, N=100 C=0.1 T=120.

## 재현 (codes/ 에서)

```bash
PYTHONPATH=. python experiments/track_c3.py c1 runs/track_c1     # (b) fidelity cross-seed stability
PYTHONPATH=. python experiments/track_c3.py c2 runs/track_c2     # 개입 outcome mean/std
PYTHONPATH=. python ../slurm/scripts/merge_oracle_a.py           # (a) 2¹⁰ retrain fidelity (spearman_a)
```

주의: `track_c3.py`는 `_aonly` 디렉토리(methods 키 없음)와 섞이면 KeyError → C1 traj와 oracle은
별도 루트로 분리 필수(여기선 `c1/` vs `c1_oracle/`).

## 실행 인프라

`slurm/` (루트): `scripts/run_array.sbatch` (제네릭 array 러너) + `grids/*.txt` (셀→env 매핑,
실제 실행 provenance) + `scripts/{probe_oracle_a,merge_oracle_a}.py` + `status.sh`. 로그는 gitignore.

타임아웃 메모: CIFAR C1 traj는 Ripple이 1.8–4.0h로 변동 → `-t ≥ 9h` 필요
(label_skew_seed2가 4.5h 한도 초과 1회 → 재제출 통과).
