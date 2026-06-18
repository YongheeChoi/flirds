# runs/ — 실험 결과 디렉토리 (명명·배치 규칙)

> 모든 실험 산출물의 **단일 루트**. 이 규칙은 강제다 — 새 실험은 반드시 여기 형식을 따른다.
> (2026-06-15 통일. 이전엔 일부 러너가 CWD-상대경로를 써서 `codes/`에서 돌리면 `codes/runs/`가
> 따로 생기는 사고가 있었음 → 전 러너를 repo-루트 상대경로로 교정 + 규칙을 여기 명문화.)

## 0. 철칙

1. **결과는 무조건 repo-루트 `runs/<group>/...` 아래.** 절대 `codes/runs/`에 쓰지 않는다.
   러너는 출력 경로를 `os.path.dirname(...repo root...)/runs/...`로 잡는다 (CWD 상대경로 금지 —
   그게 폴더가 둘로 갈린 원인이었다).
2. **개별 run-dir = RunLogger 표준 4파일**: `config.yaml` + `meta.json`(git_sha/dirty/env_hash/
   버전) + `metrics.json` + `phi.parquet`. 작고 불변이라 **git 추적**(재실행 없이 재분석 가능).
3. **파생물은 gitignore**: 실행 로그(`tier*/`, `*.out`, slurm), 합본 리포트(`RESULTS.md`),
   재생성 산출물(`analysis/`). rundir만 있으면 전부 재생성되므로.

## 1. 레이아웃

```
runs/
  README.md                                ← 이 파일 (규칙)
  phase1/rundirs/<cell>                     LLM 클린 베이스라인 (N=5, 레거시·동결; 옛 codes/runs)
  phase2_matrix/
    rundirs/<cell>                          LLM 밸류에이션 본 그리드 (25셀)
    make_analysis.py / make_report.py       재분석 툴 (rundir → CSV+차트 / 로그 → RESULTS.md)
    analysis/  RESULTS.md  tier*/           (gitignore: 파생물·로그)
  track_c/                                  CNN 표준세팅 (plan §3.11)
    {c1, c1_oracle, c2}/<cell>              c1=fidelity / c1_oracle=(a) 2^N retrain / c2=개입
    README.md  RESULTS.txt                  (track_c 자체 설명·합본)
  track_d/rundirs/<cell>                    LLM 표준세팅 (OpenFedLLM 무대)
```

cell은 항상 그룹 하위의 서브디렉토리(`rundirs/` 또는 c1/c2 같은 의미 단위) 안에 둔다 —
스크립트·리포트와 섞이지 않게.

## 2. cell 이름 문법 (전 그룹 공통)

```
[<scale>_]<setting>_<condition>[_<variant>][_seedN]
```
- **토큰 구분 = `_`**, **토큰 내부 복합어 = `-`** (예: `label-flip`, `grad-noise`, `device100-a0.5`).
  → `_`로 split하면 항상 의미 토큰 단위가 된다.
- **`seedN`은 맨 끝, dir이 단일 시드일 때만.** phase2_matrix는 한 cell에 3시드를 묶으므로
  (metrics의 `{threat}_seed{N}` 키) 이름에 seed 토큰이 없다. track_c/track_d/phase1은
  시드별 dir이라 `_seedN`이 붙는다. (규칙: 이름 = "형제 dir과 구분되는 축"만, 정해진 순서로.)

| 토큰 | 어휘 (고정) |
|---|---|
| `scale` (LLM만; CNN 없음) | `1B` `3B` `7B` — `m3b`·`1b` 금지 |
| `setting` | `silo5` · `device100-a{α}` · `std20` · `anchor5` · `mnist` · `cifar10`(+ 파티션 `_iid`/`_shard`/`_dir1`) |
| `condition` | LLM: `noisy` `frrand` `frzero` `poison` · CNN c1: `iid` `label-flip` `label-skew` `feature-noise` `quantity-skew` · CNN c2: `clean` `free-rider` `grad-noise` `label-flip` |
| `variant` (선택) | `anchor`((b)-perround+coalition) · `aonly`((a) retrain 오라클) · `str{X}`(c2 강도) · `full-lr1e-3`·`sweep-lr1e-4`·`mini`·`smoke`(phase1) |
| `seedN` | 단일-시드 dir일 때만, 맨 끝 |

**예시**: `1B_silo5_noisy` · `1B_device100-a0.5_frzero_anchor` · `3B_silo5_poison` ·
`cifar10_label-flip_seed0` · `cifar10_iid_aonly_seed0` · `cifar10_dir1_grad-noise_str0.05_seed0` ·
`1B_std20_seed0` · `1B_silo5_full-lr1e-3_seed0`.

## 3. 러너 → 출력 (이 매핑이 규칙을 강제한다)

| 러너 | 출력 루트 | 이름 생성 |
|---|---|---|
| `experiments/phase2_matrix.py` | `runs/phase2_matrix/rundirs` | `RUN_NAME` 또는 `{scale}_{setting}_{cond}[_anchor]` |
| `experiments/track_c1.py` | `runs/track_c/c1` (`C1_RUN_ROOT`) | `{ds}_{scenario}[_aonly]_seed{N}` |
| `experiments/track_c2.py` | `runs/track_c/c2` (`C2_RUN_ROOT`) | `{ds}_{part}_{threat}_str{X}_seed{N}` |
| `experiments/track_d.py` | `runs/track_d/rundirs` | `RUN_NAME` 또는 `{scale}_{regime}_seed{N}` |
| `experiments/phase1_clean_run.py` | — (레거시·동결, 재실행 비권장) | — |

전부 repo-루트 상대경로 기본값. **`RUN_NAME`을 직접 넘길 땐 위 문법을 지킬 것**(특히 `_sN`이 아니라
`_seedN`). 새 러너를 추가하면 이 표에 한 줄 추가.

## 4. 재분석 / 재생성

- **phase2_matrix**: `python runs/phase2_matrix/make_analysis.py` (rundir → `analysis/` 전체 CSV+차트;
  분류·정렬은 config 기반이라 이름이 바뀌어도 안 깨짐) + `make_report.py` (tier 로그 → `RESULTS.md`).
- **track_c 안정성**: `python codes/experiments/track_c3.py {c1|c2}` (cross-seed; `_seedN` trailing 가정).
- **phase1(레거시)**: `python codes/experiments/read_runs.py runs/phase1/rundirs`.
