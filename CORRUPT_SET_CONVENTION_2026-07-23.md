# 오염 클라 집합 추출 규약 — 감사 결과와 논문 병기 문구 (2026-07-23)

> Yonghee 검토용. 결론 = **재실행 없음, 병기로 해결.** 코드·분석 쪽 조치는 이미 반영됐고,
> 아래 §3 문구를 `paper/`에 붙이면 끝난다(`paper/`는 세션 규칙상 미수정).

## 1. 무엇을 발견했나

Track G 그리드 로그에서 `corrupt=48`(n=100, `MAL_FRAC=0.4`)이 눈에 띄어 추적한 결과:

- `track_c2.build()`의 오염 집합 추출이 **위협마다 다르다**. `label_flip`만 클라별 독립
  베르누이라 **개수가 시드에 따라 변동**하고, 나머지 셋은 정확히 40명이다.
- 실현 오염 수는 **시드에만 의존**한다: seed 0/1/2 → **39 / 48 / 47**. 3-seed 평균
  **44.7%** — 우리가 코드·문서에 써 온 "40%"와 다르다. 베르누이 3회가 우연히 전부 위로 튀었다
  (기대 40, sd 4.9).
- 이 값은 `track_c/c2`·`track_h`·`probe_signal`·현 Track G 그리드에서 **전부 일치**한다.
  즉 온디스크 전 rundir가 같은 실현 집합을 쓴다.

## 2. 버그가 아닌 이유 + 통일하지 않기로 한 이유

**버그 아님.** [FedCorr 공식 구현](https://github.com/Xu-Jingyi/FedCorr) `util/util.py::add_noise`를
직접 대조했다:

```python
gamma_s = np.random.binomial(1, args.level_n_system, args.num_users)   # 베르누이, 개수 고정 아님
gamma_c = gamma_s * ((1 - args.level_n_lowerb) * np.random.rand(n) + args.level_n_lowerb)
```

`track_c2.py`의 `rng.random(n) < rho` + `rng.uniform(TAU=0.5, 1.0)`이 이것과 동일하다. 즉
label-flip만 베르누이인 건 (ρ, τ) 규약을 지키려는 의도적 선택이다. `C2_STRENGTH`가 label-flip에서
ρ를, 나머지에서 σ/amplitude를 쓸어가는 것도 각 위협의 dose knob이 다르기 때문으로 정합적이다
(세션 중 이걸 "세 번째 비일관"으로 잘못 지목했다가 철회).

**통일 안 함.** 두 가지 이유다.

1. **비용**: 규약을 바꾸면 label-flip rundir가 전부 무효가 된다 — 현 캠페인 123런 + Track H 기존
   51런 + `track_c/c2` 30 + `probe_signal/cnn_c2` 12 + scale 6 = **222런 ≈ 155 GPU-h**.
2. **더 중요한 이유 — 바꾸는 행위 자체가 비교 가능성을 깬다.** 지금은 온디스크 전체가 한 규약
   위에 있다. 손대는 순간 구/신 규약이 갈라진다. 특히 rate rng 분리안은 strmain 72셀(온디스크)과
   69셀(큐)을 갈라놓는데, **N=100에서는 고치는 게 아무것도 없다**(§4).

**핵심 방어 논리**: 오염 축(위협)별로는 완전히 통일돼 있다. 두 규약 모두 데이터를 만지기 전
시드 전용 스트림에서 뽑히므로 **한 시드는 데이터셋·파티션·dose·트랙에 걸쳐 같은 집합**을 준다
(`tests/test_corrupt_set_canon.py::test_corrupt_set_is_seed_only`). 우리가 실제로 그리는 대조
(파티션 간·dose 간·방법 간·게이트 간)는 전부 위협을 고정하므로 두 규약이 한 비교 안에서 만나지
않는다.

**남는 통계 비용(정직하게 기록)**: `oracle_excl`이 seed0은 39명, seed1은 48명을 뺀다 → recovery
분모가 시드마다 다른 데이터량을 뜻하므로 3-seed 평균에 추가 분산이 얹힌다. 편향이 아니라 분산이며,
`n_corrupt` 컬럼으로 표에 드러난다.

## 3. 논문 병기 문구 (붙여넣기용)

### 3.1 한국어 — `paper/paper-ko.md` §D.3 corruptor 정의 **바로 뒤**에 새 문단

> **D.3+ 오염 클라이언트 집합의 추출 규약(위협별).** 오염 클라이언트 *집합*은 위협마다 출처
> 규약을 따르며 단일 규칙이 아니다. label-flip은 FedCorr의 $(\rho,\tau)$ 잡음 모델을 공식 구현
> 그대로 재현한다 — 클라이언트별 독립 베르누이 $\gamma_s \sim \mathrm{Bin}(1,\rho)$로 오염 여부를
> 뽑고, 오염 클라이언트의 라벨 잡음률은 $\tau \sim U(0.5,1)$이다(고정-dose 사다리 셀에서는 이
> draw 대신 $\{0.15,0.35,0.70\}$을 지정한다). 따라서 오염 클라이언트 **수가 시드에 따라 변동**하며,
> $N{=}100$·$\rho{=}0.4$에서 시드 0/1/2의 실현값은 각각 **39/48/47(평균 44.7\%)** 이다. 반면
> update-level 위협(free-rider, frrand, grad-noise)은 대응하는 준거 문헌이 없어 정확히
> $\lfloor \rho N \rceil = 40$명을 비복원 추출한다. 두 규약 모두 데이터를 만지기 전 시드 전용
> 난수 스트림에서 뽑히므로 **한 시드는 데이터셋·파티션·dose에 걸쳐 동일한 오염 집합**을 준다.
> 본 논문의 모든 대조는 위협을 고정한 채 이뤄지므로 두 규약이 한 비교 안에서 만나는 일은 없다.
> 표에는 명목 $\rho$가 아니라 **실현 오염 수**를 병기한다.

### 3.2 English — AAAI appendix (experimental setup)

> **Corrupt-client set construction (per threat).** The corrupt *set* follows each
> threat's source convention rather than one uniform rule. Label-flip reproduces
> FedCorr's $(\rho,\tau)$ noise model verbatim: client corruption is an independent
> Bernoulli draw $\gamma_s \sim \mathrm{Bin}(1,\rho)$ and each corrupted client's label
> noise level is $\tau \sim U(0.5,1)$ (fixed-dose ladder cells pin the level to
> $\{0.15,0.35,0.70\}$ instead). The corrupt-client *count* therefore varies with the
> seed: at $N{=}100$, $\rho{=}0.4$, seeds 0/1/2 realize **39/48/47 clients (44.7\% mean)**.
> The update-level threats (free-rider, frrand, grad-noise) have no such reference
> implementation and draw exactly $\lfloor \rho N \rceil = 40$ clients without
> replacement. Both rules draw from a seed-only stream before any data is touched, so a
> given seed fixes the corrupt set across datasets, partitions and doses. Every contrast
> we report holds the threat fixed, so the two rules never meet inside one comparison.
> Tables report the **realized** corrupt count, not the nominal $\rho$.

### 3.3 본문에서 고쳐야 할 표현

"오염 클라 40%" / "40\% corrupted" 로 쓴 곳은 label-flip 무대에 한해 **"$\rho{=}0.4$, 실현
39/48/47"** 로 바꾼다. update-level 위협은 40이 맞으므로 그대로 둔다.

## 4. 부수 발견 — rng 결합(잠복, 미발동)

dir1에서 빈 클라이언트가 생기면 보정용 `rng.integers`가 strmain의 rate 추첨과 **같은 스트림**을
소비한다 → 같은 시드인데 파티션마다 rate 벡터가 어긋난다.

- **재현성 문제가 아니다.** 같은 (seed, dataset, partition)이면 소비 순서가 결정론적이라 재실행 시
  비트 동일하게 재현된다.
- **지금까지 한 번도 발동하지 않았다.** dir1 최소 클라 크기가 187/209/224/236으로 전부 0 초과
  (`test_no_empty_clients_at_n100`이 seed 0/1/2 전부 검증 — 로그로 확인 못 했던 seed2 포함).
- **고치지 않는다.** 스트림을 분리하면 strmain rate가 바뀌어 온디스크 72셀과 큐 69셀이 갈라지는데,
  N=100에서는 얻는 게 없다. 대신 발동 시 `[WARN]` 로그 + `metrics.json.empty_clients` 기록으로
  전환했다(침묵 제거).

## 5. 반영된 조치 (전부 숫자 불변, GPU 미사용)

| | 내용 |
|---|---|
| `codes/experiments/track_c2.py` | 위협별 규약·출처·실현값 주석 명시 / 빈 클라 `[WARN]` + `metrics.json`에 `empty_clients`·`n_corrupt` 기록 |
| `codes/tests/test_corrupt_set_canon.py` (신규) | 8 tests — 39/48/47 canon 고정, 나머지 40 고정, 시드 전용성(4 파티션×2 데이터셋), 빈 클라 부재. `os.environ`+모듈 reload를 쓰므로 autouse fixture 로 전역 상태 원복(뒤에 오는 `test_partition_qskew` 보호) |
| `runs/track_g/make_analysis.py` | `n_corrupt` 컬럼(per-arm + cellmean 평균 + md 표) |
| `runs/track_c/c2fid/make_analysis.py` | `n_corrupt` 컬럼(C1 호환 스키마의 EXTRA에 추가) |

검증: 기존 59 tests + 신규 8 tests green. 편집 후 재생성한 `cnn_summary.csv`에서
label_flip 39–48 vs 나머지 40 고정 확인. 진행 중 잡 실패 0.
