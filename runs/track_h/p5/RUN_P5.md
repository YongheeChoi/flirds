# P5 실험 — 신뢰-기반 sign 정책(hard vs soft) 점수원 경쟁 (Slurm 서버 실행 지시서)

> **이 문서 = CNN P5 실험(Slurm 서버)의 실행 정본** (Yonghee 2026-07-21 역할 분담:
> CNN = Slurm 서버·이 run-dir 문서 기반 / **R4 LLM = B200 컨테이너·`REMAINING.md`
> §1.1 기반** — R4 실행 절차는 REMAINING에 있고, 이 문서 §2–4는 두 무대 공통
> 스펙·공정성·예측 정본). Slurm 세션은 §5 pre-flight → §6 sbatch 제출 → §7
> 보고·커밋 순서로 진행. 예측은 사전 등록이며 **실행 후 수정 금지**(MISS 그대로 보고).

## 1. 배경 — 왜 P5인가

Track H Tier 1(CNN dir1, overview §3.2.6)에서 strict sign 게이트(P1: cum≤0 즉시
배제)는 **경계선 분산을 확신 있는 유해와 똑같이 과금**한다는 것이 관찰자 데이터
재분석(07-20 로컬, 탐색적)으로 확인됐다: flirds의 clean 오배제 31/100 중 21은
|t|<1.645의 경계선 노이즈(진짜 기여 ≈0인 클라의 cum은 ±σ√n 랜덤워크라 절반 확률로
0 아래에 있음), 반면 corrupt 검출 신호는 t≈−5.5로 강건. GN에서 loss-heur의 검출
신호는 t≈−1.27로 marginal. → **결정을 "부호"가 아니라 "그 클라 자신의 온라인
증거"로 내리는 정책 2종을 전 점수원 공통으로 추가**하고, 성능(절대 test acc)으로
P1과 3자 비교한다. 두 정책의 서열(hard vs soft)은 사전 미확정 — 둘 다 보고.

## 2. 정책 정의 (z = 1.645 고정, 단측 95%)

클라 c의 온라인 통계(그 런이 학습 중 관측한 자기 raw 스트림만): 참여수 n, 누적
cum=Σraw, 표본표준편차 σ̂(ddof=1), t = cum/(σ̂√n).

| 정책 | arm 이름 | 규칙 | 엣지 |
|---|---|---|---|
| **P5-hard** | `<src>_cgate` (T2: `t2_csign_<src>`) | **배제 iff cum + 1.645·σ̂√n ≤ 0** (UCB 규칙 — 유의하게 음수일 때만). 선택된 cohort는 평범한 n-가중(라운드-raw 스크린 없음 — 단일 라운드 노이즈로 결정하지 않는다는 정의의 일부) | n<max(min_obs,2) → 유지(증거 없음); σ̂=0 → strict로 축퇴(exact-0 FR 배제 유지); burn-in·probation은 V2와 동일 기계 |
| **P5-soft** | `<src>_pweight` (T2: `t2_pw_<src>`) | **w ∝ n·Φ(t)** — "기여가 양수일 확률"로 가중. 방향 대칭: 확신 양수→1, 확신 음수→~0, 증거 없음(t=0)→정확히 ½ | burn-in 전/n<min_obs → 중립 1(FedAvg 기본); σ̂=0 → 1[cum>0] (exact-0 → 가중 0); 전량 0 → n-가중 fallback |

- 게이트 하이퍼는 기존 V2 공통값 그대로(burn_in=10, min_obs=2, probation=5) + 신규
  상수 `conf_z=1.645`(env `C2_CONF_Z`; **변경 금지**). config.yaml에 자동 기록됨.
- T2 버전: 관찰자 런의 **최종** 통계로 1회 판정(csign=UCB kept 재학습 / pw=고정
  Φ(t)-가중 재학습; kept·가중 dedupe, size-matched `t2_random_k` 통제 자동).

## 3. 공정성 조항 (Yonghee 2026-07-21 — 위반 금지)

1. **사전 정보 사용 금지**: 신뢰 판정에 들어가는 모든 수치(σ̂, n, cum)는 **그 런이
   training 과정에서 관측한 자기 스트림**에서만 나온다. 별도 캘리브레이션 런,
   clean-셀 사전 실측, oracle 마스크, 타 런/타 seed 정보 일절 불사용. (사전 실측
   기반 캘리브레이션 안은 07-21 기각 — 이 원칙이 우선.)
2. z=1.645는 데이터에서 고르지 않은 **보편 상수**(단측 95% 교과서 임계값)이며 전
   점수원·전 셀·전 위협 동일. z=0이 곧 기존 P1이므로 P5는 P1을 포함하는 족(族).
3. 같은 규칙을 8개 점수원 각자의 자기 스트림에 적용 — σ̂가 자기 노이즈라서 노이즈
   큰 방법일수록 오히려 넓은 보호 밴드를 받는다(규칙 자체는 flirds 편향 없음).
4. 셀별·점수원별 하이퍼 튜닝 금지(기존 Track G/H 관례 승계).
5. 07-20의 오프라인 리플레이는 **탐색적 진단**으로만 인용; 우열 판정은 이 사전
   등록 런(확증)으로만.

## 4. 사전 등록 예측 (HP-1~6; 탐색 리플레이 유래 — online 피드백으로 어긋날 수 있음, MISS 그대로 보고)

| # | 예측 | 근거(리플레이) |
|---|---|---|
| HP-1 | flirds cgate: clean acc가 P1(.6315) 대비 상승(오배제 31→~10명 수준) | 경계선 21/31 구제 |
| HP-2 | flirds cgate: corrupt 검출 유지 → FR/GN/LF acc ≥ P1 수준(GN T2 csign ≈ .61) | corrupt t≈−5.5 강건 |
| HP-3 | lossheur cgate: GN 검출 반토막(30→~16) → GN acc가 P1(.598) 대비 하락; pweight는 Φ(−1.27)≈0.10 가중으로 중간 회복 | GN 신호 marginal |
| HP-4 | renorm 4종: FR 붕괴 유지(전 정책; FR에 확신-양수 편향이라 어떤 z에서도 미검출) + clean 오배제는 감소하나 잔존 | 편향은 정책 무관 |
| HP-5 | flirds1st·fedif: GN 실명 유지(≈vanilla .244; t=+3.1 확신-오판) | 편향 |
| HP-6 | pweight: 전 소스 clean ≈ vanilla ±band(공통 Φ 인자는 재정규화로 상쇄) — soft가 hard보다 lossheur GN에 유리(약신호 부분 반영) | 대칭 규칙 성질 |

**종합 예측**: 오염-평균에서 flirds가 cgate 1위(전 위협 생존 유일). hard vs soft
서열은 미등록(둘 다 보고).

## 5. 구현 현황 + Pre-flight (서버에서 먼저)

구현·로컬 검증 완료(2026-07-21 로컬 세션): `fl/intervene.py`(SignAccumulator.stats
+ `_conf_keep`/`make_confgate_select_fn`/`make_probweight_weights_fn`/`_phi_cdf`) ·
`experiments/track_c2.py`(`_TH_POLICIES` += cgate/pweight; T2: `C2_T2_P5`/
`C2_T2_LEGACY`) · `experiments/track_g.py`(LLM 동일 arm + `T2_P5` — §8 R4용) ·
`runs/track_h/make_analysis.py`(P5h/P5s 파싱). 테스트 32개 green(신규 test_p5 10 +
기존 회귀 22) + fmnist/tiny-gpt2 e2e 스모크 green.

**Slurm 서버 환경 준비** (B200 컨테이너와 별개 머신 — 처음이면 여기부터):
1. repo clone(브랜치 main, P5 커밋 포함) → `<REPO>` 확정.
2. venv: `torch`+`torchvision`(CUDA), `numpy`, `scikit-learn`, `pandas`, `pyarrow`,
   `pyyaml` — 가능하면 B200 venv 고정판(torch 2.12.0/numpy 2.4.6 등,
   `$BATCH/PROVENANCE.md`)과 major 버전 일치. CNN 트랙은 HF/transformers **불필요**.
3. **cifar10 데이터 확인(제출 전 필수)**: `~/data/cifar-10-batches-py/` 존재 또는
   `~/data/cifar-10-python.tar.gz` **정확히 170,498,071 bytes**. 없으면 torchvision
   자동 다운로드에 맡기지 말 것(cs.toronto.edu 스트림 stall 전례 — B200 컨테이너
   07-20 원인 확정) — B200 컨테이너 `~/data`에서 scp로 가져오는 것이 안전.
   fmnist(스모크용)는 AWS 호스트라 자동 다운로드 무해.

pre-flight (`codes/`에서; 실패 시 중단·보고):
```bash
PYTHONPATH=. $PY tests/test_p5.py && PYTHONPATH=. $PY tests/test_track_h.py \
  && PYTHONPATH=. $PY tests/test_signgate.py
# CNN e2e 스모크 (~2-3분, GPU 1장; fmnist 자동 다운로드 OK)
C2_DATASET=fmnist C2_MODE=smoke C2_THREAT=free_rider C2_SEED=0 C2_BURN_IN=1 \
  C2_ARMS=vanilla,oracle_excl,flirds_cgate,flirds_pweight,gtg_cgate \
  C2_T2=1 C2_T2_P5=1 C2_T2_LEGACY=0 C2_RUN_ROOT=/tmp/p5smoke C2_PERSIST=1 \
  PYTHONPATH=. PYTHONUTF8=1 $PY experiments/track_c2.py
# 확인: t2_csign_*/t2_pw_* 라인 + dedupe + TRACK-C2 RUN OK
```

## 6. 실행 (sbatch 2개 — 동시 제출 가능, 독립)

1. `sbatch_p5_t1.sh` — T1 online: 96 task(8점수원×4위협×3seed), task당 arm 2개
   (`<src>_cgate`,`<src>_pweight`), ~10–16분/task ≈ **~21 GPU-h**.
2. `sbatch_p5_t2.sh` — T2 retrain: 12 task(관찰자+csign/pw 재학습, legacy T2는
   스킵 — 이미 디스크에 있음), ~1–1.5h/task ≈ **~15 GPU-h**.

제출 전 두 파일에서 채울 것: `--partition`(필요시 `--account`) — `sinfo`로 확인;
`<REPO>`(`--output` 경로 포함)·`<VENV_PY>` — 또는 `REPO=<abs> PY=<venv-python>
sbatch ...`로 주입. `%4` throttle은 큐/GPU 예산에 맞게 조정. **rundir 루트는
`runs/track_h/rundirs_cnn`이고 RUN_NAME은 `*_<src>p5_*`/`*_obsp5_*`로 기존 96런과
절대 충돌하지 않게 이미 설계됨**(RunLogger는 동명 디렉토리를 덮어씀 — 이름 수정
금지). 총 **~36–45 GPU-h**.

부분 실패 시: 해당 array index만 `sbatch --array=<idx> ...`로 재제출(멱등 —
RUN_NAME이 같으므로 그 셀만 덮어씀).

## 7. 종료 후 (보고 프로토콜)

1. `python runs/track_h/make_analysis.py` — P5h/P5s가 policy 열로 자동 편입.
2. 보고: **P1 vs P5h vs P5s 절대 test acc 표**(vanilla/oracle_excl 앵커 포함,
   위협별 + 오염-평균; overview §3.2.6 형식) + HP-1~6 HIT/MISS + GPU-h 실측.
3. 커밋: 신규 rundir + `analysis/` 재생성분 + (수치 반영은 overview §3.2.6에만 —
   이 문서는 스펙 정본이라 결과 수치 기입 금지).

## 8. R4(gsm50k5) 연계 — Yonghee 07-21: "R4도 동일 정책 적용"

LLM 러너(track_g.py)에 같은 arm이 배선 완료: `<src>_cgate` / `<src>_pweight` +
`T2_P5=1`(→ `t2_csign_*`/`t2_pw_*`, Φ-가중 재학습·dedupe) + `T2_LEGACY=0`(기존
t2_sign 중복 스킵). **R4의 실행 절차·시점·명령은 B200 컨테이너의 정본인
`REMAINING.md` §1.1에 있다**(Yonghee 07-21 역할 분담 — 이 문서는 CNN/Slurm 전용).
스펙·공정성 조항(§2–3)·예측 프레임(§4)은 R4 leg에도 동일 적용.

## 9. 금지

- z(=1.645)·게이트 하이퍼 변경, 셀별 튜닝, 사전 캘리브레이션 런 추가.
- oracle 마스크·타 런 정보를 게이트 입력으로 사용.
- poison 위협(전면 제외 관례), 기존 rundir 재실행·덮어쓰기(신규 RUN_NAME만).
- 예측표(§4) 사후 수정. 결과 수치를 이 문서에 기입(overview에만).
