# REMAINING (Slurm · JB) — A6000 48GB: **frrand 종료 → R4 online seed2 인수**

> 배분 정본 = **`REMAINING-00-INDEX.md`** · 수록목록 정본 = `research-wiki/survey/flirds-paper-experiment-plan.md`.
> **역할 변경(2026-07-25)**: 담당이던 **L9 frrand 가 스코프에서 빠졌다**(계획서 §0.1 오염축 5종에 frrand 없음 → §5 "위협축 제외" 목록).
> **새 역할 = L11 seed2 21셀**(현재 **아무도 안 맡고 있는 3-seed 구멍**) + HJ 꼬리 work-steal.
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 3-seed. push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만.

## 0. 환경

- torch 2.11 스택 · `HF_HOME` 자체 구성(오프라인) · 파티션 `suma_a6000,gigabyte_a6000` + `--qos=base_qos` · 8-GPU/user.
- `codes/` 에서 `PYTHONPATH=.`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **OOM knob**(A6000 48GB 고유): `VAL_CHUNK` T1=3~5 · observer=2. 청크 합산은 `flirds_estimator._chunked` 상 **수학적 exact → φ 불변**, peak 만 내려간다. **`VAL_MAXLEN` 은 φ를 바꾸므로 금지.**

## 1. L9 frrand — 스코프 아웃 · **전량 중단 확정**

**사실**: frrand(랜덤-델타 free-rider)는 확정 오염축 5종({CNN: lf@0.70, frzero, grad-noise} / {LLM: swap@0.7, frzero})에 없다 → **어떤 표에도 들어가지 않는다.** 지금 도는 24셀은 전부 그 위협이다.

**동시에**: A6000 은 클러스터 여유 **10장 / 98**(가동률 90%)로 **논문에 들어가는 LLM 작업의 유일한 병목**이다. frrand 가 점유하는 슬롯이 곧 §5.3 표가 못 채워지는 이유가 된다.

**실측 잔여**(07-25 17:40 기준 · 총 ~580 GPU-h · wall 3–4일):

| 셀 유형 | wall | 상태 | 종료 예상 |
|---|---|---|---|
| renorm-4 T1 s0 (4셀) | ~23–28h | ~14h 경과(~60%) | 07-25 23시–07-26 04시 |
| renorm-4 T1 s1·s2 + lossheur s2 | ~23–28h / 9h | 대기·초반 | 07-26 낮–밤 |
| flirds1st·fedif T1 (6셀, `VAL_CHUNK=5`) | ~11–15h | 슬롯 대기 | 07-26 |
| **observer T2 (3셀, `--time=96h`, `VAL_CHUNK=2`)** | **~44–78h** | seed0 우선 | **07-27~28** |

**결정 (2026-07-25 Yonghee): L9 전량 중단.** "논문에 수록할 실험에 포함되지 않으면 할 필요 없다" — 이 논리는 observer T2 3셀뿐 아니라 **frrand 셀 전부**에 적용된다(전부 축 밖). 거의 끝난 renorm s0 4셀도 잔여 ~10h × 4슬롯 = ~40 GPU-h 를 산출 0 에 쓰는 것이라 함께 중단한다.

```
scancel 1873996 1874031 1875968 1875969     # L9 배열 전체
squeue -u $USER                             # 잔재 확인 후 §2 로 전환
```
→ **~580 GPU-h · 8슬롯 즉시 회수.** 이미 완주한 frrand rundir 은 디스크에 **그대로 둔다**(표에서만 제외; 삭제 불요).

취소 예: `scancel <jobid>` (배열 전체) / `scancel <jobid>_[8-23]` (PD 원소만).

> frrand rundir·러너 산출은 **존속**한다 — 표에서만 빠진다. 되살릴 일은 없다(재제안 금지).

## 2. ★ L11 seed2 — R4 online 점수원 경쟁 seed2 (21셀 · **현재 무주공산**)

> **왜 JB 인가**: L11 63셀은 seed 로 3계정에 쪼개져 있었는데(HJ s0·1 = 42셀 제출 완료 / **YH s2 = 21셀**), **YH 의 seed2 몫이 취소·미제출**됐다. 그대로 두면 §5.3 online 표 7행이 **2-seed** 로 끝난다(3-seed 규칙 위반). JB 는 같은 A6000·같은 env·같은 sbatch 라 **셋업 마찰 0** 으로 인수할 수 있다.

- **셀**: 7 비-flirds(flirds1st·lossheur·fedif·gtg·fedsv·comfedsv·shapleyfl) × {clean, noisy, frzero} × **seed2** = **21**(array 42-62).
- 비-flirds 는 online 스코어링에 **HVP 불요**(값·1차) → 자체 인라인 스코어 = **B200 독립·즉시 가동**.
- **⚠ R4 = R=100**(2026-07-25; INDEX §0). sbatch 에 `ROUNDS=100`·`RUNDIR_REPLACE=1`·소스별 `VAL_CHUNK` 가 **배선 완료** → `--export` 불요.
- **비용 @R=100**: renorm-4 12셀 × ~12.5h ≈ **150** + same-game·FedIF 9셀 × 3.7–4.65h ≈ **36** → **~186 GPU-h**(8슬롯 ~23 wall-h).

```
cd $REPO && mkdir -p runs/track_h/_logs
RUNDIR_ROOT=$REPO/runs/track_h/rundirs_llm_jb \
  sbatch --array=42-62%8 runs/track_h/sbatch_l11_online.sh
```
- `RUNDIR_ROOT` 를 주지 않으면 seed2 는 `rundirs_llm_yh` 로 자동 라우팅된다(집계는 되지만 계정별 분리를 위해 명시 권장).
- `--time` 은 파일 기본값 **24:00:00** 이면 충분(R=100 최장 ~12.5h).
- **싼 것 먼저**: same-game·FedIF 는 `--array=42,45,48,51,54,57,60%8` 류로 먼저 뽑아 몇 시간 만에 3-seed 를 닫을 수 있다(인덱스 = `SRC_I = (IDX%21)/3`).

## 2b. G4c seed2 — R4 retrain renorm-4 (3셀 · ~150 GPU-h)

> **R=100 으로 부활한 레그.** §5.3 **retrain 표의 renorm-4 4칸**을 채운다(나머지 5행 = same-game 계열은 B200 L1 담당).
> seed0·seed1 은 **B200** 이 가져간다 — 이 셀은 관찰자가 매 라운드 renorm-4 를 채점하고 그게 비용의 ~92%라 B200 이 **3.6× 빠르다**(13.9h vs ~50h). seed2 만 여기.

```
sbatch --array=6-8%8 runs/track_h/sbatch_l4_renorm_t2.sh      # seed2 3셀
```
- 착지 root = `rundirs_llm_g4c`(sbatch 기본값). **⚠ 이 셀의 `observer` arm 은 L1 의 same-game observer 와 rundir 이름이 같아서** 같은 root 를 쓰면 서로 덮어쓴다 — root 를 바꾸지 말 것.
- **`--time=72:00:00`**(파일 기본값): A6000 셀 ~50h(관찰자 ~38h + 재학습 4개 ~12h). rundir 은 arm 종료 시 써지므로 timeout = 관찰자 arm 전손.
- **`VAL_CHUNK` 를 낮추지 않는다**: renorm-4 는 forward-only(@no_grad)라 grad 경로의 OOM 가드가 적용되지 않는다. 청크 합산은 exact → φ 동일.
- **순서**: §2(L11 seed2)를 먼저 닫고 착수 — L11 이 §5.3 online 표 7행의 3-seed 를 완성하는 쪽이라 우선순위가 높다.

## 3. 여유 시 — HJ 꼬리 work-steal

- HJ 의 L11 seed0·1(42셀 ~372 GPU-h @R=100)은 8슬롯이면 07-27 후반에 끝나지만, A6000 여유가 10장뿐이라 실제로는 밀릴 수 있다. JB 슬롯이 비면 흡수한다:
```
RUNDIR_ROOT=$REPO/runs/track_h/rundirs_llm_jb \
  sbatch --array=<HJ 잔여 범위>%8 runs/track_h/sbatch_l11_online.sh
```
- arm-level idempotent · 착지 root 만 계정별로 분리하면 `make_analysis` 가 dup-win 으로 병합한다. **HJ 가 이미 완주한 인덱스는 빼고** 제출할 것(중복 = GPU 낭비).

## 4. 완료 후

1. rundir 커밋(push는 Yonghee) → `runs/track_h/make_analysis.py`(LLM 로더가 `rundirs_llm_jb` 를 이미 읽는다) → `flirds-results-downstream` §5.3 R4 online 표.
2. **스택 캐비엇**: A6000(torch 2.11) vs canonical(B200 torch 2.12) — recovery 정규화로 병치(mean|Δ|≤0.006). **`timing.json` 은 §5.5 cost 표에 쓰지 않는다.**
3. **완료 판정**: 로그 `TRACK G DONE` + rundir mtime.
