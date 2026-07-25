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
- **비용**: renorm-4 12셀 × ~23–28h ≈ **~300** + same-game·FedIF 9셀 × 2.5–3.2h ≈ **~26** → **~326 GPU-h**(8슬롯 ~41 wall-h).

```
cd $REPO && mkdir -p runs/track_h/_logs
RUNDIR_ROOT=$REPO/runs/track_h/rundirs_llm_jb \
  sbatch --export=ALL,VAL_CHUNK=3 --time=24:00:00 --array=42-62%8 runs/track_h/sbatch_l11_online.sh
```
- `RUNDIR_ROOT` 를 주지 않으면 seed2 는 `rundirs_llm_yh` 로 자동 라우팅된다(그래도 집계는 되지만, 계정별 분리를 위해 명시 권장).
- **`--time` 24h**(원 기본값 08h 는 renorm 소스에서 timeout). renorm 이 24h 를 넘길 조짐이면 `42:00:00`.
- **싼 것 먼저**: same-game·FedIF 는 `--array=42,45,48,51,54,57,60%8` 류로 먼저 뽑아 몇 시간 만에 3-seed 를 닫을 수 있다(인덱스 = `SRC_I = (IDX%21)/3`).

## 3. 여유 시 — HJ 꼬리 work-steal

- HJ 의 L11 seed0·1(42셀 ~980 GPU-h)이 8슬롯으로는 07-30 까지 밀린다. JB 슬롯이 비면 흡수한다:
```
RUNDIR_ROOT=$REPO/runs/track_h/rundirs_llm_jb \
  sbatch --export=ALL,VAL_CHUNK=3 --time=24:00:00 --array=<HJ 잔여 범위>%8 runs/track_h/sbatch_l11_online.sh
```
- arm-level idempotent · 착지 root 만 계정별로 분리하면 `make_analysis` 가 dup-win 으로 병합한다. **HJ 가 이미 완주한 인덱스는 빼고** 제출할 것(중복 = GPU 낭비).

## 4. 완료 후

1. rundir 커밋(push는 Yonghee) → `runs/track_h/make_analysis.py`(LLM 로더가 `rundirs_llm_jb` 를 이미 읽는다) → `flirds-results-downstream` §5.3 R4 online 표.
2. **스택 캐비엇**: A6000(torch 2.11) vs canonical(B200 torch 2.12) — recovery 정규화로 병치(mean|Δ|≤0.006). **`timing.json` 은 §5.5 cost 표에 쓰지 않는다.**
3. **완료 판정**: 로그 `TRACK G DONE` + rundir mtime.
