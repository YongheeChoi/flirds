# REMAINING (Slurm · JB) — **LLM → CNN 전환**: (a) 재학습 오라클 seed2

> 배분 정본 = **`REMAINING-00-INDEX.md`** · 수록목록 정본 = `research-wiki/survey/flirds-paper-experiment-plan.md`.
> **역할 = RTX3090 에서 (a) 재학습 오라클 seed2**(G2·G9 의 16셀) **+ work-steal.**
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 3-seed. push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만.

## 0. 역할이 두 번 바뀌었다 — 현재 상태만 보면 된다

| 종전 | 현재 | 사유 |
|---|---|---|
| L9 frrand 24셀 | **폐기** | frrand = 확정 오염축 5종 밖 → **07-25 전량 중단 확정** |
| L11 seed2 21셀 · G4c seed2 | **폐기** | **LLM downstream 스코프 컷**(§1) — 해당 소스가 표에서 빠짐 |
| — | **G2·G9 seed2 (CNN, 3090)** | CNN 이 남은 물량의 대부분이고 3090 여유가 21장 |

**즉시 할 일**: 아직 살아 있는 L9/L11 잡이 있으면 `scancel` → §2 로 전환.
완주한 frrand rundir 은 디스크에 **그대로 둔다**(표에서만 제외; 삭제 불요).

## 1. LLM downstream 스코프 컷 (2026-07-25 Yonghee)

§5.3 LLM 개입 표의 비교 대상을 **이미 계산이 끝난 것**으로 한정한다 — vanilla(observer) · oracle_excl · random_excl · **flirds류**. 디스크 확인 결과 noisy·frzero 는 이 셋 + `t2_sign` ×4(flirds·flirds1st·loss-heur·FedIF)까지 **3-seed 완결**이라, 남은 신규 실행은 **online Flirds-1st 9셀(HJ)** 과 **clean seed1·2 4셀(B200)** 뿐이다.
→ **renorm-4 는 LLM downstream 표에서 빠진다.** 그 붕괴는 CNN §5.3(8점수원 양 표 3-seed)과 LLM §5.2 fidelity(G1 이 9방법 φ 전량 산출)가 담당한다.

## 2. G2·G9 seed2 — (a) 재학습 오라클 (16셀 · ~168 GPU-h)

> **무엇**: N=10 전원참여에서 **(a) 2¹⁰ 재학습 오라클 + (b) 2¹⁰ + 9방법 φ**. (a)는 방법-중립 참값이라 **전 방법을 채점하는 유일한 무대**인데, 현 C1 시나리오가 확정 오염축과 한 칸도 안 겹쳐서 다시 정렬하는 것이다.
> **비교불가성은 남는다**: (a)는 2^N 재학습이라 **N=100 에서 원리적으로 불가** → 1A-CNN(N=100 부분참여)과 N·참여율은 못 맞춘다. 맞출 수 있는 건 **오염축과 파티션뿐**이고 이 잡이 그걸 한다. 논문에도 명시.

- **셀**: **`--array=14-15,24-31,40-47` = 18셀 ~205 GPU-h**(mnist seed0 잔여 2 + seed1 8 + seed2 8). Slurm 4계정을 **GPU-h 로 균등화**한 몫이다(YH `0-7,16` · JW `32-39,17-23,8-13` · HJ 는 c1축 없이 G12+G10).
  인덱스 규약: `SEED=IDX/16` · 그 안에서 `0-7`=cifar10, `8-15`=mnist · 파티션 `iid,dir1` × 4위협(clean·lf@0.70·free_rider·grad_noise).
- **비용(실측)**: `t_a` = cifar10 **32,808 s ≈ 9.1 h** · mnist **41,168 s ≈ 11.4 h**(`runs/track_c/c1_oracle/*/metrics.json`). 궤적 ~103 s·전 방법 합 ~8분은 무시 가능 → **셀 ≈ t_a**.
- **⚠ 착수 게이트 = C-b 가 `origin/main` 에 있는 것.** 요구 env = `C1_PARTITION`(iid\|dir1) · `C1_THREAT`(clean\|label_flip\|free_rider\|grad_noise) · `C1_FLIP_RATE=0.70`.
  **정정(07-25)**: 구 러너로 제출하면 **실패하지 않고 조용히 틀린다** — 미지 env 는 무시되고
  `C1_SCENARIO=iid` 로 도는데 `C1_RUN_NAME`(l.432)만 먹혀 위협 라벨 붙은 이름에 iid-clean
  데이터가 들어가고 EXIT=0. **JB 의 워처 술어(`grep C1_THREAT` on `origin/main`)가 정확히
  옳은 방어**다 — 그대로 유지. 제출을 보류하고 슬롯을 idle 로 둔 판단도 맞다.
- **C-b 는 이미 구현·검증 완료(YH 워킹트리)** — 남은 건 `git push`(Yonghee) 하나다.
  JB 가 포팅하지 않기로 한 판단이 결과적으로 옳았다: JW 는 별도 구현(`989f5ca`)을 만들어
  **정본이 두 벌 되는 문제**가 생겼고, 지금 그걸 되돌리는 중이다.
- 착지 후 JB 계획대로 `C1_ORACLE_A=0` 빌드-only 스모크(별도 `_smoke` 이름)로 새 threat 경로를
  먼저 검증할 것. **⚠ 대조 기준은 `round(0.4N)` 균일이 아니다** — 오염-집합 draw 는 위협별로
  갈린다(`track_c2.py:248-259` 감사 · 논문 `paper-ko.md:863-869`):
  **label_flip = 독립 Bernoulli(ρ=0.4) 라 개수가 시드마다 변동**(강도만 0.70 고정),
  free_rider·grad_noise = 고정 `⌊ρN⌉` 비복원(update-level = 준거 문헌 없음 → 논문이 예외 기술).
  draw 는 **seed-only** 라 dataset/partition 과 무관하다(l.255-257).

  **JB 몫(mnist seed1·2)의 미리 계산된 실현값** — 스모크에서 이 값이 나와야 한다
  (같은 코드로 N=100 을 돌리면 논문 기재값 39/48/47 을 정확히 재생산 → 신뢰 가능):

  | | seed1 | seed2 | (참고) seed0 |
  |---|---|---|---|
  | **label_flip** | k=7 `[1,2,4,5,7,8,9]` | k=6 `[0,1,3,4,6,7]` | k=3 `[3,5,6]` |
  | fr · gn (고정 4) | `[0,4,6,7]` | `[3,4,6,9]` | `[1,4,6,7]` |

  전 셀 k≥2 → AUROC 가 모든 셀에서 정의된다. `rates=[0,…]` = free_rider 라벨 무접촉(정상).

```
cd $REPO && mkdir -p runs/track_c/c1/_logs
sbatch --array=24-31%8    runs/track_c/c1/sbatch_c1_axis.sh   # mnist seed1
sbatch --array=40-47%8    runs/track_c/c1/sbatch_c1_axis.sh   # mnist seed2
sbatch --array=14-15%8    runs/track_c/c1/sbatch_c1_axis.sh   # mnist seed0 잔여 2셀
```
- `--time=24:00:00` 내장(최장 11.4h + 여유). 셀 하나가 ~11.4h 라 **중도 컷 = 그 셀 전손** — `--time` 을 줄이지 말 것.
- JB 몫은 **mnist 전량 seed1·2**(부록 G9)다. 본문 G2(cifar10)는 YH·HJ·JW 가 나눠 갖는다 — 그쪽이 먼저 착지해야 본문 표가 닫힌다.
- **채우는 것**: 계획서 §2.1 "1B-CNN vs (a)"(본문) · §3.1 "1B-CNN mnist vs (a)"(부록) · §3.4 φ 부호 감사의 **CNN 레그 재감사**(현 감사에 frzero·grad-noise 없음).

## 3. 셋업 (LLM 계정 → CNN)

1. **conda env**: 기존 torch 2.11 스택 그대로. CNN 은 transformers/trl/peft 불요.
2. **데이터**: torchvision `mnist`·`cifar10`(HF 캐시 아님). YH 디렉토리 복사 또는 최초 1회 다운로드(수백 MB). `HF_HOME` 무관.
3. **파티션**: `--partition=base_suma_rtx3090`(sbatch 내장) · 8-GPU/user.
4. sbatch 상단 `REPO`/`PY` 기본값이 YH 경로다 → 다르면 `REPO=… PY=… sbatch …`.

## 4. 완주 후 — work-steal (~07-27 오전)

- 4계정을 **~23 wall-h 로 균등 배분**했으니 큰 편차는 없다. 그래도 먼저 비면 다른 계정의 `sbatch_c1_axis.sh` 잔여 → HJ 의 G10 순으로 흡수한다.
- 남은 `--array` 범위만 지정할 것(같은 rundir 이름 = last-writer-wins 라 중복 실행은 GPU 낭비).

## 5. 완료 후

1. rundir 커밋(push는 Yonghee).
2. C1 집계 재생성 → `flirds-results-fidelity`(vs (a) 절)·`flirds-results-ablation`(부호 감사 CNN 레그) → paper §5.2 sub·부록 C.
3. **스택 캐비엇 없음** — (a)는 재학습 오라클이라 하드웨어 독립이고 기존 C1 과 같은 torch 2.11. 다만 **`timing.json` 은 §5.5 cost 표에 쓰지 않는다**(canonical = B200 실측만).
