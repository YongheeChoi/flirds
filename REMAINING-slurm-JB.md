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

## 1b. ~~G14 — mnist 기준 arm(앵커) dir1 9런~~ → **YH 전량 이관 (2026-07-26)**

> 한때 JB에 dir1 9셀(`--array=9-17%8`)을 배분했으나 **같은 날 YH 전량(18셀)으로 재배분**됐다 — 총 3–5.5 GPU-h라 쪼갤 실익이 없고, JB는 §2(16셀 × ~17.6 h, ETA 07-27 09:40)로 차 있다. **JB가 할 일 없음.** 상세 = `research-wiki/survey/flirds-paper-experiment-plan.md` §4.4 · `REMAINING-slurm-YH.md` §6b.

## 2. G2·G9 seed2 — (a) 재학습 오라클 (16셀 · ~168 GPU-h)

### 진행 상황 (2026-07-26 19:15) — 8/16 완료

- **seed1 8/8 완료**(iid·dir1 × clean·label-flip·free-rider·grad-noise) · **seed2 0/8 실행 중**(`1878819_40-47`, 3–4 h 경과). 실패 0.
- **셀당 실측 ~17.6 h**(`oracle_a.time`=63,327 s = (a) 2¹⁰ 재학습 지배; 종전 11.4 h 추정은 무경합 구셀 값). **전량 ETA ~07-27 09:40**(마감 07-29 00:00 대비 ~38 h 여유). 셀당 wall 24 h ≫ 17.6 h → timeout 위험 없음.
- **검증 PASS(textbook)**: iid_free-rider `oracle_a.phi` = 프리라이더(0,4,6,7) ≈ −0.008 / 정직 ≈ +0.37. **phi_a = `metrics.json['oracle_a']['phi']`**(rundir 내장; 별도 _aonly 신규축 없음). corrupt(seed1)=`[0,4,6,7]`.
- **완료 rundir 8개 커밋**(seed1; **push=Yonghee**). ⚠ **공유 repo** — 타계정 커밋이 워킹트리에 섞여 mtime/glob 카운트가 오염된다(예: git-checkout된 `cifar10_*_seed2`). 완료판정·집계는 반드시 셀명 정규식 `mnist_(iid|dir1)_(clean|label-flip_fr0.70|free-rider|grad-noise)_seed[12]` 로 필터.

> **무엇**: N=10 전원참여에서 **(a) 2¹⁰ 재학습 오라클 + (b) 2¹⁰ + 9방법 φ**. (a)는 방법-중립 참값이라 **전 방법을 채점하는 유일한 무대**인데, 현 C1 시나리오가 확정 오염축과 한 칸도 안 겹쳐서 다시 정렬하는 것이다.
> **비교불가성은 남는다**: (a)는 2^N 재학습이라 **N=100 에서 원리적으로 불가** → 1A-CNN(N=100 부분참여)과 N·참여율은 못 맞춘다. 맞출 수 있는 건 **오염축과 파티션뿐**이고 이 잡이 그걸 한다. 논문에도 명시.

- **셀**: **`--array=24-31,40-47` = 16셀**(mnist seed1 8 + seed2 8). **seed0 잔여 2셀(`14-15` = dir1 free_rider·grad_noise) → HJ 로 이관**(2026-07-26; JB 3090 큐에서 `scancel 1878820` 후 HJ remaining §3d 에 기록). Slurm 4계정을 **GPU-h 로 균등화**한 몫이다(YH `0-7,16` · JW `32-39,17-23,8-13` · HJ = G12+G10 **+ G9 seed0 2셀**).
  인덱스 규약: `SEED=IDX/16` · 그 안에서 `0-7`=cifar10, `8-15`=mnist · 파티션 `iid,dir1` × 4위협(clean·lf@0.70·free_rider·grad_noise).
- **비용(실측)**: `t_a` = cifar10 **32,808 s ≈ 9.1 h** · mnist **41,168 s ≈ 11.4 h**(`runs/track_c/c1_oracle/*/metrics.json`). 궤적 ~103 s·전 방법 합 ~8분은 무시 가능 → **셀 ≈ t_a**.
- **✅ 착수 게이트 해소 — C-b·C-a 가 `origin/main` 에 착지했다**(`d09e528`, 07-25). `git pull` 후
  바로 제출한다. env = `C1_PARTITION`(iid\|dir1) · `C1_THREAT`(clean\|label_flip\|free_rider\|grad_noise) · `C1_FLIP_RATE=0.70`.
- **JB 의 두 판단이 결과적으로 옳았다.**
  ① 워처 술어(`grep C1_THREAT` on `origin/main`) — 구 러너로 제출하면 **실패하지 않고 조용히
  틀린다**(미지 env 무시 → `C1_SCENARIO=iid` 로 도는데 `C1_RUN_NAME` 만 먹혀 위협 라벨 이름에
  iid-clean 데이터가 들어가고 EXIT=0). 술어로 막는 게 유일한 방어였다. 슬롯 idle 도 맞다.
  ② C-b 를 포팅하지 않은 것 — JW 는 별도 구현(`989f5ca`)을 만들어 정본이 두 벌 됐고 철회했다
  (`f056b16`). JB 는 그 왕복이 없었다.
- 계획대로 `C1_ORACLE_A=0` 빌드-only 스모크(별도 `_smoke` 이름)로 새 threat 경로를 먼저 검증할 것.

  **대조 기준 = 전 위협 공통 고정 `⌊ρN⌉`=4** (`track_c1.py:187-196`; **seed-only** =
  dataset/partition/dose 무관, `default_rng(1000+seed)` 의 첫 소비):

  | **seed1 (JB)** | **seed2 (JB)** | (참고) seed0 |
  |---|---|---|
  | `[0,4,6,7]` | `[3,4,6,9]` | `[1,4,6,7]` |

  `rates=[0,…]` = free_rider 라벨 무접촉(정상).
  ⚠ **N=100 주무대(`track_c2`)는 label_flip 만 Bernoulli(ρ=0.4)라 개수가 변동한다**(실현
  39/48/47). 이 무대가 고정을 쓰는 건 의도적이다 — 도즈를 0.70 으로 고정해 FedCorr 재현이
  아니고, N=10 에서 Bernoulli 는 3/7/6 으로 흔들려 4위협이 한 fidelity 표를 공유할 수 없다.
  따라서 **`corrupt=` 가 위협마다 같게 나오는 것이 정상**이다(버그로 오해하지 말 것).

```
cd $REPO && mkdir -p runs/track_c/c1/_logs
sbatch --array=24-31%8    runs/track_c/c1/sbatch_c1_axis.sh   # mnist seed1
sbatch --array=40-47%8    runs/track_c/c1/sbatch_c1_axis.sh   # mnist seed2
# seed0 잔여 2셀(--array=14-15) → HJ 이관 (2026-07-26; scancel 1878820 · §3d HJ)
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
