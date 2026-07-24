# REMAINING (Slurm RTX3090 풀) — CNN + 작은-N LLM + LLM downstream overflow

> 실행처별 인수인계 **2부작** 중 **yonsei Slurm RTX3090** 몫. 짝 = `REMAINING-b200.md`(HVP·fidelity·timing + SFT 팩킹).
> **~~vast 폐기~~(2026-07-24)**: R4 downstream SFT는 B200 팩킹이 주력, **3090은 26일 free 후 seeds-1-2 overflow 흡수**(§6·24GB 적재).
> **마감(신): 실험 07-28 / 논문 07-29 21:00** — seed0 우선(`REMAINING-b200.md` §1a 2단계).
> **현재: 캠페인 실행 중(2026-07-24 등록·모니터링) — 아래 「실행 현황」 참조.** push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만.
> 논문·문서 정본 = `paper/workplan/00-INDEX.md`. 기존 rundir은 read-only.

## 실행 현황 (2026-07-24 저녁 갱신 · 커밋 80ebf30 시점)

완료분 rundir 커밋 `47680ec`(§4)·`80ebf30`(§1·§3) + Yonghee merge. **§5 파일럿 OOM 실패 1건**(§5 참조).

| § | 실험 | 상태 | 진척 |
|---|---|---|---|
| §1 | c2fid CNN fidelity | 🟡 seed0·1 완료·seed2 진행 | **110/144**(s0 48✓·s1 46·s2 16) 실행 중·8슬롯 점유 |
| §2 | W-B P1w observer(T2) | 🟡 파일럿 완주·게이트 대기 | seed0 파일럿 **29셀 완료**; `p1wgate`가 GPU 확보 시 판정 → 통과면 30-89 자동 |
| §3 | C-fr frrand full-method | 🟡 파일럿 완주·게이트 대기 | seed0 8/8 EXIT=0·커밋(`80ebf30`); `frgate` PD(GPU 확보 시 seeds1-2 판정) |
| §4 | C1 β0.3 재실행(30셀) | ✅ 완주·커밋 | β0.3 canonical **승격 커밋 `47680ec`**; ShapleyFL만 β0.5→0.3 변화 검증·타 10방법 비트동일 |
| §5 | L8 **silo5 (a)-leg만**(gsm5 보류) | ⏸ 3090 취소·A6000 이관 | 3090 SFT OOM 확인 → **A6000(48GB)서 다른 아이디로 실행**(별도); 이 환경 §5 잡 취소(silo5a 실패·l8gate scancel). gsm5 취소 완료 |

- **한도**: 동시 8-GPU(QOSMaxGRESPerUser) + §1 array `%8`. 현재 §1 c2fid가 8슬롯 점유 → 게이트 3종(p1w·fr·l8) PD 대기(각 GPU 확보 시 발화).
- **파일럿→GO 게이트**(§2·§3·§5)는 Slurm `--dependency`로 자동(세션 무관·fail-safe): 완주 시 GPU-h 실측 보고 후 잔여 leg 자동 제출.
- **예상 종료**(경합 유동): §1~§4(CNN) ~07-25, +§5(LLM) ~07-26.
- **seed0 우선(2026-07-24 · change #3)**: 전 실험 seed0 완주 = 논문 착수선 → seeds 1-2 보강(`REMAINING-b200.md` §1a 2단계).
  실행 순서 = §1 c2fid **seed-major**(s0 48셀✓) · §2/§3 seed0 파일럿→게이트 · §5 silo5-a **seed-major 정정** · §4 C1 저비용 dataset-major = **의도된 예외**.
  **전역 배리어(Q2)**: 게이트가 실험별이라 전역 seed0-우선은 자동 아님 → §1 seed1/2 hold 계획이었으나 **이제 무의미**
  (§1이 이미 110/144 = seed0·1 완료+seed2 진행). 남은 seed0 병목은 **§5뿐인데 그건 GPU 경합이 아니라 OOM**(§5) → hold 불요.
- **gsm5 보류 = 취소 완료**(Yonghee, 2026-07-24): gsm5 파일럿(1866894)+게이트(1866895) scancel 됨. silo5a 파일럿(1866896)은 **OOM 실패**·게이트(1866897)는 **scancel**(이 환경 §5 실행 종료; A6000 다른 아이디로 이관) → §5.

## 0. 환경

- **CNN(§1 c2fid·§2 W-B·§3 C-fr·§4 C1)**: conda `lora4cl`(`/home/chyoyhr/anaconda3/envs/lora4cl/bin/python`,
  torch 2.11.0), Slurm partition `base_suma_rtx3090`, 8-GPU QOS. sbatch 스크립트·로그 경로는 각 항목 런북.
  리포 루트(서버) = `/home/chyoyhr/projects/flirds/`.
- **작은-N LLM(§5 L8: gsm5·silo5 a-leg)**: 이 서버에선 CNN과 **동일 conda `lora4cl`**(2026-07-24 확정; 아래
  "venv 계열/다른 env"는 B200 기준). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + 24 GiB 노브 `VAL_CHUNK` + `HF_HOME=/scratch/chyoyhr/hf_home`.
- 공통: `HF_HUB_OFFLINE=1`, `codes/`에서 `PYTHONPATH=.`.
- **더 큰 GPU(2026-07-24 조사)**: 이 클러스터 `base_qos`로 **A6000 48GB**(`suma_a6000`·`gigabyte_a6000` ~96장)·**RTX6000Ada 48GB**(`asus_6000ada`)·**RTX4090 24GB**(`suma_rtx4090`) **즉시 접근**(모든 파티션 AllowAccounts=ALL — 계정 아닌 QOS가 게이트). **A100 40/80GB**(`suma_a100`)·**RTX PRO 6000 96GB**(`asus`/`gigabyte_pro6000`)만 특수 QOS(`a100_qos`/`pro6000_qos`). 8-GPU 상한은 per-user라 계정 2개면 동시 16. **24GB OOM 실험(§5 silo5-a)은 A6000로 해결.**

## 1. c2fid 본런 143셀 — CNN fidelity 주무대 (🟢 실행 중 — 70/144; seed-major=seed0 우선, s0 48셀 완료)

- 파일럿 `cifar10_dir1_grad-noise_fid_seed0` 완주·커밋(`570a93f` 그리드는 별개) —
  **실측 1.05 GPU-h/셀**(3,777s = 궤적 재생 284 + (b) 2¹⁰×120 오라클 824 + 8방법 2,669; peak 3.3 GiB)
  → 본런 **≈150 GPU-h**(fmnist 셀은 더 낮음, 8-GPU wall ~19h).
- **Yonghee GO 후** `sbatch runs/track_c/c2fid/sbatch_fid.sh`(143셀) → `runs/track_c/c2fid/make_analysis.py`
  → F-1~F-4 사전등록 대조(MISS 포함 보고). 스키마·게임 캐비엇·(b) 라운드 샤딩 = `runs/track_c/c2fid/README.md`.
- **채우는 overview ⬚**: §5.2 (메인) c2fid fidelity · §5.4 c2fid φ-AUROC · §5.6 F-4 dose 해상도 ·
  부록 C cross-game c2fid 전표 · 부록 D (b)-target 안정성 · figure F3(메인쌍)·F6(탐지, L2와 공유).

## 2. W-B — CNN P1w twin leg (W-A 판정 완료·W-B 파일럿 29셀 완주·게이트 대기)

> **정본 = `runs/track_h/RUN_P1W_CNN.md`**(W-A 판정·실행 절차·비용·수록 규칙). 스펙 =
> `paper/workplan/T4-p1w-cnn-relay.md`. 커밋 `93ee942`(무GPU 산출물). **P1w ≡ 기존 P2**
> (`gatew_v2`/`t2_signw`) — 신규 코드 없음, 신규 실행은 **T2 재학습 leg만**(T1 = skew 캠페인 재사용).

- **W-A 완료(무GPU, 로컬)**: restack 드리프트 312쌍 — recovery 앵커(oracle_excl 0.0010·
  vanilla 0.0024)·P1w(`flirds_gatew_v2` 0.0063) mean|Δ| 전부 분석 밴드 내 → **dir1 P2를 P1w로
  귀속, 재실행 불필요**. dir1 canon rundir 재생성 = overview §3.2.3 일치(P1w-T1 오염평균 .5913 /
  P1w-T2 .5959). **FedIF 역전 확인**(P1w on .6011 / re .6159 > flirds) = 수록 규칙 '타 소스 역전' 발동.
- **W-B 실행(신규 = T2 leg)**: `sbatch runs/track_h/sbatch_cnn_p1w.sh` = flirds-only observer +
  `C2_T2=1`, 확장 90셀({cifar10 shard/qskew/iid, fmnist iid/dir1} × {clean,fr,frrand,gn,lf@0.70,
  strmain} × 3seed). `track_h/rundirs_cnn` 착지 → skew T1과 셀키 병합. 게이트 HP = R1 verbatim.
  1. `mkdir -p runs/track_h/_logs` → **파일럿** `sbatch --array=0-29%8 …`(seed0, 30셀 ~10–11 GPU-h).
  2. 완주 후 병합 검증: `python runs/track_h/make_p1w_cnn_table.py`(T2 rows>0·dir1 canon 재현) +
     GPU-h 실측 → **Yonghee GO 게이트**.
  3. `sbatch --array=30-89%8 …`(seeds 1-2) → `make_analysis.py` + `make_p1w_cnn_table.py`.
  4. 보고: W-B 표(P1 vs P1w, 위협×파티션) + H-15 대조 + **FedIF 확장 재현 = W-D 대기**(flirds-only) +
     수록 의견 — **W-B 단독 판정 금지**(L7·W-A 종합 후 Yonghee 확정). 비용 전체 **~30–32 GPU-h**.
- **규칙**: 결과 = overview §3.2.3 이웃 신규 소절 → paper·T2는 그로부터 · cifar10 iid는 stack-caveat
  (clean/fr/gn/lf T1 앵커=B200; recovery로 읽기) · 다른 세션 파일 커밋 금지(이 leg 산출물 = rundir + `analysis/p1w_cnn*`만).
- **채우는 overview ⬚**: §5.3 CNN P1w(크기-가중) 규칙부.
- **W-D(후순위·별도 승인)**: 확장 무대 비-flirds 점수원 8종 → FedIF 역전 확장 재현 판정용.

## 3. C-fr — CNN frrand full-method 완성 (2026-07-23 Yonghee 신규)

> `REMAINING-b200.md` §3(R4 frrand)의 CNN 대응. 현재 CNN competition의 frrand은 **flirds 단독 leg**
> (online-only; 다른 7방법·retrain 미실행) → full-method + retrain으로 완성해 "exact-0 생존 vs
> renorm 붕괴"를 random free-rider에서도 검증(frzero 대칭화).

| # | 셀 | 내용 | 비용(GPU-h, 추정) | 상태 |
|---|---|---|---|---|
| **C-fr** | C2(CNN) frrand 완성 | cnn_competition frrand을 flirds-only → **full-method**(gtg·fedsv·comfedsv·shapleyfl·flirds1st·lossheur·fedif) **+ retrain(T2)** — dir1 × 3-seed(논문 §5.3 무대); frzero 셀과 동형 | ~15–30(dir1 3-seed online+retrain) | ⚪ **파일럿 제출됨(2026-07-24; array 0-7 seed0, PD)** → 게이트 통과 시 seeds1-2(8-23) 자동 |

- **구현**: 기존 `track_c2` competition에 frrand threat을 전-방법으로 확장 + `C2_T2`(retrain) = 신규 코드 없이 threat 커버리지만 확장.
- **채우는 overview ⬚**: §5.3 CNN 8점수원 표의 **frrand¹ 열**(현재 flirds 외 "–" → 7방법+retrain 채움) + §5.4 CNN frrand 탐지 AUROC(observer가 source별 `auroc` emit — `cnn_competition.csv` auroc 열).

### 실행 레시피 (확정 — 추측 없음)

> ⚠ **frrand 참조 셀이 "없다"는 진단은 오진.** 아래가 복제할 frzero-동형 sibling과 러너·템플릿이며 전부 리포에 있음.

- **frzero 참조 셀**(mirror 대상, 완전 명세): `runs/track_h/rundirs_cnn/cifar10_dir1_free-rider_<src>_seed{0,1,2}`
  (rundir 토큰은 `frzero`가 아니라 **`free-rider`**). config = n100·frac0.1·R120·ep5·lr0.01·b64·val2000·test8000·target0.6,
  gate{burn_in10·tau0·min_obs2·probation_every5·z_c1.5·alpha_w1}, `sources=[flirds,flirds1st,lossheur,gtg,fedsv,comfedsv,shapleyfl,fedif]`.
- **러너**: `experiments/track_c2.py`, `C2_THREAT=frrand`(1급 배선 확인 — track_c2.py L75 threat 목록·L293 `make_delta_transform(...,"frrand")`·L161 `FRRAND_MULT`). `C2_MODE=full`.
- **템플릿 = `runs/track_h/sbatch_strmain.sh`** — **이 dir1 competition에 새 threat 한 열을 RTX3090에서 추가한 선례**(label_flip strmain 편입). C-fr = 그 sbatch에서 `C2_THREAT=label_flip`→`frrand`로 바꾸고 strmain 특유 부분(strength=main) 제거. 나머지 동일:
  - `SRCS_P1=(flirds1st lossheur gtg fedsv comfedsv shapleyfl fedif)` 7소스, arms=`<src>_gate_v2,<src>_gatew_v2,<src>_mult,<src>_zgate_v2`
  - +1 `obs` 셀 = `C2_ARMS=observer` + `C2_T2=1`(T2 retrain) → **8 셀타입 × 3seed = 24 run**(strmain과 동형).
  - `C2_RUN_ROOT=$REPO/runs/track_h/rundirs_cnn`, `C2_RUN_NAME=cifar10_dir1_frrand_<tag>_seed<seed>`.
  - **P5 arms(`cgate`/`pweight`) 제외**(2026-07-23 P5 드롭) — `cnn_competition.csv`의 P5h/P5s 행은 레거시.
- **flirds 열은 이미 있음**: `runs/track_g/rundirs_cnn/cifar10_dir1_frrand_g_seed{0,1,2}`(track_g gate 그리드) → `make_analysis.py`가 셀키 병합.
- **스택 캐비엇**(이미 처리됨): 참조 frzero = B200/torch2.12, C-fr = RTX3090/torch2.11 — strmain이 동일 조건으로 이미 편입됐고 W-A가 recovery-정규화하 드리프트 무시가능(mean|Δ|≤0.006) 판정. 절대값 병치는 recovery로 읽음.
- **분석**: `python runs/track_h/make_analysis.py` → `cnn_competition.csv`에 frrand 7방법+retrain 행 추가. **파일럿-우선**(seed0=array 0-7 → GPU-h 보고 → GO 후 seeds 1-2 = 8-23), 다른 Slurm 항목(§1 c2fid·§2 W-B·§4 C1)과 우선순위는 Yonghee.

## 4. C1 30셀 β0.3 재실행 (ShapleyFL β 감사 — anchor5는 `REMAINING-b200.md` §4) — ✅ 완료·커밋 `47680ec`

논문 인용 ShapleyFL 값이 실제 **β=0.5** rundir 산출로 판명(감사 07-23): C1 30셀
(git_sha `5cb927b`, 06-12)이 β0.5→0.3 변경(`e89af94`, 06-25) **이전** — 재실행 계획 미반영.
- **셀 = C1 30셀**(track_c1; cifar10·mnist × 5시나리오 × 3seed). 오케스트레이터 = `rerun_beta03/`.
- **seed 순서**: array는 dataset-major·seed-interleaved(셀당 ~1.6h·전량 8슬롯 ~6h 1패스) → seed0-우선
  재정렬 안 함 = change #3의 **의도된 예외**(저비용이라 seed0 데이터가 전체와 거의 동시 완주).
- 실행 = `SFL_BETA=0.3` + 셀당 `RUNDIR_REPLACE=1`. **⚠ 실측(07-24): RUNDIR_REPLACE가 canonical(하이픈·무해시)을 못 덮고 `*_<hash>` 새 디렉토리 생성**(track_c1 네이밍이 06-12 이후 해시 접미사 부여) → **수동으로 canonical 승격 후 커밋(`47680ec`)**. §7 잔여 배선(track_c1 identity=None)과 연관 — 다음 재실행 전 네이밍 정합 필요.
- 완료 후: rundir 교체 커밋 → `make_analysis`/`make_fidelity` 재생성 → overview §3.1.2(C1) ShapleyFL 행
  갱신 → paper §5.2 sub(C1 표)·부록 C 값 갱신 + **B.5 재실행-대기 주석 삭제**(anchor5 B200분 함께 착지해야 완결).
- **갱신 대상 overview**(빈칸 아님 — 값 교체): §5.2 sub C1 vs (a) 표의 ShapleyFL 열(예: cifar10/qskew +0.81) · 부록 C.

## 5. L8 — retrain-(a) 스위트 (작은-N LLM: **silo5 a-leg만** · gsm5 보류) — ⏸ 이 환경(3090) 실행 취소 → **A6000(다른 아이디)에서 실행**

> **⚠ 스코프 변경 (2026-07-24 Yonghee): gsm5·anchor5 보류, silo5 (a)-leg만 활성.**
> (a) retrain 오라클 = **silo5 단독**으로 확보. 근거·정리 = 아래 및 `paper/workplan/T5-retrain-a-suite.md` §1 보류 배너.
> — **silo5**(non-IID)만이 실재 cross-seed 신호(clean +0.87 / noisy +0.93)를 갖는 유의미한 (a)-검증 무대.
> — **gsm5**(IID, 주무대 데이터)는 near-additive·ρ≈0 축퇴 무대라 anchor5 기존 0.933과 정보 중복 → 보류(코드·캐시 존치, 부활 시 §5.1 그대로).
> — **anchor5**(IID, Alpaca) 기존 0.933은 **보류 참조**(재실행 없음; `REMAINING-b200.md` §4 β0.3 재실행도 보류).
> **실행 조치(2026-07-24 확정)**: gsm5 파일럿+게이트 취소 완료(Yonghee). silo5a 파일럿(clean seed0) = **CUDA OOM 실패(EXIT=1, 17:42)** —
> SFT 재학습이 24GB 초과(아래 메모리 절). **→ 이 환경(3090) §5 실행 취소**(silo5a 실패 + `l8gate` scancel 1866897). **silo5 (a)-leg는
> A6000(48GB)에서 다른 아이디로 실행 예정** — 이 리포에 A6000 sbatch 미배선(별도 진행). fidelity leg라 하드웨어 독립이므로 A6000 산출 φ·(a)-vs-(b)는 유효.
> **⚠ 결과 여파**: silo5 (a)-leg 미완인 동안 §5.2 sub (a) 칸은 ⬚ 유지(anchor5 0.933은 보류 참조로 폴백 가능).

> 스펙 = `paper/workplan/T5-retrain-a-suite.md`. 코드 **커밋 완료**(TRACKED·clean). **전 셀에 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**(단편화 방지).
> **env(이 서버 확정 2026-07-24)**: CNN과 **동일** conda `lora4cl`(§0의 "venv 계열"은 B200 기준·무효). `HF_HOME=/scratch/chyoyhr/hf_home`
> (model+gsm8k+5도메인 캐시 완비), 런타임 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`. sbatch = `runs/phase2_matrix/sbatch_{gsm5,silo5_a,l8_gate}.sh`.

**구현 파일(커밋됨)**: `experiments/phase2_matrix.py`(REGIME=gsm5 신설 + dual (a)+(b) 오라클 + `report_vs_a`) ·
`experiments/track_a_silo5.py`(신규) · `flirds/data/llm.py`(`build_gsm8k_iid`에 `per_client`) ·
`flirds/oracle/exact_sv_llm.py`(`subset_valloss_utility`) · `runs/phase2_matrix/merge_silo5_a.py`(신규) ·
테스트 `tests/test_gsm5_a.py`(3종 green). gpt2 와이어링 스모크로 gsm5 전 경로((a)+(b)+9방법+report_vs_a) 확인 완료.

**⚠ 메모리(24 GiB 3090 — 실측):** gsm5는 SCALE==1B에서 `val_chunk`을 **자동 2**로 낮춤. B200용 기본 `val_chunk=10`은
estimator **2차 HVP**(`jvp∘grad`, eager-attn)가 **~38 GiB** 필요 → 3090 **OOM 실측**. `val_chunk`은 exact chunk-sum →
**φ 값 불변(메모리 전용)**. `val_chunk=2`에서 peak ~22 GiB로 **완주 확인**. 여유 더 필요하면 `VAL_CHUNK=1`.
(B200에서 돌리면 `VAL_CHUNK=10`으로 override해 가속 — 값 동일.) **⚠ silo5 (a)-leg 정정(2026-07-24 실측)**: HVP는 없지만
**SFT 재학습 자체가 24GB OOM**(`trl compute_loss` 활성서 5.87 GiB 추가 요구 vs 4.89 free = ~1GB 부족; `expandable_segments` 켜도 부족).
1B full SFT(모델+Adam+활성) > 24GB → **3090 부적합, A6000 48GB 필요**(base_qos 접근 가능·§0). `VAL_CHUNK`은 HVP용이라 이 학습 OOM엔 무효.

**0) 스모크(선택 — 이 세션서 이미 green; 서버 캐시에 gpt2+gsm8k 있을 때만):**
```
SMOKE_MODEL=gpt2 REGIME=gsm5 THREAT=noisy SEED=0 ROUNDS=2 MAX_STEPS=2 VAL=8 PER_CLIENT=6 \
  BATCH=2 VAL_CHUNK=4 VAL_MAXLEN=64 PERSIST=0 PYTHONPATH=. $PY -u experiments/phase2_matrix.py
```

**1) gsm5 무대 — ⏸ 보류(2026-07-24; 실행 안 함, 부활 대비 스펙 존치):**
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True REGIME=gsm5 THREAT=clean SEED=0 \
  PYTHONPATH=. $PY -u experiments/phase2_matrix.py     # THREAT∈{clean,noisy} × SEED∈{0,1,2}
```
- dual 오라클 **(a)+(b) 자동 on**(`ORACLE_A` 기본 on) · 9방법 · `report_vs_a`(spearman_a/pearson_a) 자동. 완료 = 로그 `MATRIX DONE` + rundir.
- 산출 rundir: `runs/phase2_matrix/rundirs/1B_gsm5_{clean,noisy}_nr0.7_s{seed}` — `phi.parquet`=(a)·(b)·9방법 φ,
  `metrics.json`=vs(b) + **vs(a)(spearman_a/pearson_a)** + timing.
- **비용/시간**: 셀당 (a) **2⁵=32 retrain × R30**이 지배적 → 3090서 여러 시간/셀. vs(b)만 싸게 선(先)확인하려면 `ORACLE_A=0`.

**2) silo5 (a)-leg(★활성 = 유일 (a)-무대) — {clean,noisy,frzero} × seed{0,1,2} = 9 leg (sbatch = seed-major: array 0-2 = seed0):**
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True REGIME=silo5 THREAT=frzero SEED=0 \
  PYTHONPATH=. $PY -u experiments/track_a_silo5.py     # THREAT∈{clean,noisy,frzero} × SEED∈{0,1,2}
```
- 기존 canonical `1B_silo5_{threat}`와 **동일 split·seed** 재현 → (a) 32-retrain(R=10)만 신규. 완료 = 로그 `SILO5 (a)-LEG DONE`.
- **전제**: canonical `1B_silo5_{clean,noisy,frzero}` rundir + 5도메인 데이터셋 HF캐시 존재(서버엔 있음). 기존 rundir **무수정**(read-only).
- 산출: `runs/phase2_matrix/rundirs/1B_silo5_{threat}_aonly_s{seed}`.

**3) 조인·분석:**
```
PYTHONPATH=. $PY runs/phase2_matrix/merge_silo5_a.py     # canonical ⋈ *_aonly → spearman_a/pearson_a + silo5_a_fidelity_1B.csv
```
- gsm5는 `make_analysis.py` 택소노미 밖이라 **스킵됨** → gsm5 vs(a)/vs(b) 수치는 rundir `metrics.json`에서 직접(또는 별도 롤업).
- 헤드라인(T5 §2): silo5 `(b)oracle` 행의 `rho_a` = 두 오라클의 **실재 신호 일치**(목표 clean +0.87 / noisy +0.93; overview §5.4).

**4) 완료 후**: rundir 커밋 → overview §3.1.1 이웃 신규 소절 기입 → paper §5.2 sub 표(T1)·T2 F2 갱신.
  **⚠ 방법 범위(2026-07-24 기준)**: silo5 (a) retrain 표 = **전 방법**(renorm-4+FedIF 포함) · **vs (b) 열은 same-game 3만**
  (flirds·flirds1st·lossheur; cross-game은 in-run 오라클과 다른 게임이라 vs (b) 미채점). rundir은 전 방법 φ 산출(무해) — 표기만 제한.
**사전기대(T5 §2)는 실행 전 커밋**, HIT/MISS 그대로 보고. gsm5·3B/7B (a)는 하지 않음.
- **채우는 overview ⬚**: §5.2 sub **silo5 (a)-leg**(gsm5 주표는 보류 → silo5가 유일 (a)-무대).

## 6. LLM downstream overflow — B200 보조 (seeds 1-2, 26일 free 후 · 2026-07-24 신규)

> **역할**: R4 downstream SFT(L11·L4·L9-arms·L7-arms)의 주력은 **B200 팩킹**(`REMAINING-b200.md` §1a).
> 3090은 **CNN(§1-4)+L8(§5) 완주(~26일) 후** 놀지 않게 **seeds 1-2 overflow만** 받는 **조건부 보조** — B200가 28일 창에 못 담는 꼬리 흡수.

- **왜 3090에 됨 (07-24 실측 정정)**: 24GB 적재 = **순수-SFT arm만** — T2 재학습·**cum 재사용** online-gate 적용·downstream eval = allocated **~15–18 GiB**(downstream 15.0·val-curve 17.7; T2 phase는 로깅 갭이나 구조상 동급). **VAL_CHUNK 불요**(HVP 아님).
  - ⚠ **retrain-scoring arm**(gtg·fedsv·comfedsv·shapleyfl의 online/renorm **값 산출**, exclusion 재학습)은 **~32 GiB → 24GB 불가**. 이 값 산출은 **B200가 관찰자로 전담**하고, 3090엔 그 **cum을 재사용하는 순수-SFT arm만** 넘긴다(아래 cum 전제와 동일 조건). ⟹ L11/L4/L9-arms도 "cum 재사용" 형태로만 3090행.
- **무엇을(우선순위)**: **seeds 1-2만**(seed0은 B200가 ~26일에 먼저 뽑음 = 논문 착수선). 큰 독립 물량 순 = **L11(online 42런) → L4(renorm T2 6셀) → L9-arms → L7-arms**.
- **전제 = 관찰자 cum**: downstream은 관찰자 φ 재사용(HVP 재실행 0). **B200가 산출한 cum(`metrics.json`의 `observer_cum`, 수 KB)만 3090에 복사** → arms가 읽어 실행. L11=L1 cum·L9-arms=B200 L9관찰자 cum·L7-arms=§2a 경로.
- **러너·명령**: `experiments/track_g.py`(B200과 동일 러너·코드). env = §0(작은-N LLM = conda `lora4cl`). 예:
  ```
  RUNDIR_ROOT=$REPO/runs/track_h/rundirs_llm_3090 \
  REGIME=gsm50k5 THREAT=<clean|noisy|frzero> SEED=<1|2> \
    ARMS=<src>_gate_v2 OBS_SOURCES=<...> T2=<0|1> T2_LEGACY=0 T2_P5=0 \
    PYTHONPATH=. $PY -u experiments/track_g.py
  ```
  (arm 세트·OBS_SOURCES는 B200 L11/L4/L9/L7 큐와 동일; 착지 root만 분리해 canonical `rundirs_llm` 무수정.)
- **스택 캐비엇**: 3090(torch2.11)/B200(torch2.12) 절대값 병치 금지 → **recovery 정규화로 읽음**(W-A mean|Δ|≤0.006; CNN C-fr 선례). **timing.json은 §5.5 cost에 사용 금지**(B200 실측만).
- **조건부**: B200 팩킹 배속이 ≥2.5×면 3090 보조 거의 불요(B200가 28일에 완주). ~1.5×거나 B200 공백 시 이 §6로 꼬리 흡수 = seeds 1-2 3-seed 완주 보장.
- **분석**: `make_analysis.py` LLM 로더에 `rundirs_llm_3090` root 추가(track_h dup-win) → B200 산출과 셀키 병합.

## 7. rundir 정체성 — 잔여 배선 (CNN track 관련)

처방 1+2 구현 완료(07-23): 정체성 allow-list(`check_identity`/`precheck`; 우회 `RUNDIR_REPLACE=1`) +
β 단일화(`shapleyfl.BETA = env SFL_BETA, 기본 0.3`). 배선 완료 = `track_g`·`phase2_matrix`, 테스트 6개.
- **잔여**: ① `track_c1`·`track_c2`·`track_c2_fid`·`track_d`·`phase1_*`는 아직 `identity=None`
  (레거시 통짜 비교) — 이들 config에 `sfl_beta`를 추가하려면 identity 배선을 함께 해야 함
  (§4 C1 재실행은 이 배선 없이 `RUNDIR_REPLACE=1`로 우회). ② 처방 3(`superseded.json`) 미착수.
