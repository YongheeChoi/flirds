# REMAINING (Slurm · HJ) — A6000 48GB: R4 online 점수원 경쟁 (진행 중)

> 배분 정본 = **`REMAINING-00-INDEX.md`** · 수록목록 정본 = `research-wiki/survey/flirds-paper-experiment-plan.md`.
> **역할 = R4 §5.3 online 표의 7 비-flirds 행 중 seed0·1**(42셀; seed2 = JB).
> **마감: 실험 07-28 24:00 / 논문 07-29 21:00.** 3-seed(seed-major). push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만.
>
> ## ★ 즉시 조치 — **flirds1st 3셀만 완주, 나머지 L11 취소**
>
> 07-25 19:50 진행 상황 기준. 실행 중 8셀의 정체는 `SRC_I=(IDX%21)/3` 로 갈린다:
>
> | idx | 정체 | 스코프 | 조치 |
> |---|---|---|---|
> | **0·1·2** | **flirds1st seed0** × clean/noisy/frzero (58%, ~23:00 완주) | ✅ 안 | **완주** |
> | 3·4·5 | lossheur seed0 × 3 (46%) | ❌ 밖 | `scancel` |
> | 6·7 | fedif seed0 × clean·noisy (44%, frzero=idx 8 미착수라 2/3) | ❌ 밖 | `scancel` |
>
> ```
> scancel <jobid>_[3-20]       # lossheur·fedif·renorm-4 = 스코프 밖 (실행 중 포함)
> squeue -u $USER              # 0·1·2 만 남았는지 확인
> ```
> **✅ 실행 완료 07-25 20:20 — 단 범위는 `[3-41]`.** 문서의 `[3-20]` 은 seed0(0–20)까지만 끊어 **seed1 21셀(idx 21–41)이 대기로 남는다** — `21·22·23` = flirds1st seed1(= B200 c4 몫), `24–29` = lossheur·fedif seed1, `30–41` = renorm-4 seed1 로 **전부 스코프 밖**이다. 판정 기준이 "`squeue` 에 0·1·2 만"이라 그쪽을 따랐다. 결과 = `1876764_0,1,2` 만 잔존(flirds1st seed0 clean/noisy/frzero, 확인 완료).
> - **flirds1st seed0 3셀은 B200 c4 의 해당 3줄을 지운다** — c4 는 seed1·2 6셀만 맡는다(폴백용 주석은 c4 하단에 있다).
> - **lossheur·fedif 를 끊는 이유**: LLM downstream 스코프 컷으로 표에서 빠졌고, **1-seed 는 3-seed 규칙상 논문에 못 넣는다**. 3-seed 로 승격하려면 seeds1·2 에 +50 GPU-h(B200)가 더 드는데 c4 에 그 여유가 없다(G5 를 밀어내야 한다).
> - **이미 착지한 rundir 은 지우지 않는다** — 집계는 canonical `rundirs_llm` 이 dup-win 으로 이긴다.
> - **그 뒤 HJ 는 A6000 을 그대로 이어 G12(§3) → 3090 으로 G10(§3b)** 순으로 간다.

## 0. 환경 (실제 셋업 결과)

- **공유 `lora4cl` 사용 불가**: `/home/chyoyhr` 가 `drwx------`(700) → env 읽기 불가. **동일 스펙 재생성**으로 해결: `torch 2.11.0+cu128 / transformers 5.5.4 / trl 1.2.0 / peft 0.19.1`(= `requirements-llm.txt` 의 torch-2.11 스택). 빌드 = `runs/_probe/build_env.sh`.
- **공유 HF 캐시 직접 사용 불가**: `/scratch/chyoyhr/hf_home/token` 이 600 → 오프라인에서도 HF 가 읽어 `PermissionError`. **blob 자체는 읽힘** → 자체 `HF_HOME` 구성(`hub` 하위는 공유 캐시로 **심링크** = 재다운로드 0, `datasets/` 만 복사 = `.shuffle()` 캐시 쓰기용, **token 파일 없음**).
- 파티션 `suma_a6000,gigabyte_a6000` + `--qos=base_qos` · 동시 **8-GPU/user** 확인.
- 공통: `codes/` 에서 `PYTHONPATH=.`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### ⚠ `VAL_CHUNK` — 이제 **소스별로 자동**(sbatch 배선 완료)

- **증상(원래 문제)**: 기본 `VAL_CHUNK=10` 으로 flirds1st 를 돌리면 **~82 라운드에서 CUDA OOM**(48GB 소진; `flirds_values → _chunked → grad`). B200(180GB) 기준 설계라 생기는 **A6000 고유 제약**.
- **조치**: `VAL_CHUNK=3` 이면 정상(168 라운드 통과 확인). **φ 불변** — `_chunked` 는 청크별 grad 의 가중합이 전체 val-grad 와 **정확히 동일**(근사 아님). **`VAL_MAXLEN` 은 φ를 바꾸므로 건드리지 말 것.**
- **개선(07-25)**: 이 제약은 **grad 경로에만** 필요한데 7개 소스 전부에 걸려 있었다. 이제 sbatch 가 소스별로 준다 — **flirds1st·fedif = 3**(functorch val-grad), **lossheur·renorm-4 = 10**(forward-only @no_grad, 메모리 여유). 청크 합산이 exact 라 **어느 값이든 φ 동일 = 무위험**.
  속도 이득은 **미측정**이다: `make_llm_loss` docstring 은 (b) 오라클에서 청크를 키웠을 때 **~1.0×**(FLOP-bound)로 프로파일됐다고 적고 있고, 위 재측정상 same-game 셀은 학습이 지배한다. renorm 셀(스코어링 ~92%)에서만 여지가 있다.
- 제출은 `sbatch --array=0-41%8 runs/track_h/sbatch_l11_online.sh` 로 충분(`--export` 불요).

## 1. 완료 — silo5 (a)-leg (9/9)

| threat | 실측 범위 | 평균 | peak |
|---|---|---|---|
| clean | 8.56–8.87 h | 8.72 h | 26.3 GiB |
| noisy | 8.35–8.94 h | 8.65 h | 26.3 GiB |
| frzero | 5.73–6.13 h | 5.91 h | 26.3 GiB |

- **총 69.9 GPU-h**(사전 추정 26 은 2.7× 과소평가 — 실제 7.8 GPU-h/leg). peak 26.3 GiB > 24 → **48GB 필수**가 실측으로 재확인됐다.
- 산출 = `runs/phase2_matrix/rundirs/1B_silo5_{threat}_aonly_s{seed}` · 조인 = `PYTHONPATH=. $PY runs/phase2_matrix/merge_silo5_a.py`.
- **✅ 수록 예정**(Yonghee 확인 07-25). 앞선 "수록 위치 미정" 메모는 오독이었다 — anchor5 듀얼오라클과 **별개 항목으로 논문에 들어간다**. 추가 실행 0, 조인만 하면 된다(`merge_silo5_a.py`).
- **✅ 조인 실행 완료(07-25 21:51)**: 87 (threat,seed,method) 행 → `runs/phase2_matrix/silo5_a_fidelity_1B.csv`. 재생성분이 커밋본과 **바이트 동일**(워킹트리 클린) = 재현성 확인.
  - 3위협 전부 `Flirds / Flirds1st / Banzhaf / loss-heur = ρ_b, ρ_a 모두 +1.000`. 하위는 `FedIF +0.867/+0.933/+0.900`, `ComFedSV +0.833/+0.867`(clean 은 행 자체가 없다), `FedSV +0.933`(clean·frzero).
  - **⚠ 스크립트 말미 headline 과 실측이 어긋난다**: `merge_silo5_a.py` 는 "`(b)oracle` 행의 `ρ_a` = 두 오라클의 실신호 일치도(**clean +0.87 / noisy +0.93** target; overview §5.4)"라고 찍는데, **실제 출력은 3위협 모두 `+1.000`** 이다. 즉 (a)-retrain 과 (b)-in-run 이 완전 일치했다. 둘 중 하나다 — ①기대값 +0.87/+0.93 이 다른 무대(anchor5?)의 수치라 문구가 잘못 붙었거나, ②N=5·등n 축퇴로 +1.000 이 무정보(07-04 감사의 "deferred 최상위" 항목)라 headline 자체를 재서술해야 하거나. **논문에 쓰기 전 Yonghee 판정 필요** — 지금 표를 그대로 옮기면 본문 주장과 숫자가 불일치한다.

## 3. 담당 ① — G12: A축 lever probe seed 보강 (**15셀** · ~60 GPU-h · **A6000**) — 🟢 제출 완료 `1878707`

> **B200 c4 에서 넘친 유일한 물량**이다. c4 = L1 clean(18.6) + flirds1st s1·2(25.2) + G5(20) + G12 앞 3셀(9) ≈ 72.8h 로 74h 창을 채우고, **G12 나머지 16셀만 흘러넘친다**.
> HJ 가 맡는 이유 = **A6000·HF 캐시 셋업이 이미 있고 지금도 거기 있다** → 전환 마찰 0. 끝나면 3090 으로 넘어간다.

- **셀**: anchor5 `lr{1,2,3}e-3 × st{20,30}` seed1·2 중 잔여 + anchor5 `r{32,64}` seed1·2 + `noise_1B_r64` seed1·2 = **15**(c4 가 `lr2e-3_st20` s1·s2 와 `lr3e-3_st20` s1 을 가져감).
  **16 → 15 (07-25 HJ 확인, 코드 대조 완료)**: `lr1e-3_st10 seed0` 은 **추가 실행이 필요 없다** —
  `track_d.py` 의 `ANCHOR5` 기본값이 `lr=1e-3 · max_steps=10 · r=16` 이라 기존
  `runs/track_d/rundirs/1B_anchor5_seed0` 이 곧 그 셀이고, `runs/probe_signal/make_figures.py:120`
  이 이미 그것을 `("lrsteps", 1e-3, 10)` / `("rank_anchor", 16)` baseline 으로 읽는다.
- **제출 = `--array=0-14%8`** (`1878707`). 배열 순서를 우선순위로 깔았다:
  `0-4` lr3e-3·lr2e-3 → `5-8` lr1e-3 → `9-12` r32/r64 → `13-14` noise.
  **마감에 걸리면 뒤 인덱스부터 자르면 핵심 질문은 지켜진다.** config 는 대응 seed0 셀과 일치
  (lr 격자 `MMLU_LIMIT=40` · rank 셀 MMLU 전체) — cross-seed 비교는 같은 셀의 seed 간 비교라
  **이 일치가 축의 유효성 자체**다. 현재 `0-4` 실행 중 · OOM 0.
- **⚠ 다른 계정에 파급되는 발견 — track_d 의 HF 캐시 공백**: `vicgalle/alpaca-gpt4` 와 `cais/mmlu`
  가 공유 캐시에도 HJ 캐시에도 없었다(gsm8k 계열 6종 + Llama-1B 뿐). track_d 는 이 둘이 없으면
  **오프라인에서 시작조차 못 한다**. `hf_pin.py` 의 `REVISIONS` 가 비어 있어 최신 커밋으로 받으면
  되고 둘 다 public → HJ 의 `HF_HOME` 에만 추가(공유 캐시 무수정, +238MB) · 오프라인 재로딩 검증 완료.
  → **B200 c4 의 G5·G12 도 track_d 계열이므로 같은 공백을 먼저 확인**해야 한다
  (B200 은 과거 anchor5·probe_signal 을 돌렸으니 있을 가능성이 높지만, 확인 전 제출 금지).
- 러너 `track_d.py` / `probe_val_noise.py`, 착지 `runs/probe_signal/rundirs`(noise 는 `noise_probe/`).
- **R4 무대가 아니다 → `ROUNDS` 를 주지 않는다.** anchor5 는 N=5·R=30 이라 작지만 flirds HVP 경로이므로 **`VAL_CHUNK=2`** 권장(청크 합산 exact → **φ 동일**).
```
# 예: 한 셀
REGIME=anchor5 LR=3e-3 MAX_STEPS=20 ORACLE_A=0 FIDELITY=1 ARMS=1 MMLU_LIMIT=40 \
  SEED=2 LORA_R=16 RUN_NAME=1B_anchor5_lr3e-3_st20_seed2 VAL_CHUNK=2 \
  RUNDIR_ROOT=$REPO/runs/probe_signal/rundirs PYTHONPATH=. $PY -u experiments/track_d.py
```
- **부록·최저 우선**이라 마감에 걸리면 꼬리부터 버려도 된다. 다만 핵심 질문("lr 로 커진 φ가 cross-seed 실재 신호인가", 예측 ρ≈0)에 필요한 **`lr{2,3}e-3` 계열을 먼저** 돌린다.

#### 제출 완료 — job `1878707` (07-25 20:32, `--array=0-14%8`)

- 스크립트 `runs/probe_signal/sbatch_g12_lever_seeds.sh`(신규). **16 이 아니라 15셀**이다 — `lr1e-3_st10 seed0` 은 **별칭이 유효**해서 뺐다: anchor5 기본값이 `lr=1e-3 / max_steps=10 / r=16`(`track_d.py:89`)이라 `runs/track_d/rundirs/1B_anchor5_seed0` 가 곧 그 셀이고, **`make_figures.py:120` 이 이미 그것을 baseline 으로 읽고 있다**(README 도 동일 명시). 추가 실행 0.
- **배열 순서 = 우선순위**: `0-4` lr3e-3·lr2e-3 → `5-8` lr1e-3 → `9-12` rank r32/r64 → `13-14` val-noise r64. 마감에 걸리면 **뒤 인덱스부터 `scancel`** 하면 핵심 질문은 지켜진다.
- **config 는 대응 seed0 셀과 정확히 일치**시켰다 — lr×steps 셀은 `MMLU_LIMIT=40`, rank 셀은 MMLU 전체(=0). cross-seed 비교는 같은 셀의 seed 간 비교라 이 일치가 축의 유효성 그 자체다. `VAL_CHUNK=2` 만 다른데 이건 메모리 knob 이고 청크 합산이 exact → **같은 게임**이다(참조셀 B200 peak 98.9 GiB → 48GB 에선 축소 필수).

##### ⚠ 선행 조치 — HF 캐시에 **alpaca-gpt4·cais/mmlu 가 없었다**

- 공유 캐시(`/scratch/chyoyhr/hf_home`)에도 **없다** — 거기엔 gsm8k 계열 6종 + Llama-3.2-1B 뿐이다. track_d(anchor5)는 `vicgalle/alpaca-gpt4`(학습 데이터)와 `cais/mmlu`(downstream)를 쓰므로 **오프라인으로는 시작 자체가 불가**했다.
- `flirds/hf_pin.py` 의 `REVISIONS` 가 전부 비어 있어(`rev()`→None) 최신 커밋을 받으면 되고, 둘 다 public(토큰 불요). **HJ 자체 `HF_HOME` 에만 추가**했다(공유 캐시 무수정, +238 MB → 419 MB). 오프라인 재로딩 확인 완료(alpaca 52,002 / mmlu-test 14,042 / 클라 샤드 4,000×5).
- **B200·YH 쪽에서 track_d 계열을 새로 돌린다면 같은 공백을 먼저 확인해야 한다.**

## 3b′. G10 착수 게이트 — **열렸다 · 제출 완료** (07-25 21:55)

- C-a 가 `d09e528` 로 착지(`track_c2.py:157` 에 `"mnist": LeNet5`) → HEAD 에 포함 확인. 파티션도 C-b 로 `base_suma_rtx3090,dell_rtx3090` 확장 반영됨.
- **제출**: `1878912`(`--array=0-71%8`, seed0) · `1878913`(`--array=72-215%8`, seeds1·2).

### ⚠ HJ 계정에서 제출할 때 필요한 3개 override (스크립트 파일은 **무수정**)

`sbatch_cnn_mnist_comp.sh` 는 YH 홈 기준이라 그대로 쓰면 **216런 전량이 기동 실패**한다 — `--output` 이 `/home/chyoyhr/projects/flirds/...`(700, 쓰기 불가)라 Slurm 이 출력 파일을 못 만든다. `REPO`·`PY` 도 같은 홈을 가리킨다. 제출 시 덮어썼다:

```
sbatch --qos=base_qos --output="$REPO/runs/track_h/_logs/%x_%A_%a.out" \
       --export=ALL,REPO=/home/rlaguswls186790/flirds,PY=<HJ python> \
       --array=0-71%8 runs/track_h/sbatch_cnn_mnist_comp.sh
```

### 선행 조치 2건

- **torchvision 부재** → `0.26.0+cu128` 설치. **`--no-deps` 로 넣어 `torch 2.11.0+cu128` 을 보호**했다(그냥 설치하면 torch 를 갈아끼워 실행 중인 LLM 잡의 스택이 바뀐다). `pillow` 도 없어 같이 설치. 검증: `torch 2.11.0+cu128 / torchvision 0.26.0+cu128`.
- **MNIST 미캐시** → `~/data` 에 선다운로드(60k/10k). `flirds/data/cnn.py:25` 가 `download=True` 라 **216런이 동시에 같은 경로로 내려받으면 레이스**가 난다(계산노드 인터넷 여부도 불확실).
- **CPU 스모크 통과**: mnist/dir1/label_flip@0.70, flirds 4-arm → `AUROC 1.000/0.978/0.989/1.000`, `TRACK-C2 RUN OK`. 216런 투입 전 C-a 배선을 실증했다.

### ⚠ QOS 캡은 **파티션을 가로지른다**

G10 은 3090 인데도 대기 사유가 `QOSMaxGRESPerUser` 다 — A6000 잡 8장이 3090 잡을 막는다. 즉 **파티션을 늘려도 총 8슬롯은 그대로**이고, L11 잔여 + G12 + G10 이 **같은 8슬롯을 나눠 쓴다**(총 ~200 GPU-h → ~25 wall-h → 07-26 밤~07-27 아침). 제출 순서대로 G12 → G10 으로 흘러간다.

## 3b. 담당 ② — G10: mnist downstream (216런 · ~135 GPU-h · 3090)

- **⚠ 착수 게이트 = 코드 C-a**(`track_c2.py:157` MODEL_FN 에 `"mnist": LeNet5` 1줄; YH 담당).
```
cd $REPO && mkdir -p runs/track_h/_logs
sbatch --array=0-71%8    runs/track_h/sbatch_cnn_mnist_comp.sh   # seed0 (72)
sbatch --array=72-215%8  runs/track_h/sbatch_cnn_mnist_comp.sh   # seeds 1-2
```
- 2파티션 × 4위협 × (**8**소스 + 관측자) × 3seed = 216. mnist 는 track_g 그리드가 없어 **flirds 소스도 여기서 생성**(7이 아니라 8인 이유). **P1w 는 같은 rundir 에서 동반 산출**(추가 런 0).
- CNN 은 conda `lora4cl`(torch 2.11) + torchvision 데이터만 필요 — HF 캐시 불요.
- **파티션 = `base_suma_rtx3090,dell_rtx3090`**(sbatch 내장 · 07-25 확장). base_suma 단독은 여유 0
  (총 71장, 빈 6장은 draining node01)이고 dell 에 9장이 놀고 있었다 — JW 실측. 같은 RTX3090 이라
  스택 캐비엇 동일. `sinfo -o "%P %G %a"` 로 다른 3090 파티션이 보이면 더 붙여도 된다.
- **깨끗한 상태에서 시작한다**: YH 가 이전 판 근거로 큐에 넣어 둔 216셀을 **취소했다**(전량 PD ·
  mnist rundir 0개). 중복·팬텀 없음.
- **torchvision 미설치 가능성**: JW 계정은 없어서 `0.26.0+cu128` 을 별도 설치했다(torch 2.11.0+cu128
  불변). HJ 도 CNN 첫 제출 전 `python -c "import torchvision"` 로 확인할 것.

## 3c. 완주 후

- **JW·JB 의 `sbatch_c1_axis.sh` 잔여를 work-steal**(남은 `--array` 범위만; 같은 rundir 이름 = last-writer-wins 라 중복 = GPU 낭비).

---

## (이하 참고) 종전 L11 63셀 계획 — **스코프 컷으로 폐기**

> R4 §5.3 online 표는 CNN 처럼 8방법인데 현재 online 은 **flirds 만**(B200 L1) → 나머지 **7방법 T1 부호-게이트**를 채운다.
> **손대지 않는다** — 재제출·재정렬 금지. 완료분만 커밋.

- **셀**: 7 비-flirds(flirds1st·lossheur·fedif·gtg·fedsv·comfedsv·shapleyfl) × {clean, noisy, frzero} × seed{0,1,2} = **63**.
  - **G4a = same-game+FedIF 3종**(flirds1st·lossheur·fedif) 27셀 — **싸다**.
  - **G4b = renorm-4**(gtg·fedsv·comfedsv·shapleyfl) 36셀 — **비싸다**.
- 비-flirds 는 online 스코어링에 **HVP 불요**(값·1차) → retrain-scoring ~32 GiB → 48GB, **B200 cum 불요·자체완결**.
- 착지 root `rundirs_llm_hj`(canonical `rundirs_llm` 무수정) → `make_analysis` dup-win 병합.

### 실측 단가 (200 라운드 학습부 · `VAL_CHUNK=3`)

> **⚠ 종전 표(same-game 2.5 h / lossheur 3.2 h)는 HJ 오측정이었다 — 실측은 그 ~3×.** 07-25 19:00 8셀 동시 가동분에서 `train_runtime` 호출수(=라운드×K=5)로 재측정. 라운드당 132 s(same-game) / 166 s(lossheur) — 그중 학습만 110 s 라 `VAL_CHUNK` 영향은 미미하고, **비용은 200라운드 학습 자체**다. 아래 값은 8셀 전부 ±2% 내로 일치.

| source | R=200 실측(재측정) | **R=100** | 셀 수 | 소계 @R=100 |
|---|---|---|---|---|
| flirds1st | 7.33 / 7.37 / 7.51 h → ~7.4 h | **~3.7 h** | 9 | ~33 |
| fedif | 7.41 / 7.23 h → ~7.4 h | **~3.7 h** | 9 | ~33 |
| lossheur | 9.23 / 9.15 / 9.59 h → ~9.3 h | **~4.65 h** | 9 | ~42 |
| **renorm-4**(gtg·fedsv·comfedsv·shapleyfl) | ~23–28 h(JB 실측, 미검증) | **~12.5 h** | 36 | **~450** |
| | | | **63** | **~559 GPU-h** |

> **R=100 환산이 거의 정확히 절반인 이유**: 비용이 라운드에 선형인 두 항(200라운드 학습 110 s/round + 라운드당 스코어링)으로만 이뤄져 있다. 고정비(모델 로드·말미 downstream 평가)는 소수라 셀당 약간의 하방 오차만 남는다.

- 값은 진행률 외삽(200라운드 학습부)이며 **모델 로드 + 말미 downstream 평가는 미포함** — 실제 end-to-end 는 이보다 다소 길다. 첫 셀 착지 시 확정치로 교체할 것.
- **`--time` 조치 완료(07-25 19:00)**: 대기 셀 34개 중 **renorm-4 24셀만 `scontrol update TimeLimit=42:00:00`** 으로 증액(같은 근거 = 실측 23–28 h 가 24 h 벽에 걸림; rundir 은 말미에 한 번에 써지므로 timeout = 셀당 24 GPU-h 전손). **same-game 10셀은 24 h 유지**(7–10 h 면 충분 + 긴 `--time` 은 backfill 불리). **실행 중 잡은 건드리지 않았고**(INDEX §0 운용 원칙), `scancel`·재제출 0 — 증액은 비파괴 갱신이다. 파티션 `MaxTime=14-00:00:00` 이라 42 h 는 한도 내.
- **renorm 이 느린 이유**: 라운드마다 coalition/submodel 평가(`shapleyfl_round_raw` = val forward)가 들어간다 — forward 만 하는 lossheur 대비 2.4×, same-game 대비 라운드당 ~13×. **이건 버그가 아니라 논문이 주장하는 지수 비용 그 자체**(op-count §7.3: (b)·ShapleyFL 은 K 에 지수).

### 예상 종료 · 리스크

- **HJ 몫(seed0·1 42셀) = ~372 GPU-h @R=100**(same-game 18셀 ~72 + renorm-4 24셀 ~300). 8슬롯 → **~47 wall-h → 07-27 후반**. R=200 이었으면 88–103h(07-29~30)로 마감을 못 맞췄다 — **R=100 전환이 이 레그를 마감 안으로 들여놓은 것**이다.
- **JB 가 seed2(21셀)를 병렬로 맡는다** → 63셀 전체가 07-27 후반 착지. 단 **A6000 클러스터 여유가 10장**이라 HJ+JB 가 상시 16슬롯을 채우지는 못한다 → `asus_6000ada`(RTX6000Ada 48GB 8장)를 파티션 목록에 추가해 뒀다.
- **착지 순서가 방어선**: seed-major + 소스-major(21셀/seed = flirds1st→lossheur→fedif→gtg→fedsv→comfedsv→shapleyfl)라 **싼 same-game 이 먼저 착지**한다. 07-26 오전이면 seed0 same-game 9셀이 다 들어와 §5.3 online 표의 G4a 3행은 **1-seed 로 조기 확보**된다. 물량이 마감에 걸리면 잘리는 쪽은 항상 renorm 꼬리(seed1)다.
- **⚠ clean 은 컷하지 않는다 (2026-07-25 Yonghee: "clean 은 필수").** renorm-4 의 clean 12셀을 빼면 ~300 GPU-h 를 아낄 수 있지만, clean 열은 **오발화(false-firing) 판정의 근거**라 유지한다 — renorm 은 flirds 와 달리 clean 에서도 음수 φ 로 발화해 `equals_vanilla` 스킵이 안 되고 실제 개입이 일어난다. 즉 renorm 의 clean 칸은 "비어도 되는 대조"가 아니라 **결과 그 자체**다.
- **판정 시점 = 07-27 아침**: seed0 착지 + seeds1·2 진척으로 완주 여부를 본다.

### ⚠⚠ 최대 변수 = A6000 노드 drain (07-25 19:50 관측, 신규)

- **12노드 중 7노드가 `drng`/`drain`**(`Reason=Kill task failed`, root@07-25T17:19 및 T19:35). 클러스터 총 96 GPU 중 **할당 70 / 유휴 26 인데 유휴 26장이 전부 drain 노드 위**에 있다 → **스케줄 가능한 유휴 GPU = 0**.
- `PrivateData=jobs` 라 `squeue` 로는 내 잡만 보인다. **"클러스터가 비어 있다"는 오독** — 실제로는 73% 점유 중이다.
- **`ArrayTaskThrottle` 증액은 무효 — 상한은 QOS 다.** 20:35 에 두 잡(L11 3 + G12 5)이 8 GPU 를 채우자 대기 사유가 **`QOSMaxGRESPerUser`** 로 바뀌었다. 즉 §0 의 "동시 8-GPU/user" 는 **QOS 하드 캡**이고, 배열 `%8` 은 그 캡을 넘지 않는 표현일 뿐이다(단일 배열만 보면 사유가 `JobArrayTaskLimit` 로 가려져 이 캡이 안 보인다). **8슬롯은 영구 천장이다.**
- **실행 중 8셀 가운데 5셀이 drain 노드 위**(node44×3, node51, node27). drng 는 **실행 중인 잡을 죽이지 않으므로 이 5셀은 안전**하다. 문제는 **끝난 뒤**다 — drain 노드는 새 잡을 받지 않으므로 그 슬롯이 나에게 되돌아오지 않는다. 재활용되는 건 mix 노드 3셀(node26·45·47)뿐이고 그마저 타 사용자와 경쟁한다.
- **아래 감쇠 시나리오는 취소된 42셀 계획 기준이라 지금은 무효**(잔여가 L11 3 + G12 15 로 줄었다). 20:35 현재 drain 에도 불구하고 **8/8 슬롯을 다 받고 있다** — mix 노드에서 GPU 가 풀렸다. 기록으로만 남긴다:

  | 유효 슬롯 | wall-h | 완주 |
  |---|---|---|
  | 8 (drain 해소) | 83–98 | **07-29 07:00 ~ 22:00** |
  | 5 | 133–157 | 07-31 ~ 08-01 |
  | 3 (drain 지속) | 222–262 | **08-04 ~ 08-05** |

- **`Kill task failed` 은 통상 stale 프로세스 정리로 관리자가 수 시간 내 해제**하는 상태다. 내 8셀은 전부 정상 진행 중(19:50 기준 실패 시그니처 0)이라 원인은 내 잡이 아닐 가능성이 높다. **관리자에게 undrain 요청이 필요한지 Yonghee 판단 요망** — 이 한 건이 완주일을 07-29 ↔ 08-05 로 가르는 최대 변수다.

## 3. 완료 후

1. rundir 커밋(push는 Yonghee) → `runs/track_h/make_analysis.py`(LLM 로더가 `rundirs_llm_hj` 를 이미 읽는다) → `flirds-results-downstream` §5.3 R4 online 표 7행.
2. **스택 캐비엇**: A6000(torch 2.11) vs canonical(B200 torch 2.12) — fidelity·recovery 는 stack-robust(mean|Δ|≤0.006)라 recovery 정규화로 병치. **`timing.json` 은 §5.5 cost 표에 쓰지 않는다.**
3. **완료 판정**: 로그 `TRACK G DONE` + rundir mtime.
