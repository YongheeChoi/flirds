# 시간·컴퓨팅 비용 비교 방법론 검증 (검증·감사 항목 6)

- 작성: 2026-07-04, 검증 세션 서브에이전트 (문서만 생성; 코드·위키·rundir 무수정, git 무접촉)
- 과업 출처: `PROMPT_VERIFICATION_SURVEY_2026-07.md` 항목 6
- 재료: 회계 감사 정찰 노트(cost_timers), 선행연구 원문 조사 7편(GTG/FedSV/ShapleyFL/ComFedSV/FLDetector/IRDS + Ripple 정독), 원격 실측 정찰 노트(remote_recon) — 전부 2026-07-04 세션 스크래치. 아래 인용은 원출처(file:line, 논문 절·표 번호, rundir/로그 경로)로 직접 표기.
- 표기: **[확인]** = 코드/원문/실측에서 직접 확인, **[추측]** = 정황 추정(근거 병기), **[실측 대기 — …]** = 별도 실측 에이전트 진행 중.

## 0. 요약 판정

1. **우리 회계의 실체** = "공유 frozen logs 위에서 각 valuation 함수 호출만 GPU-sync wall-clock으로 측정"(4개 러너의 동일한 `_timed`). FL 학습(로그 생성) 시간은 runtime 표 **밖**이며 phase2_matrix에서는 아예 미측정. `timing.json`/GPU-hours/peak-mem은 protocol §15.1 스펙만 있고 **미구현**. (§1)
2. **유일한 회계 예외 = Ripple**: 자체 FedAvg 궤적 학습이 타이머 안에 포함 — from-logs 방법들과 같은 표에 두면 범주 혼합. 단 알고리즘 구조상 from-logs화가 불가능해(per-step·per-sample 로컬 정보 필요) 불가피하며, 서술로 해소해야 한다. (§1.2, §3, §4)
3. **선행 7편 조사 결과, 비용 보고의 표준 관행은 사실상 없다**: wall-clock 실측이 있는 논문 3/7(그중 표 형태 1/7 = Ripple뿐), 실측 자체가 전무한 논문 3/7, 하드웨어 명시 2/7, 베이스라인 구현 주체 명시 0/7, 오차막대 0/7. 우리 현행 보고(방법별 wall-clock 초 + N·R·모델·하드웨어 명시)가 이미 선행 전부보다 상세하다. (§2)
4. **내부 공정성은 대체로 성립**(같은 로그·같은 loss 클로저·같은 GPU·같은 fp32) — 예외적 왜곡 지점 7건을 §4에 caveat 목록으로 확정(논문 명시 필요).
5. 논문용 비용 표 표준으로 **[valuation wall-clock + 학습 대비 overhead % + utility-eval/grad/HVP 카운트 + 하드웨어·정밀도 1줄 + 로그 상주량]** 병기를 제안. FLOPs는 비권고(선행 0/7 사용, 추정 부정확). (§5)

---

## 1. 회계 감사 확정

### 1.1 `_timed`의 구현과 범위 [확인]

동일한 사본이 4개 러너에 존재 (문자 그대로 동일):

```python
def _timed(fn, device):
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t
```

| 파일 | 위치 |
|---|---|
| `codes/experiments/phase1_baseline_compare.py` | :68–75 |
| `codes/experiments/phase2_matrix.py` | :280–287 |
| `codes/experiments/track_c1.py` | :108–115 |
| `codes/experiments/track_d.py` | :130–137 |

- 호출 전·후 `torch.cuda.synchronize()`(cuda일 때만) + `perf_counter` — 비동기 CUDA 커널 완료까지 기다린 wall-clock. 측정 방식 자체는 표준 관행에 부합.
- 측정 단위는 **방법 호출 1회의 wall-clock 스칼라**이며, 각 방법이 내부에서 수행하는 delta의 CPU→GPU 이동·forward/grad/HVP를 전부 포함.
- FLDetector·STD-DAGMM은 `device="cpu"`로 계산하되(phase2_matrix.py:339,341) 바깥 sync는 cuda 기준 — sync 오버헤드만 미미하게 붙고 측정은 유효.

### 1.2 타이머 안/밖 — 러너별 확정 [확인]

| 러너 | FL 학습(로그 생성) | 방법별 valuation | 예외 |
|---|---|---|---|
| phase2_matrix (:290–348) | **미측정** — `build_trajectory`→`_fl`(:226–253)에 타이머 없음(t 변수 자체 부재). rundir 어디에도 미기록 | 방법 14종 개별 `_timed`(:303–346) | FedDQC 직전의 `load_state_dict`(:345)는 타이머 밖 |
| phase1_baseline_compare (:78–162) | **미측정** — `run_llm_fedavg_logs`(:95–97) 타이머 밖 | 방법별 개별 `_timed`(:102–134) | **Ripple(:140–143)은 자체 축소-궤적 학습이 타이머 안** |
| track_c1 (:161–282) | **측정됨** — `t_traj`(:190–192)→metrics `traj_time`(:274). 단 방법 runtime에 안 섞임 | 방법별 개별(:199–229) | (a) retrain oracle(:180) = 2^N 재학습이 타이머 안(정의상 그것이 비용); Ripple(:233–235) 자체 학습 포함 |
| track_d (:348–455) | **측정됨** — `t_vanilla`(:367); fidelity `res["runtime"]`에는 미포함, arms 활성 시 `train_s`(:408)로 영속 | 방법 11종(:183–221) | (a)(:187) = 2^5 SFTTrainer 재학습 타이머 안; arm 재실행 `t_arm`·평가 `t_eval` 별도 영속(:396–408) |

**확정 결론**: 기존 runtime 표(2026-06-06/07)의 수치는 전부 phase1_baseline_compare 산으로, **"공유 로그 생성 FL 학습 ~15분(문서 서술치)은 표 밖"**이다(`research-wiki/raw/conversations/flirds/2026-06-06-sv-baseline-port-and-results.md:47` "small config → ~15min shared-logs, ~54–91min ripple"). 즉 표의 열은 "학습이 끝난 뒤 사후 valuation의 추가 비용"이고, Ripple 행만 "자체 학습 포함 end-to-end"다.

### 1.3 timing.json — 스펙만 있고 미구현 [확인]

`research-wiki/wiki/flirds-protocol.md` §15.1(:243–249)은 per-phase wall-clock(`timing.json`)+GPU-hours+peak GPU memory를 요구하나:

| §15 요구 | 실코드 상태 | 판정 |
|---|---|---|
| `timing.json` per-phase | **미구현.** grep 결과 timing.json을 쓰는 코드 0곳; `find runs/ -name timing.json` 0건. 실제 영속처 = `metrics.json`의 방법별 `runtime` 스칼라(phase2_matrix.py:387→run_logger.py:82–86) + track_c1 `traj_time`/`oracle_a.time` + track_d `train_s`/`eval_s` | 불일치 |
| GPU-hours 집계 | 코드 0곳 (grep `gpu_hours` 0건) | 불일치 |
| peak GPU memory | `experiments/phase1_hvp_profile.py:66` 한 곳(HVP 프로파일 전용) — 러너 미기록 | 불일치(부분) |
| φ-est vs oracle 같은 run에서 병렬 측정 | 성립 — 같은 frozen logs에서 (b)와 추정기를 개별 `_timed` | 정합 |
| `aggregate_runs.py` | 부재. 실제 롤업 = `runs/phase2_matrix/make_analysis.py` 등 트랙별 도구 | 이름·형태 상이 |

(a) oracle만 per-coalition `(|S|, retrain_s, eval_s)` 리스트가 opt-in으로 존재(`flirds/oracle/exact_sv_llm.py:56,97–98`) — 단 stdout 리포트(`experiments/phase2_llm_a_oracle.py:69–156`)일 뿐 파일 저장 아님.

**함의**: 논문에서 "GPU-hours consumed"를 주장하려면 현재 자료로는 드라이버 로그 타임스탬프+metrics 합산으로 재구성해야 한다(§5, 후속 제안 참조).

### 1.4 학습 대비 overhead 비율 — 기존 실측으로 계산

#### (i) 정밀 실측: track_d anchor5 r64 (1B, N=5 K=5 R=30, LoRA r=64, seed0, fp32, B200 1장)

출처: 원격 `runs/probe_signal/rundirs/1B_anchor5_r64_seed0/metrics.json`의 per-method runtime(2026-07-04 정찰 실측; config = Llama-3.2-1B-Instruct, val 200, max_steps 10, lr 1e-3, git 5c5df9e). vanilla FL wall-clock은 같은 셀 arms의 `train_s`에서 **역산**: shapleyfl_w arm 5525s = vanilla×2.96 → **vanilla ≈ 1,866s** [추측·파생 — 셀 자체 `t_vanilla` 직접 읽기 아님. 원자료는 arm 5525s와 2.96 비율만 기록(recon cost §3); 정밀치는 `metrics.json`의 anchor5_r64 `train_s("vanilla")`를 직접 읽어 대체할 것(track_d.py:394/408이 arm별 `t_vanilla` 영속)]. 교차검증: 같은 R=30 anchor 무대 noise probe의 vanilla 1,885s(`runs/probe_signal/_logs/noise_r16.log` 말미)와 ±1% 일치 — 단 이는 **rank-16 셀** 값이므로 이 r64 셀과의 대조는 same-cell이 아니라 across-rank sanity check임(rank가 vanilla 학습 시간에 미치는 영향은 작다는 가정 하의 정합).

| 방법 | valuation runtime (s) | **vanilla FL 대비 overhead** |
|---|---|---|
| Flirds-1st | 245 | **13.1 %** |
| FedIF | 249 | 13.3 % |
| Flirds (1st+2nd) | 747 | **40.0 %** |
| Fed-LOO | 820 | 43.9 % |
| loss-heur | 1,155 | 61.9 % (구현 비효율 ~2배 포함, §4-6) |
| ComFedSV | 2,327 | 124.7 % |
| ShapleyFL | 3,722 | 199.4 % |
| FedSV | 3,723 | 199.5 % |
| GTG | 3,763 | 201.7 % |
| Banzhaf | 3,777 | 202.4 % |
| (b) oracle (exact 2^5) | 3,778 | **202.5 %** |

- 읽는 법: N=5 anchor 레짐에서 Flirds의 사후 valuation은 **한 번의 FL 학습 시간의 40%**, 1차만 쓰면 13%. 코얼리션 계열((b)/Banzhaf/GTG/FedSV/ShapleyFL)은 학습을 한 번 더 도는 것의 약 2배 비용.
- caveat: 이 셀은 probe용 **LoRA r=64**(canonical r16 아님)·val 200 조건. r16 canonical 조건 및 2026-06-06 표 조건의 같은 비율은 → **[실측 대기 — GPU 해방 후 2026-06-06 표 조건(1B N=5 R=10 val=100) 스모크 재측정: 공유 로그 생성 시간을 `_timed`로 명시 측정 + 방법별 runtime 동시 산출]**.

#### (ii) 느슨한 1차 근사: 2026-06-06 표 조건 (1B N=5 R=10 val=100)

공유 로그 생성 "~15min"은 디버깅 서술의 어림값(위 :47)이므로 정밀치 아님. vanilla ≈ 900s로 두면: Flirds-1st 35s ≈ **3.9%**, Flirds 107s ≈ **12%**, loss-heur 164s ≈ 18%, GTG 537s·FedSV 532s·Banzhaf/ShapleyFL/(b) ~531s ≈ **59–60%**, Ripple 4,515s ≈ 5.0×(단 자체 축소-학습 R=4 포함이라 이 비율은 overhead가 아니라 end-to-end/학습 비율). → anchor r64 실측(위)보다 코얼리션 계열 비율이 낮은 것은 R=10 vs 30, val 100 vs 200 차이로 정성적으로 일관. 정밀치는 위 [실측 대기] 항목으로 대체.

#### (iii) 대규모 레짐 참고치: std50k5 (N=50 K=5 R=200, 진행 중)

vanilla FL 실측 = r16 10,998s / r32 11,136s / r64 11,404s(각 셀 로그 마지막 라인, 2026-07-03). phase-1 fidelity(10-method 합산)는 ETA 추정 ~38h ≈ vanilla의 **~12×** — 이 레짐에서는 사후 valuation 총비용이 학습을 압도한다(코얼리션 방법들이 지배). 방법별 분해는 **[실측 대기 — probe std50k5 3셀 완주 후 metrics.json (ETA 2026-07-05 저녁~07-06)]**.

### 1.5 방법별 로깅 요구 — `(w_r, δ)` 공유 로그로 충분한가 [확인]

공통 로그 계약: `logs = [(w_r, {client: (delta, n_c)})]` (`fl/server.py:5–8`; LLM은 LoRA 파라미터만, `fl/llm_server.py:2–15`).

| 분류 | 방법 | (w_r, δ) 충분? |
|---|---|---|
| logs만 (진짜 model-free) | FLDetector, STD-DAGMM | **충분** (val/model/loss_fn 불필요) |
| logs + 서버 val loss 클로저 | Flirds, Flirds-1st, (b), Banzhaf, loss-heur, Fed-LOO, GTG, FedSV, ShapleyFL, ComFedSV, FedIF, FLTrust | **충분** (+서버 val 셋 전제 — 우리 프로토콜의 소여) |
| 로그 밖 정보 필요 | FedDQC | 불충분 — **클라 원시 데이터 + base model + tokenizer** (`feddqc.py:19–24`; δ조차 안 씀) |
| 〃 | (a) retrain oracle | 불충분 — 클라 로컬 loaders/데이터셋으로 2^N 재학습 (`exact_sv.py:50`, `exact_sv_llm.py:52`) |
| 〃 (**per-step 정보 필요**) | Ripple | **구조적으로 불충분** — drop term은 per-sample·per-local-step gradient + 로컬 학습 중간 파라미터(Alg.1 L5 "Store intermediate parameters"), Hessian sketch는 로컬 데이터 HVP(L6). 라운드 단위 δ로 복원 불가. 원논문 Algorithm 1 (AAAI-26 p.5) 자체가 클라 참여형 온라인 프로토콜 |

**판정**: 17개 비교 방법 중 Ripple 하나만 per-step/per-sample 로컬 정보를 요구하며, 이것이 §1.2의 회계 예외(자체 학습 포함)의 근본 원인이다. 나머지 전부는 현행 공유 로그 + 서버 val로 충분 — 추가 로깅 확장 불요.

### 1.6 로그 저장 용량 [확인 + 산술]

- LoRA r=16, 7모듈(q/k/v/o/gate/up/down_proj) → 1B(Llama-3.2-1B-Instruct) LoRA 파라미터 **11,272,192 ≈ 11.27M** → fp32 **45.09 MB**/벡터 (교차검증: `ripple_llm.py:20` "P ~ 12M").
- 로그는 **디스크에 저장되지 않는다** — rundir 영속물은 config/meta/metrics/phi.parquet뿐(원격 `find runs -name '*.pt' -o '*.pkl' -o '*.npz'` 0건, 2026-07-04 확인). w_r 이력은 GPU 상주, delta 이력은 CPU RAM(`fl/llm_server.py:56` `.cpu()`, `fl/server.py:39,44`). 즉 아래는 **런타임 메모리 상주량**이다.

| 설정 (config 출처) | K | R | 라운드당 (1+K)×45.09MB | 총량 | GPU(w_r) | CPU(δ) |
|---|---|---|---|---|---|---|
| silo5/iid5 (`phase2_matrix.py:112–113`) | 5 | 10 | 270.5 MB | **2.71 GB** | 0.45 | 2.25 |
| track_d anchor5 (`track_d.py:84–87`) | 5 | 30 | 270.5 MB | **8.12 GB** | 1.35 | 6.76 |
| track_d std20 (〃) | 2 | 200 | 135.3 MB | **27.05 GB** | 9.02 | 18.04 |
| device100 (`phase2_matrix.py:118–120`) | 10 | 30 | 496.0 MB | **14.88 GB** | 1.35 | 13.53 |
| device100 poison R=60+1 (:46, :266) | 10 | 61 | 496.0 MB | **30.25 GB** | 2.75 | 27.50 |

3B(24.3M LoRA 파라미터, delta 97.3 MB): anchor5 ≈ 17.5 GB / std20 ≈ 58.4 GB / device100 R=30 ≈ 32.1 GB. 서버 RAM 2.2TB·B200 183GB 대비 전부 여유이나, **from-logs 회계의 숨은 비용**으로 논문에 1줄 보고 가치가 있다(선행 7편 중 이 항목을 보고한 논문 0; FLDetector만 O(Np) asymptotic 저장 오버헤드 언급, §2.7). LoRA rank probe(r64)는 ×4로 스케일(std50k5 r64 셀 RSS ~170GB 실측과 일관).

---

## 2. 선행연구 비용 보고 관행 (7편 원문 조사)

각 절: 셋업 / 재는 대상 / 보고 단위 / 학습 포함 여부 / 하드웨어 / 베이스라인 구현 주체. 전부 원문 직접 확인(로컬 PDF·md; 절·표 번호 병기).

### 2.1 GTG-Shapley (Liu et al., ACM TIST 2022; arXiv 2109.02053v1)

- **셋업**: MNIST만, N=10, 시나리오 5종(§5.1.1, pp.12–13). **모델 아키텍처·로컬 epoch·lr·batch·본실험 라운드 수 T·하드웨어 전부 미기재**(전문 grep 확정) — 코드 저장소(footnote 1)로만 위임.
- **재는 대상**: "**The total time of calculating SVs** is used to evaluate the efficiency of each approach"(§5.1.3(1), p.13) — SV-계산 단계의 총 wall-clock을, 공통 수렴 기준 Eq.(10)(최근 10개 샘플 상대변화 <0.05) 도달까지 잰 **time-to-convergence**. 비용 전용 표는 없고 전부 시간–정확도 수렴 곡선(Fig.6–10 본비교, Fig.11–15 ablation; 양축 log10).
- **보고 단위**: wall-clock 초(곡선) + 본문 서술 배수("faster than … TMR by 7.4 times", §5.2.1 — iid 1개 시나리오 수치) + asymptotic(§4.4: O(T log N)~O(T N log N)). **utility-eval 횟수는 이론 단위일 뿐 실측 카운트 미보고.**
- **학습 포함 여부**: 명시 없음. Fig.2(a)의 runtime 파이(Training 1.4%/Reconstruction 0.1%/Evaluation 98.5%, §3 p.7)에는 Training이 포함되어 있으나 본비교 곡선(Fig.6–10)의 Time 정의는 서술 부재. [추측: "total time of calculating SVs" 문구상 valuation만일 가능성이 높으나 단정 불가]
- **비대칭 주의**: retraining 기반 3종(Original/TMC/GroupTesting)은 utility eval마다 재학습이 시간에 내재, gradient 기반 4종(MR/TMR/Fed-SV/GTG)은 재구성+평가 — 같은 곡선 위 두 범주의 시간 의미가 다름(§5.1.2, §5.2.1).
- **하드웨어**: 전무. **베이스라인 구현**: 명시 없음 [추측: 공통 수렴 기준 일괄 적용이 단일 코드베이스=저자 재구현을 시사]. 오차막대 없음.

### 2.2 FedSV (Wang et al., Springer FL-Privacy&Incentive 2020; arXiv 2009.06192v1)

- **비용 실험 = 0건.** wall-clock·FLOPs·런타임 표/그림·하드웨어·속도배수 전부 부재(전문 검색 확정). 논문 전체에 표가 0개.
- 유일한 비용 서술 = §4.2(pp.7–8) asymptotic, 단위는 **utility-eval 횟수**: exact **O(T·2^m)**(위첨자 폰트 스팬으로 확인 — 소스노트의 O(Tm²)는 오기), permutation ≈ O(m log m)/라운드, group testing ≈ O((log m)²)/라운드.
- 효율 헤드라인은 "no extra **communication** cost"(Abstract, §1) — 연산 비용 아님.
- **실험은 permutation(Alg.2)만 사용**(p.11)하면서 실제 permutation 수 T(또는 ε,δ)를 **미보고** — 재현성 갭.
- **베이스라인**: Fed-LOO·Random 둘뿐, 둘 다 자체 정의·자체 구현, **비용 비교 없음**. 공식 코드 릴리즈 없음(우리 `fedsv.py:1–9` docstring "No official code → self-build"와 부합).

### 2.3 ShapleyFL/AFedSV+ (Sun et al., KDD 2023; DOI 10.1145/3580305.3599500)

- **시간·연산 비용 표 없음**: wall-clock 0건, FLOPs 0건, 하드웨어 0건(grep 확정; PyTorch만 언급, §6 서두).
- 효율 근거 3종: (i) 정성 문장(#P-hard 인용 §1; "enumerating all subsets…" §5.2), (ii) communication rounds 대비 accuracy 수렴 곡선(Fig.1–2, §6.2), (iii) **부록 B.5 Table 3** — 유일한 정량 "효율" 실험이나 재는 것은 시간이 아니라 **동일 permutation 예산(80–400)에서의 추정 MSE**(벤치마크 = MC 2000-perm 추정치), 그것도 "**first communication round**만"(p.2108, 비용상 명시).
- **메인 실험에서 라운드당 partial SV를 exact로 했는지 DMC 근사로 했는지 본문 미명시** — 참여 10클라면 라운드당 2^10=1024 utility eval인데 이 비용 계정이 논문에 없음.
- **학습 포함 여부**: 논할 대상 자체가 없음(시간을 안 잼). **베이스라인 구현 주체**: 명시 없음(저자 repo 존재 → 자체 재구현 [추측]); 시간 비교가 없어 구현 최적화가 결론을 왜곡할 여지도 없음.

### 2.4 ComFedSV (Fan et al., ICDE 2022; arXiv 2109.09046v3)

- **wall-clock 실측 존재** — 단 표가 아니라 **Fig.8**(§VII-D, p.10–11) 곡선뿐, 본문에 숫자 없음. 좌축 "time(mins)"(그래프 판독: N=100에서 ComFedSV ~30–150min, FedSV ~7–45min, 데이터셋별), 우축 시간비.
- **셋업**: N ∈ {10,…,100}, 라운드당 30% 참여; **T·MC 샘플 수 M 미기재**; 모델 = logistic/FCNN/simple CNN/VGG16(파라미터 수 미기재). 반복·오차막대 없음. **하드웨어 전무**(grep 확정) → wall-clock 절대값은 재현 불가 수준이며 논문 스스로도 절대값이 아니라 **ratio의 asymptotic 수렴 검증**(FedSV/ComFedSV → 참여율 K/N ≈ 0.3)을 논거로 씀(§VII-D).
- **재는 대상**: "the time required for **computing** ComFedSV and FedSV" — valuation 계산 시간. FL 학습 포함 여부 명시 없음 [추측: ratio가 K/N으로 깨끗이 수렴하는 것으로 보아 제외 또는 공통 상쇄]. 지배 비용은 본문이 직접 명시: "the main cost is evaluating the roundly utility function ut"(§VII-D) — 1급 단위는 **utility-call 수** asymptotic(ComFedSV O(TNK log N) vs FedSV O(TK² log K)).
- **자기 방법이 baseline보다 ~3.3× 느리다는 역방향 결과를 그대로 보고**한 점이 특징. **베이스라인 구현**: 저자 재구현 추정(동일 코드베이스, Huawei AI Gallery 공개; 명시 문장 없음 [추측]).
- 보고 품질 신호: Fig.8 캡션-범례 색 배정 뒤바뀜, 범례 "FedCV" 오타(원문 확인).

### 2.5 Ripple Shapley (Zeng et al., AAAI 2026; pp.28085–28093)

7편 중 **유일하게 비용 표(wall-clock)가 있는 논문** — 그래서 우리와의 회계 차이를 가장 정확히 짚어야 한다.

- **셋업**(p.5 Training Setup): MNIST(2-layer MLP)·CIFAR-10("a standard CNN", 파라미터 수 미기재), 로컬 5 epochs/라운드, 100라운드, batch 10, lr 0.01, **5회 반복 평균**. **클라 수 N·참여율은 본문 어디에도 없음**(grep 확정; "Following the protocol of (Sun et al. 2023)"로만 위임). **하드웨어 명시: Tesla V100 ×2, PyTorch** — 조사 7편 중 하드웨어를 밝힌 둘 중 하나.
- **재는 대상**(Table 1, p.5–6): "Average **Cumulative** Computation Time(s)" — **FL 학습 + valuation을 합친 누적 wall-clock**. "The baseline runtime corresponds to the duration of a conventional training run…"(p.5)으로 Plain Training(100R 누적 480.91s)이 기준선임을 명시. 공정성 장치로 전 방법 FedAvg 집계 통일.
- **원수치**(@Round 100): Ripple 984.98 / S-FedAvg 4,531.91 / FedSV 48,282.84 / AFedSV+ 61,437.57 / Plain 480.91 / FedProx 530.39 (s). 본문 배수 주장: plain의 **2.05×**, S-FedAvg·FedSV·AFedSV+ 대비 **4.6× / 49.06× / 62.37×** (검산: 4.601✓ / 49.02 근사 / 62.37✓).
- **valuation-only로 환산하면**(학습 480.91s 차감; 우리 파생 계산): Ripple 오버헤드 504.07s(≈5.04s/라운드 상수) vs S-FedAvg 4,051s(8.0×) / FedSV 47,802s(94.8×) / AFedSV+ 60,957s(120.9×) — **학습 포함 회계가 배수를 오히려 희석**한 사례. 즉 그들 셋업에서는 어느 회계 정의로도 Ripple이 최저비용이고, 우리 실측(1B에서 Ripple 최고비용)과의 역전은 회계 정의 차이가 아니라 **모델 규모 d·비교군·eigsh 비용의 차이**다(Ripple 정독 노트 §5.4; 상세 진단은 항목 2 문서 소관).
- **형식 복잡도는 자기 방법에 대해 미제시**(naive Jacobian곱 O(r·d³)만 명시, p.4); ripple 깊이 R도 Table 1 실험값 미명시. **베이스라인 구현 주체**: "all methods are implemented in PyTorch … identical environments" — 자체 재구현으로 읽힘 [추측], 공식 코드 사용 여부·코드 공개 링크 부재.
- 보고 품질 신호: "three benchmark image datasets: MNIST and CIFAR-10"(2개 나열), "total everage runtime" 오탈자, Table 1 캡션 "on Two Datasets"인데 수치 한 벌(평균 [추측]).

### 2.6 IRDS / In-Run Data Shapley (Wang et al., arXiv 2406.11011v3; 중앙집중식 — FL 아님)

- **셋업**(§5.1 + Appx E.1): GPT2-Small(124M), Pile, **80GB A100 1장**(명시 — 하드웨어 밝힌 둘 중 하나). 단 Table 1 측정의 batch/seq/val point 수는 §5.1에 미명시(throughput이 val point 수에 의존하는데도).
- **재는 대상**: in-run 방법이라 valuation이 학습에 융합 — Table 1은 "valuation을 켠 학습의 **throughput**(points/sec)" vs "일반 학습 throughput". 즉 **비용 = 학습 대비 상대 감속**이라는 프레임: Regular 76.2 / 1차 ghost 70.5(−7.5%) / 2차 ghost 34.4(≈2.2× 감속) / naive 직접구현 4.2·1.8.
- **보고 단위**: throughput + 상대배수. 절대 wall-clock·GPU-hours·FLOPs·utility-eval 횟수 전부 미보고. 오차막대·측정 반복 없음.
- **비교 대상은 전부 자기 자신**(regular training / 자기 방법의 naive 구현) — 외부 방법 런타임 0건. 외부 베이스라인 재구현 공정성 이슈가 아예 발생하지 않는 보고 구조.
- **원문 자체 결함**: 본문 ">30× faster than the naive implementation"(§5.1) vs Table 1 산출 16.8×(1차)/19.1×(2차) — 어느 비율도 30×에 못 미침(42.3×는 regular/direct-2nd). 인용 시 표 원수치를 쓸 것.
- caveat 명시 관행은 좋은 편: "with sufficient GPU memory" 전제, 메모리 부족 시 런타임 증가 인정(§6), SGD-전용 한계 명시.

### 2.7 FLDetector (Zhang et al., KDD 2022; arXiv 2207.09209v4)

- **비용 실험 전무** — wall-clock/FLOPs/하드웨어/속도배수/코드링크 0건(전 11페이지 키워드 전수 검색 확정).
- 유일한 비용 절 = §4.3 Complexity Analysis(p.4): 총 시간 **O(N³+KBn²+(6N+2n)p+Nn)/iteration** ≈ 파라미터 수 p에 선형; 저장 O(Np); "the server is powerful in FL, so … acceptable"; 클라 오버헤드 0. 전부 asymptotic+정성.
- **베이스라인 비용 비교 없음**(비교축은 탐지 정확도만). VAE 베이스라인은 저자 셋업(오히려 유리 조건 부여 명시); ablation 2종은 자체 변형.
- **함의**: 우리 표의 "FLDetector ~24s"(silo5)·211.7s(device100 α=0.01 셀) 같은 수치는 **원논문과 대조 불가능한 우리 신규 측정**이며, 각주에 "원논문은 asymptotic만 보고"를 명기하는 것이 정직하다. 우리 실측이 원논문의 정성 주장("서버 선형·저렴")을 정량화해준 관계.

### 2.8 종합 관찰 — "표준 관행이 사실상 없음"의 정량화

| 논문 (venue) | 실측 비용 | 형태 | 재는 대상 | 학습 포함 | 하드웨어 | 베이스라인 구현 주체 | 오차막대 |
|---|---|---|---|---|---|---|---|
| GTG-Shapley (TIST'22) | wall-clock | 곡선 (표 없음) | SV-계산 time-to-convergence | 불명 | ✗ | ✗ | ✗ |
| FedSV (Springer'20) | **없음** | — | asymptotic(eval 횟수)만 | — | ✗ | ✗ (비용비교 자체 없음) | — |
| ShapleyFL (KDD'23) | **없음**(시간) | MSE-vs-예산 표 | 근사 정확도(비용 아님) | — | ✗ | ✗ | ✗(효율 실험) |
| ComFedSV (ICDE'22) | wall-clock | 곡선 (표·숫자 없음) | valuation 계산 시간 | 불명 | ✗ | ✗ (자체 추정) | ✗ |
| Ripple (AAAI'26) | wall-clock | **표** (Table 1) | 학습+valuation 누적 | **포함(명시)** | ✓ V100×2 | ✗ (자체 추정) | ✗ (5회 평균만) |
| IRDS (arXiv'24) | throughput | 표 (pts/sec) | 학습 융합 처리율 | 분모=학습 | ✓ A100×1 | 자기-대조만 | ✗ |
| FLDetector (KDD'22) | **없음** | — | asymptotic/iteration | — | ✗ | ✗ (비용비교 없음) | — |

집계 (7편):
- **wall-clock 실측 존재: 3/7** (GTG·ComFedSV·Ripple) — 그중 **표 형태는 1/7**(Ripple), 나머지는 곡선에서 판독해야 하고 ComFedSV는 본문 숫자조차 없음.
- **실측 비용 전무: 3/7** (FedSV·ShapleyFL·FLDetector) — 전부 우리가 wall-clock 비교표에 넣는 방법들.
- **valuation 단계만 잰다고 명시: 2/7** (GTG·ComFedSV) — 그러나 둘 다 학습 시간 포함 여부는 불명. **학습 포함을 명시한 것은 Ripple 1편뿐**이고 그 회계는 우리와 반대(포함).
- **하드웨어 명시: 2/7** (Ripple·IRDS). 모델 파라미터 수 명시: **1/7**(IRDS 124M).
- **베이스라인 구현 주체 명시: 0/7.** 비용 측정의 반복·오차막대: **0/7**(Ripple만 5회 평균 언급, 분산 미보고).
- utility-eval **실측 카운트** 보고: **0/7** (asymptotic 단위로만 4편이 사용).
- 원문 내 자기-불일치 발견: IRDS ">30×" vs 표 수치 16.8–19.1×, ComFedSV 캡션-범례 뒤바뀜, Ripple 데이터셋 개수 오탈자, GTG ablation 명칭 혼용(GTG-Tib/Tid) — 7편 중 4편에서 비용 절 서술 결함.

**결론**: FL 기여도 평가 문헌에 비용 보고의 합의된 프로토콜은 없다. 시간의 정의(수렴 도달 vs 고정 예산 vs 학습 포함 누적 vs throughput)가 논문마다 다르고, 대부분 하드웨어·구현 출처·오차 없이 보고한다. 따라서 (i) **선행 논문의 비용 수치와 우리 수치의 직접 비교는 어떤 조합에서도 불가능**하고(같은 정의를 쓰는 쌍이 없음), (ii) 우리 논문은 비용 표에 **자체 회계 정의를 1문단으로 명시**하는 것만으로 선행 전부보다 엄격해진다.

---

## 3. 우리 비교 baseline 전수 회계 점검

현재 비교표 등장 방법 전수(변형 통합 17개; (a)·Ripple의 CNN/LLM 변형 분리 시 19행 — 과업 지시문의 "16개"는 Fed-LOO 미포함 카운트). 근거: 회계 감사 노트 §3 (구현 file:line 포함).

| 방법 | 구현 | 시점 | 필요한 입력 | 장치 | 우리 단일 회계("frozen logs 위 valuation-only wall-clock")의 공정성 판정 |
|---|---|---|---|---|---|
| Flirds | `flirds/core/flirds_estimator.py:65` | 사후 from-logs | logs + val loss 클로저 | GPU (fp32 HVP) | **공정** — 계약 그대로 |
| Flirds-1st | 〃 (`second_order=False`) | 〃 | 〃 | GPU | **공정** |
| (b) in-run oracle | `flirds/oracle/in_run_sv.py:139,157` | 〃 (frozen logs 위 2^N 또는 Σ2^{P_r} forward) | 〃 | GPU | **공정** — 재학습 없음, 같은 계약 |
| Banzhaf | `flirds/baselines/banzhaf.py:46` | 〃 ((b)와 같은 coalition utility 재사용) | 〃 | GPU | **공정** |
| GTG | `flirds/baselines/gtg.py:145` | 〃 (within-subset 재정규화 + guided-trunc MC) | 〃 | GPU | **공정** — 단 원논문 시간 정의(수렴-도달)와 다름을 인용 시 명시 (§4-5) |
| FedSV | `flirds/baselines/fedsv.py:39` | 〃 (permutation MC) | 〃 | GPU | **공정** — 단 TMC truncation+캐시는 원논문 Alg.2에 없는 우리 최적화 (§4-5) |
| ShapleyFL | `flirds/baselines/shapleyfl.py:100` | 〃 (per-round exact 2^{P_r}) | 〃 | GPU | **공정** — 원논문은 exact/DMC 모호(§2.3); 우리는 exact 구현임을 명시 |
| ComFedSV | `flirds/baselines/comfedsv.py:127` | 〃 (perm-prefix 관측 + ALS 완성) | 〃 | GPU forward + **CPU ALS(numpy)** | **대체로 공정** — ALS가 CPU지만 원논문 스스로 "main cost = utility eval"(GPU 부분)이라 했으므로 정합; 장치 혼합 각주 1줄 |
| loss-heur | `flirds/oracle/in_run_sv.py:54` (싱글턴 N회) | 〃 | 〃 | GPU | **공정하나 과대측정** — base loss를 클라마다 재계산(라운드 캐시 없음, :64) → 최적 구현 대비 ~2배. 표 각주 필요 (§4-6) |
| Fed-LOO | `flirds/oracle/in_run_sv.py:71` | 〃 | 〃 | GPU | **공정** |
| FedIF | `flirds/baselines/fedif.py:85` | 〃 (라운드당 1 val grad) | 〃 | GPU | **공정** |
| FLTrust | `flirds/baselines/fltrust.py:48` | 〃 | logs + loss_fn (model-free 아님) | GPU | **공정** |
| FLDetector | `flirds/baselines/fldetector.py:74` | 〃 | **logs만** | **CPU** (러너 지정, phase2_matrix.py:339) | **조건부 공정** — 같은 회계로 재지만 장치가 다름. wall-clock 열에 "CPU" 표기 필수 (§4-2) |
| STD-DAGMM | `flirds/baselines/std_dagmm.py:166` | 〃 (자체 AE+GMM 200ep 학습 포함) | logs만 | **CPU** | **조건부 공정** — 동일; 내부 anomaly-모델 학습이 타이머 안(방법 고유 비용이므로 정당) |
| FedDQC | `flirds/baselines/feddqc.py:35` | 사후 (base model 스코어) | **클라 원시 데이터 + model + tokenizer — logs 불필요** | GPU | **조건부 공정** — valuation 호출 자체는 같은 `_timed`로 재나, 입력 체계가 from-logs 프레임 밖(데이터 접근 전제 자체가 다름). 표에 입력 요구 열 병기 권장 |
| (a) retrain oracle | `exact_sv.py:50`/`exact_sv_llm.py:52` | **사후지만 2^N 재학습** | 클라 로컬 데이터 | GPU 학습 | **범주 자체가 end-to-end** — valuation-only 열이 아니라 별도 열(재학습 오라클)로 보고. per-coalition timing은 opt-in stdout뿐(§1.3) |
| **Ripple** | `ripple.py:148`/`ripple_llm.py` | **in-run — 자체 FedAvg 궤적 학습 + per-step drop + 클라별 eigsh가 전부 타이머 안** | 클라 로컬 데이터 (logs 안 받음) | GPU 학습+HVP (+scipy ARPACK CPU 조정) | **불공정(범주 혼합) — 표의 유일한 실질 예외.** phase1에서는 심지어 축소 자체-궤적(rip_rounds=4 vs 공유 R=10, `phase1_baseline_compare.py:64`) 기준인데도 4,515s. from-logs화는 구조적으로 불가(§1.5)이므로 분리 서술이 유일한 해법 (§4-1) |

**요약 판정**: 17개 중 12개는 무조건 공정(동일 로그·동일 loss 클로저·동일 GPU·동일 fp32), 3개는 각주 1줄로 해소되는 조건부 공정(FLDetector·STD-DAGMM의 CPU, FedDQC의 입력 체계), (a)는 범주 분리로 해소, **실질 문제는 Ripple 1건**.

---

## 4. 종합 판정 — 현행 비교 방식의 왜곡 지점과 논문 명시 caveat 목록

현행 방식(단일 `_timed` 회계, 방법별 wall-clock 표)은 내부적으로 대체로 공정하다. 논문에 명시해야 할 caveat을 우선순위 순으로 확정한다.

**C1. Ripple 회계 불일치 (유일한 범주 예외)** — Ripple runtime은 "자체 FL 학습 포함 end-to-end", 나머지는 "학습 제외 valuation-only". 같은 열에 두면 안 됨. 완화 요인: (i) Ripple 원논문 회계 자체가 학습 포함 누적(Table 1)이라 그들 프레임과는 오히려 정합, (ii) 그들 셋업에서 valuation-only로 환산해도 결론(Ripple 최저가) 불변임을 우리가 파생 계산으로 확인(§2.5) — 즉 우리 표의 역전은 회계 탓이 아님. **처방**: 표를 2열(valuation-only / end-to-end)로 분리하거나 Ripple 행에 †각주 + "우리 조건에서의 valuation-only 환산치"를 병기. 환산치는 **[실측 대기 — Ripple 분리 계측 스모크(항목 2 Ripple 감사와 공유; 학습 phase vs drop/sketch/ripple phase 분리 타이머)]**. phase1 수치가 축소 궤적(R=4) 기준임도 각주로.

**C2. FLDetector·STD-DAGMM CPU-only 비대칭** — GPU 방법들과 wall-clock을 한 열에서 비교하면 하드웨어가 다르다. "cheapest ~24s"(silo5) 주장은 CPU 수치임을 명시. 완화 요인: "model-free server-side가 GPU 없이도 돈다"는 것 자체가 이 방법들의 스토리라 **비대칭이 오히려 방법 특성의 정직한 반영** — 단 표기 없이는 오해 소지. **처방**: 장치 열 추가(§5 프로토콜).

**C3. fp32 강제의 내부 공정성 / 외부 비교 caveat** — 전 방법이 같은 fp32(matmul tf32=off 실측 확정, 원격 정찰 2026-07-04: `matmul_tf32 False`, `f32mp highest`, remote_recon §3)로 측정되므로 **방법 간 비교는 공정**. 그러나 절대 wall-clock은 tensor-core 대비 부풀려져 있어 외부 논문 수치·실무 배포 비용과 직접 비교 불가. **배수는 미검증** — 흔히 인용되는 ×3.1은 **프롬프트에서 넘어온 미검증 placeholder일 뿐 어떤 recon 노트·코드에도 1차 근거가 없다**. 유일하게 확인된 사실은 정성적 조건(`matmul_tf32=False`, remote_recon §3)뿐이고, 절대치 771ms/fwd(root CLAUDE.md baseline)는 fp32 단일 forward 시간이라 bf16/tf32 기준선이 없어 배수를 정할 수 없다(=771ms가 ×3.1을 뒷받침하지 않음). **처방**: 비용 표 캡션에 "fp32, tensor-core 미사용; bf16 대비 ×k 느림" 1줄. 배수 확정치(×3.1 대체)는 **[실측 대기 — fp32 vs bf16/tf32 마이크로벤치(항목 3 정밀도 감사와 공유); 이 실측 전까지 ×3.1은 인용 금지]**.

**C4. 원논문 시간 수치 부재 방법들 = "우리 재구현 측정"임을 명시** — FedSV·ShapleyFL·FLDetector는 원논문에 비교 가능한 실측 비용이 0건(§2.2, §2.3, §2.7), GTG·ComFedSV는 실측이 있어도 정의·셋업·하드웨어가 달라 대조 불가(§2.8). 따라서 우리 표의 모든 baseline 수치는 원논문 수치의 재현이 아니라 **우리 재구현+우리 조건의 신규 측정**이다. 표 각주 1줄("all baseline runtimes are our re-implementations measured under identical conditions; original papers report no comparable wall-clock")로 처리 — 이는 약점 고백이 아니라 §2.8상 불가피하고, 동일 조건 측정이라는 점에서 선행 관행보다 강한 주장.

**C5. 우리 재구현이 원논문 대비 유리/상이하게 동작하는 지점** — (i) FedSV: TMC truncation(trunc_eps=0.001 기본)+서브셋 utility 캐시(`fedsv.py:26–33`)는 원논문 Alg.2에 없는 우리 최적화; permutation 수 기본 `max(30, 2m)`(:51)도 자체값(원논문은 실험값 미보고라 "원설정 재현"이 애초 불가능). FedSV runtime이 과소(=유리)하게 측정될 수 있음 → 각주. (ii) GTG: 원논문 시간은 자기-수렴 도달까지(Eq.10), 우리는 고정 예산 실행 — 시간의 정의가 다름을 인용 시 명시. (iii) ShapleyFL: 원논문이 exact/DMC 중 뭘 썼는지 모호 → 우리 구현은 per-round exact임을 명시(비용을 후하게 잡은 쪽).

**C6. loss-heur runtime ~2배 과대측정** — `in_run_utility`가 싱글턴 호출마다 base loss를 재계산(`in_run_sv.py:64`, 라운드 캐시 없음). "가장 싼 계열"의 실측치에 구현 비효율이 포함 → Flirds와의 배수 주장에 유리한 방향의 왜곡(경쟁 방법이 느려 보임)이므로 각주 또는 수치 보정 필요. 코드 수정은 본 과업 범위 밖 — 후속 제안에 기재.

**C7. FL 학습 시간·GPU-hours 미기록 (총비용 주장 불가)** — phase2_matrix는 트래젝토리 생성 시간을 어디에도 안 남김(§1.2); GPU-hours 집계 코드 0곳(§1.3). protocol §15.1의 "measured wall-clock ratio" 요구 중 φ-est vs oracle 병렬 측정만 충족. 현 상태로 논문에 쓸 수 있는 것은 "방법별 valuation wall-clock + (track_c1/d 한정) 학습 시간"까지이고, 캠페인 GPU-hours는 드라이버 로그 타임스탬프 재구성이 필요. §1.4의 overhead 비율 표가 이 갭의 1차 보완이다.

부수 정합화 필요(왜곡은 아님): protocol.md §15의 timing.json/aggregate_runs.py 문구가 실코드와 불일치(§1.3) — 논문 서술이 protocol을 인용하지 않도록 하거나 protocol을 현실로 개정.

---

## 5. 보고 프로토콜 제안 (논문용 비용 표 표준)

선행 관행 조사(§2.8)에 근거해, 병기할 지표와 근거를 확정한다.

### 5.1 권고: 5개 병기 + 1개 비권고

| # | 지표 | 권고 | 근거 (선행 관행 연결) |
|---|---|---|---|
| 1 | **valuation-only wall-clock (s/run, 방법별, GPU-sync)** | **주 지표 (현행 유지)** | 선행 중 이 회계를 명시 정의한 논문 0/7(§2.8) — GTG·ComFedSV가 의도는 같으나 학습 포함 여부 불명. 정의 1문단("frozen logs 위 valuation 함수 호출, `torch.cuda.synchronize` 경계, 학습 제외")을 캡션에 넣는 것 자체가 선행 대비 개선. Ripple·(a)는 별도 열(end-to-end)로 분리(C1) |
| 2 | **학습 대비 overhead % (vanilla FL wall-clock 병기)** | **병기** | IRDS의 "학습 대비 상대 감속" 프레임(§2.6)과 Ripple의 "plain training 2.05×" 프레임(§2.5)이 문헌에서 가장 해석 가능한 두 보고였음 — 우리 §1.4 표가 이미 산출 가능(anchor r64 실측: Flirds 40%, Flirds-1st 13%, 코얼리션 계열 ~200%). 하드웨어·정밀도 의존성이 상당 부분 상쇄되는 무차원 수 |
| 3 | **연산 카운트: utility-eval / val-grad / HVP 횟수 (실측)** | **병기** | GTG·FedSV·ComFedSV·FLDetector의 1급 단위가 전부 이 계열의 asymptotic(§2.1–2.4, 2.7) — 실측 카운트를 보고한 논문은 0/7이므로, 카운트 실측+공식(예: Flirds = R×(1 HVP + Σ\|P_r\| dot), (b) = Σ_r 2^{\|P_r\|} forward)을 표로 주면 선행의 이론 단위와 우리 실측이 직접 접속됨. 하드웨어 독립이라 외부 비교 가능한 유일한 축 |
| 4 | **하드웨어 + 정밀도 1줄** (B200 1장/방법, fp32 matmul tf32-off, CPU 방법은 EPYC 코어 명시) | **필수** | 하드웨어 명시 2/7, 정밀도 명시 0/7(§2.8). C2(CPU 비대칭)·C3(fp32 배수)가 이 1줄로 해소 |
| 5 | **로그 상주량 (GB, from-logs 방법의 숨은 비용)** + peak GPU memory | **간단 병기** | 로그 저장 비용을 보고한 선행 0/7(FLDetector만 O(Np) asymptotic). §1.6 표 재사용. peak-mem은 protocol §15.1 스펙이나 현재 미구현 — **[실측 대기 — 러너에 `torch.cuda.max_memory_allocated` 계측 추가 후 재실행(후속 제안)]** |
| — | **FLOPs 추정** | **비권고** | 사용 선행 0/7(§2.8). LoRA forward+HVP+eigsh 혼합 파이프라인의 FLOPs 추정은 부정확하고 검증 불가 — 대신 #3의 연산 카운트가 같은 목적을 실측으로 달성. 리뷰어 요구 시 카운트×per-op 실측 시간으로 사후 환산 가능함을 각주 |

### 5.2 표 골격 제안

```
Method | Device | Wall-clock (s) | Overhead vs FL train (%) | #utility-eval | #val-grad | #HVP | Log residency (GB)
(valuation-only 블록: Flirds, Flirds-1st, FedIF, FLTrust, Fed-LOO, loss-heur†, GTG, FedSV‡, ShapleyFL, ComFedSV, Banzhaf, (b) oracle, FLDetector*, STD-DAGMM*, FedDQC§)
(end-to-end 블록: Ripple¶, (a) retrain oracle)
캡션: 회계 정의 1문단 + fp32/tensor-core caveat + "baseline runtimes are our re-implementations
(original papers report no comparable wall-clock — 0/7 with a comparable accounting)"
각주: †base-loss 재계산 ~2× 포함, ‡TMC trunc+cache는 우리 최적화, *CPU-only,
§클라 원시 데이터 접근 필요, ¶자체 궤적 학습 포함(원논문 회계와 동일 프레임)
```

- 캠페인 총 GPU-hours(protocol §15.1의 budget-report 수치)는 방법 표와 분리해 실험 절 말미 1문장으로 — 드라이버 로그 타임스탬프에서 재구성.
- 반복·오차: 방법별 wall-clock에 3-seed std 병기(선행 0/7이 오차를 보고 — 최소 비용으로 차별화).

---

## Yonghee 결정 필요

1. **비용 표의 회계 구조**: §5.2처럼 valuation-only 블록 + end-to-end 블록(Ripple·(a)) 2단 분리로 갈지, 단일 표 + 각주(†¶)로 갈지. (본 문서 권고: 2단 분리 — C1의 가장 깔끔한 해소이고 Ripple 원논문 프레임과도 정합.)
2. **timing.json/GPU-hours/peak-mem 구현 여부**: protocol §15.1 스펙을 논문화 전에 구현할지(러너 4곳 + run_logger 수정; 기존 rundir엔 소급 불가), 아니면 protocol.md를 현실(metrics.json runtime 기반)로 하향 개정할지. (권고: 신규 본런부터만 계측 추가 — 기존 25셀 재실행은 비용 대비 이득 없음; peak-mem은 스모크 1회로 방법별 대표치만 채워도 §5.1-5 충족.)
3. **loss-heur 수치 처리**(C6): 각주로 두기 vs base-loss 캐시 수정 후 재측정. (권고: 논문 표에는 각주; 수정은 다른 코드 결과에 영향 없는 국소 변경이라 본런 전 반영 가치 있음.)
4. **FLDetector·STD-DAGMM GPU 재측정 여부**(C2): CPU 수치 유지+장치 명시(권고)로 충분한지, GPU 측정을 추가해 두 열로 보고할지.

## 후속 실험 제안

1. **[진행 중 실측과 연결] 2026-06-06 표 조건 스모크 재측정** — 1B N=5 R=10 val=100에서 공유 로그 생성 시간을 명시 `_timed`로 측정 + 방법별 runtime 동시 산출 → §1.4-(ii)의 "~15min 어림값"을 실측으로 교체하고 overhead % 열 완성. GPU 해방 후(probe std50k5 완주, ETA 07-05 저녁~07-06) 1셀 ~30분 급.
2. **Ripple 분리 계측**(항목 2와 공유) — ripple 러너에 phase 타이머 래퍼(사본 스크립트, 원본 무수정)를 씌워 학습/drop/sketch/ripple-갱신 분해 → C1의 valuation-only 환산치 확보.
3. **fp32 vs bf16(or tf32) 배수 실측**(항목 3과 공유) — 대표 방법 2개(Flirds, (b))의 forward/HVP 마이크로벤치로 ×3.1 주장의 확정치 산출 → C3 캡션 수치.
4. **peak GPU memory 스모크** — 방법별 1회 실행에 `torch.cuda.max_memory_allocated` 리셋-측정 래퍼 → §5.1-5 열 채움 (phase1_hvp_profile.py:66 패턴 재사용).
5. **GPU-hours 재구성 스크립트** — `runs/*/_logs/_driver.log`류의 start/done 타임스탬프 + metrics.json runtime을 합산하는 read-only 롤업(§15.3 정신; `make_analysis.py`에 컬럼 추가 형태 제안) → 캠페인 총비용 1문장의 근거.
6. **std50k5 완주 후 §1.4-(iii) 방법별 overhead 표 확정** — N=50 레짐에서 "valuation이 학습을 압도"하는 교차점 서술(스케일링 논거)에 사용.

---

## 검증 처리 로그 (2026-07-04 사실검증 반영)

| # | severity | 위치 | 이슈 | 판정 | 처리 |
|---|---|---|---|---|---|
| 1 | minor | §4 C3 (line 259; §5.1 row 4·후속제안 3 연동) | ×3.1 fp32-vs-tensor-core 배수는 프롬프트 사전 확인치일 뿐 어떤 recon 노트·코드에도 1차 근거 없음; 771ms/fwd(fp32 단일 forward)는 bf16/tf32 기준선 부재라 배수를 성립시키지 못하므로 "정합" 서술이 근거를 과장 | **수용** | C3을 개정 — ×3.1을 "미검증 placeholder"로 재규정, 유일 근거는 정성적 `matmul_tf32=False`(remote_recon §3)임을 명시, 771ms 뒷받침 함의 제거, "실측 전 ×3.1 인용 금지" 추가 |
| 2 | minor | §1.4(i) (line 77) | vanilla ≈ 1,866s는 셀 `t_vanilla` 직접 읽기가 아니라 shapleyfl_w 5525s÷2.96 역산인데 [확인]으로 태깅; 교차검증 1,885s는 same-cell이 아니라 across-rank(r16 vs r64) | **수용** | [확인]→[추측·파생]으로 재라벨, 셀 `train_s("vanilla")` 직접 읽기로 대체하라는 지침 병기, 1,885s 대조가 across-rank sanity check임을 명시 |

- 판정 근거 원자료: remote_recon §3(`matmul_tf32 False`/`f32mp highest`; fp32-vs-bf16 벤치 부재), cost recon §3(arm 5525s=vanilla×2.96 기록, `t_vanilla` 직접 미기록), remote_recon 노이즈 probe 블록(r16 vanilla 1,885s).
- 두 건 모두 minor·수용, 기각 0건, critical 0건. 두 이슈 모두 Ripple 무관(eigsh 정정 해당 없음).
