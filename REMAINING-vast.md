# REMAINING (vast.ai 풀) — R4 downstream SFT 물량

> 실행처별 인수인계 **3부작** 중 vast.ai 몫.
> 짝 = `REMAINING-b200.md`(HVP·fidelity·timing) · `REMAINING-slurm.md`(CNN + 작은-N LLM).
> **상태: 대여 미개시 — Yonghee 승인·대여 후 실행.** push는 Yonghee 직접.
> 수치 = rundir/analysis 재생성 값만(수기 기입 금지). 논문·문서 정본 = `paper/workplan/00-INDEX.md`.

## 0. 배치 규칙 — 왜 이 실험들이 여기인가

**메모리가 실행처를 가른다** (2026-07-24 B200 실측):

| 클래스 | nvidia-smi reserved | `timing.json` allocated | 32GB(5090) 적재 | 실행처 |
|---|---|---|---|---|
| **HVP** (flirds 2차 φ, `jvp∘grad`) | ~140 GiB | **95.5 GiB** | ✗ (VAL_CHUNK 강제 감량 시 5–10× 지연) | **B200 전담** |
| **LOW/SFT** (게이트·T2 재학습·online·renorm·탐지기) | ~40 GiB | **~27 GiB(추정)** | ✓ 네이티브 | **vast** |

- reserved ≠ allocated(비 ~1.47) — B200 192GB에선 캐싱 할당자가 과다 예약. **판단은 allocated로.**
- ⚠ **착수 전 검증 1건(필수)**: 기존 T2/online rundir `timing.json`의 LOW 페이즈 `peak_gib` 확인
  → **≤30이면 RTX 5090(32GB) 확정**, **>32면 48GB+ 급으로 상향 대여**(recipe 변경 금지 — batch 축소 X).
- HVP는 여기 **안 보냄**: 95.5 GiB라 32GB에 안 들어가고, VAL_CHUNK로 우겨넣으면 B200 대비 5–10× 느려 대여비만 태움.

**정책 (07-24 Yonghee)**: seed0(canonical 앵커) + **전 timing 셀은 B200 고정** · seeds 1-2의 non-timing
fidelity/downstream 복제만 vast · **`timing.json`은 vast 산출 사용 금지**(§5.5 cost = B200 실측만).

> ⚠ **정책 예외 = 승인 대기 항목**: B200 잔여 창(§2 note)이 HVP·fidelity만으로 이미 포화라
> **downstream seed0까지 vast로 내려야 물리적으로 완주**함. downstream EM은 W-A에서 스택-강건 판정
> (recovery 정규화하 mean|Δ|≤0.006)이고 대상이 foil(비-flirds) 레그라 근거는 있으나, **Yonghee 명시 승인 필요**.
> 승인 전이면 §2의 **V-A(seeds1-2)만** 실행하고 V-B는 보류.

## 1. 환경 (대여 박스 1회 구성 → 이미지로 구워 전 셀 재사용)

B200 컨테이너와 **동일 버전 고정**(rundir meta 정합):
`torch 2.12.0+cu130` · `transformers 5.9.0` · `trl 1.5.1` · `peft 0.19.1` · `accelerate 1.13.0` ·
`datasets 4.8.5` · `numpy 2.4.6`. (5090=Blackwell/CUDA 13.x → cu130 휠 정상.)

- `HF_HOME=<box>/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`, `codes/`에서 `PYTHONPATH=.`.
- **gated meta-llama**: 유효 토큰 없음 → B200과 동일한 **해시 교차검증 공개 미러**로 캐시 구성.
  검증 체인 = `$BATCH/PROVENANCE.md`. 캐시를 이미지에 구워 넣어 셀마다 재다운로드 방지.
- 코드 sync: **fix(`8598cea`)+revert(`fa2c167`) 이후 커밋**에서 체크아웃(git clean) — rundir meta의
  `git_sha`가 canonical 조건.
- **스택 캐비엇**: vast rundir는 B200과 다른 GPU/드라이버 → **절대값 병치 금지, recovery 정규화로 읽음**
  (CNN C-fr가 동일 처리 선례). Spearman·recovery·EM은 스택-강건(W-A).

## 2. 배정 실험

전부 **downstream SFT**(관찰자 φ 재사용 → HVP 재실행 0). 비용은 **B200 환산 GPU-h**.

### V-A — 정책 준수분 (seeds 1-2, 즉시 실행 가능)

| # | 실험 | 셀/런 | GPU-h | 의존성 |
|---|---|---|---|---|
| **V1a** | **L11** non-flirds 7방법 T1 부호게이트 | 42런 (3threat × **seed{1,2}** × 7src) | ~165–200 | L1 observer cum(noisy·frzero 산출됨 / clean seed0 산출됨) |
| **V2a** | **L4** renorm-4 T2-only | 6셀 (3threat × **seed{1,2}**) | ~135–155 | 없음(renorm=value-only, 자체완결) |
| **V3a** | **L9** frrand full-8 **arms**(T1/T2) | seed{1,2} | ~113 | **B200가 L9 flirds 관찰자 선(先)산출** → cum 복사 |
| **V4a** | **L7** P1w arms(`flirds_gatew_v2`·`t2_signw_flirds`) | 12런 (seed{1,2} × 3threat × {T1,T2}) | ~53 | L1/L7 observer cum |
| | **소계** | | **~466–521** | |

### V-B — 승인 시 추가 (해당 레그 seed0; §0 예외 항목)

| # | 실험 | 셀/런 | GPU-h |
|---|---|---|---|
| V1b | L11 seed0 | 21런 | ~85–100 |
| V2b | L4 seed0 | 3셀 | ~65–75 |
| V3b | L9 arms seed0 | — | ~57 |
| V4b | L7 arms seed0 | 6런 | ~27 |
| | **소계** | | **~234–259** |

> **B200 잔여 창 계산(근거)**: 07-24 11:00 → 07-26 24:00 ≈ 61h × 4 GPU = **~244 GPU-wall-h**.
> B200 필수분(HVP·fidelity·timing) = L1-clean s1·2(19) + L2 잔여 4셀(60–100) + L10(30–45) +
> L9 관찰자(30) + L7 관찰자(40) + anchor5(40) ≈ **~220–275**, HVP는 **1셀/GPU**라 팩킹 불가
> → **창을 이미 채움**. ⟹ downstream seed0(V-B)이 B200에 들어갈 자리가 없음.

**여기서 안 하는 것**: HVP 셀 전부(L2·L10·L5·L6·각 레그 관찰자) · timing 측정 · fidelity canonical.
전부 `REMAINING-b200.md`.

## 3. 비용·시간

**1셀/GPU**(LOW allocated ~27 GiB, 32GB에 2개는 불가) → 16장 = 16-wide.

| 범위 | V-A만 (~490) | V-A+V-B (~730) |
|---|---|---|
| s=1.3 | ~40h / ~$400 | ~59h / ~$590 |
| **s=1.5 (예상)** | **~46h ≈ 1.9일 / ~$460** | **~68h ≈ 2.9일 / ~$685** |
| s=2.0 | ~61h / ~$610 | ~91h / ~$915 |

`s` = 5090/B200 속도비. fp32 워크로드(`bf16=False`, [llm_server.py:83](codes/flirds/fl/llm_server.py:83))라
B200 텐서코어 우위 미사용 → 5090 FP32 raw가 오히려 셈(~105 vs ~80 TFLOPS)이나 대역폭은 B200 ~4.5×.
SFT는 대역폭 의존이 HVP보다 낮아 `s`가 유리한 쪽. 요금은 16장 박스 ~$10/hr 기준.

**⚠ 대여 전 캘리브레이션(30분, 필수)**: 5090 **1장**에서 L11 online **1런** 실행 → wall ÷ 같은 셀의
B200 rundir `timing.json` wall = **`s` 확정** → 위 표 한 줄이 점 추정이 됨. $400–900 불확실성을 여기서 제거.

## 4. 실행 순서 (의존성)

1. **§0 검증**(LOW `peak_gib` ≤30) → GPU 티어 확정 → 박스 대여 → §1 이미지 구성.
2. **`s` 캘리브레이션 1런** → 예산 확정 → Yonghee 보고.
3. **즉시 착수(cum 대기 없음)**: **V2a(L4)** = 자체완결 · **V1a(L11)** = B200 산출 cum 복사분으로 가동.
   - cum 복사 = 해당 L1 rundir의 `metrics.json`(`observer_cum`)만. 파일 수 KB, rundir 전체 불요.
4. **V3a(L9 arms)**: B200가 **L9 flirds 관찰자**를 프론트로드해 cum 넘겨준 뒤 착수 — 그전엔 블록.
   → B200 큐에서 L9 관찰자 우선순위를 올려둘 것(`REMAINING-b200.md` 실행처 배치 §).
5. **V4a(L7 arms)**: observer 재사용 경로(§2a 권장(b)) 사용 — `ARMS=flirds_gatew_v2`만 분리 실행 시
   관찰자 불요. rundir root 충돌 없음(P1w arm명 신규).
6. 승인 시 **V-B** 이어서.

**착지 root**: `RUNDIR_ROOT=<repo>/runs/track_h/rundirs_llm_vast` — B200 canonical(`rundirs_llm`) **무수정**.
`make_analysis.py` LLM 로더에 이 root 추가 필요(dup-win 규칙 유지).

## 5. 산출물 회수·커밋 규칙

- [ ] rundir 전량 회수(박스 종료 전) — **arm 단위 영속**이라 완료 arm은 중도 종료에도 생존.
- [ ] meta `git_sha` = fix-후 커밋 · config `identity` 가드 통과 확인.
- [ ] **`timing.json`은 회수하되 §5.5 cost 표에 사용 금지** — vast 산출임을 rundir README/커밋 메시지에 명기.
- [ ] `runs/track_h/make_analysis.py` 재생성 → H-12/H-13/H-14 사전등록 대조(MISS 포함 보고).
- [ ] overview §3.2.4 이웃 소절 기입 → paper §5.3은 그로부터.
- [ ] **push는 Yonghee 직접.**

## 6. 하지 않는 것 (재제안 금지)

`REMAINING-b200.md` §2 목록 그대로 승계: gnoise LLM 재시도 · LIE/sign-flip · (a) 3B/7B · P0 전면 소급 ·
Fed-LOO · **poison** · P5h/P5s · std20/anchor5-vs(b) 재실행 · E5 N=10 확장 · β0.3 잔여(device100·3B·7B).
추가로 **HVP 셀 vast 이관 금지**(§0) · **batch 축소로 메모리 맞추기 금지**(recipe 변경).
