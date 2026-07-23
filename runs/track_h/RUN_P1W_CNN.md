# T4 — P1w CNN 검증 (W-A 판정 + W-B 실행 인수인계)

> 스펙 정본 = `paper/workplan/T4-p1w-cnn-relay.md`. 이 문서 = 구현 산출물 + W-A 판정 +
> W-B 실행 절차. **P1w ≡ 기존 P2(sign+크기가중)** — 신규 코드 없음(track_c2 `gatew_v2`/`C2_T2`).
> 구현 세션(2026-07-23, Windows): 코드·판정 로컬 완료, GPU 실행은 RTX3090 Slurk 서버 몫.

정책 매핑(arm 라벨 유지, 분석 표기만 P1w):

| | P1 (sign gate, n-가중) | **P1w (=P2, sign+크기가중)** |
|---|---|---|
| **T1 online** | `flirds_gate_v2` | `flirds_gatew_v2` |
| **T2 retrain** | `t2_sign_flirds` | `t2_signw_flirds` |

---

## W-A — dir1 기존 P2 재사용 판정 (재실행 0)

**드리프트 = 무시 가능(귀속-임계 지표 한정) → dir1 P2를 P1w로 귀속, 재실행 불필요.**

근거 = `runs/track_g/rundirs_cnn`(원본 B200/torch2.12) vs `rundirs_cnn_restack`(RTX3090/
torch2.11) 동일 config·seed 12셀(cifar10 iid·dir1)×arm×3seed = 312쌍 절대-acc 드리프트:

| 지표 | mean drift | mean\|drift\| | max\|drift\| | 판정 |
|---|---|---|---|---|
| 전체(312쌍) | −0.0022 | 0.0088 | 0.233 | — |
| **oracle_excl**(recovery 천장) | −0.0007 | **0.0010** | 0.0018 | ✅ 안정 |
| **vanilla**(recovery 바닥) | +0.0009 | **0.0024** | 0.024 | ✅ 안정 |
| **flirds_gatew_v2 (P1w)** | +0.0005 | **0.0063** | 0.032 | ✅ 밴드 내(±0.02) |
| flirds_gate_v2 (P1) | −0.0053 | 0.0144 | 0.095 | ⚠ clean 0.041 |
| flirds_gate_v1 (V1, **미사용**) | −0.0079 | 0.0175 | **0.233** | 무관(P1/P1w 아님) |

- **recovery 정규화 앵커(vanilla·oracle_excl)와 P1w arm은 스택 간 안정**(mean\|drift\| ≤ 0.006,
  분석 밴드 RECOVERY_MIN_GAP 0.02·clean-parity 0.006 이내) → dir1 P2 값은 스택-강건.
- **최대 드리프트(0.233)는 flirds_gate_v1**(per-round raw 게이트) — P1/P1w 비교에 **미사용**.
- **grad-noise 가 가장 seed-민감**(per-seed 게이트 arm 드리프트 최대 0.055; V1은 0.23) —
  단 **3-seed 평균에서 상쇄**(dir1 GN P1 mean drift +0.005·P1w −0.004). 표는 3-seed 평균이라 강건.
- **핵심**: dir1 P1·P1w 는 **둘 다 같은 B200 스택**(track_h/rundirs_cnn 재사용) → dir1 내부
  P1-vs-P1w 비교는 드리프트와 무관하게 자기일치. 드리프트는 dir1(B200)↔확장(RTX3090) 절대값
  병치 시에만 유효하고, 그때도 recovery(스택-강건)로 읽으면 무해.

**canon 재확인(재생성 값, 수기 아님)**: `make_p1w_cnn_table.py` 가 merge(track_g T1 +
track_h T2)로 overview §3.2.3 flirds 행을 **정확히 재현** →

| | clean | 오염평균(fr·gn·lf@0.7) |
|---|---|---|
| P1 online (`flirds_gate_v2`) | .6315 | **.5843** |
| P1w online (`flirds_gatew_v2`) | .6188 | **.5913** (P1 대비 +0.7pt) |
| P1 retrain (`t2_sign_flirds`) | .6277 | **.6107** |
| P1w retrain (`t2_signw_flirds`) | .6123 | **.5959** (P1 대비 −1.5pt) |

**FedIF 역전(수록 규칙 '타 소스 역전' 조항)**: dir1 P1w 오염평균 FedIF **online .6011 /
retrain .6159 > flirds .5913 / .5959** — 실측 확인. 판정 보고에 명시 필수(아래 판정 §).

> 참고: T4 스펙 W-A note-2 의 손기입 수치는 위 재생성 값과 일치(T1/T2 라벨은 online/retrain).

---

## W-B — flirds P1w twin leg 실행 (확장 무대, 신규 = T2 leg만)

**무대**(90셀 = 5 (ds,part) × 6 위협 × 3 seed; 전 셀 downstream twin 有):
`{cifar10 shard·qskew·iid, fmnist iid·dir1} × {clean, free_rider, frrand, grad_noise,
label_flip@0.70, label_flip strmain} × seed{0,1,2}`. **cifar10 dir1 은 제외**(= W-A).

**신규 실행 = T2 leg만.** T1(`flirds_gate_v2`/`gatew_v2`)은 skew 캠페인
(`track_g/rundirs_cnn`)에 이미 있음 → 재사용. 이 leg 는 flirds-only observer + `C2_T2=1`:
- 산출: `observer`(flirds 채점) + `t2_sign_flirds`(P1-T2) + `t2_signw_flirds`(P1w-T2) +
  `t2_random_k<size>`(순위-가치 통제).
- 착지 = `runs/track_h/rundirs_cnn` → `make_analysis` 가 skew twin 의 vanilla·oracle_excl·
  T1 게이트 arm 과 **셀키(ds,part,threat,flip_rate,seed)로 병합**. (dir1 에서 이 병합이 위
  canon 을 정확히 재현함으로 검증됨.)
- **비-flirds 점수원(W-D)은 후순위 별도 승인** — flirds leg 보고 후.

### 제출

```bash
mkdir -p runs/track_h/_logs                            # 로그 dir(gitignore line 54 = 서버 생성)
sbatch --array=0-29%8 runs/track_h/sbatch_cnn_p1w.sh   # seed-0 파일럿(30셀) → GPU-h 보고
sbatch --array=30-89%8 runs/track_h/sbatch_cnn_p1w.sh  # GO 후 seeds 1-2
# 또는 한번에:  sbatch runs/track_h/sbatch_cnn_p1w.sh
```

### 비용 (그리드 실측 준용)

- arm ~3.5–8분(track_g CNN 실측; observer=flirds 1소스라 게이트 arm 1개 등가).
- 오염셀 75개 × 4 arm(observer + t2_sign + t2_signw + t2_random) + clean셀 15개 × 1 arm
  (observer; T2 kept=전원 → `equals_vanilla` 스킵) = **315 arm-등가**.
- **≈ 30–32 GPU-h**(범위 ~18–42; 스펙 개산 25–35 부합). 8-GPU 동시 → wall ~4–5h.
- 파일럿(seed0, 30셀 = 오염 25 + clean 5 = 105 arm-등가) ≈ **10–11 GPU-h** → 실측 보고 후 seeds 1-2.

### 분석 (실행 후)

```bash
python runs/track_h/make_analysis.py         # 병합 competition CSV(policy 컬럼 = 기존 스키마)
python runs/track_h/make_p1w_cnn_table.py     # W-B 전용: P1 vs P1w 표(위협×파티션) + H-5 대조
```
산출: `analysis/p1w_cnn.csv` + `p1w_cnn_README.md`(절대 acc 표·오염평균 gap·guard-recovery·
clean parity·FedIF 역전 노트). T1 은 지금도 채워짐(skew on disk); T2 는 sbatch 완주 후 채워짐.

---

## 사전등록 (실행 전 커밋) = README §2 H-15

dir1 H-5 트레이드오프의 확장 재현: ① 오염평균 P1w ≥ P1 online(크기-skew 큰 qskew/shard서 gap
최대, fmnist iid ≈0) ② P1w ≤ P1 retrain 또는 근사동률 ③ clean P1w 오발화 소폭↑(±0.006 경계)
④ FedIF 역전 확장 재현 = **W-D 대기**. **T1-only 예비 관측**(on-disk, T2 전): 온라인 오염평균
gap(P1w−P1) = **+0.003**(dir1 +0.007 대비 방향 일치·소폭), guard-recovery P1w +0.69 > P1 +0.54,
clean ΔAcc P1w −0.007(P1 −0.006; 밴드 경계). ⚠ 예비치 — 실행 전 커밋 후 T2 포함 재산.

## 판정 (완료 후 = 00-INDEX §1 수록 규칙)

- **승**(본문 승격): W-A·W-B·L7 전 범위서 P1w ≥ P1(오염) & clean parity 유지.
- **동률**: "부호가 가치의 대부분" ablation 1문장(P1w 미본문화, rundir 존속).
- **미수록**(P1만): 열세 또는 **타 소스 역전** — dir1 에서 **FedIF > flirds(P1w)** 이미 관측 →
  이 조항이 이미 발동 상태. 확장 무대 FedIF 재현은 W-D 필요. **W-B 단독 판정 금지**: L7(LLM
  P1w)·W-A 종합 후 Yonghee 확정.
- 결과 기입처: overview §3.2.3 이웃(신규 소절) → paper·T2 는 그로부터.
