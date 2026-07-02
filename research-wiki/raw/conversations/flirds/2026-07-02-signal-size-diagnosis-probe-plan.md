---
type: conversation
date: 2026-07-02
topic: flirds
participants: [Yonghee, Claude]
tags: [signal-size, diagnosis, lora-rank, participation, probe]
---

# 2026-07-02 — 신호 크기 진단(Phase 1) + rank/참여 probe 계획·실행

> 세션 유형: 분석(재실행 없음) + 실험 설계. Yonghee 가설: "val-loss 절대 변화량이
> 너무 작아 fidelity·intervention 정밀도가 떨어진다" 검증. lever는 LoRA rank와
> round당 참여 수 둘만; 무대 변경 없음.

## Phase 1 진단 (기존 산출물만: track_d 18셀 / phase2_matrix 25셀 / track_c 150셀)

방법 요점 — 새 계산 없이 저장물에서 유도:
- loss-heur φᵢ = U({i}) (in_run_utility, U(∅)=0), Σφ_(b) = U(N) → **가산성 갭
  v(N)−Σv({i})를 phi.parquet만으로 계산** (별도 coalition 재계산 불필요).
- vanilla arms의 val_curve/MMLU/ROUGE = base vs trained 절대 성능.
- per-seed rundir을 cellbase로 묶어 paired arm-효과와 (b) oracle 자기 순위의
  cross-seed 안정성 산출.

결론(상세 수치·표는 wiki/flirds-signal-size-diagnosis.md):
1. fp32 정밀도 병목 아님 (최소 신호가 ulp의 10²⁺배).
2. 절대 학습: val-loss −0.03~−0.13, ROUGE +5~13pp 실재 / **MMLU 0/음수**
   (std20 −0.8pp, 7B std20 −1.4pp 유의; SFT는 포맷 학습, capability 아님).
3. **주 병목 = IID-clean 무대에 클라 간 진짜 신호가 구조적으로 없음**: (b) oracle
   자기 순위 cross-seed ρ ≈ −0.37~+0.73(대부분 ~0); CNN 대조군(track_c)
   label_flip 0.968 vs iid −0.042. 추정기는 oracle의 노이즈까지 값-수준
   0.9999+로 재현 — fidelity 측정은 성립, 무대에 랭킹할 실체가 없는 것.
4. 게임 사실상 가산적(갭 0.1–0.9%, Banzhaf−Shapley d∞ ~2e-6) → anchor에서 방법
   전부 +1.000 붕괴; 방법 구별은 부분참여(std20)가 만듦.
5. intervention: paired val-loss 축에서만 검출(flirds_w −0.0009~−0.0036,
   SNR 2.4–4.5; 7B가 최대), MMLU/ROUGE는 효과 < 표본 SE라 원리적 검출 불가.
   r2t는 7B std20에서만 해상도(127–142 vs 159).
6. 4-i(val bootstrap)는 체크포인트 미보존으로 artifact만으론 불가 → probe에
   per-chunk (b) 효용 dump 삽입으로 설계.

## Phase 2 probe 계획 (보고 후 승인 대기; 실행 안 함)

- LLM 1B: A=anchor5 rank{32,64}×3seed(6셀, rank16 재사용), B=std N=50/5-per-round
  R=200 rank{16,32,64}×3seed(9셀 전부 신규; (b)=per-round 2⁵),
  C=noise-probe(anchor rank16/64 seed0 bootstrap SE). α=2r 유지.
  trimmed+GTG 스위트 추천. 비용 ≈ A 21h + B ~250h + C 4h ≈ **270 GPU-h**
  (B가 92%; 4 GPU ~3일). 파일럿 = 전 셀 seed0 먼저 → 게이트.
- CNN(yonsei SLURM, sbatch 작성/제출 분리 — 이 박스 sbatch 없음): 용량 파라미터 =
  폭 배수 w{0.5,1,2,4}(disanalogy caveat), C1-probe 72셀(iid+label_flip ×
  k{2,5,10}) ~25 GPU-h, C2-probe 36셀 ~50–70 GPU-h. grids/probe_c{1,2}.txt +
  기존 run_array.sbatch 재사용.
- 코드 변경은 전부 env-gated 기본값=현행 (track_d: LORA_R/LORA_ALPHA/
  N_CLIENTS/K_ABS/FID_METHODS/ARMS_LIST; track_c1/c2: WIDTH/KFRAC;
  models/cnn.py width 인자). RUNDIR_ROOT=runs/probe_signal 분리. 커밋은 요청 시.

## 운영 메모
- 진단 시점 GPU 0–2 = Yonghee의 3B_anchor5 재실행(ORACLE_A=0, RUN_NAME=
  3B_anchor5_seed{0,1,2}) 진행 중. **완료 시 기존 3B_anchor5 rundir((a)oracle
  포함, 커밋본 a5f5893 계열)을 (a) 없는 버전으로 덮어씀** — 커밋본은 안전하나
  working tree에서 (a) 행이 사라짐. Yonghee 인지 필요.
- 3B_std20 rundir working-tree 수정본(06-19 Pearson 네이티브 재실행, git sha
  1c02fcd, 커밋 대기)을 진단에 사용.

## 승인 및 실행 개시 (같은 세션, 이어서)
- Yonghee 결정: **seed0 파일럿 먼저 / 방법 스위트 = full 11종 / CNN C1+C2 모두**.
  → full 반영 비용: 파일럿 ~150 GPU-h, 3-seed 전체 ~450 GPU-h (wiki §2.1 갱신).
- 구현(전부 env-gated, 기본값=현행): track_d.py LORA_R/LORA_ALPHA(α=2r)+
  N_CLIENTS/K_ABS override+config에 lora 기록; experiments/probe_val_noise.py 신규
  (chunk-resolved 2⁵ (b) 효용 → val bootstrap SE·half-split ρ; track_d 로더 재사용);
  models/cnn.py width 인자; track_c1 C1_WIDTH/C1_KFRAC/C1_RIPPLE/C1_RUN_NAME +
  **ComFedSV partial=KFRAC<1 수정**(부분참여 KeyError); track_c2 C2_WIDTH/C2_FRAC/
  C2_RUN_NAME. 스모크: track_d(gpt2 N8/k3/r32) green, noise-probe(gpt2) green,
  track_c1(mnist w2 k0.5) green, track_c2(fmnist w2 f0.3) green, CNN bit-identical
  가드 green(width 기본 1.0 = 동일 치수).
- 실행: `runs/probe_signal/run_pilot.sh` nohup 가동 — GPU3에서 1B_anchor5_r32_seed0
  학습 개시 확인; GPU0/1/2는 3B 재실행 종료 대기 후 std50k5 r16/32/64 자동 시작.
- CNN: slurm/grids/probe_c1.txt(66셀)+probe_c2.txt(30셀) 생성(기존 w=1 셀 중복 제외);
  제출 커맨드는 runs/probe_signal/README.md (yonsei에서 Yonghee).

## 정정 (같은 세션, Yonghee)
- Yonghee: "shapley 0.3으로 돌린 건 덮어써야지. 지금 우리 세션에서 새로 돌릴 실험이
  덮어쓰냐고 물어본 거였는데." → **β0.3 캠페인의 canonical 덮어쓰기는 의도된 설계
  (단일 provenance)이며 유지**; 앞선 분리 조치는 오해로, 전부 원복 —
  3B_std20 canonical = β0.3본 재반영(β0.5 원본은 git 히스토리), preserve 가드 중단·삭제,
  v2 큐 RUNDIR_ROOT 리다이렉트 제거, rundirs_beta03/ 제거(커밋 amend로 흔적 제거).
- **원 질문에 대한 답: 이번 세션 probe는 기존 결과를 일절 덮어쓰지 않음** —
  LLM `runs/probe_signal/rundirs/` 신규 셀명, CNN `runs/probe_signal/cnn_c{1,2}/`
  `pc*` 셀명, 기준점 셀(rank16 anchor·CNN w=1)은 재실행 없이 기존 rundir 재사용.
- 부수 정정: 3B_anchor5 rundir은 원래 (a)oracle을 포함하지 않음(1B anchor만 (a) 보유)
  — 앞선 "(a) 소실" 경고는 이중으로 잘못된 flag였음.
