# T5 — retrain-(a) 특성화 스위트 (= L8): gsm5 신설 + silo5 (a)-leg

> 목적(paper §5.2 sub): (a) exact retrain oracle 비교를 **주무대에 최대 정합**(gsm5) + **비IID 신호 무대**(silo5)로
> 확보. (a)는 2^N 재학습이라 작은-N 별도 무대가 불가피 — 논문 설명 문구의 실체.
> 실행처: **RTX3090×8**(1B fp32 LoRA 안착; B200은 L1·L2·L7 전용). Yonghee 직접 가동 가능하도록 명령까지 명시할 것.

## 1. gsm5 무대 신설 (주 표 후보 — 주무대와 데이터·위협·심판 동일)

- **스펙**: GSM8K·Llama-3.2-1B-Instruct·LoRA r16/α32·plain SGD lr 1e-3·10 steps·b16·maxlen 512·warmup 3
  (= R4 verbatim), **N=5 전원 참여**(= R4의 라운드-cohort 크기 5와 동일 — "라운드-cohort 축소판" 프레임), **R=30**,
  **클라당 149문항**(R4의 per-client 크기 유지 — seed-셔플 후 149×5=745만 사용, 잔여 버림), val=200(공식 test 카브,
  R4 규칙 동일). 위협 2셀: clean / noisy=answer_swap@0.7 **오염 2/5(=40% 유지, 클라 0–1)**. seeds {0,1,2}.
- **oracle**: dual — (b) exact 2⁵ per-round + **(a) exact 2⁵ retrain**(부분집합별 R=30 재학습, val-loss 게임 식 (5)).
  방법 스위트는 전 9종 채점(싸다; 본문/부록 배치는 T1 규칙 — 본문 same-game·vs(a)는 전 방법).
- **구현 포인트**: `build_gsm8k_iid`에 per-client 크기 파라미터(또는 n=5·745-서브샘플 경로) 추가;
  (a) 러너는 `experiments/track_d.py`의 32-retrain 기계 이식(**`_guard` 훅 위생 필수** — SFTTrainer↔HVP 충돌 전례);
  무대 러너는 track_g GSM50K5 레짐의 N=5 변형 또는 phase2_matrix REGIME 추가 중 싼 쪽. 스모크(tiny-gpt2, PERSIST=0) 후 제출.
- **비용**: (a) ≈ 32 retrains×R30 ≈ 8.5 B200-h/seed·셀(anchor5 실측 30,817s의 동일 단가) → 2셀×3-seed ≈ **~51 B200-h**
  (3090 ~2×; 6 leg 병렬 → wall ~1일). 무대런((b)+방법) ≈ +8 B200-h.
- **사전 기대(등록 후 커밋)**: 근가산 + 신호(오염 40%) 무대 → (a)↔(b) 및 same-game vs (a) **높은 일치 예상**(anchor5
  0.933과 silo5-류 안정성의 결합); noisy 셀에서 (a)도 오염 클라를 하위로 매길 것. MISS 그대로 보고.

## 2. silo5 (a)-leg (비IID 보조 — 기존 무대에 (a)만 추가)

- **셀**: silo5 {clean, noisy nr1.0, frzero} × seeds {0,1,2} — **기존 canonical rundir(β0.3 재실행판 ce0b454,
  `runs/phase2_matrix/rundirs/1B_silo5_*`)와 동일 split·seed**로 (a) 32-retrain(R=10)만 신규 실행.
- **조인**: (a)는 재학습이라 실현 궤적 무관 — 기존 phi.parquet(방법·(b) φ)와 seed별 조인해 spearman_a/pearson_a 산출
  (CNN `merge_oracle_a.py` 패턴; `*_aonly_*` rundir 명명 관례 재사용). 기존 rundir **무수정**(read-only) — 파생 CSV만.
- **비용**: ≈ 2.9 B200-h/seed·셀 → 3셀×3-seed ≈ **~26 B200-h**(3090 9 leg 병렬 → wall ~반나절).
- **의미**: N=5 무대 중 silo5만 (b) 타깃이 seed-재현(clean +0.87·오염 +0.93; overview §5.4) — "(a)가 그 실재 신호를
  같게 매기는가"의 첫 측정. frzero 셀은 (a)-쪽 null-player((a) 게임에서 free-rider 재학습 기여 0) 검증 겸용.

## 3. 배치·우선순위·산출

- 우선순위: **gsm5 먼저**(본문 주 표 후보) → silo5. anchor5 기존 값(0.933 셋)은 IID 참조로 존치(재실행 없음).
- 3090 배치: leg(=셀×seed) 단위 1 GPU 1 job; (a) 재학습 32회는 leg 내 직렬(체크포인트 재개 가능하게 subset-단위 저장 권장).
- 산출: rundir(config+meta+phi.parquet[(a)·(b)·방법 φ]+metrics[spearman_a/pearson_a 포함]) + 분석 CSV →
  overview §3.1.1 이웃 신규 소절 기입 → paper §5.2 sub 표 채움(T1) → T2 페이지 F2 갱신.
- 옵션 처리: 본문 주 표 = gsm5(착지 시) / silo5 = 비IID 보조 / anchor5 = IID 참조 각주·병기. gsm5 미착지 동안은
  anchor5+CNN 시나리오별 표가 본문 임시 주인(⬚ 교체 전제).
- 3B/7B (a)는 **하지 않음**(여유 시 재논의 — 00-INDEX §0).
