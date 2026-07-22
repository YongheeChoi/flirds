# gnoise dose 진단 (2026-07-22) — 등방 가우시안 위협의 LLM-LoRA 무대 미성립 판정

`gn_full`(GN_ABS γ\*=5) 밴드 체크포인트 FAIL 후, **γ를 더 올려야 하는가**를 6 GPU-h짜리
도박 대신 계산으로 결정하기 위한 진단. 결론은 **γ 축을 닫는다** — 아래 근거 3종이 일치.

## 산출물

| 파일 | 내용 |
|---|---|
| `gnoise_dose_diag.py` | forward-only 섭동 응답 측정. R=30 학습 후 스냅샷 {10,30} × γ {1…1000} × 섭동 3종 |
| `diag30.json` | 위 실행 결과 (1B, gsm50k5 clean, seed0) |

재실행: `REGIME=gsm50k5 R_TRAIN=30 SNAPS=10,30 GAMMAS=1,5,20,50,100,200,500,1000 \
MEM_FRAC=0.22 CACHE=<path>.pt OUT=<path>.json python -u gnoise_dose_diag.py`
(`CACHE`가 있으면 학습 40분을 건너뛰고 섭동 단계만 수 초에 재계산 — γ 범위 바꿔 재판독할 때 사용.
`MEM_FRAC`은 하드 상한이라 프로덕션 런과 GPU를 공유해도 그쪽을 굶기지 않는다.)

## 섭동 3종 (모듈별 ‖Ξ‖_F 매칭)

- **(a)** A·B 각각에 독립 가우시안 = `_add_gnoise`가 실제로 하는 것
- **(b)** ΔW 공간 등방 가우시안 (base weight 직접 섭동 후 복원)
- **(c)** gradient 방향 (참조 상한; chunk별 backward로 누적 = 전체 val gradient)

## 판정 (r=30, val 0.688 기준)

| γ | ‖Ξ‖ | Ξ/ΔW | Ξ/W0 | (a) | (b) | (c) |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.038 | 0.17× | 0.006% | +0.00002 | −0.00001 | +0.188 |
| 20 | 0.153 | 0.66× | 0.025% | +0.00006 | −0.00004 | +0.748 |
| 100 | 0.763 | 3.3× | 0.126% | +0.00036 | −0.00018 | +4.870 |
| 1000 | 7.837 | 34× | **1.297%** | **+0.00830** | −0.00061 | +20.084 |

- **(a)≈(b)≈0, (c)만 큼** — 같은 norm에서 (c)/(a) = **2,400~38,000배**. 즉 dose가 아니라 **방향** 문제.
- (a)가 (b)보다 크므로 "ΔW 공간으로 dose를 재지정" 처방은 **역효과**(H1의 문제제기는 타당, 처방은 틀림).
- 기하: ‖A‖ 24.449가 r=10↔r=30 **불변**(학습은 B만 움직임), ‖A‖/‖B‖ 291→125, ‖ΔW‖/‖W0‖ 0.016%→0.038%.
- 이론식 `E‖Ξ‖² = s²σ²(d_in‖B‖² + d_out‖A‖² + σ²·r·d_in·d_out)` 예측 0.042 vs 실측 0.038(γ=5).

## 실무대 대조 (rundir)

| | val | gsm8k_em | 비고 |
|---|---|---|---|
| clean observer | 0.60219 | 0.3771 | 기준 |
| **gnoise γ=5** (`rundirs_llm/gsm50k5_gnoise_observer_seed0`) | 0.60246 | 0.3753 | oracle보다 **높음**(부호 반대) |
| **gnoise γ=20** (`rundirs_llm_gn20/…`) | 0.60247 | 0.3718 | 부호는 정상화, 여전히 밴드 밖 |
| oracle_excl (γ-무관 재사용) | 0.60239 | 0.3735 | |
| *(참고)* noisy=answer_swap observer | 0.60882 | 0.3342 | **무대 성립 사례** — Δval +0.0066에 EM −4.3pt |

진단(1회 섭동) 대비 실무대(200라운드 반복 주입) Δval이 **4.7배**(γ=20: +0.00006 → +0.00028)
= 누적이 라운드당 오염 2/5 희석보다 우세. 이 배율로 밴드 진입 γ를 역산하면 **γ≈470**.

## 문헌 대조 (2026-07-22 조사)

- **Fang et al. USENIX Sec'20**: 가우시안 공격의 σ를 benign 클라 간 좌표별 분포에 moment-matching
  (= **canonical γ≈1**). 그리고 이 공격을 *"crafting compromised local models randomly can **not
  effectively attack**"*를 보이기 위한 **음성 대조군**으로 명시적으로 도입. MNIST LR에서 Krum
  0.14→0.13 등 오차가 오히려 **감소**. ⇒ 우리 결과는 실패가 아니라 **문헌이 예측한 결과**.
- **LoRASC**(Findings-EMNLP'24): 노이즈를 **LoRA 곱 BA의 std 기준**으로 스케일(우리와 동일 축),
  λ=10 ≈ **γ2.9**에서 GSM8K 19.5→27.5 **성능 향상**. ⇒ 우리 γ=5는 문헌의 *정규화 레시피* 자리.
- **양자화 좌표**: γ=1000의 ‖Ξ‖/‖W0‖=1.3%는 **INT8 수준**(무변화가 정상). 유해 구간(INT4≈15%)엔
  γ≈11,500 필요.
- **LLM/LoRA FL에 노이즈 dose를 재조정한 선행 0건**. OpenFedLLM §5.4가 *"uncertainty on the
  effectiveness of previous robust methods on FedLLM"*을 open problem으로 선언해 둔 그 빈칸.
- ⚠️ 검증 실패로 **인용 금지**: "Krum 원논문 σ=200"(refuted), arXiv 2509.09097·2602.19926·
  2605.07961(ID 확인 실패 — 인용 전 직접 검증 필요).

## 결론

등방 가우시안 노이즈는 **LLM LoRA FL에서 성립하는 위협이 아니다**(dose 문제가 아니라 구조 문제:
저랭크 부분공간 × Hessian spiked spectrum → 랜덤 방향 곡률 ≈ 0). negative result로 서술 가치 있음.
위협을 교체한다면 문헌 표준은 **LIE**(`μ_j + z·σ_j`, z≈1σ, 조정된 mean-shift) 또는 **sign-flip**
(크기 파라미터 없음) — 둘 다 (c)가 보여준 방향-정렬 영역. 프로젝트 인벤토리 **I-28**(direction-aligned
poison = 2차항 flagship 후보)과 같은 안건.

**H-10(1차 실명 vs 2차 포착)은 CNN에서 이미 성립**(vanilla .244 vs Flirds .567~.607 vs
Flirds-1st/FedIF .244~.248)하므로 서사 자체는 유지되고, LLM leg만 공백으로 남는다.
