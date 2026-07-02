# Corruption-axis × non-IID-axis matrix (2×2)

신호 실재성(B축) 실험. 계획: `research-wiki/wiki/flirds-signal-size-diagnosis.md` §2.4.
"신호는 학습 강도(A축)가 아니라 클라 간 실제 차이(B축)가 만든다"를 오염축과 비IID축을
**분리**해 확정한다(기존 silo5는 둘이 항상 결합돼 분리 불가).

## 셀 (silo5 급 통일: N=5 full, val20/R10, (b)=exact 2⁵, 3-seed)

| 무대 \ 클라 | clean | noisy | free-rider(rand/zero) | poison |
|---|---|---|---|---|
| **IID** (alpaca) | `1B_iid5_clean` | `1B_iid5_noisy` | `1B_iid5_frrand`·`1B_iid5_frzero` | `1B_iid5_poison` |
| **non-IID** (5-domain) | `1B_silo5_clean` | 기존 `1B_silo5_noisy` | 기존 `1B_silo5_frrand`·`frzero` | 기존 `1B_silo5_poison` |

신규 6셀(굵게 아닌 것 포함 IID 5 + silo5 clean 1). silo5 오염 3셀은 기존 재사용(비교용).
전부 신규 셀명 → **기존 결과 안 덮어씀**. `runs/phase2_matrix/rundirs`에 저장(make_analysis 통합).

## 측정 (핵심 질문 위계)

- **1차 fidelity**: (b)oracle 자기 순위 cross-seed ρ. 미지수 = non-IID clean이 오염 없이도
  ρ 높은가(도메인 순수 신호), IID clean은 ρ≈0 재현되는가.
- **2차-③ 탐지 AUROC**: 오염 클라 이진 탐지. 핵심 대조 = IID+noisy vs non-IID+noisy
  ("도메인 이질성이 탐지를 돕나/방해하나").

## 실행

**GPU0-2** (GPU3은 rank-probe = `run_pilot.sh`가 계속 사용). 캠페인(3B_anchor5) 종료 후:

```bash
# 조율: run_pilot.sh의 GPU0-2 B셀(std50k5) waiter를 멈춰야 매트릭스가 그 GPU를 잡는다
#       (Yonghee 결정 "매트릭스 먼저"). GPU3 A셀 체인은 유지.
nohup bash runs/matrix_cxni/run_matrix.sh > runs/matrix_cxni/_nohup.out 2>&1 &
```

개별 셀 수동 실행 예:
```bash
cd codes; PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 REGIME=iid5 THREAT=clean RUN_NAME=1B_iid5_clean \
  RUNDIR_ROOT=../runs/phase2_matrix/rundirs python -u experiments/phase2_matrix.py
# poison만 별도 config:
CUDA_VISIBLE_DEVICES=2 REGIME=iid5 THREAT=poison LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=0.8 \
  POISON_TRAIN=1000 RUN_NAME=1B_iid5_poison RUNDIR_ROOT=../runs/phase2_matrix/rundirs \
  python -u experiments/phase2_matrix.py
```

중단: `pkill -f run_matrix.sh; pkill -f 'phase2_matrix.py'`

## 분석

`runs/phase2_matrix/make_analysis.py` 재실행(rundir만으로 CSV+차트). cross-seed ρ는
세션 스크래치 스크립트(진단 문서 §1.4 표 생성 로직)로 phi.parquet에서 재계산.
