---
type: checkpoint-index
title: "Flirds 체크포인트 2026-06-10 — 인덱스"
created: 2026-06-10
updated: 2026-06-10
---

# Flirds 체크포인트 (2026-06-10)

> **목적**: Phase 0–2 구현(코드 + step5 matrix orchestrator)이 끝난 시점에서, **연구자 본인이 전체 그림을 다시 잡기 위한** 재오리엔테이션. 외부 발표용 아님.
> **방법**: 문서 요약이 아니라 **실제 코드(`codes/`) · raw 로그(`research-wiki/raw/`) · 논문 PDF**를 7개 병렬 reader로 직접 읽고 대조 + load-bearing 항목 직접 spot-check. 추측 없음, 모든 정량주장에 파일경로 근거.
> **규율**: ⓐ 코드+smoke green(값 coarse) / ⓑ 실제 실험결과(seed·N·model 명시) / ⓒ 설계 락·미실행 — 3-state 구분. 문서 과장은 검증값 채택 + 충돌 기록.

---

## 문서별 1줄 요약

| 문서 | 요약 |
|---|---|
| [00-overview](00-overview.md) | **전체 그림 한 장** — end-to-end 파이프라인(mermaid), 구성요소 지도(estimator/oracle/9 valuation/4 detector/2×4 매트릭스), 용어집, 3-state 상태표. **제일 먼저 읽을 것.** |
| [01-research-value](01-research-value.md) | **알고리즘 + 노벨티** — `flirds_estimator.py` 코드 기준 1차+2차 Taylor·round당 HVP 1회·forward HVP·true Hessian 정확본 + 선행한계→차별점→노벨티(각 근거). |
| [02-experimental-setup](02-experimental-setup.md) | **최종 실험 세팅** — 모델/regime/threat 4종/α-sweep/seed/LoRA·fp32/(b) 비용공식 + 위협 정의의 논문근거(answer_swap·free_rider·backdoor) + #7 selection 실측(metrics.json 직접 대조). |
| [03-baselines-and-prior-work](03-baselines-and-prior-work.md) | **baseline + 선행연구 (PDF 1:1 대조)** *— 가장 무거움.* 경쟁 baseline 7종 + detector 4종 + 선행 6편, 원문 vs 우리구현 차이+왜. Xu·Bagdasaryan은 PDF 부재→web-extract. |
| [04-plan-vs-implementation-divergences](04-plan-vs-implementation-divergences.md) | **plan 대비 분기 11건** — GGN→Hessian, momentum→plain SGD, ROUGE→val-loss, detector 재설계, per_client 40→300 등 + 각 raw 근거. |
| [05-open-issues-and-next](05-open-issues-and-next.md) | **미해결 + 다음** — poison-vs-Flirds framing(verification ruling: matrix=EVADED 맞음, headline은 미결) + claimed-vs-verified 교정표 + caveat + real grid 실행계획(cost-tiered). |
| [06-closest-competitors-fedif-fedtsv-ripple](06-closest-competitors-fedif-fedtsv-ripple.md) | **직접 경쟁자 3종 포지셔닝** — FedIF/FedTSV/Ripple을 'LLM+FL+기여도' 교집합 축에 놓은 근접도+장단점. ★ **Ripple "2차항 없음" 주장을 코드(`ripple.py:191`)로 정밀화**(within-round vs cross-round 곡률; ⓒ Yonghee 결정). 03의 경쟁 3종 심화판. |
| [07-novelty-limitations-analysis](07-novelty-limitations-analysis.md) | **novelty·한계 분석 + 개선 제안** (06-10 오후) — novelty 방어력 등급(강: dual-oracle·per-round 분해·비용구조 / 사활처: 2차항), 한계 4범주 15건, 개선 17건 우선순위. ★ **§7.0이 00·05의 'real grid 미실행'을 SUPERSEDE**: tier1 완료+tier2 진행 중(.log-only ⚠), poison서 동률 첫 붕괴 — Flirds-1st 0.000 완전회피, **2nd seed별 {0,0.25,1.0}**(seed2=2차가 1차 이긴 최초 LLM 데이터포인트). |

---

## 추천 읽기 순서

**00 → 01 → 02 → 03 → 04 → 05** (→ **06** 경쟁자 포지셔닝 → **07** novelty·한계 판정)

빠르게 현황만: **00**(상태표 §0.5) → **05**(다음 단계 §5.5) → **07**(§7.0 real-grid UPDATE — 00·05의 '미실행'을 교체).
방법론 핵심만: **01**(알고리즘 §1.1) → **03**(IRDS 대조 §C.1).
"내 주장이 진짜인가" 점검: **05**(§5.2 교정표) → **02**(§2.6 #7 실측).
경쟁자 포지셔닝만: **06**(축별 근접도 §6.1 → Ripple 2차 정밀화 §6.5).

---

## 핵심 한 줄 (전 문서 종합)

- **방법은 검증됨(ⓑ)**: (a)-valloss = (b) in-run = estimator **Spearman +1.000** (1B N=5 fp32). free-rider φ 정확0. N=5 near-additive서 Flirds가 프론티어 지배(5–15× 싸게 같은 ranking).
- **코드는 단단함(ⓐ)**: estimator/oracle/backend/FL/데이터/baseline 11종/detector 4종/matrix orchestrator 전부 구현 + smoke green.
- ~~**real grid는 미실행(ⓒ)**~~ → **STALE(06-10 오후, [07](07-novelty-limitations-analysis.md) §7.0이 교체)**: tier1(silo5 4-threat 3-seed) 완료 + tier2(α-sweep) 진행 중. poison서 동률 첫 붕괴(Flirds-1st 0.000 / 2nd seed별 {0,0.25,1.0}). ⚠ .log-only — run-dir 영속화 필요.
- **교정 적용**: poison detector 0.75(≠1.0), FedSV tiny +0.900(real +1.000), STD-DAGMM ~360s, D2b "REFUTED"→matrix "EVADED(Flirds AUROC 0.0)"가 맞음.
- **즉시 다음**: cost-tiered real grid (poison은 `LR=2e-3` D2b config); §3.9 headline framing은 Yonghee 결정.

---

*근거 추적: 코드 인용은 `path:line`, 실측은 `codes/runs/.../metrics.json` 또는 `research-wiki/raw/conversations/flirds/*.md`, 논문은 `research-wiki/raw/papers/flirds/*.pdf`. 검증/교정은 `~/.claude/.../memory/phase2-step5-verification.md`.*
