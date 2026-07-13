# Flirds Paper Draft — 상태 문서

생성: 2026-07-12 (Claude 세션). 계획 문서: `research-wiki/survey/paper-readiness-plan-2026-07-12.md`.

## 빌드

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

venue 스타일 미적용(결정 D-2 대기) — 현재 self-contained article class로 어디서든 컴파일됨.
PDF에서 빨강 `[TODO:]` / 주황 `[PENDING DECISION:]` / 보라 `[VERIFY vs rundir:]`가 전부 보임.

## 파일 구조

```
main.tex                  — 프리앰블, \todo 매크로, 섹션 include
sections/abstract.tex     — 측정학-우선 초록
sections/introduction.tex — 서사 + 기여 4개 (+E1 후 5번째 자리)
sections/related.tex      — 5문단 related work
sections/setup.tex        — FedAvg 세팅, (a)/(b) 게임 정의, fp32 근거, metric
sections/method.tex       — 추정기 식 + 명제 5개 (P1~P8 요약판)
sections/experiments.tex  — hardness ladder 무대, 프로토콜, 11+4 방법
sections/results.tex      — fidelity ladder / dual-oracle / boundary map / poison / cost / 2차축
sections/discussion.tex   — 측정학의 가치, trajectory-특이성, 한계 6개
sections/conclusion.tex
sections/appendix.tex     — 6개 부록 스텁 (소스 문서 매핑됨)
references.bib            — ~30 entries; [CHECK] 표시 = 원문 검증 필요
```

## 표기 규약 (본 드래프트에서 채택한 잠정 결정)

- **부호**: φ>0 = 유익 (내부 코드 규약의 반전; presentation에서 통일 적용). D-3 확정 필요.
- **novelty 문구**: "to our knowledge / observed gap" 수위 유지 (D-15).
- **Ripple**: fidelity 표 제외 + runtime 별도·회계 각주 (D-6 권고안 선반영).
- **비용 주장**: cohort-조건화 서술 (std20 역전 명시).
- **poison**: 두 프레이밍 병기 + PENDING 표시 (D-1이 정해야 최종 문장 확정).

## Placeholder 인벤토리 (실험/결정 → 논문 삽입 지점)

| 마커 | 위치 | 채우는 것 | 의존 |
|---|---|---|---|
| judge cell (E1) | abstract, intro 기여5, results §6.2, appendix B | own-game 표 + (a) 심판 | GPU ~20h |
| Fed-LOO (E4) | Table 1 두 칸, experiments | Fed-LOO fidelity 행 | GPU ~6-9h |
| Taylor 1B (E2) | method Prop 3 | 잔차 스케일링·2차 우위 | GPU ~1h |
| (b) N=10 (E5) | results §6.1 rung1 문단 | 고검정력 fidelity | GPU ~10h |
| 3B poison seeds (E8) | results §6.4 | "스케일 의존 회피" 3-seed | GPU ~4h |
| delta free-rider (E7) | results §6.6, discussion | exact-0 스트레스 | GPU ~3h |
| 회계 스모크 (E3) | results §6.5 | ×3.1 확정, Ripple 환산 | GPU ~0.5h |
| bootstrap CI | 모든 표 | 95% CI | GPU-0 재집계 |
| per-scale fidelity 표 | Table 1 + appendix C | 1B/3B/7B 분리 수치 | GPU-0 (make_fidelity) |
| CNN (a)-발산 해명 | results §6.2 | P5-analysis 서술 | GPU-0 분석 |
| std50k5 seeds 1-2 | results §6.1 | 3-seed화 또는 각주 | GPU 수 h |
| bib [CHECK] ~12건 | references.bib | 원문 서지 확정 | GPU-0 |

## 수치 출처 및 주의

- 모든 수치는 survey 문서(results-overview 06-25, analysis 06-26, diagnosis, cost/precision
  audits) 기준. **논문 제출 전 rundir/CSV 재검증 필수** (`[VERIFY]` 마커).
- **금지 수치** (리뷰 R1/R2/R9 반영): "3B (a) vs (b) +0.900" (3B에 (a) 없음 — 미사용),
  task6 "+1.000" (rundir 미커밋 — 미사용; track_d anchor5 (a)=0.933±0.047만 사용),
  CNN ComFedSV 구수치 {1.0,0.96,...} (미사용), fp32/bf16 "×3.1" (placeholder — 미사용).
- ShapleyFL β: 1B/3B는 β=0.3 반영본; 7B·robustness matrix는 β=0.5 프로비넌스 (D-9 각주/재실행).

## 검증 패스 (2026-07-12, 3-way 적대 검증 완료·반영)

금지 수치 스크리닝 **통과**. 발견 불일치 21건 전부 수정 반영. 주요 교정:
- [치명] anchor(IID Alpaca)와 silo5(5-domain non-IID) 무대 혼동 → 분리 서술 (리뷰 R3 재유입 차단)
- [치명] 비용 표 device 열: 방법별 분리 (FedSV ~4,970 / GTG ~16–18k / ShapleyFL ~24.9k /
  Banzhaf 불가) + 3-seed 정본 평균으로 교체 (예: (a) 30,817±244s)
- gpt2 스모크 수치 귀속 (5.83e-12 = 구현 대조, 7.61e-12 = 2^N Shapley 대조)
- 잔차 bound 레벨 (utility-level 1차 = M₂/2·‖Δ‖²)
- MMLU/ROUGE 스코프를 std20으로 한정 (7B anchor vanilla +0.31pp 예외 명기)
- xseed "null" 주장을 1B 한정 + 7B anchor +0.73 예외 명기

**미해소 정합 확인 필요 2건** (rundir에서 확정):
1. (b) oracle 비-IID noisy AUROC: overview 0.604±.041 vs analysis 0.660 → master_metrics.csv로 확정
2. (b) 1B anchor seed당 ~1h 실측의 정확 조건 (r16 canonical vs r64 probe 셀)
