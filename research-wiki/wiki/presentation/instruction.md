# 작업: 교수님 미팅용 Flirds 발표자료 (담백한 팩트 전달형)

## 배경

- 이전 세션에서 만든 발표자료 2종(세미나용+교수님용)은 **회사 프레젠테이션/세일즈 톤이라 전부 폐기**했다 (git 542da17 추가 → 2aeed0a 삭제; 히스토리에 남아 있지만 톤이 잘못된 버전이니 참고하지 말 것).
- 이번에는 **교수님 미팅용 1종만** 만든다. 연구실 세미나용은 나중에 별도 작업.
- 청중: 지도교수님. 연구 첫 빌드업부터 전 과정을 봐오셔서 배경은 압축 recap이면 충분하다.
- 언어: 한국어 (기술 용어는 영어 그대로).

## 톤 — 가장 중요한 요구사항

담백하게 팩트만 전달한다. 구체적으로:

- 과장·설득 어휘 금지: "완벽", "지배", "단독", "혁신", "강력" 류 금지. "최초"는 꼭 필요할 때만 범위를 정확히 한정해서. 카피라이팅식 헤드라인 금지.
- 평서문 서술. **모든 정량 주장에 설정 병기**(모델/N/regime/seed/config)와 출처.
- **3-state 라벨을 자료 전체 규약으로**: ⓐ 구현+smoke(값 coarse) / ⓑ 실측(설정 명시) / ⓒ 설계만·미실행. (checkpoint 문서들이 쓰는 규율 그대로.)
- 약점·실패·불확실성도 같은 톤으로 그대로 보고. 잠정치는 "잠정" 표기.
- 디자인: 흰 배경, 무채색 위주(강조색 최대 1개), 표·수식·짧은 문장 중심. 그라데이션/큰 숫자 카드/배지/아이콘 장식/이모지 금지. 학술 발표 핸드아웃에 가깝게.

## 구성 (Yonghee 지정 — 각 번호가 꼭 1페이지일 필요 없음, 내용에 맞게 분할)

1. **Recap: 현재 Flirds의 알고리즘 정리**
   - 무슨 문제를 푸는지 (FL 클라이언트 기여도 평가; 서버는 update만 받음)
   - 이 문제를 푸는 데 있어 선행 연구들이 놓쳐온 빈 부분 (retrain Shapley는 LLM-FL서 비용 불가 / FL-SV 계열은 coalition 비용+CNN 한정 / IF는 iHVP 불안정 / IRDS는 centralized·per-step·sample-level / Ripple·FedIF·FedTSV 각각의 한계 — client-level × in-run × closed-form 1+2차 × post-hoc × LLM 교집합이 빔)
   - method가 어떤 알고리즘으로 동작하는지 (frozen FedAvg 궤적, val-loss 1+2차 Taylor, round당 HVP 1회, 핵심 수식 1개, free-rider φ=0의 구조적 이유, 왜 2차항이 FL에서 의미를 갖는지)
2. **실험 설계**: 어떤 실험들을 설계해서 돌렸고, 돌릴 예정이며, 왜 그런 실험들이 필요하고 뭘 확인하려고 하는 건지
   - 각 실험 항목에 "현재까지의 결과 또는 상태" 1줄씩 같이 표기 (ⓑ 수치 / ⓒ 예정)
   - 포함할 실험 축: dual-oracle 검증(왜 같은 게임 val-loss여야 하는지), baseline 9종 같은 frozen 궤적 비교, 2 regime × 4 threat matrix + threat-matched detector 4종(라벨은 채점 key지 입력이 아님 — 순환 아님), cross-device N=100(per-round exact 분해 + α-sweep + anchor), selection run, scale ladder 1B→3B→7B, N=10 retrain oracle 연기 사유
3. **여전히 남아있는, 더 풀 만한 여지가 있는 문제들·한계들**
4. **앞으로의 계획**
5. (선택 — 원치 않으면 이 항목 삭제) 끝에 "논의·결정이 필요한 항목" 반 장: threat-matrix headline framing, N=10 retrain oracle 투자 여부(2–5일/GPU 또는 multi-GPU 샤딩), real grid lr 선택 등 Yonghee-결정 대기 항목

## 반드시 직접 읽을 소스 (요약을 믿지 말고 원문 대조)

1. `research-wiki/wiki/checkpoint-2026-06-10/` — 00(전체 그림)·01(알고리즘+노벨티)·02(실험 세팅)·05(미해결+다음)·06(경쟁 3종 FedIF/FedTSV/Ripple)·07(novelty·한계 + §7.0 real-grid 시작). 03(baseline PDF 대조)·04(plan 분기)는 필요시.
2. `runs/phase2_matrix/RESULTS.md` — 최신 real grid 결과. `make_report.py`가 로그에서 자동 생성하므로 **세션 시작 시점에 다시 생성/확인할 것** (tier2가 그새 더 완료됐을 수 있음; "running" 셀 수치는 잠정).
3. `~/.claude/projects/-NHNHOME-26msit001-A-edge-ai-lab-yonghee-flirds/memory/phase2-step5-verification.md` — 문서 과장 교정값.

## 정확성 주의 (이전 세션에서 확정한 팩트 — 데크에 그대로 반영)

- tier1(silo5 4-threat 3-seed)의 최신본은 **FedIF baseline 추가 + 결과 영속화 때문에 재실행된 버전** (`runs/phase2_matrix/tier2/silo5_*.log`; 원본은 `tier1_orig/`). FedIF는 코드에 정식 편입됨(1903a58) — checkpoint 07의 "FedIF 비교 suite 부재" 지적은 **해소된 상태**이니 한계로 쓰지 말 것.
- **poison Flirds-2차 AUROC는 동일 config·seed 재실행 간 0.417±0.425 ↔ 0.917±0.118로 갈림** (LLM run간 비결정성; train_loss 소수점 셋째 자리부터 궤적이 다름). 1차는 양쪽 다 0.000으로 일관. → "2차가 잡는다" 단정 금지. "1차 완전 회피는 재현 일관, 2차는 부분 회복하나 run/seed 분산이 큼 — 원인 규명 중"까지만 쓰고 두 run 값을 병기.
- noisy/free-rider에서는 모든 valuation 방법이 Spearman +1.000 동률 (N=5 near-additive) — "Flirds가 더 정확"이 아니라 **"같은 랭킹을 더 싸게"**(런타임 ⓑ: Flirds-1st 35s / Flirds 107s / loss-heur 164s / GTG·FedSV·ShapleyFL·Banzhaf·(b)oracle ~530s / Ripple ~4515s 별도 세션)가 정확한 서술.
- dual-oracle: retrain val-loss = in-run oracle = estimator **Spearman +1.000** (1B N=5 fp32, lr 2종; 3B는 estimator +1.000 유지, retrain vs in-run +0.900 1-seed). retrain-ROUGE는 다른 게임이라 발산(+0.4@1B/−0.9@3B) — val-loss로 검증해야 하는 근거로만 사용.
- N=100: per-round exact 분해는 2^N과 수학적 동일(Δφ≈3e-16 검증), α=0.5 anchor에서 Flirds vs per-round oracle +1.000 (1-seed). free-rider φ=0은 N=100에서도 exact.
- 비용 팩트: retrain oracle 1B N=5 fp32 = 126분, N=10 = 2–5일/1-GPU; (b) oracle은 2^N·R·val·seq FLOP-bound; estimator는 round당 HVP 1회.

## 산출물

- 위치: `research-wiki/presentations/2026-06-advisor-meeting/`
- self-contained HTML 슬라이드 1개(1280×720, ←/→ 키보드 내비, 외부 의존성 없이 시스템 폰트) + 같은 내용의 PDF.
- 재실행 가능한 빌드 스크립트로 생성 (python `/home/korea_bupj/miniconda3/envs/flirds/bin/python`; playwright print-to-PDF 사용 가능 확인됨; 한글 시스템 폰트 OK). tier2/3B 완료 시 갱신할 지점을 스크립트 주석으로 표시.
- 완성 후 playwright 스크린샷으로 전 슬라이드 레이아웃 검증 (겹침/overflow).
- 커밋은 Yonghee가 시킬 때만. push는 불가(Yonghee가 직접).

## 진행 방식

소스를 읽은 뒤 **섹션별 슬라이드 분할안(슬라이드 제목 + 들어갈 핵심 팩트)을 먼저 제시하고 Yonghee 컨펌 후** 제작에 들어갈 것. 디자인 톤이 위 요구와 맞는지는 첫 1–2장 스크린샷으로 중간 확인 받는 것을 권장.