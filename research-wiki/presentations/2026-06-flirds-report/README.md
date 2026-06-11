# Flirds 발표자료 (2026-06)

high-level 연구 소개용 슬라이드 2종. 체크포인트(`wiki/checkpoint-2026-06-10/`)를
바탕으로 하되, 결과 수치는 **최신 real grid**(`runs/phase2_matrix/RESULTS.md`,
FedIF+persistence 재실행분) 기준.

| 파일 | 용도 | 구성 |
|---|---|---|
| `flirds-seminar.html/.pdf` | 연구실 세미나 | 표지 + 본편 13장 + 부록 4장 (18장) |
| `flirds-advisor.html/.pdf` | 교수님 미팅 | 표지 + **recap 1장**(문제·방법 압축) + 본편 9장 + 부록 4장 (15장) |

두 버전의 차이는 앞부분뿐: advisor 버전은 "왜 기여도/Shapley 장벽/핵심 아이디어/2차항"
4장을 recap 한 장으로 대체 (교수님은 빌드업을 여러 번 보셨음). 이후 슬라이드는 동일.

## 보는 법 / 발표

- HTML을 브라우저로 열기 → ←/→(또는 Space), Home/End로 이동. `#7`처럼 URL 해시로 점프.
- PDF는 인쇄·배포용 (슬라이드당 1쪽, 1280×720).

## 재빌드 (수치 갱신 시)

```bash
cd research-wiki/presentations/2026-06-flirds-report
/home/korea_bupj/miniconda3/envs/flirds/bin/python build.py   # HTML+PDF 4파일 재생성
```

슬라이드 내용은 전부 `build.py` 안의 `SLIDES` dict에 한 번만 정의 → 고치면 두 버전에
모두 반영. 차트(runtime/poison AUROC)는 같은 파일의 데이터 리스트에서 생성.

## 실험 완료 시 업데이트할 곳 (슬라이드에 "업데이트 예정" 칩 표시)

- **슬라이드 "결과 ④ cross-device"** (`SLIDES["device"]`): tier2 α-sweep 완료 수치,
  anchor(α=0.5, oracle 동반) 결과, tier 상태표.
- **슬라이드 "현재 상태와 다음 단계"** (`SLIDES["status"]`): 진행→완료 이동, 3B/7B 결과.
- **슬라이드 "결과 ③ backdoor"** (`SLIDES["backdoor"]`) + `poison_chart()`:
  poison Flirds-2차 분산의 원인 규명이 끝나면 "불안정(0.42↔0.92)" 서술을 결론으로 교체.

## 수치 출처 (검증 노트)

- tier1 표/런타임/poison: `runs/phase2_matrix/RESULTS.md` + `runs/phase2_matrix/tier2/silo5_*.log`
  (FedIF+영속화 재실행분, 3-seed). **주의**: poison Flirds-2차는 동일 config 재실행 간
  0.417±0.425(`tier1_orig/`) ↔ 0.917±0.118 — run간 비결정성. 발표에선 "부분 회복하나
  불안정"으로만 주장 (둘 다 부록 D에 병기).
- dual-oracle 삼중 일치/비용/CNN/N=100 anchor: `wiki/checkpoint-2026-06-10/` 00·01·02.
- tier2 잠정치(FR α=0.1 AUROC 1.0): RESULTS.md running 셀 — "잠정" 표기 유지.
