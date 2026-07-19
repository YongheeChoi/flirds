---
type: survey
title: "Ripple baseline 제외 결정 — 근거 정리 + 리뷰어 Q&A"
created: 2026-07-19
updated: 2026-07-19
tags: [ripple, baseline, exclusion, decision, reviewer-defense]
---

# Ripple baseline 제외 결정 (2026-07-19)

> **결정**: Ripple Shapley(Zeng et al., AAAI 2026)를 논문의 baseline 스위트(fidelity·runtime
> 비교표)에서 **제외**한다 — Yonghee 결정, 2026-07-19. [[ripple-audit]] §7-1의 대기 결정
> (감사는 (ii) "runtime 표 유지+각주"를 기본안으로 제안했었음)이 **(i) 완전 제외**로 닫힘.
> 관련연구에서의 위치 서술은 유지(§4 처리 방침).
>
> 이 문서의 용도: **리뷰어가 "왜 Ripple이 없나"를 물었을 때의 답변 원고.** 기술 상세의 정본은
> [[ripple-audit]](구현 감사·속도 진단)과 `measurements-eigsh-cpu.md`(eigsh 실측)이고, 여기는
> 제외 사유를 결정문 형태로 모은 요약이다. 실측 데이터(CNN C1 rundir·acct 로그)는 전부 보존 —
> 리뷰어가 수치를 요구하면 제시 가능하다.

---

## 1. 제외 사유 — 서로 독립적인 다섯 겹

각 사유는 단독으로도 제외를 정당화하며, 순서는 방어 강도순이다. 핵심 골자(감사 §5):
**"방법의 측정 대상과 프로토콜 구조가 우리 비교 축과 양립하지 않음을 확인했고, 포트 완성은
그 판정을 바꾸지 못한다"** — "우리 포트가 미완이라 뺐다"가 아니다.

### ① 공개 코드가 없어 전면 자체 구현했고, 구현 결정 다수가 논문에 부재

- 논문에 **코드 공개 링크 없음**(감사 §2.1 확인). 우리 port(`ripple.py` CNN + `ripple_llm.py`
  LLM)가 유일한 구현이며 docstring부터 "no public code"를 명시.
- 논문이 침묵하는 구현 결정들: 고유분해 방법·수렴 판정·tol 일체 부재(전문 grep "tol"/"Lanczos"
  0건), drop 항의 z_val("a validation sample" — 1개인지 전체 val인지 불명), eigsh which=LA/LM,
  심지어 **자기 실험의 클라 수 N·참여율도 본문에 없음**(Sun et al. 2023 프로토콜로 위임).
  비교 baseline도 저자 자체구현으로 추정되고 그 코드도 없음.
- 귀결: 우리가 낸 수치는 "Ripple"이 아니라 "우리 해석의 Ripple"이다. 나쁜 수치가 방법 탓인지
  해석 탓인지 분리할 수 없어, 표에 넣으면 **원 방법을 부당하게 폄하할 위험**이 있다(감사 §5-③).

### ② 측정 대상(게임)이 다르다 — fidelity 표에 넣으면 범주 오류

- 우리 1차 축 = (a)/(b) exact oracle 대비 fidelity, 즉 **라운드별 marginal-contribution 게임**.
- Ripple φ = drop(즉시 기여) + **ripple(그 업데이트가 이후 라운드 val-loss에 미친 영향의 소급
  가산; temporal 전파)** — 정의상 다른 대상을 잰다. 같은 이유로 개입 실험에서도 이미 제외돼
  있었다("its full value is non-causal", `track_c2.py:18–19`).
- 또한 논문은 **sample-level**, 우리는 client-level 무대 — 우리 port는 per-batch·per-step으로
  축약해 클라에 귀속(선형성)한 **적응**이지 재현이 아니다.

### ③ 논문 스스로 우리 비교 축을 거부한다 — 대조할 참조점이 없음

- 원문 인용(p.6, 감사 확인): "we argue that such metrics fail to reflect their practical
  utility in FL … numerical alignment of Shapley scores may not correspond to meaningful
  training dynamics" — **Spearman/Kendall류 oracle-fidelity 수치가 논문 전체에 없다.**
- 논문의 "high attribution fidelity"는 poisoning 하 가중집계 downstream 정확도(Eq.20) 근거의
  주장. 즉 "논문 보고만큼 나오는가"를 잴 공통 자가 애초에 존재하지 않는다.

### ④ 프로토콜이 우리 실험 인프라와 양립 불가 — from-logs 재구성 불가 (확정)

- Ripple은 **온라인·클라 참여형 프로토콜**(클라가 per-step 양을 계산·업로드; 업링크 모델
  크기의 k배). 우리 공유 로그 `(w_r, δ)`로는 drop 항(per-step gradient·중간 파라미터)과
  로컬 Hessian sketch를 복원할 수 없음이 확정됐다(감사 §4.2 항목별 판정).
- 따라서 25셀×3-seed 공유-로그 그리드에 편입하려면 **셀마다 자체 궤적 재실행**이 필수 —
  구조적으로 불가능하거나(streaming projection 미구현으로 N=100 셀 실행 불가) 예산상
  비현실적이다. from-logs로 만들면 "Ripple의 재현"이 아니라 "Ripple에서 영감을 받은 서버-측
  변형"이 된다(감사 §4.2 판정문).

### ⑤ 방법 고유 비용이 우리 무대에서 지배적 — 회계를 어느 쪽으로 통일해도 최고비용

- LLM silo5 3-seed(B200, 통일 회계 실측): **Ripple 3,536s(2,366–4,363) = Flirds valuation의
  ~33× · coalition-sweep의 ~6.6×**. CNN C1: MNIST ~1.1–2.3k s/CIFAR ~7.5–11.1k s = FL 학습
  자체의 10–130×.
- **회계 공정성 양방향 점검(07-19 대화에서 확정)** — "Ripple만 자기 궤적이 타이머에 포함"
  비대칭은 실재하나 어느 교정도 결론을 못 바꾼다:
  - Ripple에서 궤적을 빼면(valuation-only 통일): CNN 계측상 궤적 몫 **0.4%** — 거의 그대로.
    관대하게 공유 궤적치(410s)를 통째로 빼도 ~29× Flirds.
  - 나머지 전 방법에 공유 궤적(~410s 실측)을 더하면(논문식 end-to-end): 33×→**~6.8×**,
    6.6×→~3.7× — 여전히 전 방법 중 최고비용.
  - 논문 자체 셋업에선 역방향: valuation-only 환산 시 Ripple 우위가 94.8×/120.9×로 커짐.
    즉 회계 정의 선택은 어느 쪽 결론도 뒤집지 못한다.
- **"우리 포트가 느린 탓" 반론은 실측으로 차단**: eigsh는 정상·빠르게 수렴(12/12 호출, 호출당
  ~115 matvec, |λ|max≈1.019 — CPU-spin/stall 가설 반박). 비용의 실체 = 클라×라운드만큼의
  고유분해 volume + 매 로컬스텝 full-val gradient(full C1 셀 기준 47,000회) = **방법 고유**.
  port 개선 여지(tol=0 → matvec ~2배; z_val 전체-val 해석 축소)를 다 먹어도 volume이 남는다.
- 참고: 62×/49× 주장과 우리 실측은 **모순이 아니라 양립** — 그들 수치는 소형 모델·학습 포함
  누적·느린 coalition-형 자체구현 대비의 조건부 실측이고, 절대 주장은 plain training 대비
  2.05×뿐(그것도 우리 무대에선 10–130×로 불성립).

---

## 2. 실측 성능 — 돌려봤고, 잘 안 나왔다 (보조 근거)

제외의 주 근거는 위 ①–⑤(구조적 사유)이고 성능 저조는 **보조 근거**로만 쓴다 — 축소-config
자체 구현이라 "방법 탓 vs 포트 탓"을 분리할 수 없기 때문(사유 ①과 같은 논리). 그러나 실측
기록은 남긴다:

- **CNN C1 fidelity (10 시나리오 × 3-seed, rundir 영속)**: vs (b) exact 2¹⁰ Spearman
  **0.373±.444** (same-game 방법: Flirds 0.919·Banzhaf 0.989·loss-heur 0.860); vs (a) retrain
  0.213±.462(전 방법 최하); Kendall 0.311. 신호가 강한 칸만 생존(mnist label-flip 0.97·
  quantity-skew 0.96), 나머지는 ≈0–0.4. sample-level 스케일이라 값-수준 거리(euclid ~113)는
  비교 무의미.
- **E3 비용-성능 동시 실측(iid seed0)**: 셀 **최고 비용**(위상 95%)을 쓰고도 fidelity
  0.345/0.406 — same-game 최상위권에 못 미침 = dominated.
- **LLM silo5 탐지(acct 3-seed)**: noisy AUROC **0.500/0.750/0.250(seed별) = seed-불안정**
  (다른 valuation 방법 0.750 일관); free-rider는 1.000. 구서버(06-06)에서도 noisy 0.50±0.20으로
  valuation 방법 중 최저.
- 종합: 우리 무대에서 Ripple은 **최고 비용으로 최하위권 fidelity·불안정 탐지** — "비용 대비
  성능" 어느 축에서도 baseline 스위트에 남길 실익이 없다.

---

## 3. 리뷰어 예상 Q&A

**Q1. 가장 가까운 single-run FL attribution 방법(AAAI 2026)인데 왜 baseline에 없나?**
A. 세 가지 독립 사유다. (1) Ripple은 우리 1차 평가축(exact-oracle 대비 fidelity)과 **다른
게임**을 잰다 — drop+temporal-ripple 누적값은 라운드별 marginal-contribution이 아니고, 논문
스스로 Shapley-score 정렬 지표를 거부하며 관련 수치를 보고하지 않는다. (2) 온라인·클라
참여형 프로토콜이라 우리 공유-로그 실험 인프라로 재구성이 불가능함을 항목별로 확인했다(궤적
재실행 필수). (3) 공개 코드가 없어 전면 자체 구현했는데 핵심 구현 결정 다수(고유분해 방법·
tol·z_val 범위·클라 수까지)가 논문에 부재해, 우리 수치를 "Ripple의 성능"으로 표에 싣는 것이
오히려 원 방법에 불공정하다. 관련연구에서 위치와 차별점(sample-level temporal 전파 vs 우리
client-level per-round Shapley)은 서술한다.

**Q2. 구현이 느려서/성능이 안 나와서 뺀 것 아닌가?**
A. 아니다 — 그 가설을 우리가 직접 실측으로 배제했다. 계측 결과 eigsh는 전 호출 정상 수렴
(호출당 ~115 matvec)했고, 큰 비용은 포트 결함이 아니라 방법 고유의 클라×라운드 고유분해
volume + per-step val-gradient다. 회계 비대칭(Ripple만 자체 궤적 포함)도 양방향으로 교정해
봤다: Ripple에서 궤적을 빼면 0.4% 감소뿐, 전 방법에 학습시간을 더해도 여전히 최고비용
(~3.7–6.8×). 성능 실측(CNN Spearman 0.373 vs same-game 0.86–0.99)은 제외의 보조 근거로만
기록해 뒀다 — 자체 구현이라 방법 자체의 한계로 단정하지 않는다.

**Q3. 논문은 62× speedup을 주장한다. 당신들 비용 결론과 모순 아닌가?**
A. 모순이 아니다. 그들의 62×/49×는 "FL 학습 포함 누적 시간, 소형 모델(MLP/소형 CNN),
느린 coalition-평가형 자체구현 baseline 대비"의 조건부 실측이고(원수치: Ripple 985s vs FedSV
48,283s vs plain 481s @R100), 절대 주장은 plain training의 2.05×다. 우리 무대(LLM/from-logs
소비형 비교)에서는 방법 고유 비용이 그 구도를 뒤집으며, 이는 그들 주장과 양립한다 — 다만
"우리 세팅으로 전이되지 않는다"가 우리의 실측 결론이다.

**Q4. 그래도 fidelity를 재서 보여줄 수 있지 않았나?**
A. 쟀다 — CNN 트랙(exact 2¹⁰ 듀얼 oracle)에서 10 시나리오×3-seed로 실측했고 vs (b) 0.373,
vs (a) 0.213이다. 다만 두 이유로 본 표엔 싣지 않는다: 논문이 그 지표를 명시적으로 거부해
비교 참조점이 없고, 우리 client-level·축소-config 적응의 수치라 방법 본연의 성능으로 읽힐
위험이 있다. 요청 시 부록/응답에 수치를 제공할 수 있다.

**Q5. 공정한 비교를 하려면 무엇이 필요했나?**
A. 둘 중 하나다. (1) 저자 코드+완전한 실험 명세로 그들 무대를 재현 — 코드·명세 부재로 불가.
(2) from-logs 호환 파생 방법 설계(drop→라운드 단위 1차 내적, Hessian sketch→서버-측 HVP
재활용) — 정의 가능하나 이는 명시적으로 "Ripple이 아닌 파생"이며(감사 §6-3), temporal 전파
항의 from-logs 계열은 문헌 공백이라 별도 기여로 다뤄야 한다.

---

## 4. 처리 방침 (이 결정의 실무 반영)

- **논문**: fidelity·runtime baseline 표에서 Ripple 제거. 관련연구에 위치 서술 유지 —
  "in-run per-sample temporal-propagation 방법; oracle-fidelity를 스스로 측정하지 않으며(인용
  ③), 우리 1차 축과 게임이 다름(②)". 필요 시 제외 각주 1줄: "no public code; online
  client-side protocol incompatible with our shared-log grid; see rebuttal materials".
- **데이터**: 기존 실측(C1 rundir 30셀·E3·acct 로그·eigsh 계측)은 전부 보존 — 재실행 불필요,
  리뷰어 요구 시 제시.
- **문서**: [[ripple-audit]] §7-1 대기 결정 닫힘(이 문서가 결정 기록);
  [[baseline-original-paper-verification]] §3.7은 원 논문 대조 기록으로 유지(제외 결정 링크 병기).

---

## 5. (영문 초안) 리뷰어 응답용 문단

> We ported Ripple (Zeng et al., AAAI 2026) ourselves — no public code exists — and evaluated
> it on our CNN track with exact 2^10 dual oracles. We ultimately exclude it from the baseline
> tables for three independent reasons. First, it measures a different quantity: its value
> accumulates a temporal "ripple" of each update into later rounds, which is not the per-round
> marginal-contribution game our oracles define; the paper itself explicitly rejects
> Shapley-score alignment metrics and reports none. Second, it is an online, client-side
> protocol (per-step quantities computed on local data) that provably cannot be reconstructed
> from the shared round-level logs our 25-cell grid replays, requiring a full trajectory re-run
> per cell. Third, key implementation choices (eigensolver, tolerance, validation-sample scope,
> even client counts) are unspecified in the paper, so our numbers would reflect our
> interpretation rather than the method. For completeness: in our measurements the port —
> whose eigensolver we verified converges normally, i.e., the cost is method-inherent — was
> the most expensive method under every accounting convention we tried (3.7–33x our method)
> while ranking near the bottom in oracle fidelity (Spearman 0.37 vs 0.86–0.99 for same-game
> baselines); we report these only as supporting evidence, since a reduced-configuration port
> should not be read as the method's ceiling.

---

## 6. 근거 문서 지도

| 주장 | 정본 위치 |
|---|---|
| 코드 부재·논문 침묵 지점 목록 | [[ripple-audit]] §1(대조표 #6–#18)·§2.1 |
| 62×/49× 측정 조건·원수치·2.05× | [[ripple-audit]] §2.1 |
| 4-요인 격차 분해(회계·d-스케일·기각된 eigsh 가설) | [[ripple-audit]] §2.3 |
| eigsh 정상 수렴 실측(CPU-spin 반박) | `measurements-eigsh-cpu.md` · [[ripple-audit]] §3 |
| from-logs 불가 항목별 판정 | [[ripple-audit]] §4.2 |
| LLM 3-seed 통일 회계(3,536s·6.6×·33×) | [[ripple-audit]] §4.3 |
| 회계 양방향 교정(0.4% / ~6.8×) | 이 문서 §1-⑤ · [[baseline-original-paper-verification]] §3.7 |
| CNN fidelity 실측(0.373/0.213) | `runs/track_c/c1/*/metrics.json` · overview §3.1.2 |
| 제외 3겹 논거의 원형 | [[ripple-audit]] §5 |
