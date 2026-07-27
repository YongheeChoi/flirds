# Flirds 용어 통일 — 결정 안건

- 작성: 2026-07-26 (Claude). 대상: `paper-ko.md`(기준본) + 두 변경 문서(`paper-ko-gpt.md`, `paper-ko-claude.md`) + `main.tex`.
- **상태: 전 항목 확정 + 기계 적용 가능분 일괄 적용 완료(07-26 — `paper-ko.md`·통합본
  `paper-ko-claude.md`; main.tex는 내용 변경 예정이라 추후).** 적용 백업 =
  scratchpad `paper-ko.md.bak-pre-terminology`. A3·A4·A5 = 신조어
  회피(서술형 우회), C9 = **정산 관련 내용 전면 제거**, D2(세팅)·D4(score source)·
  D5(recovery)·E3(client) = 한글판도 영문 표기, 나머지 승인.
- **용어 대원칙(Yonghee 07-26 확정; 적용 대상 = paper-ko 류(legacy 제외)와 main.tex):**
  ① 새 용어의 정의는 **그 용어가 새로운 개념을 내포하거나 연구에서 결정적 역할을 할 때만**
  허용한다. ② 일반적으로 쓰는 단어이거나 커뮤니티 컨센서스가 있는 경우가 아니라면, **한
  단어에 많은 의미를 압축하기보다 풀어서 설명**하는 쪽이 읽기 좋다. — A3·A4·A5 폐기와 C9
  전면 제거가 이 원칙의 적용례. 남는 용어성 표현은 정의 조건을 통과한 것만:
  selection-retrain(D1, 표 라벨 — 정의 1회 후 사용), 게임 불일치·공통 외부
  기준값(평서술구). (고정-가중은 07-26엔 통과 예였으나 아래 07-27 갱신으로 회피.)
  이후 새 문안 제안 시 이 테스트를 먼저 통과할 것.
- **갱신(2026-07-27, Yonghee 확정):** ① **"고정-가중" 회피** — C6·A4의 07-26 결정을
  대체한다. "고정-가중"·용어화된 "고정 가중(치)" 합성어 미사용. 게임 이름 = **"연합 라운드
  게임"**(서론 기여 2의 문구; §4·§4.1 제목 포함), 가중치 고정은 서술형 동사구로만 —
  "실제 집계 가중치를 고정한", "가중치는 $S$ 안에서 다시 정규화하지 않고 런 자신이 사용한
  값 $p_k^r$ 그대로 고정한다". paper-ko.md 전면 적용 완료(§4·§5.1·§6.2·§7·부록 A —
  "고정-가중/고정 가중" 0건). ② **영문도 같은 우회**: "fixed-weight" 합성 수식어 미사용 —
  이름은 "(federated) round game"·"the game over the observed round", 고정은 "with the
  server's actual aggregation weights held fixed"·"without renormalizing within a
  coalition" 류 서술구. main.tex 이관 시 적용, 그림 프롬프트는 v2에 반영 완료. ③ §4.1
  재배치·분량 절감으로 **식 번호 재편**: 라운드 게임 = 식 (5), 폐형식 추정기 = 식 (6),
  retraining utility $U^{\mathrm{re}}$는 무번호 인라인 — A3 라벨은 "식 (5)를 겨냥하는
  계열"로 읽는다(기준본 일괄 치환 완료).
- **갱신(2026-07-27 ②, Yonghee 확정 — A1 개정: "exact" 이름 성분 폐기):** 기준값의
  이름에서 "exact"를 제거한다 — "exact in-run Shapley"/"exact retraining Shapley" 풀네임
  미사용. 양의 지칭 = 서술형("(전수 열거로 계산한) 라운드별 Shapley 값"), 역할명(in-run
  reference / retraining reference·"기준값")은 유지. retraining 계열 지칭은 IRDS 원문 용어
  **"재학습 기반(retraining-based)"** 준수. 상세 규칙 = A1 하단 갱신 블록. 적용: 초록·서론
  (Yonghee 정리+07-27 보완) + §5–7·부록 기계 치환분 완료; **§4(별도 세션)·main.tex(이관 시)
  미적용**.
- **갱신(2026-07-27 ③, §4 착지 — 이름 확정 + 역할명 분리):** ②의 §4 미적용분을 처리하며
  양의 지칭을 서술형 대신 **고정 명칭**으로 확정했다. **"in-run Shapley"**(식 (5) 라운드
  게임의 Shapley 값 — §4.1에서 "라운드당 $2^{|P_r|}$개 부분집합 전수 열거로 계산해 합한
  값"으로 정의)와 **"retraining-based Shapley"**(전수 재학습; IRDS 원문 용어 "Retraining-based
  Data Shapley"를 따름 — 원문 초록 32행·§B.2 제목·Fig 5 캡션에서 직접 확인). 역할명
  **reference는 §5 이후에만 사용**하고 그 정의도 §5.1 첫 문단("**기준값 표기**")으로 이관 —
  §4는 reference/기준값을 쓰지 않는다(§4.1 제목 = "연합 라운드 게임과 두 Shapley 값").
  "exact"는 부사·술어 자리에만 남긴다(영문 "computed exactly"·"exactly zero"; 명칭 자리
  금지), 정확성은 절차어로 표현 — 전수 열거(exhaustive enumeration)·전수 재학습(exhaustive
  per-subset retraining). paper-ko.md §4·§5.1 적용 완료. **main.tex §4 이관 완료(07-27)**:
  "exact retrain/in-run Shapley" 이름 13곳 전부 교체·Terminology 블록 삭제(역할명은 §5.1
  이관 시 도입), $U^{\mathrm{re}}$ 무번호 인라인화로 식 번호 = 라운드 게임 (5)·추정기 (6).
- **갱신(2026-07-27 ④, 절 제목 담백화 — Yonghee 확정):** §3 이후 절·소절 제목에서 콜론
  복합·물음형·em-dash 부연·위계 태그((1차)/(2차 ①))를 제거하고 전형적 제목으로 통일 —
  위계는 §5 도입 문단이 이미 명시하므로 제목에서는 뺀다. 확정: §3 "문제 설정과 배경 /
  Problem Setup and Background", 3.1 "연합학습 설정과 표기 / Federated Learning Setup and
  Notation", 3.2 "Shapley 값 기반 기여도 정의 / Contribution via the Shapley Value",
  3.3 "In-Run Data Shapley", **§4 "방법 / Method"**, **4.1 "연합 라운드 게임 / The
  Federated Round Game"**(③의 "…과 두 Shapley 값" 제목을 대체 — 두 값 대비는 본문 첫
  문단이 담당), 4.3 "Shapley 공리와 근사 오차"(07-27 재개정 — 구 "근사 오차와 적용 범위";
  비고(적용 범위)는 §6 한계로 이동), 5.2 "Fidelity", 5.3 "기여도 기반 개입 /
  Contribution-Guided Intervention", 5.4 "오염 client 탐지 / Detecting Corrupted Clients".
  §6·§7·부록 제목은 현행 유지. paper-ko.md 적용 완료 + main.tex §3·§4 적용 완료(§5.x
  영문 제목은 이관 시).
- **갱신(2026-07-27 ⑤, 역할명 "reference" 폐지 — Yonghee 확정):** ③이 유지했던 역할명
  in-run reference / retraining reference와 명사 "기준값"을 §5 이후에서도 제거한다. 근거 =
  §4.1이 이름·기호(in-run Shapley $\phi^{\mathrm{in}}$ / retraining-based Shapley)로 세
  대상(Flirds·전수 열거값·재학습값)을 이미 완전히 분리하므로 별도 역할어는 불필요한
  신조어. 규칙: ⓐ 본문 서술 = 풀네임 그대로. ⓑ 표 안 = **in-run SV / retrain SV**(SV =
  Shapley value; §5.1 "표기" 문단에서 선언). ⓒ 수식 = $\phi^{\mathrm{in}}$(§4.1 정의; 부록
  F 사전 등록 기준식 그대로). ⓓ 집합·역할 지칭 = "두 Shapley 값"·"채점 기준"·"타깃 Shapley
  값" 등 보통명사/동사구 — "기준값" 명사도 "Shapley 값"으로(Yonghee: 그쪽이 더 정확). ⓔ
  존치: oracle-제외(개입 참조 arm 이름), CSV·코드 열 이름(spearman_a/b, phi_(b)oracle 등).
  적용(07-27): paper-ko.md §5–결론·부록 A~G 전량 — "reference" 169→0건, 부록 A 머리말의
  reference 매핑 문장 삭제 포함. 잔존 "기준값" = §1 서론 2곳(L26·L76)도 Yonghee 예외
  승인으로 당일 수정 완료(전 파일 0건). 영문(main.tex §5 이후·supplement.tex)은 이관 시
  같은 규칙 적용.
- **갱신(2026-07-27 ⑥, baseline 표시명 — Yonghee 확정):** **individual utility →
  singleton utility** 전면 교체(paper-ko 76곳; 방법이 계산하는 값이 singleton utility
  $u_r(\{k\})$ 그 자체이므로 이름을 값에 맞춤). CSV·코드 키 `loss-heur`·`in_run_singletons`
  등 산출물 식별자는 불변 — 표시명만 바뀐다. 영문 이관 시 동일.
- **갱신(2026-07-27 ⑦, 세팅 코드명 폐지 — Yonghee 확정):** `LLM-Main`·`CNN-Main`·`Silo`·
  `Anchor`·`CNN-Grid`·`Partial-Probe`·`Cross-device` 등 세팅 고유명을 본문·표·캡션에서 전면
  제거. 지칭 = ⓐ 주 세팅 → "LLM/CNN 주 세팅"(캡션이 $N$·참여·$R$·오염율 파라미터를 직접
  보유) ⓑ 보조 → 서술구("5-도메인 비IID (세팅)"·"alpaca IID-clean (세팅)"·"$N{=}10$ (전원참여)
  격자"·"부분참여 probe"·"cross-device 보조 세팅") ⓒ 반복 지칭 자리는 표 참조("표 [F3b]에서",
  "이것이 [F3b]를 주 표로 두는 이유다") ⓓ 표 행 라벨 = 서술 라벨(B.1 "LLM 주"·"CNN $N{=}10$
  격자" 등). HTML 주석·rundir 경로의 이름은 존속(산출물 식별자). 영문 이관 시 동일 원칙.
- **갱신(2026-07-27 ⑧, 문체 규칙 — Yonghee 확정; 전 구역 공통, 각 세션 관장 구역에 적용):**
  ⓐ 본문 산문에서 em-dash(—) 수식 구문 금지 — 문장 분리·괄호·"즉/때문이다"로 대체(표
  캡션·부록 헤더의 라벨 구분자 "표 [X] — 제목"과 HTML 주석은 허용) ⓑ bold 남발 금지 —
  run-in 문단 라벨(예: **CNN 주 세팅.**)과 용어 최초 정의 자리만, 강조 bold 금지 ⓒ 분량:
  **§5 단독 < 2.5페이지**(§6·§7은 이 예산에서 제외 — 07-27 밤 정정). 감축 방식 = 표현력
  저하 없이 중복 제거·부록 이동(§5.2 서사는 "주장 → 표 인용 → 메커니즘 → 한계" 아크 유지).
  적용분: [F1b]→C.1-MNIST 일원화, [F2]→C.1-LLM, [I1]→G.0 신설, §5.4 실측 2표→E.3 신설.
- **갱신(2026-07-27 ⑨, §6·§7 재설계 — Yonghee 확정; ④의 "§6·§7 제목 현행 유지"를 대체):**
  §6 = **"한계"**(부제 없는 3문단: 가치의 심판인 검증셋 / 값의 의미와 검증의 상한 / 적용
  범위를 정하는 시스템·운영 가정 — 검증셋 공정성 전제·OOD 평가 불능 포함), 통찰은 전부
  §7 결론으로(통찰 3개: 두 게임 조건부 일치·게임 선택의 다운스트림 발현·2차항과 비용의
  조건부성). 결론 분량 = 한계의 1/3~1/2. §4.3 "비고(적용 범위)"는 삭제하고 §6이
  흡수($C^3$·A.5·A.8·공리화 상속). §6.x 하위절 번호가 사라졌으므로 상호참조는 "§6"으로.
- **갱신(2026-07-27 ⑩, §5 본문 실험 스코프 대축소 — Yonghee 확정):** 본문 표 = fidelity
  3개 {[F1] CNN 주(cifar10/**dir1 전용**, 각 열 n=3 재집계 — {dir1,iid} 풀 n=6 판은 C.1
  본표가 유지), [F2] LLM 주(3-seed 완주 6열판, 본문 복귀), **[F5] 모델 규모 1B/3B/7B
  신설**(alpaca IID std20 = research-wiki flirds-results-fidelity "표준 부분참여 충실도"
  실험; 원본 = runs/track_d/fidelity.csv 파생 CSV[1B metrics 에 pearson 키 없음], ρ+r 병기,
  위키 표와 값 일치 — 그쪽 ±는 모집단 std; anchor5 레그 전 규모 1.000은 주석)} + 개입 2개 {[I2]·[I1] 전부 **sign-gating
  selection-retrain 시점만**} + §5.4 연산수 모델 소형 표. online 시점 전표 = **G.0**(LLM
  paired·세부 + CNN dir1 online 표), retraining 특성화([F3b]·alpaca·$N{=}10$ 격자 전체) =
  **C.0 신설**, removal-curve = **G.7 신설**. 나머지 실험(반복·통제·크기-가중 포함)은 전부
  부록. B.7에 F5 행 추가·F1 행에 dir1 주석.
- **갱신(2026-07-28 ⑪, fidelity 축에서 singleton utility 제외 — Yonghee 확정):** 본문
  fidelity 표 [F1]/[F2]/[F5]와 그 해석에서 singleton utility 행·서술 제거. 존속 위치 =
  개입 §5.3(score source)·§5.4 비용·부록 C 전표(C.0 [F3b]·C.1·C.1-MNIST·C.2·C.3 등).
  본문 fidelity 지칭은 "식 (5) 겨냥 3종" 대신 **"Flirds·Flirds-1st"**로 쓴다(§5.1 비교
  방법의 3종 정의 자체는 유지 — 개입이 사용). 제외 직전 값은 각 표 HTML 주석에 보존.
- **갱신(2026-07-28 ⑫, 개입 표 stub 헤더 공란화 — D4 "score source" 폐기, Yonghee 확정):**
  개입 표의 첫 열(행 레이블 열) 헤더를 **공란**으로 둔다. D4(07-26 "한글판도 score
  source")를 대체한다. 근거 ⓐ 그 열은 통제 arm(vanilla (observer)·oracle-제외·
  selection-random)과 점수를 공급하는 방법(Flirds 외)을 함께 담으므로 한 단어로 정확히
  덮이지 않는다 — "score source"는 통제 3행을, "방법"은 통제의 성격을 흐린다. ⓑ 대안
  "baseline"은 제안 방법 Flirds를 baseline으로 강등하고, §5.2·§5.3 본문이 이미 쓰는
  "baseline·비교군 = Flirds를 뺀 경쟁 방법"(§5.1 말미·[I2] 해석 문단)과 한 절 안에서
  충돌한다. ⓒ stub head 공란은 표준 표기 관행이고 행 레이블이 자명하며 표의 종류는 캡션이
  밝히므로 정보 손실이 없다. 적용 완료: `paper-ko.md` 표 [I2]·[I1], `paper-ko 부록.md`·
  `appendix_cdef.md` 각 13개 표(HEADA 11 + [I7] + [I8]), 생성기 `make_appendix.py`
  (HEADA·[I7]·[I8] `table()` 호출) — 두 부록 md 는 생성기 출력과 비트 동일한 `|  |` 형식,
  본문은 그 파일의 정렬 패딩 유지. 본문 §5.3 서술의 "score source" 2곳은 Yonghee 가 직접
  "두 방법" 치환·문장 삭제로 처리(기준본 "score source"·"점수원" 0건). **미적용 =
  main.tex L691** "the score source" 괄호 정의(§5 이관 시 함께 제거). **잔여 판단 대기 =
  부록 F 서술의 "점수원" 5곳**(원문 = `make_appendix.py`) — 표 헤더가 아니라 이번 결정
  범위 밖이나, 기준본에서 이 명사가 사라졌으므로 부록만 남는 상태.
- **⚠ 영문 동기화 대기(supplement.tex — 담당 세션 몫):** 기이관 부록 A 분량(283행)에
  ⑤·⑥ 이전 판 용어 7곳 잔존 — "retraining reference"/"in-run reference"(L49 Notation
  문단 포함)·"individual utility". 이관 재개 전 ⑤(풀네임/in-run SV·retrain SV)·⑥(singleton
  utility)으로 재동기화할 것. main.tex는 §4까지라 해당 없음.
- 출처 태그: **[기준]** = 기준본(구 paper-ko2)에 이미 있음 / **[legacy]** = 옛 원본에 있음 / **[tex]** = main.tex 유래 / **[G]** = GPT 변경 문서 도입 / **[C]** = Claude 변경 문서 도입.

먼저 사실관계: **"같은-게임 계열 / cross-game 계열"은 기준본과 legacy 원문에 원래 있던 분류
용어**다(§5.1·부록 C). 반면 **영문 "same-game"의 전면화(제목·기여명)와 "realized-round /
realized-update"는 원문에 없고** main.tex 제목·GPT/Claude 변경 문서가 들여온
표현이다(legacy에는 "realized 귀속" 1곳만). "measure-first"도 main.tex에서 온 표현이다. 즉
아래 A그룹은 전부 "채택할지 말지"부터가 결정 대상이다.

---

## 문헌 경향 데이터 (2026-07-26 조사; 개별 안건의 "문헌:" 줄이 여기를 인용)

우리 논문이 걸쳐 있는 두 커뮤니티의 용어 관행을 1차 자료(제목·본문)로 확인한 결과다.

| 계보 | 과업을 부르는 이름 | 대표 근거 (전부 원문 확인 또는 검증된 인용표 기준) |
|---|---|---|
| **FL 기여도 계보** (우리의 홈) | **contribution evaluation / assessment** | 전용 서베이 존재: "Shapley-value-based **Contribution Evaluation** in Federated Learning: A Survey"(IEEE, 2023; 평가 접근을 self-reporting/개별성능/utility game/SV 4분류). 제목 사용례: GTG-Shapley(TIST'22 "Participant **Contribution Evaluation**"), SPACE(NeurIPS'23 "**Contribution Evaluation**"), ShapFed(IJCAI'24 "contribution **assessment**"), Song et al.(BigData'19 "**contribution index**") |
| FL 데이터-가치 계보 | **data valuation** | FedSV(2020 "Data **Valuation** for FL"), ComFedSV(ICDE'22 "Data Valuation in HFL"), FedIF(2025 "Federated Data Valuation"), Data Shapley·Beta Shapley·Data Banzhaf·Jia et al.(AISTATS'19) — 경제/마켓 프레임과 결합 |
| 중앙집중 단일-런 계보 | **(training data) attribution / influence** | TRAK(ICML'23 "**Attributing** Model Behavior at Scale"), Ripple(AAAI'26 "Data Influence **Attribution**"), ProToken(2026 "Token-Level **Attribution**"), IRDS는 "Data Shapley in One Training Run". 서베이(Hammoudeh & Lowd, *Machine Learning* 2024 "Training Data **Influence** Analysis and Estimation")는 influence ≈ data valuation ≈ data attribution을 준동의어로 정리하고, 방법 분류는 **retraining-based vs gradient-based** |
| 참값 명칭 관행 | **exact / original / actual Shapley value** | GTG 원논문은 전수 재학습 기준선을 "**Original Shapley**"로 부르고 "actual Shapley values"에 근접함을 주장 — "ground truth"를 참값 명칭으로 쓰지 않음. SV-근사 문헌 일반의 채점 관행 = 거리 지표(GTG: cosine/Euclid/max)·RMSE·**Spearman rank correlation**. 예외: 중앙집중 datamodel 계보(TRAK의 LDS)는 재학습 counterfactual을 "ground truth"로 부르기도 함 — 즉 GT 표현은 홈 커뮤니티(FL-SV) 관행이 아님 |
| free-rider 위협 분류 | zero / random / **delta weights** / disguised | Lin, Du & Liu(2019 "Free-riders in FL: Attacks and Defenses"; zero·random 업데이트 + STD-DAGMM), **"delta weights attack"** = 직전 글로벌 모델 차 재제출의 확립 명칭(Lin 계보; Delta-DAGMM(2022)이 명칭 재사용), Fraboni et al.(AISTATS'21 "Free-rider Attacks on Model Aggregation"; plain/disguised·시변 교란), Zhu(NeurIPS-FL WS'21 "Advanced Free-rider Attacks") |
| FL 규모 축 | **cross-silo / cross-device** | Kairouz et al. 2021 (Advances and Open Problems in FL) — 표준 구분; 무대 영문화 시 재사용 대상 |

핵심 시사 세 가지: ① 참값을 "ground truth"라 부르는 것은 우리 홈 계보의 관행이 아니며,
관행은 **"exact/original Shapley value (of game X)"라는 이름 + 비교 역할일 때 reference**다
→ A1. ② 과업 이름은 FL 계보(contribution evaluation)와 단일-런 계보(attribution)가 갈리는데
우리 논문은 정확히 그 다리 위에 있다 → A2의 역할 분담 규칙이 문헌 분포와 정합. ③ free-rider
하위 유형은 이미 문헌에 이름이 있다(zero/random/delta weights) — 코드명(frzero/frrand/frdelta)
대신 그 이름을 쓰면 인용으로 방어된다 → D3.

---

## A. 논문 정체성 용어 (최우선)

### A1. 참값 명칭: `GT` 유지 vs `reference` 전환 — **가장 큰 갈림**

- 현황: 기준본 = **retrain GT / in-run GT** 규약(§4.1 용어 규약: 수식어 없는 GT·exact 금지,
  oracle은 개입 arm 전용). GPT 문서는 **공통 치환**으로 `in-run GT→in-run reference`,
  `retrain GT→retraining reference`, `exact retrain Shapley→exact retraining Shapley`를
  제안. main.tex는 현재 "ground truth (**oracle**)"로 한글 규약과 **정반대**(audit-v2 P0-2).
- 쟁점: "GT/참값"은 보편 참값으로 읽혀 리뷰 공격면("스스로 정의한 GT")이 되고, "reference"는
  그 공격을 원천 차단하나 표·본문 전면 치환 비용 + "exact"가 이름에서 빠지면 강도가 죽는 느낌.
- **문헌:** SV-근사 계보의 관행은 참값의 **이름 = "exact/original/actual Shapley value"**이고
  (GTG는 "Original Shapley" 기준선과 거리 3종으로 비교), "ground truth"는 이름으로 쓰지 않는다.
  "ground truth"를 쓰는 쪽은 중앙집중 datamodel 계보(TRAK/LDS) 정도. 즉 **양(量)의 이름은
  exact-계열, 비교에서의 역할은 reference**가 문헌 정합 — GPT안과 방향은 같되, "exact
  in-run Shapley (value)"라는 풀네임을 버리지 않는 형태가 관행에 가장 가깝다.
- 선택지: ① 현행 유지(retrain GT/in-run GT) ② GPT안(retraining/in-run **reference**)
  ③ 절충 — 이름은 "exact in-run Shapley"/"exact retraining Shapley"(SV-관행), 역할·약칭은
  "in-run reference"/"retraining reference".
- **결정(권고): ③.** 구체 규칙 — ⓐ 양의 이름(정의·명제·서술) = **exact in-run Shapley /
  exact retraining Shapley**(GPT 제안대로 retrain→retraining 문법 교정 포함) ⓑ 비교
  역할·표 약칭 = **in-run reference / retraining reference** ⓒ 한글 총칭이 필요하면
  "참값" 대신 **"기준값(reference)"** ⓓ 금지: 단독 "ground truth"·"GT"·"참값", 참값 의미의
  "oracle"(B1) ⓔ 일괄 치환: `in-run GT→in-run reference`, `retrain GT→retraining
  reference`, §4.1 용어 규약 문단은 이 규칙으로 재작성, main.tex "(oracle)" 삭제.
- **갱신(확정 07-27 ②): 위 ③ⓐ를 대체 — "exact"는 이름 성분에서 전면 제거.** 근거: 게임이
  고정되면 그 Shapley 값은 유일하므로 "exact"는 양의 이름이 아니라 계산 방식의 서술이다
  (이름에 넣으면 매 언급마다 3단 합성명을 끌고 다니게 되는 부담이 실제 어색함의 원인).
  ⓐ 양의 지칭 = 서술형: 한글 **"(전수 열거로 계산한) 라운드별 Shapley 값"**, 영문 "the
  Shapley value of the round game(, computed by exhaustive enumeration over all
  coalitions)". 정의 시점에 계산 방식을 1회 명시하면 이후 "exact" 수식어 불필요.
  ⓑ 역할명 유지(③ⓑ·ⓒ 그대로): in-run reference / retraining reference·한글 "기준값".
  비교 문장에서 기준값 쪽은 **항상 역할명**으로 — Flirds도 같은 게임의 Shapley 추정치를
  내므로 맨몸 "in-run Shapley"는 기준값 지칭으로 쓰지 않는다.
  ⓒ retraining 계열 지칭 = **"재학습 기반(retraining-based)"** — IRDS 원문(ICLR 2025)이
  "Retraining-based Data Shapley"를 절 제목 용어로 쓰고 "algorithm에 대한 평균 기여 vs
  특정 run에 대한 기여"라는 우리와 같은 구분을 이미 공식화하므로 인용으로 방어된다
  (대체어 탐색 불필요 — 07-27 확인). 서론 기여 1의 "재학습 기반 데이터 가치평가"가 적용례.
  ⓓ "exact"의 허용 용법(이름 아님): 계산 서술 부사("근사 없이", "computing the Shapley
  value exactly requires $2^N$…", "전수 열거"), 대수 성질 **"exact-0"**(zero-update),
  지표 exact-match(EM). 혼성형("exact in-run reference"류)은 계속 금지(B2 강화).
  ⓔ §4.1의 공식 명명·기호($\phi^{\mathrm{in}}$) 재작성과 부록 A 도입부의 표기 정의문
  ("표기: retraining reference = exact retraining Shapley, …")은 **§4 수정 세션에서**.
  적용 현황: 초록·서론 완료(07-27), §5–7·부록 기계 치환 완료(07-27 — §5.2 제목 "exact
  in-run Shapley 재현"→"in-run reference 재현", 혼성형 4곳, "exact 열거/전수" 5곳,
  §5.2 sub 제목, A.2 첫머리, E5 fidelity 정의문), main.tex은 이관 시.

### A2. 논문의 자기 명칭: contribution **evaluation** / valuation / estimation / attribution

- 현황: 확정 제목 = "…Client-Level Contribution **Evaluation**…". 그러나 main.tex 초록 첫
  문장은 "Client contribution **valuation**", 본문엔 valuation 17회·contribution 14회·
  attribution 9회·evaluation 8회·estimation 4회 혼재. 한글도 "기여도 평가"(4)·"가치평가"(3)·
  "귀속"(10) 혼용.
- **문헌:** FL 홈 계보의 우산 용어는 **contribution evaluation**(전용 서베이 제목이 그대로
  이 표현; GTG·SPACE·ShapFed도 제목에 사용)이고, **data valuation**은 FedSV·ComFedSV·FedIF와
  Data Shapley 계보(경제 프레임), **attribution/influence**는 단일-런 중앙집중
  계보(TRAK·IRDS·Ripple·ProToken)의 용어다. 우리 논문은 FL 과업 이름(evaluation)에 단일-런
  방법론(attribution)을 접붙인 위치라, 역할 분담이 곧 계보 표시가 된다.
- **결정(권고):** 4-역할 규칙으로 확정 — ⓐ **과업/문제 = contribution evaluation(기여도
  평가)**: 제목·초록 첫 문장·서론 도입(main.tex 초록 첫 단어 valuation→evaluation 교정)
  ⓑ **방법이 하는 일 = attribution(귀속)**: "라운드 개선을 업데이트에 귀속" 류 서술
  ⓒ **추정기 = contribution estimator(기여도 추정기)** ⓓ **data valuation**은 Data
  Shapley·마켓 계보를 지칭할 때만(§2·§3.2). "가치평가"라는 한글은 그 인용 맥락에서만.

### A3. same-game / cross-game 표기

- 현황: 한글 "같은-게임 계열" + 영문 "cross-game 계열"로 **한/영 비대칭** [기준]. 영문
  same-game은 기준본 B.5 1곳 + main.tex 다수 + 두 변경 문서(기여명 "Same-game 오차 해석" [C],
  GPT C09 "same-game fidelity" 등).
- **문헌:** same-game/cross-game도 확립 용어는 아니다(신조어). 다만 가리키는 구분 자체 —
  "자기 목표 게임에 대한 근사 오차"와 "다른 utility를 겨냥해 생기는 불일치"의 분리 — 는
  SV-근사 문헌이 전자만 채점해 온 관행(GTG의 Original-Shapley 대비 거리 채점)과 정확히
  잇닿아, 신조어지만 정의 1회로 방어 가능한 부류다.
- **결정(확정 07-26): 신조어 페어 폐기 — 서술형으로 우회.**
  ⓐ 방법 분류 라벨: **"식 (6)을 겨냥하는 계열(3종)"** vs **"다른 게임을 겨냥하는
  계열(5종)"**(표 축약: "식 (6) 겨냥 3종" / "타 게임 5종"; 영문 methods targeting Eq. (6) /
  methods targeting other games).
  ⓑ 오차 서술: "같은-게임 오차/fidelity" → **"게임 불일치가 섞이지 않은 (순수) 근사 오차"**,
  "cross-game 성분" → **"게임 불일치(game mismatch) 성분"**(이미 초록이 쓰는 평서술).
  ⓒ 주의 — **in-run/retraining으로는 대체 불가(직교 축)**: GTG·FedSV도 재학습 없이 같은 런의
  업데이트로 계산하므로 넓은 의미의 in-run이다. 갈리는 것은 실행 방식이 아니라 **겨냥하는
  게임의 정의**(coalition 재정규화 여부 등)라서, 이 구분을 in-run/retraining이라 부르면
  오히려 틀린 말이 된다.
  ⓓ 잔여 치환 대상: 기준본 §5.1 계열 라벨·표 제목("같은-게임 3종")·RQ1·결론·B.5("same-game
  주장")·C.1, [C] 기여 2 제목·K13, [G] C09 전반.
  ⓔ **갱신 07-27**: §4.1 재배치(식 번호 교환)로 라운드 게임 = 식 (5) — 이 라벨은 이제
  **"식 (5)를 겨냥하는 계열"**이다(기준본 일괄 치환 완료; 헤더 갱신 ③).

### A4. realized-round (game) / realized-update (attribution) — **원문에 없던 신조어**

- 현황: [G]+[C] 동시 도입(기여 1 "realized-round Shapley 게임" [C·K07], GPT C02/C11
  "realized-round validation-loss game", gap 문장 "realized-update 게임" [C·K03, GPT C03-1
  유사]). 기준본의 기존 표현 = "라운드 게임", "실현된 궤적", "고정-가중 라운드 게임".
- 쟁점: 영문 개념어로는 정확하고 강하지만(realized = 실행에서 실제로 발생한), 한글 본문에
  영문 합성어가 늘어난다. 대안 한글: "실현 라운드 게임", "실현-업데이트 귀속".
- **문헌:** "realized-round/realized-update"는 두 계보 어디에도 확립된 용어가 아니다(신조어).
  문헌에 있는 대립축은 **retraining-based vs gradient-based**(Hammoudeh & Lowd)와
  **counterfactual vs 단일-런(in-run/one training run)**(IRDS 제목·Ripple 제목)이다. 즉
  문헌-정합 최댓값은 이미 확보한 "**in-run**"을 확장하는 것("per-round in-run game" 류)이고,
  realized-*를 쓰려면 첫 등장 정의가 필수다.
- **표현하려던 의미(질문 답):** "가상의 재구성·재학습이 아니라, **실제로 일어난 그 라운드**의
  **실제 수신 업데이트**와 **실제 사용 가중치**로 정의된 게임"(→ grand coalition이 실제 집계와
  일치). 즉 전달할 내용은 ①관측된(실제) ②라운드별 ③가중치 고정 — 셋 다 기존 서술어다.
- **결정(확정 07-26): 신조어 철회 — 기준본의 기존 표현 재사용.** 게임 이름 =
  **"고정-가중 라운드 게임(fixed-weight round game)"**(기준본 §5.1·§7 결론에 이미 존재),
  실제성 강조가 필요한 곳에만 **"관측된(observed)"** 수식(예: "관측된 라운드의 고정-가중
  검증손실 게임"). "실현-업데이트 귀속"류는 "관측된 업데이트에 대한 귀속"으로.
  realized-round/realized-update는 전면 미사용 — [C] K03·K07과 [G] C02·C03-3·C11의 해당
  문구 수정 대상.
- **갱신(확정 07-27): "고정-가중 라운드 게임"도 회피 — 헤더 갱신 ①②가 대체.** 게임 이름 =
  **"연합 라운드 게임"**(영문 **(federated) round game**), 필요 시 "관측된 라운드의 검증손실
  게임"·"the game over the observed round"류 서술. realized-* 전면 미사용은 그대로 유지.

### A5. measure-first — [tex]→[G]·[C] 역수입

- 현황: main.tex "measure-first principle" → GPT C03-2·Claude K06이 한글 서론에 도입
  ("measure-first 원칙").
- **문헌:** 무전례 신조어(슬로건). 다만 내용 자체("다운스트림 성과는 근사 정확성의 대체물이
  아니다")는 SV-근사 문헌의 표준 채점 관행(exact 값 대비 거리·순위 상관 — GTG 등)과 일치하므로,
  슬로건 없이도 관행 인용으로 같은 주장이 선다. 슬로건은 기억성 장치로만.
- **결정(확정 07-26): 표어 철회(③) — 이름 없이 내용 문장만.** 예: "다운스트림 성과를 추정
  정확성의 대체물로 삼지 않는다 — 추정값을 먼저 자신이 겨냥한 게임의 exact 값 대비로 채점한
  뒤 실효성을 본다." 근거: 슬로건이 연구의 핵심 기여처럼 보이는 부작용(Yonghee) + 내용은
  SV-근사 문헌의 표준 채점 관행이라 이름 없이도 선다. main.tex의 "measure-first principle"
  문장, [G] C03-2, [C] K06②를 동일하게 무명화.

---

## B. 참값·비교 체계 (규약 정비)

| # | 항목 | 현황 | 결정(권고) |
|---|---|---|---|
| B1 | oracle 규약 | 한글: 개입 arm 전용 [기준]. main.tex:446 "ground truth (oracle)"로 **역전**. 문헌: FL 탐지·강건화 실험 관행에서 oracle = "오염 집합을 아는 상한 arm" — 기준본 규약이 관행 정합, main.tex가 이탈 | **기준본 규약 확정**: oracle = §5.3 개입 arm 전용(영문 arm명 "oracle-exclusion (reference arm)"), 기준값 명칭으로 사용 금지. main.tex "(oracle)" 삭제(A1ⓔ와 함께) |
| B2 | "exact" 사용 규칙 | 규약 = 수식어 없는 exact 금지 [기준]. 위반 후보: K02 초록의 "exact in-run 참값"(혼성 축약) | ~~**"exact + 대상 명시"만 허용**: exact in-run Shapley / exact retraining Shapley / exact 열거(enumeration). 혼성형("exact in-run 참값"·"exact 참값") 금지~~ → **갱신 07-27 ②: 대체** — exact는 **이름 성분으로 전면 미사용**(A1 갱신 블록). 허용 = 계산 서술 부사("전수 열거", "근사 없이", "computing … exactly")·대수 성질 exact-0·지표 exact-match(EM)뿐. 혼성형 금지는 유지 |
| B3 | "중립 참값" 표현의 교체 [C·K18/K26] | **무엇이 문제인가(상세)**: §5.1은 retrain 기준값 비교표를 정당화하며 "어느 방법의 목표값도 아닌 **중립** 참값"이라 부르는데, 같은 논문의 C.3 각주가 "retrain 기준값의 게임 자체가 부분집합 크기로 재정규화되는 구조라 재정규화 계열(GTG·FedSV류)에 유리하게 작동하는 칸이 있다"고 스스로 인정한다. 심판이 일부 선수와 같은 규칙을 공유하면 중립이 아니므로 자기모순(리뷰 공격면)이고, "중립 심판은 같은 정책 하 다운스트림뿐"(Yonghee 07-19 결정)과도 충돌 | **"공통 외부 기준값(shared external reference)" 채택** — 주장 강도를 "비교되는 어떤 방법의 목표값도 아니고(외부), 전 방법에 같은 잣대로 적용된다(공통)"까지로 낮춘 역할 명칭. 공정성·중립성은 주장하지 않으며, 구조적 근접성 각주를 §5.1로 승격(K18). "중립(neutral)"은 금지어 |
| B4 | 두 기준값의 표 표기 | 표에서 "(b)oracle" 같은 코드명 노출 여부(분석 CSV는 (b)oracle 사용) | **논문 표에 코드명 금지** — A1 약칭(in-run reference / retraining reference)만. 코드명↔논문명 대응표는 부록 B.7(provenance)에 1회 |
| B5 | "게임의 답" ([D1]·[D2] 행 이름) | in-run GT (게임의 답) [기준] | **유지.** 영문 = "the game's own answer"(행 이름 "in-run reference (the game's own answer)") |
| B6 | fidelity 채점 지표 이름 | Spearman·Pearson 본문 + Kendall·거리 3종 부록 [기준]. 문헌: SV-근사 채점 관행 = exact SV 대비 거리(GTG: cosine/Euclid/max)·RMSE·Spearman — **우리 구성이 관행 정합** | **현행 유지** + §5.1 지표 문단에 "SV-근사 문헌의 표준 채점축(GTG 인용)" 반 문장 추가(관행 방어) |

## C. 방법·수학 용어

| # | 항목 | 현황(빈도) | 결정(권고) |
|---|---|---|---|
| C1 | 폐형식 vs closed-form | 폐형식 4 / closed-form 13 혼용 | **한글판 "폐형식" 통일**(§4 제목 포함; 첫 등장 "폐형식(closed-form)" 병기 1회). 영문판 closed-form |
| C2 | surrogate | ①우리의 "2차 Taylor surrogate 게임" ②선행연구의 "대체(surrogate) 값" ③"근사 게임 $\hat u_r$" 3용법 혼재 | **우리 것 = "2차 surrogate 게임"으로 통일**(첫 등장에 "= 근사 게임" 동치 1회 명시, 이후 "근사 게임" 표현은 수식 $\hat u_r$ 지칭에만). **선행 지칭은 surrogate 단어 회피** — "대체 utility/대체 점수"로. **갱신 07-27 (이행·보강)**: 정의 지점 = §4.2 $\hat u_r$ 도입부("…Taylor 전개한 근사 게임, 곧 2차 surrogate 게임 $\hat u_r$") 바인딩 적용; 표면형 변주(2차 Taylor surrogate·surrogate 게임·라운드 surrogate 등)는 동일 의미 하에 맥락별 허용(Yonghee 확정). 근거 보강: SVAkADD(arXiv 2502.04763)가 "k-additive surrogate game"을 같은 구도(대리 게임의 exact Shapley를 원 게임 추정치로)로 사용 — SV-근사 문헌의 기성 용어라 인용 방어 가능 |
| C3 | 절단 vs 잔차 | "Taylor 절단(truncation)"과 "Taylor 잔차(remainder)" 혼용(명제 3 제목=잔차) | **행위 = "Taylor 절단", 오차량 = "(절단) 잔차"로 역할 고정.** "절단 오차"라는 제3 표현은 "잔차"로 치환. §5.2 "정산 잔차"[C·K19]도 이 규약에 정합 |
| C4 | Flirds-1st | 한글판 Flirds-1st / main.tex "Flirds (first-order)" | **Flirds-1st로 두 언어 통일**(표 공간·본문 빈도 우세). 영문 첫 등장에 "Flirds-1st (the first-order variant)" 병기. main.tex의 "Flirds (first-order)" 치환 |
| C5 | individual utility | 방법명 [기준]. singleton utility 설명과 병존 | **유지**(표·분석 CSV와의 일관 비용 고려). 첫 등장 정의 고정: "라운드 게임의 singleton utility $u_r(\{k\})$를 직접 평가·합산하는 가산 근사(Shapley 아님)" |
| C6 | 고정 가중치/고정-가중/fixed-weight | 세 표기 혼재. **중요도(질문 답): 이 논문의 핵심 설계 선택 그 자체** — coalition 안에서 가중치를 재정규화하지 않는다는 선택에서 ①grand coalition=실제 집계 ②폐형식 성립(P4(i): 재정규화하면 합-구조 붕괴) ③zero-update exact-0이 전부 나오고, GTG·FedSV는 반대 선택(coalition 내 재정규화 $n_k/\sum_{j\in S}n_j$). 커뮤니티 브랜드 용어는 아니지만 "가중치를 고정한다"는 **평서술 형용사**라 A4·A5류 신조어와 성격이 다름 — 개념 대비(재정규화 여부)는 FL-SV 문헌이 실제로 다루는 차이. A4 결정으로 게임 공식 명칭("고정-가중 라운드 게임")이 이 표현에 기대게 됨 | ~~명사 = "고정 가중치", 수식어 = "고정-가중" 규칙화, 영문 fixed-weight~~ → **갱신 07-27: 회피 확정**(헤더 갱신 ①②) — 합성어 미사용, 서술형 동사구만("가중치는 $S$ 안에서 다시 정규화하지 않고 런 자신이 사용한 값 $p_k^r$ 그대로 고정한다" / 영문 "weights held fixed"·"without renormalizing within a coalition"). paper-ko 전면 적용 완료 |
| C7 | 고정 궤적/동결 로그/frozen log/동결된 궤적 | 혼용 ([기준] 고정 궤적·frozen log; [C] 동결 로그) | **"고정 궤적(frozen trajectory)"을 표준어로.** 로그 자료구조 지칭은 A.1에서 "동결 로그(frozen log)" 1회 정의 후 그 문맥에만. "동결된 궤적" 표현은 "고정 궤적"으로 치환 |
| C8 | 감사 가능성 | [기준·C]. GPT 피드백: 실제 보장은 "재계산 가능성(log replayability)" | **유지하되 첫 등장 한정 정의**: "감사 가능성(= 동결 로그에서의 재계산 가능성, log replayability)" — 외부 감사 전반의 보장으로 읽히지 않게 |
| C9 | 정산 (요건/잔차/신호) | 기준본 자체에는 §6.1 한 곳뿐 — "감사 가능한 **정산 신호**를 제공하지만 payment rule은 아니다"(그 외 main.tex·[G]에는 없음). 확장 두 곳(서론 「정산 요건」 [C·K04], §5.2 「정산 잔차」 [C·K19])은 legacy 동기를 복원한 [C] 제안이었음 | **확정 07-26: 정산 관련 내용 전면 제거.** ⓐ [C] K04(서론 정산 요건 문단)·K06①(정산 요건 충족 문장) **철회** ⓑ 잔차 실측(K19)은 정산 명명 없이 "**efficiency 잔차**"로 재서술해 유지 — 측정 자체는 클로드 피드백 ②의 요청이자 §6.1 efficiency 두 층의 실측 연결(제거 원하면 K19도 삭제 가능, Yonghee 한마디면 됨) ⓒ 기준본 §6.1 문장은 정산 없이 재서술(신규 K27: "동결 로그에서 언제든 같은 값으로 재계산할 수 있는 기여도 신호이지만 … payment rule은 아니다" — 한계 문장 자체는 피드백 방어선이라 유지) ⓓ "정산/settlement" 금지어 등록 |
| C10 | efficiency | "예산 정합"은 이미 소거, efficiency로 통일됨 | **유지**(게임이론 표준). surrogate-게임 기준임을 병기하는 현행 규약 유지 |

## D. 실험 용어

| # | 항목 | 현황 | 결정(권고) |
|---|---|---|---|
| D1 | **"retrain"의 3중 의미** | ① retrain GT(기준값) ② §5.3 시점 이름 "retrain"(선택 후 재학습 arm) ③ retrain-random(통제) — 같은 단어가 기준값과 개입 arm에 모두 사용 | **②·③ 개명으로 충돌 제거**: 시점 이름 "retrain" → **"selection-retrain(선별 재학습)"**(§1이 이미 "selection 실험"이라 부름 — 그 이름을 승계), "retrain-random" → **"selection-random"**. ①은 A1에 따라 retraining reference가 되므로 3중이 완전 해소. 표 헤더 "sign-gating · retrain" → "sign-gating · selection-retrain" |
| D2 | 무대 (주무대/보조무대) | [기준] 전반. 문헌: FL 규모 축 표준어 = **cross-silo / cross-device**(Kairouz et al. 2021) | **확정 07-26: 한글판도 "세팅"** — 무대→세팅, 주무대→주 세팅(main setting), 보조무대→보조 세팅(auxiliary setting). 규모 수식어는 Kairouz 용어. "stage"·"무대" 미사용 |
| D3 | 위협 이름 | free-rider(zero)/free-rider(rand)/frdelta(§6.1)/gnoise 등 코드명 혼입. B.6은 "LLM에 gradient noise 무대 미성립"인데 track_h LLM에 gnoise 실행 중 — **본문과 실측 정합 확인 필요**. 문헌: zero/random 업데이트(Lin et al. 2019), **delta weights attack**(Lin 계보·Delta-DAGMM 2022), plain/disguised(Fraboni et al. AISTATS 2021) | **문헌 명칭 채택**: **zero-update free-rider / random-update free-rider**(Lin et al. 인용), frdelta = **delta-weights free-rider**(Lin 계보 인용), gradient noise·label-flip@r·variable-intensity label-flip·feature-noise는 현행 유지. 코드명(frzero 등)은 부록 B.7 대응표에만. + 액션: track_h LLM gnoise 결과와 B.6 서술의 정합을 착지 후 판정(모순 시 B.6 갱신) |
| D4 | 점수원 (11회) | [기준] §5.3 경쟁 구조의 핵심 명사 | ~~확정 07-26: 한글판도 "score source" — 점수원→score source 치환~~ → **폐기(07-28 ⑫): 개입 표 첫 열 헤더 = 공란**. 표에 이름을 붙이지 않으므로 풀이 문장도 없다. 상세·근거 = 상단 갱신 ⑫ |
| D5 | 회복률 vs 회수율 | 1 vs 2회 혼용 (§5.3/E.1) | **확정 07-26: 한글판도 "recovery"** — 회복률·회수율→recovery(첫 등장 "recovery(회복률)" 병기 1회). 분석 CSV 열 이름과 일치 |
| D6 | arm 이름 | vanilla/관찰자(observer)/oracle-제외/random-제외/retrain-random. **vanilla vs observer(질문 답): 같은 무개입 궤적의 두 역할 이름** — vanilla = 성능 비교의 무개입 기준 arm, observer(관찰자 런) = 그 **비트동일 궤적**에 점수원들을 부착해 φ만 뽑는 실행(§5.1 고정-궤적 채점 문단; 개입 실험의 점수 출처). 성능 수치는 동일하고 역할만 다름 — 그래서 표에 "vanilla(관찰자)"로 병기돼 있음 | **대응 고정** + 혼동 방지: 표 표기는 **"vanilla (observer)"로 단일화**하고 §5.1에 두 역할 정의 1문장. oracle-제외 = oracle-exclusion (reference) / random-제외 = random-exclusion (control) / selection-random(D1 연동) |
| D7 | 오염-평균, macro-평균 | [기준] | **유지** + 영문 고정: 오염-평균 = corrupted-threat average, macro-평균 = macro-average |
| D8 | 내부 은어 | "착지", "⬚ 채움", RQ 블록, Tier, P1/P5s 정책 코드 | **작업본 전용으로 확정** — 제출 전 제거·환언 체크리스트에 등록(RQ 블록은 본문 소절 첫 문장으로 환언, 정책 코드는 B.7에만) |

## E. 표기 스타일

| # | 항목 | 현황(빈도) | 결정(권고) |
|---|---|---|---|
| E1 | coalition vs 부분집합 | 30 vs 22 | **게임론 맥락(플레이어 집합·utility 인자·재정규화 논의) = coalition, 순수 집합 연산·개수($2^N$개) = 부분집합.** 첫 등장 "부분집합(coalition)" 병기 1회 |
| E2 | 검증손실 vs 검증 손실 | 15 vs 5 | **"검증손실" 붙여쓰기 통일**(검증셋·검증 forward는 현행 유지). 5곳 치환 |
| E3 | 클라이언트 vs client | §6에서 client 다수 | **확정 07-26: 한글판도 "client" 통일**(첫 등장 "client(클라이언트)" 1회; client-level 등 합성어와 자동 정합). 클라이언트→client 일괄 치환 |
| E4 | 조사 오류 | "in-run GT을"(§5.1) | A1 치환 시 자동 해소("in-run reference를") — 치환 스크립트에 조사 검사 포함 |
| E5 | fidelity | 영문 그대로 사용 중. 문헌: SV-근사 계보는 "approximation accuracy/error"가 다수, "fidelity"는 XAI 유래 | **유지.** 첫 등장 정의 1회 필수: "충실도(fidelity) = **기준값** 대비 순위·값 재현도"(07-27 ②: "exact 값 대비"에서 갱신 — §5 도입부 치환 완료) |
| E6 | 하이픈 합성어 | 고정-가중, 라운드-cohort, 신호-부재 등 | **규칙: 합성 수식어에만 하이픈, 명사구에는 금지.** 공인 목록(A3·A4·07-27 갱신 반영): 신호-부재, 오염-평균, 라운드-cohort — 고정-가중은 07-27 회피 확정으로 목록에서 제거. 목록 외 신규 하이픈어는 이 문서에 추가 후 사용 |
| E7 | 영한 병기 원칙 | Flirds 풀네임 등은 정비됨 | **모든 기술 용어 = 첫 등장 1회 병기 후 한 표기 고정.** 이 문서의 결정값이 병기 원칙의 원본(치환 스크립트의 사전으로 사용) |

---

## 문헌 조사 출처 (07-26 확인)

- [Shapley-value-based Contribution Evaluation in Federated Learning: A Survey (IEEE)](https://ieeexplore.ieee.org/document/10365410/) — FL 우산 용어·4분류의 근거
- [GTG-Shapley (arXiv:2109.02053)](https://arxiv.org/pdf/2109.02053) — "Original Shapley" 기준선·거리 3종 채점 관행
- [Hammoudeh & Lowd, Training Data Influence Analysis and Estimation: A Survey](https://www.semanticscholar.org/paper/c164ec3b73eb108f861fcd60df73b62d2482c3a6) — influence ≈ valuation ≈ attribution, retraining-based vs gradient-based
- [Lin, Du & Liu, Free-riders in Federated Learning (arXiv:1911.12560)](https://arxiv.org/pdf/1911.12560) — zero/random 업데이트·STD-DAGMM
- [Fraboni et al., Free-rider Attacks on Model Aggregation (AISTATS 2021)](https://proceedings.mlr.press/v130/fraboni21a/fraboni21a.pdf) — plain/disguised 분류
- [Delta-DAGMM (2022)](https://www.hindawi.com/journals/scn/2022/8928790/) — "delta weights attack" 명칭 재사용례
- [Zhu, Advanced Free-rider Attacks (NeurIPS-FL WS 2021)](https://neurips2021workshopfl.github.io/NFFL-2021/papers/2021/Zhu2021.pdf)
- 제목 사용례(전부 프로젝트 `citations-table.md`에서 서지 검증됨): FedSV·ComFedSV·FedIF(data valuation) / GTG·SPACE·ShapFed·Song 2019(contribution evaluation/assessment/index) / TRAK·IRDS·Ripple·ProToken(attribution/influence/one-run) / Kairouz 2021(cross-silo·cross-device)

## 적용 순서 (결정 확정 후)

1. **A1·A2 먼저**(치환 규모 최대; B2·B3·B4·D1·E4가 종속) → 치환 사전 확정: `in-run
   GT→in-run reference`, `retrain GT→retraining reference`, `exact retrain
   Shapley→exact retraining Shapley`, `중립 참값/중립 참조→공통 외부 기준값`,
   main.tex `valuation→evaluation`(초록·과업 맥락만) + "(oracle)" 삭제.
2. A3·A4·A5·C9 확정값으로 두 변경 문서(K/C 항목)의 문구를 먼저 정렬한 뒤 기준본에 적용
   (K02의 B2 위반 교정 포함; [C] K02·K03·K04(철회)·K06·K07·K13·K19·K27은 07-26 정렬 완료,
   [G] C02·C03-2·C03-3·C09·C11은 적용 시 같은 규칙으로 수정).
3. B·C·D·E 일괄 반영 — 기계 치환 가능: A1·C1·C3(절단 오차→잔차)·D2(무대→세팅)·
   D4(점수원→score source)·D5(회복률/회수율→recovery)·E2·E3(클라이언트→client).
   수동 확인 필요: A3(같은-게임/cross-game 문맥 치환)·C2(surrogate 3용법 분리)·D1(표
   헤더)·D3(위협명+인용 추가).
4. main.tex는 확정 제목 동기화 + A1ⓔ·A2ⓐ를 §1–4에 즉시 반영, 나머지는 §5–7 이관 시.
   **07-27 갱신 3건(고정-가중/fixed-weight 회피 · 게임명 "(federated) round game" · 식 번호
   교환 (5)↔(6))도 이관 시 함께 반영.** + **07-27 ② (exact 이름 폐기)**: main.tex 초록·§1·
   §3.3의 "exact in-run Shapley value" 5곳과 §4의 명칭 블록은 §4 수정 세션·이관 시 함께.
