---
type: survey
title: "Flirds 결과 — Downstream (선택→성능)"
created: 2026-07-25
updated: 2026-07-25
tags: [flirds, results, downstream]
---

# Flirds 결과 — 2. Downstream

> **축**: 측정한 φ로 클라를 선택/가중해 학습하면 성능이 오르나(2차 ①). 어느 φ 정의가 학습을 잘 만드나 = **점수원 경쟁**. 분류·순서는 [[flirds-experiment-axis-map]] §2.
> **읽는 법**(공통 규약 [[flirds-results-fidelity]] §읽는 법): 최소단위 · mean±std(ddof0) · **각 위협 열에서 최고 점수원=볼드·2위=<u>밑줄</u>**(앵커 vanilla/oracle_excl/random_excl는 경쟁서 제외). 절대 acc(CNN)·EM(LLM). ● 3-seed/◐ 부분/⬚ 미실행.
> 자매 페이지: [[flirds-results-fidelity]] · [[flirds-results-detection]] · [[flirds-results-ablation]] · [[flirds-results-cost]]

앵커: **vanilla/observer**(개입 없음=바닥) · **oracle_excl**((b) 오라클로 오염 배제=천장) · **random_excl**(무작위 동수 배제=통제).

---

## 2-CNN · 점수원 경쟁 — 개입 정확도 `[본문·주무대]`

> **세팅**: FedSVCNN · cifar10/dir1 · N=100 · 10/100 · R=120 · P1 부호-게이트 · 절대 test acc · seed{0,1,2}. **online**=배포 중 게이팅(burn-in→probation) · **retrain**=관찰자 최종부호로 kept 확정 후 init부터 재학습. label-flip은 @0.70. 평균=오염 3종(frzero·grad-noise·lf@0.70; frrand는 retrain이 seed0¹이라 대칭 위해 제외).

**online (배포 게이팅)** (● 3-seed)

| arm | clean | frzero | frrand | grad-noise | lf@0.70 | 평균(오염3) |
|---|---|---|---|---|---|---|
| vanilla (바닥) | 0.6389±0.0043 | 0.5879±0.0024 | 0.5876±0.0035 | 0.2436±0.0181 | 0.5247±0.0236 | 0.4521 |
| oracle_excl (천장) |  | 0.6203±0.0023 | 0.6195±0.0023 | 0.6203±0.0023 | 0.6236±0.0025 | 0.6214 |
| random_excl (무작위) |  | 0.5838±0.0165 | 0.5839±0.0158 | 0.2590±0.0170 | 0.5018±0.0497 | 0.4482 |
| Flirds | 0.6315±0.0061 | <u>0.6148±0.0002</u> | 0.5895±0.0065 | 0.5668±0.0213 | 0.5712±0.0069 | <u>0.5843</u> |
| Flirds-1st | <u>0.6384±0.0014</u> | **0.6216±0.0031** | <u>0.6125±0.0046</u> | 0.2479±0.0108 | <u>0.5717±0.0066</u> | 0.4804 |
| loss-heur | 0.6264±0.0034 | 0.6114±0.0099 | 0.6015±0.0078 | <u>0.5981±0.0079</u> | 0.5670±0.0219 | **0.5922** |
| FedIF | **0.6386±0.0008** | 0.6143±0.0096 | **0.6130±0.0049** | 0.2479±0.0108 | **0.5728±0.0071** | 0.4783 |
| GTG | 0.6051±0.0061 | 0.3915±0.0066 | 0.3902±0.0067 | 0.5972±0.0043 | 0.5479±0.0094 | 0.5122 |
| FedSV | 0.5982±0.0063 | 0.3966±0.0092 | 0.3974±0.0097 | 0.5972±0.0050 | 0.5164±0.0068 | 0.5034 |
| ComFedSV | 0.5963±0.0048 | 0.3918±0.0085 | 0.3956±0.0101 | 0.5871±0.0044 | 0.5152±0.0227 | 0.4981 |
| ShapleyFL | 0.6045±0.0043 | 0.4020±0.0175 | 0.4018±0.0041 | **0.6115±0.0085** | 0.5278±0.0135 | 0.5138 |

**retrain (관찰자 최종부호 → init 재학습)** (● 3-seed · frrand¹만 ◐ seed0)

| arm | clean | frzero | frrand¹ | grad-noise | lf@0.70 | 평균(오염3) |
|---|---|---|---|---|---|---|
| vanilla (바닥) | 0.6389±0.0043 | 0.5879±0.0024 | 0.5876±0.0035 | 0.2436±0.0181 | 0.5247±0.0236 | 0.4521 |
| oracle_excl (천장) |  | 0.6203±0.0023 | 0.6195±0.0023 | 0.6203±0.0023 | 0.6236±0.0025 | 0.6214 |
| Flirds | 0.6277±0.0079 | 0.6063±0.0059 | 0.5925±0.0000 | 0.6065±0.0027 | 0.6192±0.0066 | **0.6107** |
| Flirds-1st | <u>0.6386±0.0043</u> | **0.6252±0.0019** | <u>0.6034±0.0000</u> | 0.2436±0.0181 | **0.6236±0.0025** | 0.4975 |
| loss-heur | 0.6293±0.0041 | 0.6125±0.0047 | 0.5940±0.0000 | 0.4518±0.0401 | 0.6205±0.0063 | 0.5616 |
| FedIF | **0.6417±0.0026** | <u>0.6252±0.0019</u> | **0.6102±0.0000** | 0.2436±0.0181 | <u>0.6217±0.0048</u> | 0.4968 |
| GTG | 0.6265±0.0070 | 0.5158±0.0038 | 0.5164±0.0000 | **0.6203±0.0023** | 0.5991±0.0120 | <u>0.5784</u> |
| FedSV | 0.6166±0.0078 | 0.5140±0.0066 | 0.5164±0.0000 | <u>0.6203±0.0023</u> | 0.5904±0.0082 | 0.5749 |
| ComFedSV | 0.6232±0.0100 | 0.5200±0.0125 | 0.5258±0.0000 | 0.6203±0.0023 | 0.5921±0.0083 | 0.5775 |
| ShapleyFL | 0.6223±0.0096 | 0.5113±0.0134 | 0.4925±0.0000 | 0.6203±0.0023 | 0.6028±0.0055 | 0.5781 |

> ¹ retrain frrand = seed0 파일럿(std 0.0000). ² **읽기**: **grad-noise를 잡는 유일한 estimator = Flirds**(online .5668/retrain .6065; 1차-계열 flirds1st·fedif .244~.248 = vanilla 수준 실명, loss-heur 부분 .598/.452) = 2차항 존재 이유의 다운스트림 재현. **frzero에서 exact-0 계열 생존**(flirds1st·lossheur·fedif .61~.62 ≈ 천장) vs **renorm 붕괴**(gtg·fedsv·comfedsv·shapleyfl .39~.40 online / .51~.52 retrain — free-rider 못 잡고 clean만 오배제). retrain flirds **평균 0.6107 최고**. 정직 보고: **clean 오발화 flirds −0.7pt**(개입 없어야 할 곳서 소폭 감점). **출처**: `runs/track_h/analysis/cnn_competition.csv`(`make_analysis.py`).

### fmnist·iid 경쟁 확장 `[후보]` — ⬚ 미실행

> observer(obsf seed0) 궤적만 존재 → 개입 arm 0개. 채움 = 8점수원 × {online,retrain} × 6위협 × 3seed (W-fm). 예시 골격(online):

| arm | clean | frzero | frrand | grad-noise | lf@0.70 | 평균 |
|---|---|---|---|---|---|---|
| vanilla |  |  |  |  |  |  |
| oracle_excl |  |  |  |  |  |  |
| Flirds |  |  |  |  |  |  |
| Flirds-1st |  |  |  |  |  |  |
| loss-heur |  |  |  |  |  |  |
| FedIF |  |  |  |  |  |  |
| GTG |  |  |  |  |  |  |
| FedSV |  |  |  |  |  |  |
| ComFedSV |  |  |  |  |  |  |
| ShapleyFL |  |  |  |  |  |  |

### 완전참여·동적재추첨·신뢰게이트 확증 `[부록E]` ● 3-seed

> Scale 100/100 완전참여(비용선형 확증)·Dyn 매라운드 오염 재추첨(신호파괴 한계)·P5 신뢰게이트. **P5-hard retrain(csign) = 오염-평균 0.6207 ≈ oracle_excl 0.6214**(P1 retrain 0.6107 대비 +0.010). 상세 수치는 구 카탈로그 §4.8(git 이력). (스코프상 보조 확증.)

### φ 부호-게이팅 그리드 `[제외]`

> sign/z/V2w/V3 게이트 skew 레짐 완주(144셀). 점수원 경쟁과 내용 중복이라 표 미게재(rundir·`track_g/analysis` 존속).

---

## 2-LLM

### 주무대 정확도 개입 (R4 GSM8K EM) `[본문·주무대]`

> **세팅**: Llama-3.2-1B-Instruct LoRA r16/α32 · N=50 · 5/50 · R=200 · GSM8K test 1,119 EM · seed{0,1,2}(noisy·frzero) / clean ◐ seed0. renorm 4종=L4 ⬚ · online 7방법=L11 ⬚ · frrand·strmain 열=신규 ⬚.

**retrain (T2 부호-게이트, 절대 EM)** (● noisy·frzero 3-seed)

| arm | clean ◐ | noisy(swap@.7) | frzero | frrand | strmain |
|---|---|---|---|---|---|
| observer (바닥) | 0.3727±0.0000 | 0.3274±0.0057 | 0.3560±0.0129 |  |  |
| oracle_excl (천장) | – | 0.3625±0.0099 | 0.3625±0.0099 |  |  |
| random_excl (무작위) | – | 0.3280±0.0048 | 0.3476±0.0146 |  |  |
| Flirds | 0.3727±0.0000 | 0.3479±0.0030 | **0.3625±0.0099** |  |  |
| Flirds-1st | 0.3727±0.0000 | 0.3458±0.0015 | 0.3625±0.0099 |  |  |
| loss-heur | 0.3727±0.0000 | **0.3497±0.0055** | 0.3625±0.0099 |  |  |
| FedIF | 0.3727±0.0000 | <u>0.3491±0.0033</u> | 0.3625±0.0099 |  |  |

**online (배포 게이팅 gate_v2, 절대 EM)** (● noisy·frzero 3-seed · online 7방법 ⬚)

| arm | clean ◐ | noisy(swap@.7) | frzero |
|---|---|---|---|
| observer (바닥) | 0.3727±0.0000 | 0.3274±0.0057 | 0.3560±0.0129 |
| oracle_excl (천장) | – | 0.3625±0.0099 | 0.3625±0.0099 |
| random_excl (무작위) | – | 0.3280±0.0048 | 0.3476±0.0146 |
| Flirds (gate v2) | 0.3664±0.0000 | **0.3479±0.0044** | 0.3566±0.0088 |
| Flirds-1st |  |  |  |
| loss-heur |  |  |  |
| FedIF |  |  |  |

> **읽기**: noisy — 4 estimator가 vanilla .3274 대비 회수(Flirds +.0206·loss-heur +.0223·FedIF +.0217·1st +.0184) vs random_excl +.0006(천장 oracle_excl +.0351). **순위정보의 가치** = vs-무작위 +2pt. frzero — 4 estimator 전부 **.3625 = oracle_excl 동값**(free-rider 만장일치 배제 → kept=oracle 집합). **online vs retrain**(Flirds): noisy 동급(둘 다 .3479), frzero online .3566<retrain .3625(배포 게이팅 burn-in 지연), **clean online −0.63pt**(probation 오배제) vs retrain 무해(kept=전원). ⚠ EM 노이즈 바닥 ±0.5pt라 안전한 주장은 vs-random·vs-1차까지(flirds↔loss-heur 0.36pt는 미분리). **출처**: `runs/track_h/analysis/llm_competition.csv`(regime=gsm50k5).

### 표준 개입 무해성 (clean do-no-harm) `[본문·근거]` ● 3-seed

> Llama-3.2-1B·N=20·alpaca IID. clean-IID에서 φ-가중/선택 arm이 성능을 **안 깎음**(MMLU·ROUGE parity; 게이트·V3 arm max Δ최종손실 0.00056). 오염이 없으면 개입 이득도 원리적 부재(효과 < 표본 SE) = 기대대로 do-no-harm. 수치 상세는 구 카탈로그 §3.2.1(git 이력).

### 온라인 φ-게이팅 + 재학습 회수 `[제외 표 / R4로 흡수]` ● 3-seed(일부 seed0)

> silo5·iid5 부호-게이팅: **frzero 자동배제 recovery 1.000 정확**(오배제 0쌍, oracle_excl과 최종손실 소수4자리 동일 = φ=0 공리의 배포 활용)·clean 게이트 무발화(max|Δ| 0.00056)·CNN grad-noise 회수 0.86~0.94. 독립 표는 제외(결과는 위 R4 online leg로 흡수). 상세는 구 카탈로그 §3.2.2(git 이력).

---

## 출처·재생성

- CNN: `runs/track_h/analysis/cnn_competition.csv`(dataset=cifar10·partition=dir1; online arm=`<m>_gate_v2`·retrain=`t2_sign_<m>`) → (arm,threat) seed 평균.
- LLM: `runs/track_h/analysis/llm_competition.csv`(regime=gsm50k5; gsm8k_em).
- **⬚ 미실행**: fmnist·iid 경쟁(W-fm) · R4 renorm 4종(L4)·online 7방법(L11)·frrand·strmain 열.
- 축 지도: [[flirds-experiment-axis-map]] (구 카탈로그 §3.2 = git 이력)
