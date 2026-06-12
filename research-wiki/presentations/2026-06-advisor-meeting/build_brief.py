#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flirds 교수님 미팅 발표자료 — 요약판(brief) 빌더

상세판(build.py)과 같은 표지+10장 구조에서 부연 설명을 걷어내고
핵심 수치·키워드만 남긴 한눈에 들어오는 버전. 상세판 파일은 건드리지 않는다.

사용:
    python build_brief.py              # HTML + PDF + 스크린샷 + overflow 리포트
    python build_brief.py --html-only
    python build_brief.py --no-shots

산출 (이 디렉토리):
    flirds-advisor-2026-06-brief.html / .pdf
    screenshots_brief/s00.png ...

────────────────────────────────────────────────────────────────────
2026-06-12 갱신: silo5 재실행본·α-sweep 12셀·3B 4-threat 반영 (rundirs, 8d364cc).
⚠ TODO-UPDATE — 잔여 5셀 확정 시 갱신 (build.py와 동일):
  [1] S3/S6 anchor 3셀   [2] S5/S6 device100 poison 2셀   [3] S3/S6 7B
────────────────────────────────────────────────────────────────────
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_HTML = HERE / "flirds-advisor-2026-06-brief.html"
OUT_PDF = HERE / "flirds-advisor-2026-06-brief.pdf"
SHOT_DIR = HERE / "screenshots_brief"

TITLE = "Flirds 진행 보고 (요약판)"
DATE = "2026-06-12"

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
  --ink: #1a1a1a; --sub: #555; --line: #c8c8c8; --faint: #888;
  --accent: #17508c;            /* 유일한 강조색 */
  --ph: #999;                   /* placeholder(확정 대기) */
}
html, body { background: #e9e9e9; }
body { font-family: "Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
       color: var(--ink); }
.slide { width: 1280px; height: 720px; background: #fff; display: none;
         flex-direction: column; padding: 48px 60px 42px; position: relative; overflow: hidden; }
.slide.active { display: flex; }
@media screen { .slide.active { margin: 24px auto; box-shadow: 0 1px 6px rgba(0,0,0,.25); } }

.kicker { font-size: 15px; color: var(--accent); letter-spacing: .04em; margin-bottom: 4px; }
h2 { font-size: 30px; font-weight: 600; margin-bottom: 12px; }
.rule { border: none; border-top: 1px solid var(--line); margin: 0 0 20px; }
.foot { position: absolute; left: 60px; right: 60px; bottom: 14px; display: flex;
        justify-content: space-between; font-size: 11.5px; color: var(--faint); }

.body { flex: 1; min-height: 0; font-size: 20px; line-height: 1.6;
        word-break: keep-all; overflow-wrap: break-word; }
.body.dense { font-size: 17.5px; line-height: 1.55; }
ul { list-style: none; }
li { padding-left: 16px; position: relative; margin-bottom: 10px; }
li::before { content: "–"; position: absolute; left: 0; color: var(--sub); }
b { font-weight: 600; }
.lbl { font-weight: 600; }
.muted { color: var(--sub); }
.ph { color: var(--ph); font-style: italic; }
.sm { font-size: .82em; color: var(--sub); }      /* 설정 병기용 축소 표기 */

.bt { font-size: 16px; font-weight: 700; color: var(--accent); margin: 16px 0 7px; }
.bt:first-child { margin-top: 0; }
.cols { display: flex; gap: 40px; }
.cols > div { flex: 1; min-width: 0; }

table { border-collapse: collapse; width: 100%; font-size: 17px; line-height: 1.45; }
th, td { border: 1px solid var(--line); padding: 8px 12px; text-align: left; vertical-align: top; }
th { font-weight: 600; background: #f4f4f4; }
table.dense { font-size: 15.5px; }
table.dense th, table.dense td { padding: 6px 10px; }
sup, sub { line-height: 0; }   /* 위첨자가 줄박스를 키워 불릿 정렬이 틀어지는 것 방지 */

.formula { border: 1px solid var(--line); background: #fafafa; padding: 16px 18px;
           font-size: 23px; text-align: center; margin: 12px 0;
           font-family: "Cambria Math", Cambria, "Times New Roman", serif; }
.formula .note { display: block; font-size: 14px; color: var(--sub); margin-top: 8px;
                 font-family: "Segoe UI", "Malgun Gothic", sans-serif; }

.box { border: 1px solid var(--ink); padding: 12px 16px; font-size: 18px; margin-top: 14px; }

.cover { justify-content: center; }
.cover h1 { font-size: 44px; font-weight: 700; margin-bottom: 10px; }
.cover .subtitle { font-size: 21px; color: var(--sub); margin-bottom: 6px; }
.cover .meta { font-size: 16px; color: var(--sub); margin-bottom: 44px; }
.cover .conv { border: 1px solid var(--line); padding: 14px 18px; font-size: 15px;
               line-height: 1.6; max-width: 880px; }
.cover .conv .t { font-weight: 700; margin-bottom: 4px; }

@media print {
  html, body { background: #fff; }
  .slide { display: flex !important; page-break-after: always; break-after: page; margin: 0; box-shadow: none; }
}
@page { size: 1280px 720px; margin: 0; }
"""

NAV_JS = """
const slides = Array.from(document.querySelectorAll('.slide'));
let cur = 0;
function show(i) {
  cur = Math.max(0, Math.min(slides.length - 1, i));
  slides.forEach((s, j) => s.classList.toggle('active', j === cur));
  history.replaceState(null, '', '#' + cur);
}
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') show(cur + 1);
  else if (e.key === 'ArrowLeft' || e.key === 'PageUp') show(cur - 1);
  else if (e.key === 'Home') show(0);
  else if (e.key === 'End') show(slides.length - 1);
});
show(parseInt((location.hash || '#0').slice(1), 10) || 0);
"""


def foot(n: int, total: int) -> str:
    return (f'<div class="foot"><span>{TITLE} · {DATE}</span>'
            f'<span>{n} / {total}</span></div>')


SLIDES = []

# S0 ── 표지
SLIDES.append(('cover', """
<h1>Flirds 진행 보고</h1>
<div class="subtitle">Client-level In-Run Data Shapley for Federated LLM Fine-tuning</div>
<div class="meta">2026-06-12 · Yonghee Choi · 요약판</div>
<div class="conv">
  <div class="t">표기 규약</div>
  <span class="lbl">ⓐ</span> 구현+smoke (값 coarse) · <span class="lbl">ⓑ</span> 실측 (설정 병기) · <span class="lbl">ⓒ</span> 설계만, 미실행<br>
  <span class="ph">기울임 회색</span> = 잔여 셀(device100 poison · α=0.5 anchor · 7B) 확정 대기. 상세 근거는 상세판 데크 참조.
</div>
"""))

# S1 ── 문제 + 빈틈
SLIDES.append(("§1 Recap", "문제와 선행 연구의 빈 부분", "dense", """
<ul>
  <li>server가 보는 것은 Δw<sub>k</sub>, n<sub>k</sub>뿐 → <b>post-hoc · 재학습 0 · 추가 통신 0</b>으로 client 기여도 φ<sub>k</sub></li>
  <li>용도: ① corrupt/free-rider 식별 &nbsp;② selection(상위 client 재학습)</li>
</ul>
<table class="dense" style="margin-top:10px">
<tr><th style="width:34%">선행</th><th>한계</th></tr>
<tr><td>retrain Shapley (G&amp;Z '19)</td><td>재학습 비용 — <b>126분</b> <span class="sm">(1B N=5 fp32 ⓑ)</span>, N=10 ≈ 2–5일 <span class="sm">(추정, ⓒ)</span></td></tr>
<tr><td>FL-SV 계열 ('20–'23)</td><td>coalition/MC 비용 · 전부 CNN</td></tr>
<tr><td>Influence Functions ('17)</td><td>iHVP(H<sup>−1</sup>v) 역산 — 불안정·고비용</td></tr>
<tr><td>IRDS</td><td>centralized · per-step · sample-level</td></tr>
<tr><td>Ripple (AAAI '26)</td><td>sample-level · CNN · 2차는 cross-round 전파(클라 상호작용 항 없음)</td></tr>
<tr><td>FedIF ('25)</td><td>순수 1차 · aggregation 변경</td></tr>
<tr><td>FedTSV (ECC '26)</td><td>0차 기하 · aggregation 물건</td></tr>
</table>
<p style="margin-top:12px"><b>빈 교집합:</b> client-level · in-run · closed-form 1+2차 · HVP 상호작용항 · 통신 0 · LoRA/LLM <span class="sm">(서술적 서베이 근거)</span></p>
"""))

# S2 ── 알고리즘
SLIDES.append(("§1 Recap", "알고리즘", "", """
<div class="formula">
  φ<sub>k</sub> = Σ<sub>r</sub> p<sub>k</sub><sup>r</sup> [ ⟨g<sup>r</sup>, Δw<sub>k</sub>⟩ + ½ ⟨Δw<sub>k</sub>, u<sup>r</sup>⟩ ], &nbsp; u<sup>r</sup> = H<sup>r</sup> ΔW<sup>r</sup>
  <span class="note">frozen FedAvg 궤적만 입력 · g<sup>r</sup> = ∇ val-loss · val-loss를 내리면 φ &lt; 0</span>
</div>
<div class="cols" style="margin-top:14px">
<div>
  <ul>
    <li><b>round당 HVP 1회</b> — forward H·v, H<sup>−1</sup> 불사용, true Hessian</li>
    <li>비용 <b>N-독립</b> ↔ in-run oracle 2<sup>N</sup>·R forward</li>
    <li>Flirds-1st(2차 off): 35s — coalition 계열(~530s) 대비 ≈15×</li>
  </ul>
</div>
<div>
  <ul>
    <li><b>free-rider φ = 0 exact</b> <span class="sm">(구조적; 실측 매 seed 0 ⓑ)</span></li>
    <li>2차의 근거: FL per-round Δw 큼 — CNN 0.962 &gt; 0.924 <span class="sm">ⓑ</span>; momentum서 역전 → plain SGD 고정</li>
    <li>제약: fp32 · eager attention · LoRA-subspace</li>
  </ul>
</div>
</div>
"""))

# S3 ── 실험 프레임 표
# TODO-UPDATE [1]: FedIF 수치 확정 시 '기존 대비' 행 placeholder 교체
SLIDES.append(("§2 실험 설계", "실험 프레임 — 질문 → 실험 → 상태", "", """
<table>
<tr><th style="width:30%">질문</th><th style="width:40%">실험</th><th>상태</th></tr>
<tr><td>Shapley 계산이 옳은가</td><td>dual-oracle 삼중 비교</td><td><span class="lbl">ⓑ</span> +1.000 <span class="sm">(1B N=5 fp32 3-seed)</span></td></tr>
<tr><td>기존 방법 대비</td><td>같은 궤적 9-method (+FedIF)</td><td><span class="lbl">ⓑ</span> 3-seed (FedIF 포함)</td></tr>
<tr><td>위협을 식별하나</td><td>2 regime × 4 threat + detector 4종</td><td><span class="lbl">ⓑ</span> silo5·3B·device100 · <span class="ph">poison 2셀 잔여</span></td></tr>
<tr><td>N=100서 성립하나</td><td>per-round exact 분해 + α-sweep</td><td>sweep 12셀 <span class="lbl">ⓑ</span> · <span class="ph">anchor 3셀 잔여</span></td></tr>
<tr><td>실용 가치</td><td>selection run</td><td><span class="lbl">ⓑ</span> 3-seed</td></tr>
<tr><td>scale 유지</td><td>1B → 3B → 7B</td><td>1B <span class="lbl">ⓑ</span> · 3B 4-threat <span class="lbl">ⓑ</span>(1-seed) · 7B <span class="lbl">ⓒ</span></td></tr>
<tr><td>무변별 해소</td><td>N=10 retrain oracle</td><td><span class="lbl">ⓒ</span> 연기 — 논의 ②</td></tr>
</table>
"""))

# S4 ── 방법 검증
# TODO-UPDATE [2]: FedIF 수치 확정 시 교체
SLIDES.append(("§2 실험 설계", "방법 검증", "dense", """
<div class="bt">dual-oracle — 검증은 같은 게임(val-loss)으로</div>
<ul>
  <li>retrain(val-loss) = in-run oracle = estimator — <b>Spearman +1.000</b> <span class="sm">ⓑ (1B N=5 fp32 3-seed, lr 2종)</span> · 3B<span class="sm">(1-seed)</span>: estimator +1.000 유지, retrain↔in-run +0.900</li>
  <li>retrain-ROUGE는 발산(+0.4 / −0.9) = 다른 게임 → val-loss가 옳은 검증 지표</li>
</ul>
<div class="bt">baseline 9종 — 같은 frozen 궤적</div>
<ul>
  <li>noisy/FR: <b>전 방법 +1.000 동률</b> <span class="sm">(N=5 near-additive)</span> → <b>"같은 랭킹을 더 싸게"</b></li>
  <li>runtime: <b>35s</b>(1st·FedIF) · <b>107s</b>(Flirds) · 164s(loss-heur) · ~530s(GTG·FedSV·Banzhaf·ShapleyFL·oracle) · ~4515s(Ripple, 별도 세션)</li>
  <li>free-rider φ=0 exact: Flirds·oracle·Banzhaf·loss-heur <span class="sm">(GTG·FedSV는 ≠0)</span> · FedIF(정규화 1차): noisy/FR AUROC 1.0 <span class="sm">ⓑ(silo5 재실행 3-seed)</span></li>
</ul>
<div class="bt">selection — 검출을 성능으로</div>
<ul>
  <li>noisy+FR 정확히 드롭 <span class="sm">ⓑ (3-seed, 양 lr 일관)</span> · vs full 항상 우세 · vs random cross-seed 우세 <span class="sm">(noisy AUROC lr 반전 있어도 결론 불변)</span></li>
</ul>
"""))

# S5 ── threat matrix
# TODO-UPDATE [3]: silo5 재실행본 수치 확정 시 교체
SLIDES.append(("§2 실험 설계", "위협 매트릭스 — 2 regime × 4 threat", "dense", """
<table class="dense" style="margin-bottom:12px">
<tr><th style="width:16%">위협</th><th>정의</th><th style="width:26%">matched detector</th></tr>
<tr><td>noisy</td><td>answer_swap — 정직하지만 나쁜 데이터</td><td>FedDQC</td></tr>
<tr><td>free-rider <span class="sm">(2모드)</span></td><td>zero: Δw=0 · random: benign-std</td><td>STD-DAGMM · FLTrust</td></tr>
<tr><td>poison</td><td>backdoor (trigger + γ-scaled replacement)</td><td>FLDetector · FLTrust</td></tr>
</table>
<p class="sm" style="margin-bottom:8px">라벨은 AUROC 채점 key — method 입력 아님(순환 아님). 결과: silo5 3-seed 재실행본 ⓑ (FedIF 포함, 8d364cc)</p>
<ul>
  <li>noisy / free-rider: <b>전 방법 AUROC 1.0</b> <span class="sm">(Sp 예외: FedIF 0.90–0.93 · FedSV FR-random 0.933)</span></li>
  <li><b>poison(ASR=1.00) = 동률 첫 붕괴</b> — <b>raw 1차만 0.000 회피</b> · 2차 0.917±0.118 · 정규화 1차(FedIF·FLTrust)·exact 계열 <b>1.0</b><br>
  <span class="sm">2차는 이전 동일 config run서 0.417±0.425(run간 분산, 원인 규명 중) · 3B(1-seed)는 1차·2차 모두 0.000 회피 · exact 계열 = oracle·loss-heur·Banzhaf·ShapleyFL</span></li>
  <li>detector: poison — FLDetector·FLTrust·FedDQC <b>1.0</b>, STD-DAGMM 0.75 · FR-zero STD-DAGMM 0.250 <b>실패</b> · noisy FedDQC 0.917</li>
</ul>
<div class="box">Flirds = noisy + free-rider는 잡는다. backdoor는 raw 1차 회피·2차 불안정 → 정규화 1차·secant·matched detector가 보완 <span class="sm">(채택은 논의 ①)</span></div>
"""))

# S6 ── scale-up
# TODO-UPDATE [4]: α-sweep 결과 확정 시 교체  [5]: 3B/7B tier 완료 시 라벨 갱신
SLIDES.append(("§2 실험 설계", "scale-up", "", """
<div class="bt">cross-device N=100</div>
<ul>
  <li>per-round exact 분해 ≡ 2<sup>N</sup> <span class="sm">(Δφ≈3e-16 ⓑ)</span> · oracle ~11h/4-GPU → <b>α=0.5 anchor만</b>, 나머지는 Flirds proxy-truth</li>
  <li>α-sweep 12셀 <span class="sm">ⓑ(3-seed)</span>: free-rider — Flirds 계열 전 α <b>AUROC 1.000</b> · noisy — 0.57–0.77 약함 <span class="sm">(matched FedDQC 0.96–1.0이 보완)</span></li>
  <li>anchor: Flirds vs oracle <b>+1.000</b> <span class="sm">ⓑ (1B, 1-seed smoke)</span> · φ=0 exact 유지 · <span class="ph">anchor 3셀 + poison 2셀 잔여</span></li>
</ul>
<div class="bt">model ladder</div>
<ul>
  <li>1B <span class="lbl">ⓑ</span> → 3B 4-threat <span class="lbl">ⓑ</span><span class="sm">(1-seed)</span>: noisy/FR valuation 전부 1.0 <span class="sm">(detector 예외: STD-DAGMM 0.25/0.00, FedDQC 0.75)</span>, <b>poison은 1차·2차 모두 0.000 회피</b> → 7B <span class="lbl">ⓒ</span> · 전부 fp32</li>
</ul>
<div class="bt">N=10 retrain oracle</div>
<ul>
  <li>재학습 <b>64×</b> ≈ 2–5일/1-GPU <span class="sm">(추정)</span> → <span class="lbl">ⓒ</span> 연기 — 투자 판단(논의 ②)</li>
</ul>
"""))

# S7 ── 한계
SLIDES.append(("§3 한계", "남은 문제", "dense", """
<div class="cols">
<div>
  <div class="bt">방법 내재</div>
  <ul>
    <li><b>Taylor 절단</b> — clean-preserving backdoor 회피(tangent 맹점)</li>
    <li><b>plain SGD 강제</b> — AdamW 비호환(최대 일반성 제약)</li>
    <li><b>fp32 강제</b> — 신호 &lt; bf16 정밀도</li>
    <li><b>적용 범위</b> — eager · LoRA-subspace · vanilla FedAvg 한정</li>
    <li><b>단일 val-loss 게임</b> — val 오염 미고려 <span class="lbl">ⓒ</span></li>
  </ul>
</div>
<div>
  <div class="bt">증거력</div>
  <ul>
    <li><b>N=5 무변별</b> — 유일한 분리(poison)가 Flirds에 불리</li>
    <li><b>noisy AUROC lr 반전</b> — lr 미확정(논의 ③)</li>
    <li><b>N=100 noisy 약함</b> — 0.57–0.77 <span class="sm">(FedDQC 보완)</span></li>
    <li><b>1-seed 항목</b> — 3B tier · anchor 미실행</li>
    <li><b>proxy-truth 순환</b> — off-anchor truth = Flirds 자신</li>
    <li><b>poison 인위성</b> — 별도 config서만 ASR&gt;0</li>
    <li><b>최약 공격자만</b> — stealthy·DBA·advanced FR 제외</li>
    <li><b>detector 개조 confound</b> — 원형 ablation 부재</li>
  </ul>
</div>
</div>
"""))

# S8 ── 계획 ①
SLIDES.append(("§4 계획", "① 진행 중 + main 트랙 보강", "", """
<ul>
  <li><b>real grid</b> — 20/25셀 완료 <span class="sm">(8d364cc)</span> · 잔여: device100 poison 2 + α=0.5 anchor 3 + 7B</li>
  <li><b>poison 2차 거동 규명</b> — 1B run 분산 + 3B 회피 · 기실행 궤적 재분석(비용 ~0) — <b>최우선</b></li>
  <li><b>two-sided |φ| 산출물화</b> · orientation·lr 양쪽 보고 규약</li>
  <li><b>2차항 결정 실험</b> — PGD/direction-aligned poison <span class="sm">(FedIF 공인 blind spot)</span></li>
  <li><b>advanced free-rider</b>(Lin II/III) · <b>원형 ablation</b> + bootstrap CI</li>
  <li><b>N=10 retrain oracle</b> — 투자 판단(논의 ②)</li>
</ul>
"""))

# S9 ── Track C/D
SLIDES.append(("§4 계획", "② 추가 실험 Track C/D — 표준 세팅 비교 (설계 확정 ⓒ)", "dense", """
<p style="margin-bottom:12px">동기: main 실험은 선행과 직접 비교가 어렵다 → 선행 다수의 <b>일반 학습 세팅</b>으로 비교 트랙 추가 <span class="sm">(선행 13편 프로토콜 조사 기반)</span></p>
<table class="dense">
<tr><th style="width:10%">트랙</th><th style="width:34%">세팅</th><th>핵심</th></tr>
<tr><td><b>C1</b></td><td>CNN cross-silo, N=10 full</td><td>fidelity &amp; cost — <b>(a)+(b) 듀얼 oracle</b>, GTG 5-시나리오, 9-method + Ripple</td></tr>
<tr><td><b>C2</b></td><td>CNN cross-device, N=100 <span class="sm">(메인)</span></td><td>일반 성능 — <b>개입 3종 × 3 파티션 × 4 위협</b><br><span class="sm">개입: 가중집계(곱셈형 w∝n·s 메인) · selection · bottom-q%</span></td></tr>
<tr><td><b>C3</b></td><td>(비용 0)</td><td>cross-seed 안정성 — Banzhaf·Volatility 처방 응답</td></tr>
<tr><td><b>D</b></td><td>LLM, Alpaca-GPT4 20k IID N=5</td><td>표준 세팅 — answer_swap 50% → 필터링 재학습 → <b>MMLU</b> <span class="sm">(전부 API-free; 7B=Llama-2)</span></td></tr>
</table>
<p class="sm" style="margin-top:12px">조사에서 본 공백: fidelity+학습개선 동시 커버 없음 · (a)+(b) 듀얼 oracle 없음 · LLM-scale FL valuation 직접 경쟁자 없음</p>
"""))

# S10 ── 논의
SLIDES.append(("§5 논의", "결정이 필요한 항목", "", """
<table>
<tr><th style="width:30%">항목</th><th>선택지</th></tr>
<tr><td>① threat-matrix headline</td><td>"noisy+FR은 잡고 backdoor엔 회피(detector 보완)" 채택 여부</td></tr>
<tr><td>② N=10 retrain oracle</td><td>2–5일/1-GPU vs 샤딩 구축(11–22h) vs detection-only 차선</td></tr>
<tr><td>③ real grid lr</td><td>2e-5 vs 1e-3/3e-3 — noisy 결론이 갈림</td></tr>
<tr><td>④ Ripple 차별화 문구</td><td>"2차 없음" → <b>"다른 종류의 2차"</b>로 교체 승인</td></tr>
</table>
"""))


def build_html() -> str:
    total = len(SLIDES)
    parts = []
    for i, s in enumerate(SLIDES):
        if s[0] == 'cover':
            parts.append(f'<section class="slide cover" id="s{i}">{s[1]}{foot(i + 1, total)}</section>')
        else:
            kicker, title, dense, body = s
            cls = ' dense' if dense else ''
            parts.append(
                f'<section class="slide" id="s{i}">'
                f'<div class="kicker">{kicker}</div><h2>{title}</h2><hr class="rule">'
                f'<div class="body{cls}">{body}</div>{foot(i + 1, total)}</section>')
    return (f'<!DOCTYPE html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            f'<title>{TITLE} · {DATE}</title>\n<style>{CSS}</style>\n</head>\n<body>\n'
            + '\n'.join(parts)
            + f'\n<script>{NAV_JS}</script>\n</body>\n</html>\n')


def render(shots: bool = True) -> None:
    url = OUT_HTML.resolve().as_uri()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('[warn] playwright 미설치 → Edge headless로 PDF만 시도')
        _edge_pdf(url)
        return

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1280, 'height': 720})
        page.goto(url)
        page.emulate_media(media='print')
        page.pdf(path=str(OUT_PDF), width='1280px', height='720px',
                 print_background=True, prefer_css_page_size=True)
        page.emulate_media(media='screen')
        print(f'[ok] PDF → {OUT_PDF.name}')

        n = page.evaluate('slides.length')
        report = []
        if shots:
            page.add_style_tag(content='.slide.active{margin:0 !important; box-shadow:none !important}')
            SHOT_DIR.mkdir(exist_ok=True)
            for i in range(n):
                page.evaluate(f'show({i})')
                page.screenshot(path=str(SHOT_DIR / f's{i:02d}.png'))
                ov = page.evaluate("""() => {
                    const s = document.querySelector('.slide.active');
                    const r = s.getBoundingClientRect();
                    let worst = 0;
                    for (const el of s.querySelectorAll('*')) {
                        const b = el.getBoundingClientRect();
                        worst = Math.max(worst, b.bottom - r.bottom, b.right - r.right);
                    }
                    return {sw: s.scrollWidth, sh: s.scrollHeight, worst: Math.round(worst)};
                }""")
                flag = ' OVERFLOW' if (ov['sh'] > 722 or ov['sw'] > 1282 or ov['worst'] > 2) else ''
                report.append(f's{i:02d}: scroll {ov["sw"]}x{ov["sh"]}, beyond-edge {ov["worst"]}px{flag}')
            print(f'[ok] screenshots → {SHOT_DIR.name}/ ({n}장)')
            print('\n'.join('  ' + line for line in report))
        browser.close()


def _edge_pdf(url: str) -> None:
    for exe in (r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                r'C:\Program Files\Microsoft\Edge\Application\msedge.exe'):
        if Path(exe).exists():
            subprocess.run([exe, '--headless=new', '--disable-gpu',
                            f'--print-to-pdf={OUT_PDF}', '--no-pdf-header-footer', url],
                           check=True, timeout=120)
            print(f'[ok] PDF (Edge) → {OUT_PDF.name}')
            return
    print('[warn] Edge 미발견 — PDF 생략')


if __name__ == '__main__':
    OUT_HTML.write_text(build_html(), encoding='utf-8')
    print(f'[ok] HTML → {OUT_HTML.name} ({len(SLIDES)}장)')
    if '--html-only' not in sys.argv:
        render(shots='--no-shots' not in sys.argv)
