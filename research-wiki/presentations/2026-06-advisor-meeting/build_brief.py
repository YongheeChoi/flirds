#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flirds 교수님 미팅 발표자료 — 요약판(brief) 빌더 v2 (2026-06-12 재구성)

구조 (Yonghee 지정):
  S0 표지 / S1 문제 정의 + 알고리즘 / S2 선행 연구(계열 경향 표 + baseline 속성 표)
  S3 실험 구조 + 상태 / S4 결과: cross-silo / S5 결과: cross-device·3B
  S6 고려 중인 사항(novelty 방향)

사용:
    python build_brief.py              # HTML + PDF + 스크린샷 + overflow 리포트
    python build_brief.py --html-only
    python build_brief.py --no-shots

산출 (이 디렉토리):
    flirds-advisor-2026-06-brief.html / .pdf
    screenshots_brief/s00.png ...

수치 출처: runs/phase2_matrix/rundirs/*/metrics.json (8d364cc, 20/25셀) — 3-seed mean±pstdev.
────────────────────────────────────────────────────────────────────
⚠ TODO-UPDATE — 잔여 5셀 확정 시 갱신 (이 파일에서 "TODO-UPDATE" 검색):
  [1] S3 표 anchor·poison·7B 행 상태   [2] S5 각주(anchor·poison 잔여)
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
table.result { font-size: 13.5px; line-height: 1.35; }
table.result th, table.result td { padding: 4px 8px; }
table.result td.num, table.result th.num { text-align: center; }
table.result td.num { vertical-align: middle; }
.nw { white-space: nowrap; }
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
<div class="meta">2026-06-12 · Yonghee Choi</div>
<div class="conv">
  <div class="t">표기 규약</div>
  <span class="lbl">ⓐ</span> 구현+smoke (값 coarse) · <span class="lbl">ⓑ</span> 실측 (설정 병기) · <span class="lbl">ⓒ</span> 설계만, 미실행<br>
  <span class="ph">기울임 회색</span> = 잔여 셀(device100 poison · α=0.5 anchor · 7B) 확정 대기.
</div>
"""))

# S1 ── 문제 정의 + 알고리즘
SLIDES.append(("§1 Recap", "문제 정의와 알고리즘", "dense", """
<ul>
  <li><b>문제</b>: FedAvg 연합학습에서 server가 보는 것은 client별 update Δw<sub>k</sub>와 데이터 수 n<sub>k</sub>뿐(raw data 접근 불가). 이 정보만으로 <b>각 client의 데이터 기여도 φ<sub>k</sub></b>를 매기는 data valuation 문제 — 대상은 LoRA-LLM fine-tuning(1B–7B; CNN은 검증·표준비교 트랙).</li>
  <li><b>요건</b>: 학습 종료 후 post-hoc · 재학습 0 · 추가 통신 0. 용도 = corrupt/free-rider 식별 + selection(상위 client 재학습).</li>
</ul>
<div class="formula">
  φ<sub>k</sub> = Σ<sub>r</sub> p<sub>k</sub><sup>r</sup> [ ⟨g<sup>r</sup>, Δw<sub>k</sub>⟩ + ½ ⟨Δw<sub>k</sub>, u<sup>r</sup>⟩ ], &nbsp; u<sup>r</sup> = H<sup>r</sup> ΔW<sup>r</sup>
  <span class="note">frozen FedAvg 궤적만 입력 · g<sup>r</sup> = ∇ val-loss · val-loss를 내리면 φ &lt; 0 (더 가치 있음)</span>
</div>
<ul style="margin-top:14px">
  <li>server-side validation loss의 <b>1차+2차 Taylor</b>로 in-run Shapley를 closed-form 근사 — <b>round당 HVP 1회</b>(forward H·v, true Hessian), 비용 N-독립 ↔ exact oracle은 2<sup>N</sup>·R forward.</li>
  <li><b>free-rider φ = 0 exact</b>(구조적: Δw=0 → 전 내적 0) · Flirds-1st(2차 off, ~35s) = 자체 ablation.</li>
  <li>제약: fp32 · eager attention · plain SGD mom=0 · LoRA-subspace.</li>
</ul>
"""))

# S2 ── 선행 연구: 계열 경향 + baseline 속성
SLIDES.append(("§1 Recap", "선행 연구 — 계열별 경향과 비교 baseline", "dense", """
<table class="result" style="margin-bottom:10px">
<tr><th style="width:17%">계열</th><th style="width:27%">대표</th><th>경향과 남는 한계</th></tr>
<tr><td>retrain 기반 semivalue</td><td>Ghorbani &amp; Zou '19 · Data Banzhaf '23</td><td>coalition 재학습으로 utility 정의 — 정석이나 LLM-FL서 비용 불가(1B N=5 fp32 실측 126분 <span class="lbl">ⓑ</span>, N=10 <span class="nw">≈2–5일</span> 추정)</td></tr>
<tr><td>FL-SV (게임이론적 FL 평가)</td><td>FedSV '20 · GTG '21 · ComFedSV '21 · ShapleyFL '23</td><td>retrain-free(로그로 coalition 재구성)지만 MC/열거 비용이 크고, 실험은 전부 소규모 비전 벤치마크(비-LLM)</td></tr>
<tr><td>gradient/influence 기반</td><td>IF (Koh &amp; Liang '17) · TracIn → FedIF '25</td><td>미분 신호로 저렴하게 점수화 — IF는 iHVP 역산 불안정, FedIF는 순수 1차 + aggregation 변경, CNN</td></tr>
<tr><td>single-run / in-run</td><td>IRDS '24 · Ripple (AAAI '26)</td><td>한 궤적 안에서 재학습 없이 valuation — IRDS는 centralized·per-step·sample-level, Ripple은 sample-level·CNN이며 2차가 local-Hessian 시간 전파(within-round client 상호작용 항 없음)</td></tr>
<tr><td>aggregation 개입형</td><td>FedTSV (ECC '26) 등</td><td>점수로 다음 라운드 가중치를 조정 — 평가 회계가 아닌 학습 개입, 서버 추가 연산</td></tr>
</table>
<table class="result">
<tr><th style="width:13%">비교 baseline</th><th class="num">client-level</th><th class="num">in-run<br><span style="font-weight:400">(재학습 0)</span></th><th class="num">closed-form<br><span style="font-weight:400">(비 MC/열거)</span></th><th class="num">명시적 2차 항<br><span style="font-weight:400">(client-interaction)</span></th><th class="num">추가 통신 0</th><th class="num">LLM·LoRA<br>실험</th></tr>
<tr><td>GTG / FedSV / ComFedSV / ShapleyFL</td><td class="num">✓</td><td class="num">✓</td><td class="num">✗ <span class="sm">MC·열거</span></td><td class="num">✗</td><td class="num">✓</td><td class="num">✗</td></tr>
<tr><td>Data Banzhaf <span class="sm">(원논문)</span></td><td class="num">✗ <span class="sm">sample</span></td><td class="num">✗ <span class="sm">재학습</span></td><td class="num">✗ <span class="sm">MSR</span></td><td class="num">✗</td><td class="num">—</td><td class="num">✗</td></tr>
<tr><td>Ripple</td><td class="num">✗ <span class="sm">sample</span></td><td class="num">✓</td><td class="num">✗ <span class="sm">저랭크 재귀</span></td><td class="num">△ <span class="sm">시간 전파만</span></td><td class="num">✓</td><td class="num">✗</td></tr>
<tr><td>FedIF</td><td class="num">✓</td><td class="num">✓</td><td class="num">✓ <span class="sm">1차 내적</span></td><td class="num">✗</td><td class="num">✓</td><td class="num">✗</td></tr>
<tr><td><b>Flirds</b></td><td class="num">✓</td><td class="num">✓</td><td class="num">✓ <span class="sm">1+2차 Taylor</span></td><td class="num">✓ <span class="sm">HVP</span></td><td class="num">✓</td><td class="num">✓ <span class="sm">1B·3B</span> <span class="ph">(7B 예정)</span></td></tr>
</table>
<p class="sm" style="margin-top:8px">비교는 전부 같은 frozen 궤적 위 from-logs 포팅(공정비교; loss-heur는 자작 floor baseline으로 별도) — 표는 원논문 기준 속성, 비점유 주장 근거는 서술적 서베이. ✗(2차)는 명시적 항의 부재 — 열거/MC 계열은 상호작용을 비용으로 암묵 포착. ComFedSV는 round-0 전원 참여 가정(Assumption 1).</p>
"""))

# S3 ── 실험 구조 + 상태
# TODO-UPDATE [1]: anchor·poison·7B 행 — 잔여 셀 완료 시 상태 갱신
SLIDES.append(("§2 실험", "실험 구조와 현재 상태", "dense", """
<table class="result">
<tr><th style="width:21%">실험</th><th style="width:51%">구성</th><th>상태</th></tr>
<tr><td>dual-oracle 검증</td><td>retrain(val-loss) = in-run oracle = estimator 삼중 비교 — 같은 게임(val-loss)으로 방법 자체를 검증</td><td><span class="lbl">ⓑ</span> 완료 — <b>+1.000</b> <span class="sm">(1B N=5 fp32 3-seed, lr 2종)</span> · <span class="nw">3B<span class="sm">(1-seed)</span></span> est +1.000, retrain↔in-run +0.900</td></tr>
<tr><td>matrix ① cross-silo</td><td>silo5(N=5, 도메인당 1 client) × 4 threat × 3 seed — 9 valuation + FedIF + 4 detector + (b)oracle, 같은 frozen 궤적</td><td><span class="lbl">ⓑ</span> 완료 <span class="sm">(재실행본·영속화 8d364cc)</span> → 결과 4장</td></tr>
<tr><td>matrix ② cross-device sweep</td><td>device100(N=100, Dir(α) 혼합, K=10/round) × α{0, 0.01, 0.1, 5.0} × 3 threat × 3 seed — cheap suite + proxy-truth</td><td><span class="lbl">ⓑ</span> 완료 → 결과 5장</td></tr>
<tr><td>matrix ③ anchor / poison</td><td>α=0.5 anchor 3셀((b) per-round exact 동반) · device100 poison 2셀(working-backdoor config)</td><td><span class="ph">ⓒ 잔여 5셀 — 실행 대기(가장 비쌈)</span></td></tr>
<tr><td>scale ladder</td><td>1B → 3B → 7B (전부 fp32; 3B는 (b)oracle 동반 4-threat)</td><td>1B·3B <span class="lbl">ⓑ</span> 완료 <span class="sm">(3B 1-seed)</span> · 7B <span class="lbl">ⓒ</span></td></tr>
<tr><td>selection</td><td>φ 하위 드롭 후 재학습 — full / flirds-topk / random 3-arm</td><td><span class="lbl">ⓑ</span> 완료 — corrupt 드롭 일관, vs full 우세, vs random cross-seed 우세 <span class="sm">(1B N=5 3-seed, 양 lr)</span></td></tr>
<tr><td>N=10 retrain oracle</td><td>N=5 near-additive 무변별 해소용 직접 증거</td><td><span class="lbl">ⓒ</span> 연기 <span class="sm">(≈2–5일/1-GPU 추정)</span></td></tr>
<tr><td>Track C1 — CNN fidelity</td><td>MNIST/CIFAR, N=10, (a)+(b) 듀얼 oracle, GTG 5-시나리오, 9-method+Ripple</td><td><span class="lbl">ⓒ</span> 설계 확정 · 구현 착수</td></tr>
<tr><td>Track C2 — CNN 일반 성능</td><td>N=100, 개입 3종(가중집계 w∝n·s · selection · bottom-q%) × 3 파티션 × 4 위협</td><td><span class="lbl">ⓒ</span> 〃 <span class="sm">(C1→C2 stage-gate)</span></td></tr>
<tr><td>Track C3 — 안정성</td><td>cross-seed Spearman + top/bottom-k% 일관성 (비용 0)</td><td><span class="lbl">ⓒ</span> 〃</td></tr>
<tr><td>Track D — LLM 표준 세팅</td><td>Alpaca-GPT4 20k IID N=5, answer_swap 50% → 필터링 재학습 → MMLU (API-free)</td><td><span class="lbl">ⓒ</span> 〃</td></tr>
</table>
"""))

# S4 ── 결과: cross-silo
SLIDES.append(("§2 실험", "결과 ① — cross-silo (1B N=5, 4 threat × 3 seed, 재실행본 ⓑ)", "dense", """
<table class="result">
<tr><th style="width:17%">AUROC <span class="sm">↑ 높을수록 좋음</span></th><th class="num">noisy</th><th class="num">freerider-zero</th><th class="num">freerider-random</th><th class="num">poison <span style="font-weight:400">(ASR=1.00)</span></th><th class="num">runtime/seed <span class="sm">↓ 낮을수록 좋음</span></th></tr>
<tr><td>(b) in-run oracle</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">~532s</td></tr>
<tr><td><b>Flirds</b> (1+2차)</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num"><b>0.92±0.12</b></td><td class="num">107s</td></tr>
<tr><td><b>Flirds-1st</b> (1차만)</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num"><b>0.000</b></td><td class="num">35s</td></tr>
<tr><td>FedIF <span class="sm">(정규화 1차)</span></td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">35s</td></tr>
<tr><td>GTG / FedSV / ShapleyFL / Banzhaf</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">530–538s</td></tr>
<tr><td><span class="nw">loss-heur</span> <span class="sm">(secant floor)</span></td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">164s</td></tr>
<tr><td>FLDetector</td><td class="num">0.75</td><td class="num">0.75</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">~30s</td></tr>
<tr><td>STD-DAGMM</td><td class="num">0.42±0.31</td><td class="num"><b>0.25±0.20</b></td><td class="num">1.0</td><td class="num">0.75±0.20</td><td class="num">~136s</td></tr>
<tr><td>FLTrust</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">~36s</td></tr>
<tr><td>FedDQC</td><td class="num">0.92±0.12</td><td class="num">0.75</td><td class="num">0.75</td><td class="num">1.0</td><td class="num">~22s</td></tr>
</table>
<ul style="margin-top:10px">
  <li class="sm" style="color:var(--ink)">Spearman vs (b) <span class="sm">↑ +1 = oracle과 동일 랭킹</span>: noisy/FR 전 방법 <b>+1.000</b>(예외 FedIF 0.90–0.93 · FedSV FR-random 0.933) — N=5 near-additive → 주장은 "같은 랭킹을 더 싸게". poison: ShapleyFL·Banzhaf·loss-heur +1.000 · Flirds·FedIF +0.967 · GTG +0.867 · <b>FedSV +0.367</b> · Flirds-1st 0.000.</li>
  <li class="sm" style="color:var(--ink)"><b>poison = 동률의 첫 붕괴</b>: raw 1차만 완전 회피(0.000), 정규화 1차·exact 계열은 잡음. Flirds-2차는 이전 동일 config run서 0.417±0.425 — run간 분산, 원인 규명 중(단정하지 않음).</li>
</ul>
"""))

# S5 ── 결과: cross-device + 3B
# TODO-UPDATE [2]: anchor·poison 잔여 셀 완료 시 각주 갱신
SLIDES.append(("§2 실험", "결과 ② — cross-device sweep (N=100, 3-seed ⓑ) · 3B (1-seed ⓑ)", "dense", """
<div class="cols">
<div>
  <table class="result">
  <tr><th style="width:26%">AUROC <span class="sm">↑</span></th><th class="num">noisy<br>α=0</th><th class="num">noisy<br>α=0.01</th><th class="num">noisy<br>α=0.1</th><th class="num">noisy<br>α=5.0</th><th class="num">free-rider<br><span style="font-weight:400">전 α·양 모드</span></th></tr>
  <tr><td><b>Flirds</b> / 1st / <span class="nw">loss-heur</span></td><td class="num">0.77</td><td class="num">0.57</td><td class="num">0.60–0.61</td><td class="num">0.60</td><td class="num"><b>1.00</b></td></tr>
  <tr><td>FLTrust</td><td class="num">1.00</td><td class="num">0.60</td><td class="num">0.72</td><td class="num">0.99</td><td class="num">1.00</td></tr>
  <tr><td>FedIF</td><td class="num">0.97</td><td class="num">0.57</td><td class="num">0.69</td><td class="num">0.97</td><td class="num">0.98–0.99</td></tr>
  <tr><td>FedDQC <span class="sm">(matched)</span></td><td class="num"><b>0.96</b></td><td class="num"><b>1.00</b></td><td class="num"><b>1.00</b></td><td class="num"><b>1.00</b></td><td class="num">0.14–0.55</td></tr>
  <tr><td>STD-DAGMM</td><td class="num">0.86</td><td class="num">0.65</td><td class="num">0.66</td><td class="num">0.76</td><td class="num">0.51–0.96</td></tr>
  <tr><td>FLDetector</td><td class="num">0.53</td><td class="num">0.48</td><td class="num">0.53</td><td class="num">0.53</td><td class="num">0.51–0.61</td></tr>
  <tr><td>ComFedSV</td><td class="num">0.44</td><td class="num">0.42</td><td class="num">0.43</td><td class="num">0.40</td><td class="num">0.39–0.45</td></tr>
  </table>
  <p class="sm" style="margin-top:8px">free-rider는 Flirds 계열·FLTrust가 전 α에서 1.000. noisy는 cheap valuation이 0.57–0.77로 약함 — matched FedDQC가 보완. 1차=2차=loss-heur 랭킹 동일(proxy-truth Sp 1.000).</p>
</div>
<div>
  <table class="result">
  <tr><th style="width:30%">AUROC (3B) <span class="sm">↑</span></th><th class="num">noisy</th><th class="num">FR-zero</th><th class="num">FR-rand</th><th class="num">poison</th></tr>
  <tr><td>(b) oracle</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td></tr>
  <tr><td><b>Flirds</b></td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num"><b>0.000</b></td></tr>
  <tr><td><b>Flirds-1st</b></td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num"><b>0.000</b></td></tr>
  <tr><td>FedIF</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td></tr>
  <tr><td>loss-heur</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td></tr>
  <tr><td>FLDetector / FLTrust</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td><td class="num">1.0</td></tr>
  <tr><td>FedDQC</td><td class="num">1.0</td><td class="num">0.75</td><td class="num">0.75</td><td class="num">1.0</td></tr>
  <tr><td>STD-DAGMM</td><td class="num">0.25</td><td class="num">0.00</td><td class="num">1.0</td><td class="num">0.75</td></tr>
  </table>
  <p class="sm" style="margin-top:8px"><b>3B poison(ASR=1.0)서는 Flirds 1차·2차 모두 회피</b>(1B의 2차 부분 회복이 사라짐; 1-seed). Flirds vs (b) Spearman: noisy/FR +1.000(FedIF 0.600).</p>
</div>
</div>
<p class="sm" style="margin-top:10px">표 값은 seed 평균(±std 생략; 원본 = runs/phase2_matrix/rundirs/*/metrics.json). N=100 α=0.5 anchor smoke에서 Flirds vs per-round exact oracle +1.000 <span class="lbl">ⓑ</span>(1-seed) · free-rider φ=0은 N=100서도 exact. <span class="ph">α=0.5 anchor 3셀 · device100 poison 2셀 · 7B 잔여.</span></p>
"""))

# S6 ── 고려 중인 사항
SLIDES.append(("§3 방향", "고려 중인 사항 (novelty 방향)", "", """
<table>
<tr><th style="width:30%">항목</th><th>내용</th></tr>
<tr><td>① 탐지된 client의 처리</td><td>noisy·free-rider·backdoor로 탐지된 client를 어떻게 처리할지. 현재 구현: post-hoc selection(φ 하위 드롭 후 재학습) <span class="lbl">ⓑ</span> 실측 완료 · Track C2의 개입 3종(가중집계 w∝n·s / selection / bottom-q% 제외) <span class="lbl">ⓒ</span> 설계 확정·구현 착수.</td></tr>
<tr><td>② 탐지 전용 데이터 분리</td><td>validation data와 구별되는 탐지 전용 데이터(추가 정보원)를 따로 두는 설계 검토.</td></tr>
<tr><td>③ validation data 약점 개선</td><td>단일 server val-loss 게임이 갖는 약점 — 공정성(도메인 편향), OOD, 오염 민감도 — 의 개선.</td></tr>
<tr><td>④ utility 함수 교체</td><td>val-loss 외의 utility로 게임 정의를 바꿔 사용해볼 여지.</td></tr>
</table>
"""))

# S7 ── 앞으로의 계획
SLIDES.append(("§4 계획", "앞으로의 계획", "", """
<ul>
  <li><b>① baseline 연구 심층 탐구</b> — 비교 baseline 원논문들의 메커니즘·실험 프로토콜 정밀 대조(특히 최근접 경쟁 Ripple·FedIF).</li>
  <li><b>② ablation study 실험 설계</b> — 주요 설계 선택에 대한 ablation 설계.</li>
  <li><b>③ novelty를 추가할 만한 요소 탐색</b> — 앞 장 '고려 중인 사항'과 연결해 구체화.</li>
</ul>
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
