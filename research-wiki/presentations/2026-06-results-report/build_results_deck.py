#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flirds 실험 결과 보고 데크 (독립 상세판) — 표 중심, 담백한 학술 핸드아웃 톤.

빌드:  python3 build_results_deck.py
출력:  flirds-results-2026-06.pptx  (16:9, 편집 가능 네이티브 표)

데이터 출처:
  - Track C C1 (b)fidelity / 거리 : runs/track_c/c1/<cell>/metrics.json  에서 직접 재집계
  - Track D                        : runs/track_d/rundirs/<cell>/metrics.json 에서 직접 재집계
  - Phase 2 / C1 (a) / C2          : runs/phase2_matrix/RESULTS.md, runs/track_c/RESULTS.txt 의 검증 수치 (상수)
톤: presentation-style-plain-facts — 흰 배경, 강조색 1개, 표/수식 중심, 과장수식어 금지,
    모든 수치에 설정 병기, 약점·잠정치도 같은 톤.
"""
import glob, json, statistics
from collections import defaultdict
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu

ROOT = Path("/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds")
HERE = Path(__file__).resolve().parent
OUT  = HERE / "flirds-results-2026-06.pptx"

# ── 색/폰트 (plain-facts) ───────────────────────────────────────────
INK   = RGBColor(0x1A, 0x1A, 0x1A)
SUB   = RGBColor(0x55, 0x55, 0x55)
FAINT = RGBColor(0x88, 0x88, 0x88)
ACCENT= RGBColor(0x17, 0x50, 0x8C)   # 강조색 1개
LINE  = RGBColor(0xC8, 0xC8, 0xC8)
HDRBG = RGBColor(0xEF, 0xEF, 0xEF)
ZEBRA = RGBColor(0xF8, 0xF8, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT  = "Malgun Gothic"
NO_STYLE_GRID = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # No Style, Table Grid

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
PW = 13.333

# ── helpers ─────────────────────────────────────────────────────────
def slide():
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = WHITE
    return s

def _box(s, l, t, w, h):
    return s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))

def _set(par, text, size, color=INK, bold=False, font=FONT, align=PP_ALIGN.LEFT):
    par.alignment = align
    r = par.add_run(); r.text = text
    f = r.font; f.size = Pt(size); f.bold = bold; f.color.rgb = color; f.name = font
    return r

def header(s, kicker, title, sub=None):
    if kicker:
        tb = _box(s, 0.55, 0.34, 12.2, 0.34); tf = tb.text_frame; tf.word_wrap = True
        tf.margin_top = 0; tf.margin_bottom = 0
        _set(tf.paragraphs[0], kicker.upper(), 12, ACCENT, bold=True)
    tb = _box(s, 0.55, 0.62, 12.2, 0.7); tf = tb.text_frame; tf.word_wrap = True
    tf.margin_top = 0; tf.margin_bottom = 0
    _set(tf.paragraphs[0], title, 25, INK, bold=True)
    # accent rule
    ln = s.shapes.add_shape(1, Inches(0.55), Inches(1.30), Inches(12.23), Pt(1.6))
    ln.fill.solid(); ln.fill.fore_color.rgb = ACCENT; ln.line.fill.background()
    y = 1.44
    if sub:
        tb = _box(s, 0.55, 1.40, 12.2, 0.5); tf = tb.text_frame; tf.word_wrap = True
        tf.margin_top = 0
        _set(tf.paragraphs[0], sub, 13.5, SUB)
        y = 1.86
    return y

def footnote(s, text):
    tb = _box(s, 0.55, 7.06, 12.2, 0.34); tf = tb.text_frame; tf.word_wrap = True
    _set(tf.paragraphs[0], text, 10.5, FAINT)

def notes(s, l, t, w, lines, size=12.5, gap=6):
    tb = _box(s, l, t, w, 0.4); tf = tb.text_frame; tf.word_wrap = True
    tf.margin_top = 0; tf.margin_left = 0
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap); p.space_before = Pt(0)
        if isinstance(ln, tuple):       # (text, bold, color?)
            txt, bold = ln[0], ln[1]; col = ln[2] if len(ln) > 2 else INK
            _set(p, txt, size, col, bold=bold)
        else:
            _set(p, ln, size, INK)
    return tb

def table(s, left, top, total_w, rows, ratios, aligns=None, size=10.5,
          header_row=True, row_h=0.0, zebra=False, accent_first=False):
    nr, nc = len(rows), len(rows[0])
    aligns = aligns or (["l"] + ["c"] * (nc - 1))
    h = Inches(row_h * nr) if row_h else Inches(0.3 * nr)
    gtbl = s.shapes.add_table(nr, nc, Inches(left), Inches(top), Inches(total_w), h)
    tb = gtbl.table
    # plain grid style, no banding
    tblPr = tb._tbl.find(qn('a:tblPr'))
    tblPr.set('firstRow', '0'); tblPr.set('bandRow', '0')
    sid = tblPr.find(qn('a:tableStyleId'))
    if sid is None:
        sid = tblPr.makeelement(qn('a:tableStyleId'), {}); tblPr.append(sid)
    sid.text = NO_STYLE_GRID
    tot = sum(ratios)
    for j, rt in enumerate(ratios):
        tb.columns[j].width = Emu(int(Inches(total_w) * rt / tot))
    amap = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}
    for i in range(nr):
        if row_h: tb.rows[i].height = Inches(row_h)
        for j in range(nc):
            cell = tb.cell(i, j)
            cell.margin_left = Pt(5); cell.margin_right = Pt(5)
            cell.margin_top = Pt(1.5); cell.margin_bottom = Pt(1.5)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            is_hdr = header_row and i == 0
            if is_hdr:
                cell.fill.solid(); cell.fill.fore_color.rgb = HDRBG
            elif zebra and i % 2 == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = ZEBRA
            else:
                cell.fill.solid(); cell.fill.fore_color.rgb = WHITE
            txt = str(rows[i][j])
            tf = cell.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]
            bold = is_hdr or (accent_first and j == 0 and i > 0)
            col = ACCENT if is_hdr else INK
            _set(p, txt, size, col, bold=bold, align=amap[aligns[j]])
    return tb

def fmt(v, sign=False, nd=3):
    if v is None: return "–"
    if isinstance(v, str): return v
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s

# ════════════════════════════════════════════════════════════════════
# 데이터 — Track C C1 (b)/거리 + Track D : rundir 직접 재집계
# ════════════════════════════════════════════════════════════════════
SCEN_ORDER = ["iid", "label_skew", "quantity_skew", "label_flip", "feature_noise"]
def _scen_key(ds, sc): return f"{ds}_{sc}"

def load_c1():
    sp = defaultdict(lambda: defaultdict(list)); eu = defaultdict(lambda: defaultdict(list))
    for c in glob.glob(str(ROOT / "runs/track_c/c1/*/metrics.json")):
        m = json.load(open(c)); k = _scen_key(m["dataset"], m["scenario"])
        for meth, e in m["methods"].items():
            if isinstance(e, dict):
                if "spearman_b" in e: sp[k][meth].append(e["spearman_b"])
                if "euc_b" in e: eu[k][meth].append(e["euc_b"])
    mean = lambda d: {k: {me: sum(v)/len(v) for me, v in mv.items()} for k, mv in d.items()}
    return mean(sp), mean(eu)
C1_SP, C1_EU = load_c1()

def load_track_d(regime):
    fid = defaultdict(lambda: defaultdict(list)); arms = defaultdict(lambda: defaultdict(list))
    for r in sorted(glob.glob(str(ROOT / f"runs/track_d/rundirs/1B_{regime}_seed*/metrics.json"))):
        d = json.load(open(r)); m = d[list(d.keys())[0]]
        for mt in ("spearman", "kendall", "cosine_d", "euclid_d", "max_diff", "runtime"):
            for me, v in m.get(mt, {}).items(): fid[mt][me].append(v)
        for arm, av in m.get("arms", {}).items():
            for f, v in av.items(): arms[arm][f].append(v)
    fm = {mt: {me: sum(v)/len(v) for me, v in mv.items()} for mt, mv in fid.items()}
    am = {}
    for arm, av in arms.items():
        am[arm] = {}
        for f, vs in av.items():
            xs = [x for x in vs if isinstance(x, (int, float))]
            am[arm][f] = (sum(xs)/len(xs)) if xs else None
    return fm, am
D_STD_F, D_STD_A = load_track_d("std20")
D_ANC_F, D_ANC_A = load_track_d("anchor5")

# ════════════════════════════════════════════════════════════════════
# 데이터 — 상수 (RESULTS.md / RESULTS.txt 검증분)
# ════════════════════════════════════════════════════════════════════
# Track C C1 (a) 2^10-retrain fidelity (rho_a_mean), RESULTS.txt
C1_A = {
 "cifar10_iid":(-0.232,-0.131,-0.196,-0.176), "cifar10_label_skew":(-0.176,-0.075,0.438,0.143),
 "cifar10_quantity_skew":(0.568,0.556,0.701,0.568), "cifar10_label_flip":(0.515,0.592,0.455,0.580),
 "cifar10_feature_noise":(0.632,0.503,0.438,0.564),
 "mnist_iid":(0.358,0.519,0.192,0.483), "mnist_label_skew":(-0.285,-0.063,-0.220,-0.164),
 "mnist_quantity_skew":(0.846,0.774,0.560,0.842), "mnist_label_flip":(0.964,0.968,0.968,0.968),
 "mnist_feature_noise":(0.325,0.438,0.402,0.438),
}  # (Flirds, Flirds1st, GTG, loss-heur)

# Track C C2 개입 — 대표 셀 (RESULTS.txt). arm: (acc, auroc)
C2 = {
 "cifar10·dir1·clean":      [("vanilla",0.638,None),("flirds_mult",0.642,None),("flirds_select",0.634,None),("shapleyfl",0.635,None),("fedif",0.638,None)],
 "cifar10·dir1·grad_noise": [("vanilla",0.245,None),("flirds_mult",0.433,0.993),("flirds_select",0.313,0.986),("shapleyfl",0.550,1.000),("fedif",0.527,0.998)],
 "cifar10·dir1·label_flip": [("vanilla",0.515,None),("flirds_mult",0.585,0.926),("flirds_select",0.559,0.905),("shapleyfl",0.568,0.965),("fedif",0.567,0.959)],
 "cifar10·dir1·free_rider": [("vanilla",0.587,None),("flirds_mult",0.597,0.385),("flirds_select",0.589,0.479),("shapleyfl",0.560,0.000),("fedif",0.611,0.996)],
}

# Phase 2 silo5 N=5 (RESULTS.md). method -> (AUROC: noisy,frR,frZ,poison), (Spearman: ...), runtime(s)
SILO_M = ["in-run oracle","Flirds","Flirds1st","FedIF","GTG","FedSV","ShapleyFL","Banzhaf","loss-heur","FLDetector","STD-DAGMM","FLTrust","FedDQC"]
SILO_AUROC = {
 "in-run oracle":(1.000,1.000,1.000,1.000),"Flirds":(1.000,1.000,1.000,0.917),"Flirds1st":(1.000,1.000,1.000,0.000),
 "FedIF":(1.000,1.000,1.000,1.000),"GTG":(1.000,1.000,1.000,1.000),"FedSV":(1.000,1.000,1.000,1.000),
 "ShapleyFL":(1.000,1.000,1.000,1.000),"Banzhaf":(1.000,1.000,1.000,1.000),"loss-heur":(1.000,1.000,1.000,1.000),
 "FLDetector":(0.750,1.000,0.750,1.000),"STD-DAGMM":(0.417,1.000,0.250,0.750),"FLTrust":(1.000,1.000,1.000,1.000),
 "FedDQC":(0.917,0.750,0.750,1.000),
}
SILO_SP = {
 "Flirds":(1.000,1.000,1.000,0.967),"Flirds1st":(1.000,1.000,1.000,0.000),"FedIF":(0.933,0.900,0.933,0.967),
 "GTG":(1.000,1.000,1.000,0.867),"FedSV":(1.000,0.933,1.000,0.367),"ShapleyFL":(1.000,1.000,1.000,1.000),
 "Banzhaf":(1.000,1.000,1.000,1.000),"loss-heur":(1.000,1.000,1.000,1.000),
}
SILO_RT = {"in-run oracle":532,"Flirds":106,"Flirds1st":35,"FedIF":35,"GTG":538,"FedSV":533,"ShapleyFL":530,
 "Banzhaf":533,"loss-heur":164,"FLDetector":30,"STD-DAGMM":136,"FLTrust":36,"FedDQC":22}

# Phase 2 device100 N=100 (RESULTS.md)
ALPHAS = ["0.0","0.01","0.1","0.5","5.0"]
DEV_NOISY = {  # AUROC by alpha
 "Flirds":(0.774,0.575,0.605,0.604,0.596),"FedIF":(0.973,0.568,0.693,0.830,0.973),
 "STD-DAGMM":(0.856,0.652,0.659,0.671,0.760),"FLTrust":(1.000,0.602,0.720,0.854,0.994),
 "FedDQC":(0.960,1.000,1.000,1.000,1.000),"FLDetector":(0.535,0.482,0.525,0.539,0.532),
 "ComFedSV":(0.442,0.419,0.432,0.371,0.396),
}
DEV_POISON = {  # (alpha0, alpha0.5) AUROC
 "Flirds":(1.000,1.000),"Flirds1st":(1.000,0.670),"loss-heur":(1.000,1.000),"FLDetector":(0.987,0.983),
 "STD-DAGMM":(1.000,0.983),"FedDQC":(1.000,1.000),"FLTrust":(0.650,0.498),"FedIF":(0.542,0.458),"ComFedSV":(0.778,0.727),
}
DEV_ANCHOR_SP = {"GTG":0.784,"FedSV":0.752,"ShapleyFL":0.582,"FedIF":0.721,"Flirds1st":1.000,"loss-heur":1.000}

# Phase 2 3B silo5 N=5 seed0 (RESULTS.md). (AUROC noisy,frR,frZ,poison),(Sp ...),rt(noisy)
M3B = ["in-run oracle","Flirds","Flirds1st","FedIF","loss-heur","FLDetector","STD-DAGMM","FLTrust","FedDQC"]
M3B_AUROC = {"in-run oracle":(1.000,1.000,1.000,1.000),"Flirds":(1.000,1.000,1.000,0.000),"Flirds1st":(1.000,1.000,1.000,0.000),
 "FedIF":(1.000,1.000,1.000,1.000),"loss-heur":(1.000,1.000,1.000,1.000),"FLDetector":(1.000,1.000,1.000,1.000),
 "STD-DAGMM":(0.250,1.000,0.000,0.750),"FLTrust":(1.000,1.000,1.000,1.000),"FedDQC":(1.000,0.750,0.750,1.000)}
M3B_SP = {"Flirds":(1.000,1.000,1.000,0.000),"Flirds1st":(1.000,1.000,1.000,0.000),"FedIF":(0.600,0.600,0.600,0.600),
 "loss-heur":(1.000,1.000,1.000,1.000)}
M3B_RT = {"in-run oracle":1240,"Flirds":251,"Flirds1st":82,"FedIF":82,"loss-heur":384,"FLDetector":250,"STD-DAGMM":530,"FLTrust":85,"FedDQC":49}

def auroc4(d, me): return tuple(fmt(x) for x in d[me])

# ════════════════════════════════════════════════════════════════════
# 슬라이드
# ════════════════════════════════════════════════════════════════════

# ── S1 표지 ─────────────────────────────────────────────────────────
s = slide()
tb = _box(s, 0.9, 2.55, 11.5, 1.1); tf = tb.text_frame; tf.word_wrap = True
_set(tf.paragraphs[0], "Flirds — 연합학습 클라이언트 기여도 측정", 36, INK, bold=True)
tb = _box(s, 0.9, 3.55, 11.5, 0.7)
_set(tb.text_frame.paragraphs[0], "실험 결과 보고 (이미지 · 언어모델)", 21, ACCENT, bold=True)
tb = _box(s, 0.9, 4.35, 11.5, 0.9); tf = tb.text_frame; tf.word_wrap = True
_set(tf.paragraphs[0], "방법: 검증손실의 1차+2차 테일러 전개로 클라이언트 Shapley 추정 (라운드당 HVP 1회)", 14.5, SUB)
p = tf.add_paragraph(); _set(p, "client-level FL Shapley via 1st+2nd-order Taylor of validation loss", 13, FAINT)
ln = s.shapes.add_shape(1, Inches(0.92), Inches(2.42), Inches(4.2), Pt(2.2))
ln.fill.solid(); ln.fill.fore_color.rgb = ACCENT; ln.line.fill.background()
tb = _box(s, 0.9, 6.5, 11.5, 0.5)
_set(tb.text_frame.paragraphs[0], "2026-06-18 · 수치는 모두 ⓑ실측(세팅 병기) · 미실행분은 ⓒ로 표기", 12.5, FAINT)

# ── S2 실험 개요 ────────────────────────────────────────────────────
s = slide()
header(s, "Overview", "실험 개요 — 세 트랙, 모두 완료")
rows = [["트랙","도메인 / 모델","연합 셋업","셀","seed","핵심 산출"],
 ["Image (CNN)","MNIST+LeNet5 / CIFAR-10+CNN","N=10 (fidelity) · N=100 (intervention)","150","3","fidelity (retrain+in-run oracle) · 개입 8arm"],
 ["LLM robustness","Llama-3.2 1B · 3B (LoRA)","N=5 cross-silo · N=100 cross-device(α)","25","1–3","4 threat × fidelity·탐지·개입"],
 ["LLM standard (clean·IID)","Llama-3.2-1B / OpenFedLLM","N=20 standard · N=5 reference","6","3","IID·clean fidelity·성능·수렴"]]
table(s, 0.55, 1.55, 12.23, rows, [1.55,2.5,2.2,0.5,0.5,2.7], ["l","l","l","c","c","l"], size=11.5, row_h=0.62, accent_first=True)
notes(s, 0.55, 4.5, 12.2, [
 ("핵심 질문 위계 (모든 표·서술 순서에 적용):", True),
 "  1차 — 기여도를 얼마나 정확히 측정하는가 (오라클 대비 fidelity: Spearman·Kendall·값-거리)",
 "  2차 — 측정한 기여도의 실효성: ① 일반 성능 → ② 수렴 속도 → ③ 오염 클라 탐지 (이 순서)",
 ("비교 대상 11종 + 탐지기 4종 / 듀얼 오라클: retrain oracle(부분집합 재학습→검증손실), in-run oracle(학습궤적 exact 2ᴺ 분해).", False, SUB),
], size=12.5, gap=7)
footnote(s, "공통 규약: SGD momentum=0 · 시드별 seed_everything · 모든 셀 config+meta(git/env)+phi+metrics 영속화. 실패 0.")

# ── S3 공통 설계 ① 방법·오라클 ──────────────────────────────────────
s = slide()
header(s, "Shared design · 1/2", "비교 방법 11종과 오라클 정의")
rows = [["방법","유형","핵심","라운드당 비용"],
 ["Flirds (제안)","valuation","검증손실 1차+2차 테일러","HVP 1회"],
 ["Flirds-1st","valuation","1차항만 (곡률 제외)","grad 1회"],
 ["loss-heur","heuristic","클라별 손실차","fwd 1회"],
 ["GTG / FedSV","Shapley(MC)","truncated/permutation MC","수십–수백 평가"],
 ["Banzhaf","exact","2ᴺ Banzhaf","2ᴺ 평가"],
 ["ShapleyFL / ComFedSV","Shapley(변형)","uniform-util / 부분합","수십–수백"],
 ["FedIF","influence","검증그래디언트 정렬","grad 1회"],
 ["Ripple","2nd-order","고유분해 기반 (이미지만)","eigsh (수십분)"],
 ["탐지기 4종","detector","FLDetector·STD-DAGMM·FLTrust·FedDQC","모델-free/grad"]]
table(s, 0.55, 1.55, 7.05, rows, [2.0,1.2,2.5,1.5], ["l","l","l","l"], size=10.5, row_h=0.43, accent_first=True)
notes(s, 7.85, 1.62, 4.9, [
 ("오라클 (정답 기여도)", True),
 ("in-run oracle", True, ACCENT),
 "  학습 궤적을 라운드별 exact Shapley로 분해 (2ᴺ).",
 "  추정량이 재현해야 할 게임. N=5에서 2⁵.",
 "",
 ("retrain oracle", True, ACCENT),
 "  부분집합 S를 FedAvg 재학습 → 배포모델 검증손실.",
 "  '배포 가치' 게임. 이미지 N=10에서 2¹⁰=1024 재학습.",
 "",
 ("핵심: Flirds는 in-run oracle 게임의 추정량.", True),
 "retrain oracle와의 일치 여부는 'in-run 게임 = 재학습 게임'인지를",
 "묻는 별개 질문 (S8·S18에서 다룸).",
], size=11.5, gap=4)
footnote(s, "출처: codes/core·oracle·baselines. 모든 valuation은 동일 logs=[(w_r, {client:(Δ,n_c)})] 위에서 계산 (모델·검증셋 비접근).")

# ── S4 공통 설계 ② 지표 ─────────────────────────────────────────────
s = slide()
header(s, "Shared design · 2/2", "평가 지표")
rows = [["질문","지표","정의 / 방향"],
 ["1차 fidelity (순위)","Spearman ρ · Kendall τ","오라클 φ 대비 순위 상관 (↑좋음)"],
 ["1차 fidelity (값-거리)","cosine_d · euclid_d · max_diff","오라클 φ 대비 값-크기 오차 (↓좋음)"],
 ["2차-① 일반 성능","test acc (이미지) · MMLU·ROUGE-L (LLM)","개입 arm 최종 성능 (↑좋음)"],
 ["2차-② 수렴","rounds-to-target · final val-loss","목표 도달 라운드 / 최종 검증손실"],
 ["2차-③ 탐지","AUROC","오염 클라=높은 의심점수 (1.0=완전분리)"]]
table(s, 0.55, 1.55, 12.23, rows, [2.3,3.2,4.5], ["l","l","l"], size=12, row_h=0.5, accent_first=True)
notes(s, 0.55, 4.65, 12.2, [
 ("3-state 라벨", True),
 "  ⓐ 구현 완료   ·   ⓑ 실측(이 데크의 모든 수치)   ·   ⓒ 미실행",
 ("위협 모델 (오염 종류)", True),
 "  noisy = answer_swap(데이터 품질) · free-rider = zero/random 업데이트 · poison = 백도어(스케일 모델교체)",
 ("탐지에서 valuation의 입장", True),
 "  방법은 라벨에 blind; 라벨은 평가 KEY(AUROC)일 뿐. 오염이 검증손실을 낮추면 φ가 '기여 높음'으로 나오는 것이 정직한 답.",
], size=12, gap=6)
footnote(s, "값-거리 지표는 순위가 포화(ρ=+1.000)된 구간에서 방법을 변별 — 특히 1차 vs 2차 테일러 비교에 사용.")

# ── S5 Track C 명세 ─────────────────────────────────────────────────
s = slide()
header(s, "Image (CNN)", "실험 명세 — fidelity / intervention")
rows = [["항목","fidelity","intervention (일반 성능·탐지)"],
 ["데이터·모델","MNIST+LeNet5 / CIFAR-10+CNN","CIFAR-10 / FMNIST + CNN"],
 ["클라이언트","N=10 (full 참여)","N=100, 라운드당 C=0.1 (10개)"],
 ["라운드 / local","R=10 / epochs=5","R=120 / epochs=5"],
 ["옵티마이저","SGD mom=0, lr=0.01, batch=64","SGD mom=0, lr=0.01, batch=64"],
 ["검증 / 테스트","val=2000 / test=8000","val=2000 / test=8000 (target acc=0.6)"],
 ["시나리오·위협","iid·label_skew·quantity_skew·label_flip·feature_noise","partition{iid·dir1·shard} × {clean·label_flip·grad_noise·free_rider}"],
 ["오라클","in-run oracle (exact 2¹⁰) + retrain oracle (2¹⁰)","— (개입 결과로 평가)"],
 ["개입 arm","—","vanilla·flirds{mult·repl·add·select}·shapleyfl·fedif·sfedavg"]]
table(s, 0.55, 1.55, 12.23, rows, [1.5,3.6,4.5], ["l","l","l"], size=11, row_h=0.52, accent_first=True)
footnote(s, "3 seed(0,1,2), 150셀 실패 0. retrain oracle efficiency-gap ≤1e-15 (전 셀). 환경: conda lora4cl, RTX 3090. · 내부코드 fidelity=C1·intervention=C2·in-run oracle=(b)·retrain oracle=(a)")

# ── S6 C1 fidelity vs (b) ───────────────────────────────────────────
s = slide()
header(s, "Image (CNN) · 1차 fidelity", "fidelity — in-run oracle 대비 (순위 Spearman, 3-seed 평균)")
meths = ["Flirds","Flirds1st","Banzhaf","GTG","FedSV","loss-heur"]
rows = [["시나리오 (CIFAR-10 / MNIST)"] + meths]
for ds in ("cifar10","mnist"):
    for sc in SCEN_ORDER:
        k = _scen_key(ds, sc)
        rows.append([f"{ds.replace('cifar10','CIFAR')}·{sc}"] + [fmt(C1_SP[k].get(m), sign=True, nd=2) for m in meths])
table(s, 0.55, 1.55, 7.5, rows, [2.0,0.95,0.95,0.95,0.95,0.95,1.05], ["l"]+["c"]*6, size=9.8, row_h=0.355, zebra=True)
notes(s, 8.25, 1.62, 4.5, [
 ("값-거리 (euclid_d, ↓좋음)", True),
 "2차항이 값-크기를 얼마나 좁히나 (label_flip):",
 ("  CIFAR : Flirds 0.031 vs 1차 0.046", False, SUB),
 ("  MNIST : Flirds 0.207 vs 1차 0.370", False, SUB),
 "→ 2차항이 값 오차를 1차 대비 ~25–50% 축소",
 "   (전 시나리오 일관).",
 "",
 ("읽기", True),
 "순위만 보면 Flirds·Banzhaf·loss-heur 모두 우수",
 "(Banzhaf=exact 2¹⁰). MC 기반 GTG/FedSV는 noisy.",
 "Flirds의 우위는 정확도가 아니라 비용(HVP 1회).",
], size=11, gap=4)
footnote(s, "iid·feature_noise는 in-run oracle 자체 신호가 약함(클라 동질). 거리 지표는 S4 정의. 출처: runs/track_c/c1/*/metrics.json 재집계.")

# ── S7 C1 fidelity vs (a) ───────────────────────────────────────────
s = slide()
header(s, "Image (CNN) · 1차 fidelity", "fidelity — retrain oracle (2¹⁰) 대비, 그리고 in-run oracle와의 괴리")
rows = [["시나리오","Flirds (in-run)","Flirds (retrain)","Flirds1st (retrain)","GTG (retrain)","loss-heur (retrain)"]]
for ds in ("cifar10","mnist"):
    for sc in SCEN_ORDER:
        k = _scen_key(ds, sc); a = C1_A[k]
        rows.append([f"{ds.replace('cifar10','CIFAR')}·{sc}", fmt(C1_SP[k].get("Flirds"),True,2),
                     fmt(a[0],True,2), fmt(a[1],True,2), fmt(a[2],True,2), fmt(a[3],True,2)])
table(s, 0.55, 1.55, 7.7, rows, [2.0,1.1,1.1,1.2,1,1.3], ["l"]+["c"]*5, size=9.8, row_h=0.355, zebra=True)
notes(s, 8.45, 1.62, 4.3, [
 ("핵심 — 게임 정의 차이", True),
 "Flirds는 in-run oracle 게임을 거의 완벽 재현하나,",
 "label_skew에서 in-run과 retrain oracle가 갈린다:",
 ("  CIFAR·label_skew : in-run +0.98 → retrain −0.18", False, ACCENT),
 ("  MNIST·label_skew : in-run +0.71 → retrain −0.29", False, ACCENT),
 "",
 "이는 Flirds 추정오차가 아니라 모든 in-run",
 "방법이 공유하는 한계. 희귀라벨 클라는 매 라운드",
 "한계기여는 낮아도 최종 재학습 모델엔 필수.",
 "(iid 음수는 retrain oracle 신호 자체가 미약 → 노이즈)",
], size=11, gap=4)
footnote(s, "retrain 열은 rho_a_mean(3-seed). 동일 시나리오에서 LLM standard의 reference (N=5)는 retrain oracle와 +1.000 일치 → 괴리는 이질성 구동(S14).")

# ── S8 C2 개입 ──────────────────────────────────────────────────────
s = slide()
header(s, "Image (CNN) · 2차-① 성능·③ 탐지", "intervention — 개입 결과 (CIFAR-10, dir1, 3-seed 평균)")
rows = [["arm"]]
cells = list(C2.keys())
rows = [["arm"] + [c.replace("cifar10·dir1·","") for c in cells]]
arms = ["vanilla","flirds_mult","flirds_select","shapleyfl","fedif"]
for ai, arm in enumerate(arms):
    row = [arm]
    for c in cells:
        rec = [x for x in C2[c] if x[0]==arm][0]
        acc, au = rec[1], rec[2]
        row.append(f"{acc:.3f}" + (f"  ({au:.2f})" if au is not None else "  (–)"))
    rows.append(row)
table(s, 0.55, 1.65, 12.23, rows, [1.5,2.4,2.7,2.7,2.7], ["l","c","c","c","c"], size=11, row_h=0.5, accent_first=True)
notes(s, 0.55, 4.7, 12.2, [
 ("셀당 = test acc  (탐지 AUROC).  clean은 위협 없음→AUROC 없음.", True),
 "• clean: 전 arm ≈ vanilla → 무해(do-no-harm) 확인.",
 ("• grad_noise: 전 arm > vanilla(0.245). 단 shapleyfl(0.550)·fedif(0.527) > flirds_mult(0.433).", False, INK),
 ("• label_flip: flirds_mult(0.585)이 최고. free_rider: fedif AUROC 0.996, flirds_mult 0.385 — 탐지 약함(아래).", False, INK),
 ("→ Flirds 가중은 항상 robust+무해하나 정확도에서 shapleyfl/fedif를 압도하진 않음.  soft(mult) > hard(select).", True, ACCENT),
], size=11.5, gap=6)
footnote(s, "free-rider(zero-delta) AUROC가 N=100 CNN에서 0.33–0.48로 낮음 — LLM(1.0)과 상충, 미해결 항목(S18). 출처: runs/track_c/RESULTS.txt.")

# ── S9 Phase 2 명세 ─────────────────────────────────────────────────
s = slide()
header(s, "LLM robustness", "실험 명세")
rows = [["항목","cross-silo (N=5)","cross-device (N=100)"],
 ["모델","Llama-3.2-1B-Instruct (+ 3B)","Llama-3.2-1B-Instruct"],
 ["클라이언트","N=5 (1 도메인/클라)","N=100, 라운드당 K=10"],
 ["분할","5-도메인 cross-silo","per-client Dir(α), α∈{0,0.01,0.1,0.5,5.0}"],
 ["라운드","R=10","R=30"],
 ["옵티마이저","SGD mom=0, lr=1e-3 (poison 2e-3)","동일"],
 ["정밀도·attn","fp32 master · eager attention","동일"],
 ["검증","val=100 (5도메인)","domain-pool val"],
 ["오라클","in-run oracle (exact 2⁵)","in-run oracle (per-round exact, reference point α=0.5)"],
 ["위협","noisy · freerider{zero,random} · poison","동일 (poison: install config)"]]
table(s, 0.55, 1.5, 12.23, rows, [1.5,3.6,4.6], ["l","l","l"], size=10.8, row_h=0.475, accent_first=True)
footnote(s, "LoRA 어댑터, attn=eager(전진모드 AD 위해). cross-silo/3B 3seed (3B는 1seed). cross-device α별 1–3seed. 기준점 외 α는 Flirds proxy(reference point α=0.5서 in-run oracle와 +1.000 검증). · 내부코드 cross-silo=silo5·cross-device=device100")

# ── S10 silo5 결과 ──────────────────────────────────────────────────
s = slide()
header(s, "LLM robustness · cross-silo (N=5, 1B)", "탐지 AUROC · fidelity Spearman vs in-run oracle · 런타임")
rows = [["방법","noisy","fr-rand","fr-zero","poison","ρ noisy","ρ poison","런타임"]]
for me in SILO_M:
    au = SILO_AUROC[me]; sp = SILO_SP.get(me)
    rows.append([me, fmt(au[0]),fmt(au[1]),fmt(au[2]),fmt(au[3]),
                 fmt(sp[0],True,2) if sp else "–", fmt(sp[3],True,2) if sp else "–",
                 f"{SILO_RT[me]}s"])
table(s, 0.55, 1.44, 12.23, rows, [1.7,1,1,1,1,1.1,1.1,1.0], ["l"]+["c"]*7, size=9, row_h=0.30, zebra=True)
notes(s, 0.55, 6.02, 12.2, [
 ("noisy·free-rider: 전 valuation +1.000, AUROC 1.0 — N=5 near-additive로 모두 동률. Flirds 우위=비용(106s vs 530s).", True),
 ("poison(ASR≈1.00): Flirds-1st AUROC 0.000 완전회피 → 2차항 추가 시 0.917 회복(불안정, 아래). FedSV ρ 0.367로 동률 붕괴.", True, ACCENT),
], size=11, gap=5)
footnote(s, "poison Flirds(2차)는 run간 비결정(동일 config·seed에서 0.42↔0.92) → '부분 회복하나 불안정'. loss-heur·전 탐지기는 1.0. 출처: RESULTS.md.")

# ── S11 device100 결과 ──────────────────────────────────────────────
s = slide()
header(s, "LLM robustness · cross-device (N=100, 1B)", "α-sweep — 탐지 AUROC")
rows = [["방법 (noisy)"] + [f"α={a}" for a in ALPHAS]]
for me in ["Flirds","FedIF","STD-DAGMM","FLTrust","FedDQC","FLDetector","ComFedSV"]:
    rows.append([me] + [fmt(x) for x in DEV_NOISY[me]])
table(s, 0.55, 1.5, 6.6, rows, [1.8,1,1,1,1,1], ["l"]+["c"]*5, size=10, row_h=0.4, zebra=True)
rows2 = [["방법 (poison)","α=0.0","α=0.5"]]
for me in ["Flirds","Flirds1st","loss-heur","FLDetector","STD-DAGMM","FedDQC","FLTrust","FedIF"]:
    rows2.append([me, fmt(DEV_POISON[me][0]), fmt(DEV_POISON[me][1])])
table(s, 7.45, 1.5, 5.3, rows2, [2.0,1,1], ["l","c","c"], size=10, row_h=0.4, zebra=True)
notes(s, 0.55, 5.1, 12.2, [
 ("free-rider (random·zero, 전 α): Flirds·Flirds1st·loss-heur·FLTrust = AUROC 1.000.  FedDQC 0.14–0.57(off-threat)·ComFedSV ~0.40.", True),
 ("noisy: 데이터품질 탐지는 FedDQC 영역(스케일에서 1.0). Flirds ~0.60. fidelity는 Flirds1st·loss-heur ρ=+1.000(전 α);", False, INK),
 ("        reference point α=0.5서 GTG +0.78·FedSV +0.75·ShapleyFL +0.58·FedIF +0.72 (vs in-run oracle per-round exact).", False, INK),
 ("poison(install config): Flirds AUROC 1.0(both α), 1차도 α=0서 1.0 — cross-silo와 달리 cross-device에선 1차가 회피되지 않음.", True, ACCENT),
], size=11, gap=5)
footnote(s, "cross-device φ는 한 번이라도 선택된 클라만 기록(K=10/R=30 → ~96–98/100). 출처: RESULTS.md.")

# ── S12 3B 스케일 ───────────────────────────────────────────────────
s = slide()
header(s, "LLM robustness · 스케일 (3B, cross-silo N=5)", "Llama-3.2-3B — 탐지 AUROC · ρ vs in-run oracle · 런타임 (seed 0)")
rows = [["방법","noisy","fr-rand","fr-zero","poison","ρ noisy","ρ poison","런타임"]]
for me in M3B:
    au = M3B_AUROC[me]; sp = M3B_SP.get(me)
    rows.append([me, fmt(au[0]),fmt(au[1]),fmt(au[2]),fmt(au[3]),
                 fmt(sp[0],True,2) if sp else "–", fmt(sp[3],True,2) if sp else "–", f"{M3B_RT[me]}s"])
table(s, 0.55, 1.6, 12.23, rows, [1.7,1,1,1,1,1.1,1.1,1.0], ["l"]+["c"]*7, size=10, row_h=0.375, zebra=True)
notes(s, 0.55, 5.42, 12.2, [
 ("noisy·free-rider: Flirds ρ=+1.000 유지(1B→3B), 런타임 251s vs in-run oracle 1240s (~5× 저렴).", True),
 ("poison: Flirds·Flirds-1st 모두 AUROC 0.000 — 1B서 보인 2차 회복이 3B(seed 1개)에선 안 나타남 → 경계의 스케일 취약성(미확정).", True, ACCENT),
 ("FedIF ρ는 3B서 +0.600으로 하락(1B +0.90). loss-heur·FLDetector·FedDQC·FLTrust는 poison 1.0 유지.", False, INK),
], size=11.5, gap=6)
footnote(s, "3B는 seed 1개 → 잠정. 7B·N=10 retrain oracle은 설계상 deferred(ⓒ). 출처: RESULTS.md.")

# ── S13 Track D 명세 ────────────────────────────────────────────────
s = slide()
header(s, "LLM standard (clean·IID)", "실험 명세 — 오염축 제거, OpenFedLLM 표준 셋업")
rows = [["항목","standard (N=20, 주 무대)","reference (N=5, 듀얼 오라클)"],
 ["모델·데이터","Llama-3.2-1B / alpaca-gpt4 20k (IID)","동일"],
 ["클라이언트","N=20, 라운드당 2개","N=5 (full)"],
 ["라운드 / local","R=200 / 10 steps × batch16","R=30 / 동일"],
 ["옵티마이저","SGD mom=0, lr=1e-3, seq=512","동일"],
 ["검증 / 테스트","val=200 / test=1000","동일"],
 ["오라클","in-run oracle (per-round exact)","retrain oracle (2⁵) + in-run oracle + Banzhaf"],
 ["개입 arm","base·vanilla·flirds_w(×β.5)·flirds_sel·shapleyfl_w·fedif_w","base·vanilla·flirds_w·shapleyfl_w·fedif_w"],
 ["성능 지표","MMLU(full 0-shot) · Alpaca-test ROUGE-L · 수렴","동일"]]
table(s, 0.55, 1.5, 12.23, rows, [1.5,4.3,3.6], ["l","l","l"], size=10.8, row_h=0.48, accent_first=True)
footnote(s, "무대=OpenFedLLM run_sft.sh verbatim(deviation: SGD mom=0/lr1e-3 상수/r16/fp32). clean·IID → 개입은 do-no-harm parity가 기대값. 3 seed. · 내부코드 standard=std20·reference=anchor5·in-run oracle=(b)·retrain oracle=(a)")

# ── S14 Track D fidelity ────────────────────────────────────────────
s = slide()
header(s, "LLM standard (clean·IID) · 1차 fidelity", "standard (N=20) vs in-run oracle · reference (N=5) vs retrain oracle (3-seed 평균)")
def drow(F, me):
    return [me, fmt(F["spearman"].get(me),True), fmt(F["kendall"].get(me),True),
            f"{F['cosine_d'][me]:.1e}", f"{F['euclid_d'][me]:.1e}", f"{F['max_diff'][me]:.1e}"]
hdr = ["방법","Spearman","Kendall","cosine_d","euclid_d","max_diff"]
order = ["Flirds","Flirds1st","loss-heur","GTG","FedSV","ComFedSV","ShapleyFL","FedIF"]
rows = [["standard (N=20) vs in-run oracle"]+hdr[1:]] + [drow(D_STD_F, m) for m in order if m in D_STD_F["spearman"]]
table(s, 0.55, 1.55, 6.25, rows, [1.5,1.1,1,1,1,1], ["l"]+["c"]*5, size=9.6, row_h=0.355, zebra=True)
order2 = ["Flirds","Flirds1st","Banzhaf","loss-heur","GTG","FedSV","ShapleyFL","ComFedSV","FedIF"]
rows2 = [["reference (N=5) vs retrain oracle"]+hdr[1:]] + [drow(D_ANC_F, m) for m in order2 if m in D_ANC_F["spearman"]]
table(s, 7.05, 1.55, 6.25, rows2, [1.5,1.1,1,1,1,1], ["l"]+["c"]*5, size=9.6, row_h=0.355, zebra=True)
notes(s, 0.55, 5.5, 12.2, [
 ("Flirds: in-run oracle ρ=+1.000(cosine_d 3.5e-7) · retrain oracle ρ=+1.000 — IID·clean에선 두 오라클 게임이 일치 → S7의 괴리는 이질성 구동임을 확인.", True, ACCENT),
 ("2차항 효과: cosine_d Flirds 3.5e-7 vs Flirds-1st 4.4e-5 (~125× 타이트). ShapleyFL·FedIF·ComFedSV는 스케일서 fidelity 약함.", False, INK),
], size=11.5, gap=6)
footnote(s, "reference (N=5)는 retrain oracle (2⁵) 기준. 출처: runs/track_d/rundirs/*/metrics.json 재집계.")

# ── S15 Track D 성능·수렴 ───────────────────────────────────────────
s = slide()
header(s, "LLM standard (clean·IID) · 2차-① 성능 ② 수렴", "개입 arm — MMLU · ROUGE-L · 수렴 (standard N=20, 3-seed 평균)")
arm_order = ["base","vanilla","flirds_w","flirds_sel","shapleyfl_w","fedif_w"]
rows = [["arm","MMLU (0-shot)","ROUGE-L","final val-loss","rounds→target"]]
for a in arm_order:
    if a not in D_STD_A: continue
    A = D_STD_A[a]
    rows.append([a, fmt(A.get("mmlu"),nd=4), fmt(A.get("rouge_l"),nd=4),
                 fmt(A.get("final_val_loss"),nd=4) if A.get("final_val_loss") else "–",
                 f"{A['rounds_to_target']:.0f}" if A.get("rounds_to_target") else "–"])
table(s, 0.55, 1.6, 9.0, rows, [1.6,1.6,1.2,1.5,1.5], ["l","c","c","c","c"], size=11, row_h=0.46, accent_first=True)
notes(s, 0.55, 5.0, 12.2, [
 ("base=학습 전. FL-SFT 후 ROUGE-L +0.067(0.217→0.284), MMLU −0.008 — 과제 적합 향상, 일반지식 미세 하락.", True),
 ("전 가중 arm(flirds_w·flirds_sel·shapleyfl_w·fedif_w) ≈ vanilla (MMLU ±0.0006·ROUGE ±0.001·수렴 ±1라운드).", False, INK),
 ("→ IID·clean에서 do-no-harm parity 확인 — 고칠 오염이 없을 때 가중이 성능·수렴을 해치지 않음 (설계 기대값 충족).", True, ACCENT),
], size=11.5, gap=6)
footnote(s, "rounds-to-target=검증손실 목표 도달 라운드(전 arm ~199/200). reference (N=5)도 동일 parity. 출처: 재집계.")

# ── S16 종합 1차 ────────────────────────────────────────────────────
s = slide()
header(s, "종합 · 1차 fidelity", "기여도 측정 정확성 — 트랙 횡단 요약")
rows = [["측면","결과","근거"],
 ["순위 vs in-run oracle","Flirds ρ≈+1.000 (전 트랙). Banzhaf·loss-heur도 우수; MC-SV는 noisy","cross-silo·standard +1.000 / 이미지 +0.71–1.00"],
 ["값-거리 vs in-run oracle","2차항이 1차 대비 값오차 ~2× 축소; loss-heur은 순위만 비기고 크기는 빗나감","euclid_d: 이미지 ~0.5× · standard cosine 125×"],
 ["순위 vs retrain oracle","IID·clean은 +1.000 일치; label_skew(이질성)에서 in-run oracle와 괴리(−)","reference (N=5) +1.000 / 이미지 label_skew −0.18~−0.29"],
 ["비용","Flirds 1 HVP/round; in-run oracle exact의 5–15× 저렴, 동일 순위","cross-silo 106s vs 530s · 3B 251s vs 1240s"]]
table(s, 0.55, 1.55, 12.23, rows, [1.7,4.7,3.5], ["l","l","l"], size=11.2, row_h=0.62, accent_first=True)
notes(s, 0.55, 5.35, 12.2, [
 ("한 줄 요약", True),
 "Flirds는 in-run oracle 게임을 순위·값 모두에서 충실히, exact 대비 저비용으로 재현한다. 2차항의 고유 기여는",
 "순위가 아니라 값-크기 정밀도. retrain oracle 게임과의 일치는 데이터 이질성에 의존 — 동질이면 일치, label_skew면 갈림.",
], size=12, gap=6)
footnote(s, "retrain oracle 괴리는 Flirds 특정 오차가 아니라 모든 in-run 방법 공유. 상세 S6·S7·S14.")

# ── S17 종합 2차 ────────────────────────────────────────────────────
s = slide()
header(s, "종합 · 2차 실효성", "① 일반 성능 → ② 수렴 → ③ 탐지")
rows = [["질문","요약"],
 ["① 일반 성능","오염 하 전 가중 arm > vanilla(이미지). clean/IID는 무해 parity(이미지·LLM). 정확도 최고는 아님 — grad_noise서 shapleyfl/fedif 우위, label_flip서 flirds 우위. soft>hard."],
 ["② 수렴","IID·clean(LLM standard)서 전 arm 동일 수렴(rounds→target ~199/200, parity). 고칠 오염 없을 때 가중이 수렴을 해치지 않음."],
 ["③ 탐지","free-rider: LLM AUROC 1.0(전 α)이나 N=100 CNN서 0.4(상충, S18). noisy: FedDQC 영역(1.0), Flirds ~0.6. poison: 1차 회피→2차 부분회복(1B)·실패(3B); loss-heur·전 탐지기 1.0."]]
table(s, 0.55, 1.55, 12.23, rows, [1.4,9.0], ["l","l"], size=11.5, row_h=1.0, accent_first=True)
notes(s, 0.55, 5.6, 12.2, [
 ("탐지는 위계상 마지막 — 기여도≠탐지. clean-val-loss를 낮추는 공격자를 φ가 '기여 높음'으로 보는 것은 valuation의 정직한 답이며,", True),
 "  바로 그 지점(clean-preserving poison)이 valuation 접근의 경계다 (S18).",
], size=11.5, gap=5)

# ── S18 한계·미해결 ─────────────────────────────────────────────────
s = slide()
header(s, "한계 · 미해결", "솔직한 항목 (같은 톤으로)")
rows = [["항목","상태","내용"],
 ["오라클 게임 괴리 (retrain↔in-run)","ⓑ 실측","label_skew(이질성)서 in-run φ가 재학습 가치와 역상관. 모든 in-run 방법 공유. 프레이밍 필요."],
 ["free-rider 탐지 상충","ⓑ 미해결","동일 zero-delta가 LLM AUROC 1.0 vs N=100 CNN 0.4(IID칸 포함). 버그 배제용 드릴다운 필요."],
 ["poison 2차 불안정","ⓑ 잠정","1B서 1차 회피(0.000)→2차 부분회복하나 run간 0.42↔0.92(동일 config·seed). 3B(1seed)선 회복 실패."],
 ["미실행 (deferred)","ⓒ 미실행","LLM standard 3B/7B · LLM N=10 retrain oracle · 7B 전반 — 설계상 보류."]]
table(s, 0.55, 1.55, 12.23, rows, [1.9,1.1,7.0], ["l","c","l"], size=11, row_h=0.78, accent_first=True)
notes(s, 0.55, 5.55, 12.2, [
 ("다음 단계 (제안)", True),
 "  1차: 오라클 게임 괴리(retrain↔in-run) 규명(φ 직접 분석, 희귀라벨 가설) — 핵심질문 위계상 우선.",
 "  병행: free-rider CNN-vs-LLM 상충 드릴다운(버그/실재 판별) · poison 2차 비결정성 재현 확인.",
], size=11.5, gap=6)
footnote(s, "ⓐ구현 ⓑ실측 ⓒ미실행. 잠정·미해결치도 결과와 같은 비중으로 기재.")

prs.save(str(OUT))
print(f"saved: {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
