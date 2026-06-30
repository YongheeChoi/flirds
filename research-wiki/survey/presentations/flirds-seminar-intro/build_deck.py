#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flirds 세미나 입문 데크 빌더 (재현용).

목적: 처음 듣는 청중을 위한 입문→연구소개 데크. 핵심 내용만(말로 보완),
영어 키워드 제목, 모든 수식은 LaTeX(mathtext)로 렌더해 PNG 임베드, 참조 표기 없음.

수식 렌더: matplotlib mathtext → 투명 PNG → add_picture. 임시 폴더 사용(재현 안전).
실행: /home/korea_bupj/miniconda3/envs/flirds/bin/python build_deck.py
"""
import os
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---- palette ----
C_TITLE   = RGBColor(0x1F, 0x2A, 0x37)
C_HEADER  = RGBColor(0x33, 0x47, 0x5B)
C_HEADTXT = RGBColor(0xFF, 0xFF, 0xFF)
C_ROWALT  = RGBColor(0xF2, 0xF4, 0xF7)
C_BODY    = RGBColor(0x22, 0x22, 0x22)
C_MUTED   = RGBColor(0x7A, 0x7A, 0x7A)
C_SETTING = RGBColor(0x3C, 0x46, 0x54)
C_BG_DARK = RGBColor(0x1F, 0x2A, 0x37)
C_WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
C_ICE     = RGBColor(0xCA, 0xDC, 0xFC)
C_CARD    = RGBColor(0xED, 0xF1, 0xF6)
C_CARDBD  = RGBColor(0xD3, 0xDD, 0xE8)
C_ACCENT  = RGBColor(0x33, 0x47, 0x5B)
C_GREEN   = RGBColor(0x2C, 0x5F, 0x2D)
C_GREENBG = RGBColor(0xE6, 0xEF, 0xE6)
C_GREENBD = RGBColor(0xC4, 0xD8, 0xC4)
C_TERRA   = RGBColor(0xB8, 0x50, 0x42)

# formula colors (hex for matplotlib)
F_DARK  = "#1F2A37"
F_WHITE = "#FFFFFF"
F_GREEN = "#2C5F2D"

FONT = "NanumGothic"
FDIR = tempfile.mkdtemp(prefix="flirds_formulas_")

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height
_pageno = [0]
_fi = [0]


def _set_run(r, text, size, bold=False, color=C_BODY, italic=False):
    from pptx.oxml.ns import qn
    r.text = text
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    rPr = r._r.get_or_add_rPr()
    ea = rPr.find(qn('a:ea'))
    if ea is None:
        ea = rPr.makeelement(qn('a:ea'), {})
        rPr.append(ea)
    ea.set('typeface', FONT)


def add_textbox(slide, left, top, width, height, lines, align=PP_ALIGN.LEFT,
                anchor=MSO_ANCHOR.TOP, line_space=1.0):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(2); tf.margin_right = Pt(2)
    tf.margin_top = Pt(1); tf.margin_bottom = Pt(1)
    tf.vertical_anchor = anchor
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_space
        txt, size, bold, color = ln[0], ln[1], ln[2], ln[3]
        ital = ln[4] if len(ln) > 4 else False
        _set_run(p.add_run(), txt, size, bold, color, ital)
    return tb


def _page_number(slide):
    _pageno[0] += 1
    if _pageno[0] == 1:
        return
    tb = slide.shapes.add_textbox(SW - Inches(0.7), SH - Inches(0.38),
                                  Inches(0.5), Inches(0.3))
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    _set_run(p.add_run(), str(_pageno[0]), 9, False, C_MUTED)


def slide_header(slide, title, subtitle=None):
    add_textbox(slide, Inches(0.45), Inches(0.32), Inches(12.4), Inches(0.8),
                [(title, 26, True, C_TITLE)])
    y = 1.18
    if subtitle:
        add_textbox(slide, Inches(0.45), Inches(y), Inches(12.4), Inches(0.45),
                    [(subtitle, 12, False, C_SETTING)], line_space=1.05)
        y = 1.62
    return Inches(y)


def dark_bg(slide):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    bg.fill.solid(); bg.fill.fore_color.rgb = C_BG_DARK; bg.line.fill.background()
    bg.shadow.inherit = False
    slide.shapes._spTree.remove(bg._element); slide.shapes._spTree.insert(2, bg._element)
    return bg


def card(slide, left, top, width, height, fill=C_CARD, line=C_CARDBD):
    sp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    sp.line.color.rgb = line; sp.line.width = Pt(0.75)
    sp.shadow.inherit = False
    sp.adjustments[0] = 0.06
    return sp


def chip(slide, left, top, width, height, text, fill=C_ACCENT, txt=C_WHITE, size=12):
    sp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    sp.line.fill.background(); sp.shadow.inherit = False
    sp.adjustments[0] = 0.5
    tf = sp.text_frame; tf.word_wrap = True
    tf.margin_left = Pt(4); tf.margin_right = Pt(4)
    tf.margin_top = Pt(0); tf.margin_bottom = Pt(0)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    _set_run(p.add_run(), text, size, True, txt)
    return sp


def arrow(slide, left, top, width, height, fill=C_ACCENT):
    sp = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    sp.line.fill.background(); sp.shadow.inherit = False
    return sp


def render_formula(latex, color=F_DARK, fontsize=30, dpi=300):
    _fi[0] += 1
    fig = plt.figure()
    fig.text(0.5, 0.5, f"${latex}$", fontsize=fontsize, ha='center', va='center', color=color)
    path = os.path.join(FDIR, f"f{_fi[0]}.png")
    fig.savefig(path, dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    return path


def add_formula(slide, latex, cx, top, height, color=F_DARK, fontsize=30):
    """Center a rendered formula horizontally at cx (EMU); top + height in EMU."""
    path = render_formula(latex, color, fontsize)
    pw, ph = Image.open(path).size
    w = Emu(int(int(height) * pw / ph))
    left = Emu(int(int(cx) - int(w) / 2))
    slide.shapes.add_picture(path, left, top, height=height)
    return w


def add_table(slide, top, columns, rows, col_widths_, font_size,
              manual_bold=None, left=Inches(0.45), row_h=Inches(0.3)):
    ncol = len(columns); nrow = len(rows) + 1
    gfx = slide.shapes.add_table(nrow, ncol, left, top, Emu(int(sum(col_widths_))), row_h * nrow)
    tbl = gfx.table; tbl.first_row = False; tbl.horz_banding = False
    for c, w in enumerate(col_widths_):
        tbl.columns[c].width = Emu(int(w))
    for c, hdr in enumerate(columns):
        cell = tbl.cell(0, c)
        cell.fill.solid(); cell.fill.fore_color.rgb = C_HEADER
        cell.margin_left = Pt(3); cell.margin_right = Pt(3)
        cell.margin_top = Pt(1); cell.margin_bottom = Pt(1)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        _set_run(p.add_run(), hdr, font_size, True, C_HEADTXT)
    bold = manual_bold or set()
    for r, cells in enumerate(rows):
        for c in range(ncol):
            cell = tbl.cell(r + 1, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = C_ROWALT if (r % 2 == 1) else C_WHITE
            cell.margin_left = Pt(3); cell.margin_right = Pt(3)
            cell.margin_top = Pt(0); cell.margin_bottom = Pt(0)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            _set_run(p.add_run(), str(cells[c]), font_size, (r, c) in bold, C_BODY)
    for r in range(nrow):
        tbl.rows[r].height = Emu(int(row_h))
    return gfx


def col_widths(weights, total=Inches(12.45)):
    s = sum(weights)
    return [int(total * w / s) for w in weights]


def newslide():
    return prs.slides.add_slide(BLANK)


# ================================================================ 1. Title
s = newslide()
dark_bg(s)
add_textbox(s, Inches(0.9), Inches(2.05), Inches(11.5), Inches(1.2),
            [("Flirds", 52, True, C_WHITE)])
add_textbox(s, Inches(0.95), Inches(3.25), Inches(11.5), Inches(0.7),
            [("Federated Learning + In-Run Data Shapley", 22, True, C_ICE)])
add_textbox(s, Inches(0.95), Inches(4.15), Inches(11.5), Inches(0.7),
            [("연합학습 한 학습 궤적에서 client-level 데이터 기여도를 추정한다.", 15, False, C_WHITE)])
add_textbox(s, Inches(0.95), Inches(6.4), Inches(11.5), Inches(0.4),
            [("세미나 발표 · 2026-06-30", 13, True, C_ICE)])
_page_number(s)

# ================================================================ 2. Outline
s = newslide()
slide_header(s, "Outline")
items = [
    ("01", "Motivation & Problem"),
    ("02", "Data Shapley"),
    ("03", "In-Run Data Shapley"),
    ("04", "Flirds: FL Adaptation"),
    ("05", "Approximation Error Analysis"),
    ("06", "Results"),
    ("07", "Novelty"),
]
y = 1.7
for num, title in items:
    card(s, Inches(0.6), Inches(y), Inches(0.78), Inches(0.6), fill=C_ACCENT, line=C_ACCENT)
    add_textbox(s, Inches(0.6), Inches(y), Inches(0.78), Inches(0.6),
                [(num, 17, True, C_WHITE)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(s, Inches(1.6), Inches(y), Inches(10.5), Inches(0.6),
                [(title, 18, True, C_TITLE)], anchor=MSO_ANCHOR.MIDDLE)
    y += 0.72
_page_number(s)

# ================================================================ 3. Motivation & Problem
s = newslide()
top = slide_header(s, "Motivation & Problem")
lx = Inches(0.55); lw = Inches(6.0); cy = Inches(1.95); ch = Inches(4.6)
card(s, lx, cy, lw, ch)
add_textbox(s, lx + Inches(0.35), cy + Inches(0.3), lw - Inches(0.7), Inches(0.6),
            [("필요성 (Why)", 18, True, C_ACCENT)])
add_textbox(s, lx + Inches(0.35), cy + Inches(1.1), lw - Inches(0.7), Inches(3.2),
    [("• 불량 클라이언트 식별·가중·배제", 15, True, C_TITLE),
     ("   라벨 오류·노이즈·free-rider(무임승차) 등을 모두 포함", 13, False, C_BODY),
     ("", 8, False, C_BODY),
     ("• 기여 비례 보상·인센티브 분배", 15, True, C_TITLE),
     ("   데이터 시장·연합 컨소시엄의 핵심 동기", 13, False, C_BODY)],
    line_space=1.25)
rx = Inches(6.85); rw = Inches(5.9)
card(s, rx, cy, rw, ch, fill=C_GREENBG, line=C_GREENBD)
add_textbox(s, rx + Inches(0.35), cy + Inches(0.3), rw - Inches(0.7), Inches(0.6),
            [("문제 정의 (Our Problem)", 18, True, C_GREEN)])
add_textbox(s, rx + Inches(0.35), cy + Inches(1.1), rw - Inches(0.7), Inches(3.2),
    [("• FL에서 데이터 기여도를 평가", 15, True, C_TITLE),
     ("• 클라이언트 단위로 평가", 15, True, C_TITLE),
     ("• Shapley value를 정확하고 효율적으로 추정", 15, True, C_TITLE),
     ("", 10, False, C_BODY),
     ("→ 추가 통신 없이, 한 학습 궤적만으로.", 14, False, C_GREEN)],
    line_space=1.5)
_page_number(s)

# ================================================================ 4. Data Shapley
s = newslide()
top = slide_header(s, "Data Shapley")
add_formula(s, r"\phi_i = \frac{1}{n}\sum_{S \subseteq D\setminus\{i\}} \binom{n-1}{|S|}^{-1}\left(U(S\cup\{i\}) - U(S)\right)",
            cx=int(SW / 2), top=Inches(2.3), height=Inches(0.95), color=F_DARK, fontsize=30)
add_textbox(s, Inches(1.0), Inches(3.7), Inches(11.3), Inches(0.5),
            [("각 데이터의 가치 = 모든 조합에서의 한계기여 U(S∪{i})−U(S) 의 평균.  U(S) = S로 학습한 모델의 검증 성능.",
              14, False, C_BODY)], align=PP_ALIGN.CENTER)
card(s, Inches(2.4), Inches(4.7), Inches(8.5), Inches(1.3), fill=C_CARD)
add_textbox(s, Inches(2.7), Inches(4.95), Inches(7.9), Inches(0.9),
            [("한계: 부분집합마다 모델 재학습 필요 → 2ⁿ 비용", 16, True, C_TERRA),
             ("대형 모델에선 비현실적 → In-Run으로 해결", 13, False, C_BODY)],
            align=PP_ALIGN.CENTER, line_space=1.3)
_page_number(s)

# ================================================================ 5. In-Run Data Shapley
s = newslide()
top = slide_header(s, "In-Run Data Shapley")
add_textbox(s, Inches(1.0), Inches(2.0), Inches(11.3), Inches(0.5),
            [("재학습 없이, 실제로 진행된 한 학습 과정 안에서 스텝별 기여를 누적한다.", 16, True, C_TITLE)],
            align=PP_ALIGN.CENTER)
add_formula(s, r"\phi_i = \sum_t \phi_i^{(t)}",
            cx=int(SW / 2), top=Inches(2.95), height=Inches(1.0), color=F_DARK, fontsize=30)
add_textbox(s, Inches(1.0), Inches(4.4), Inches(11.3), Inches(0.5),
            [("스텝 t의 기여 φᵢ⁽ᵗ⁾ 를 학습 전 구간에 걸쳐 합산.  비용 ≈ 학습 1회.", 14, False, C_BODY)],
            align=PP_ALIGN.CENTER)
_page_number(s)

# ================================================================ 6. Taylor Expansion
s = newslide()
top = slide_header(s, "Taylor Expansion",
                   "스텝 효용 변화 U(θₜ₊₁)−U(θₜ) 를 데이터에 대해 Taylor 전개 → 스텝별 닫힌형 Shapley.")
lx = Inches(0.55); lw = Inches(6.0); cy = Inches(2.2); ch = Inches(3.1)
card(s, lx, cy, lw, ch)
add_textbox(s, lx + Inches(0.35), cy + Inches(0.3), lw - Inches(0.7), Inches(0.5),
            [("1차항", 17, True, C_ACCENT)])
add_formula(s, r"\phi_i^{(t)} \propto \langle\, g^{val},\ g_i\,\rangle",
            cx=int(lx) + int(lw) // 2, top=cy + Inches(1.1), height=Inches(0.6), color=F_DARK, fontsize=28)
add_textbox(s, lx + Inches(0.35), cy + Inches(2.05), lw - Inches(0.7), Inches(0.8),
            [("검증 gradient와 데이터 gradient의 내적 (방향 일치도)", 13, False, C_BODY)],
            align=PP_ALIGN.CENTER)
rx = Inches(6.85); rw = Inches(5.9)
card(s, rx, cy, rw, ch)
add_textbox(s, rx + Inches(0.35), cy + Inches(0.3), rw - Inches(0.7), Inches(0.5),
            [("2차항", 17, True, C_ACCENT)])
add_formula(s, r"\phi_i^{(t)} \supset g_i^\top H\, g",
            cx=int(rx) + int(rw) // 2, top=cy + Inches(1.1), height=Inches(0.6), color=F_DARK, fontsize=28)
add_textbox(s, rx + Inches(0.35), cy + Inches(2.05), rw - Inches(0.7), Inches(0.8),
            [("곡률(Hessian) 반영 → 데이터 상호작용·중복 포착", 13, False, C_BODY)],
            align=PP_ALIGN.CENTER)
_page_number(s)

# ================================================================ 7. FL Setup
s = newslide()
top = slide_header(s, "Federated Learning")
boxes = [
    (Inches(0.7),  "서버 모델  wʳ", "전체에 배포"),
    (Inches(4.5),  "클라이언트 k", "로컬 학습 → Δwₖ"),
    (Inches(8.3),  "서버 집계", "wʳ⁺¹"),
]
by = Inches(2.05); bw = Inches(3.4); bh = Inches(1.4)
for i, (bx, h, sub) in enumerate(boxes):
    fill = C_GREENBG if i == 1 else C_CARD
    line = C_GREENBD if i == 1 else C_CARDBD
    card(s, bx, by, bw, bh, fill=fill, line=line)
    add_textbox(s, bx + Inches(0.2), by + Inches(0.28), bw - Inches(0.4), Inches(0.5),
                [(h, 16, True, C_TITLE)], align=PP_ALIGN.CENTER)
    add_textbox(s, bx + Inches(0.2), by + Inches(0.82), bw - Inches(0.4), Inches(0.4),
                [(sub, 12.5, False, C_BODY)], align=PP_ALIGN.CENTER)
    if i < 2:
        arrow(s, bx + bw, by + Inches(0.5), Inches(0.4), Inches(0.4))
add_formula(s, r"w^{r+1} = w^r + \sum_k p_k\,\Delta w_k",
            cx=int(SW / 2), top=Inches(4.0), height=Inches(0.75), color=F_DARK, fontsize=28)
add_textbox(s, Inches(1.0), Inches(5.1), Inches(11.3), Inches(0.5),
            [("서버가 받는 것 = 클라이언트 업데이트 Δwₖ 뿐 → Flirds의 유일한 입력 (추가 통신 0)", 15, True, C_GREEN)],
            align=PP_ALIGN.CENTER)
_page_number(s)

# ================================================================ 8. Flirds: FL Adaptation
s = newslide()
top = slide_header(s, "Flirds: FL Adaptation")
choices = [
    ("(a) 단위", "데이터포인트 → 클라이언트"),
    ("(b) 분해", "스텝 → 라운드  (라운드당 HVP 1회)"),
    ("(c) 게임", "효용 = 검증 손실 · 곡률 = true Hessian · SGD momentum=0"),
]
cy = Inches(1.85); chh = Inches(0.78)
for i, (h, body) in enumerate(choices):
    y = cy + i * (chh + Inches(0.16))
    card(s, Inches(0.55), y, Inches(12.2), chh)
    add_textbox(s, Inches(0.85), y, Inches(2.0), chh,
                [(h, 15, True, C_ACCENT)], anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(s, Inches(2.9), y, Inches(9.6), chh,
                [(body, 14, False, C_BODY)], anchor=MSO_ANCHOR.MIDDLE)
fy = cy + 3 * (chh + Inches(0.16)) + Inches(0.1)
card(s, Inches(0.55), fy, Inches(12.2), Inches(1.45), fill=C_BG_DARK, line=C_BG_DARK)
add_formula(s, r"\phi_k^{(r)} \approx -\nabla\ell(w^r, z_{val})\cdot \Delta w_k + \frac{1}{2}\,\Delta w_k^\top H_{val}(w^r)\,\Delta W^{(r)}",
            cx=int(SW / 2), top=fy + Inches(0.22), height=Inches(0.62), color=F_WHITE, fontsize=30)
add_formula(s, r"\phi_k = \sum_r \phi_k^{(r)}",
            cx=int(SW / 2), top=fy + Inches(0.98), height=Inches(0.36), color=F_WHITE, fontsize=26)
_page_number(s)

# ================================================================ 9. Error Analysis
s = newslide()
top = slide_header(s, "Approximation Error Analysis",
                   "Taylor 잔차는 변위 크기에 좌우된다 — IRDS와 FL은 변위 규모가 다르다.")
lx = Inches(0.55); lw = Inches(6.0); cy = Inches(2.2); ch = Inches(2.7)
card(s, lx, cy, lw, ch)
add_textbox(s, lx + Inches(0.3), cy + Inches(0.25), lw - Inches(0.6), Inches(2.2),
    [("IRDS — 중앙집중 per-step", 15, True, C_ACCENT),
     ("• 변위 ‖Δθ‖ 작음 (작은 한 스텝)", 13.5, False, C_BODY),
     ("• 1차 잔차 O(‖Δθ‖²) → 이미 작음", 13.5, False, C_BODY),
     ("• 2차 항 역할 미미", 13.5, False, C_BODY)],
    line_space=1.35)
rx = Inches(6.85); rw = Inches(5.9)
card(s, rx, cy, rw, ch, fill=RGBColor(0xF7, 0xEC, 0xE8), line=RGBColor(0xE3, 0xC9, 0xC0))
add_textbox(s, rx + Inches(0.3), cy + Inches(0.25), rw - Inches(0.6), Inches(2.2),
    [("FL — 라운드당 multi-step", 15, True, C_TERRA),
     ("• 변위 ‖Δw_round‖ ≫ per-step", 13.5, False, C_BODY),
     ("• 1차 잔차 O(‖Δw_round‖²) 커짐", 13.5, False, C_BODY),
     ("• 2차(Hessian) 항으로 잔차 ↓", 13.5, True, C_TITLE)],
    line_space=1.35)
add_formula(s, r"O(\|\Delta w_{round}\|^2)\ \longrightarrow\ O(\|\Delta w\|^3)",
            cx=int(SW / 2), top=cy + ch + Inches(0.25), height=Inches(0.6), color=F_DARK, fontsize=28)
_page_number(s)

# ================================================================ 10. Fidelity (LLM)
s = newslide()
top = slide_header(s, "Fidelity (LLM)",
                   "정답 = in-run 정확 오라클 (b) 대비 Spearman (↑). LoRA · 3-seed.  std=N20·2/round, anchor=N5·full.")
cols = ["method", "1B std", "3B std", "7B std", "1B anchor", "7B anchor"]
rows = [
    ["Flirds",     "1.000", "1.000", "0.999", "1.000", "1.000"],
    ["Flirds-1st", "0.999", "1.000", "0.998", "1.000", "1.000"],
    ["loss-heur",  "1.000", "0.999", "0.999", "1.000", "1.000"],
    ["GTG",        "0.975", "0.988", "0.977", "1.000", "1.000"],
    ["FedSV",      "0.910", "0.952", "0.968", "0.700", "0.933"],
    ["ShapleyFL",  "0.194", "0.227", "0.406", "0.700", "0.833"],
    ["FedIF",      "0.157", "0.211", "0.480", "0.067", "0.200"],
]
mb = {(0, 1), (0, 2), (0, 4), (0, 5), (2, 1), (2, 3)}
add_table(s, Inches(2.45), cols, rows, col_widths([1.8, 1.5, 1.5, 1.5, 1.6, 1.6]), 12, manual_bold=mb, row_h=Inches(0.4))
add_textbox(s, Inches(0.55), Inches(5.95), Inches(12.2), Inches(0.5),
            [("Flirds · Flirds-1st · loss-heur 가 천장(≈1.000) — 정확 오라클의 순위를 그대로 재현.", 14, True, C_TITLE)])
_page_number(s)

# ================================================================ 11. 2nd-order Effect
s = newslide()
top = slide_header(s, "2nd-order Effect",
                   "2차 항의 효과는 변위가 큰 무대에서 드러난다 (LLM은 near-additive로 1·2차 포화).")
add_textbox(s, Inches(0.55), Inches(2.0), Inches(12.2), Inches(0.4),
            [("CNN fidelity — Flirds(1차+2차) vs Flirds-1st, vs (b) 오라클 Spearman", 14, True, C_ACCENT)])
cols = ["CNN 시나리오", "Flirds", "Flirds-1st", "차이"]
rows = [
    ["cifar10 / iid",        "0.95",  "0.54",  "+0.41"],
    ["cifar10 / feature_noise", "1.00", "0.89", "+0.11"],
    ["mnist / label_skew",   "0.71",  "0.61",  "+0.10"],
    ["pool 평균 (10×3 seed)", "0.919", "0.832", "+0.087"],
]
mb = {(0, 1), (1, 1), (2, 1), (3, 1), (3, 0), (3, 3)}
add_table(s, Inches(2.45), cols, rows, col_widths([3.4, 2.0, 2.0, 1.4]), 12, manual_bold=mb, row_h=Inches(0.38))
card(s, Inches(0.55), Inches(4.85), Inches(12.2), Inches(1.4), fill=RGBColor(0xF7, 0xEC, 0xE8), line=RGBColor(0xE3, 0xC9, 0xC0))
add_textbox(s, Inches(0.85), Inches(5.1), Inches(11.6), Inches(0.9),
            [("poison(clean-보존 backdoor) 탐지 — LLM 1B, cross-silo N=5", 14, True, C_TERRA),
             ("Flirds(2차) AUROC 0.917  vs  Flirds-1st 0.000 (완전 회피) → 2차 Hessian 항이 부호를 복원", 14, False, C_BODY)],
            line_space=1.35)
_page_number(s)

# ================================================================ 12. Cost & Scalability
s = newslide()
top = slide_header(s, "Cost & Scalability",
                   "라운드당 HVP 1회 vs 2ᴺ 부분집합 평가.  LLM 1B cross-silo N=5 런타임.")
cols = ["방법", "런타임 (N=5, 1B)", "성격"]
rows = [
    ["Flirds-1st",                 "~35 s",   "1차만 — 가장 쌈"],
    ["Flirds (1차+2차)",           "~107 s",  "추정기 본체"],
    ["GTG / FedSV / ShapleyFL",    "~530 s",  "부분집합 스윕"],
    ["(b) in-run 정확 오라클",      "~530 s",  "정답(검증용)"],
    ["Ripple",                     "~4515 s", "최약·최고가"],
]
mb = {(0, 1), (1, 1)}
add_table(s, Inches(2.1), cols, rows, col_widths([3.8, 2.4, 3.0]), 12.5, manual_bold=mb, row_h=Inches(0.44))
card(s, Inches(0.55), Inches(4.9), Inches(12.2), Inches(1.3), fill=C_GREENBG, line=C_GREENBD)
add_textbox(s, Inches(0.85), Inches(5.15), Inches(11.6), Inches(0.9),
            [("정확 오라클 대비 5~15× 저렴 + 대규모 단말 연합(N=100)까지 확장", 15, True, C_GREEN),
             ("N=100 cross-device에서도 Flirds vs per-round 오라클 Spearman +1.000", 13.5, False, C_BODY)],
            line_space=1.35)
_page_number(s)

# ================================================================ 13. Robustness & Detection
s = newslide()
top = slide_header(s, "Robustness & Detection",
                   "위협별 1명 오염 · AUROC = corrupt를 high-φ로 잡는 탐지력(↑) · 1B cross-silo N=5.")
cols = ["method", "noisy", "free-rider(rand)", "free-rider(zero)", "poison"]
rows = [
    ["Flirds (1차+2차)", "1.000", "1.000", "1.000", "0.917"],
    ["Flirds-1st",       "1.000", "1.000", "1.000", "0.000"],
    ["loss-heur",        "1.000", "1.000", "1.000", "1.000"],
    ["(b) oracle",       "1.000", "1.000", "1.000", "1.000"],
    ["FLDetector",       "0.750", "1.000", "0.750", "1.000"],
    ["STD-DAGMM",        "0.417", "1.000", "0.250", "0.750"],
]
add_table(s, Inches(2.5), cols, rows, col_widths([2.6, 1.5, 2.4, 2.4, 1.5]), 12, row_h=Inches(0.4))
add_textbox(s, Inches(0.55), Inches(5.7), Inches(12.2), Inches(0.6),
            [("noisy · free-rider 는 거의 AUROC 1.0.  poison(clean-보존 backdoor)이 분리점 — Flirds-1st 완전 회피, 2차가 버팀.",
              13.5, False, C_BODY)])
_page_number(s)

# ================================================================ 14. Novelty
s = newslide()
top = slide_header(s, "Novelty",
                   "아래 6가지 속성을 동시에 만족하는 방법이 선행 연구에 없다 — 그 교집합이 novelty.")
props = [
    "client-level",
    "in-run (재학습 없음)",
    "닫힌형 1차+2차 Taylor",
    "2차 HVP 상호작용 항",
    "추가 통신 0",
    "LoRA · LLM 스케일",
]
gx0 = Inches(0.7); gw = Inches(3.85); ggap = Inches(0.2); gy = Inches(2.5); gh = Inches(1.1)
for i, p in enumerate(props):
    r, c = divmod(i, 3)
    x = gx0 + c * (gw + ggap)
    y = gy + r * (gh + Inches(0.3))
    card(s, x, y, gw, gh, fill=C_GREENBG, line=C_GREENBD)
    add_textbox(s, x + Inches(0.2), y, gw - Inches(0.4), gh,
                [(p, 14.5, True, C_GREEN)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
_page_number(s)

# ================================================================ 15. Limitation
s = newslide()
top = slide_header(s, "Limitation")
card(s, Inches(0.55), Inches(2.0), Inches(12.2), Inches(3.4), fill=C_CARD)
add_textbox(s, Inches(0.9), Inches(2.35), Inches(11.5), Inches(2.8),
    [("clean-preserving backdoor", 18, True, C_ACCENT),
     ("", 8, False, C_BODY),
     ("• clean 검증 손실을 실제로 낮추는 공격자 → valuation은 정직하게 '기여 높음'으로 평가.", 15, False, C_BODY),
     ("• 즉 탐지 실패가 아니라, 검증 손실을 게임 효용으로 쓴 정의상 옳은 답.", 15, False, C_BODY),
     ("• 기여도(valuation) ≠ 오염 탐지(detection) — 방법론의 일관된 경계.", 15, True, C_TITLE),
     ("• 조건부 경계(큰 scaled-update × 큰 모델) → 상보적 탐지기가 필요한 지점.", 15, False, C_BODY)],
    line_space=1.45)
_page_number(s)

# ================================================================ 16. Thank you / Q&A
s = newslide()
dark_bg(s)
add_textbox(s, Inches(0.9), Inches(2.7), Inches(11.5), Inches(1.2),
            [("Thank you", 46, True, C_WHITE)])
add_textbox(s, Inches(0.95), Inches(3.95), Inches(11.5), Inches(0.8),
            [("Q & A", 30, True, C_ICE)])
_page_number(s)

# ---------------------------------------------------------------- save
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flirds-seminar-intro.pptx")
prs.save(OUT)
print("saved:", OUT, "slides:", len(prs.slides._sldIdLst))
