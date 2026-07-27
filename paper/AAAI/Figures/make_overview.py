# -*- coding: utf-8 -*-
"""Figure 1 (overview) for Flirds — generated, not hand-drawn.

Regenerate:  python make_overview.py   (writes overview.png @300dpi + overview.pdf)

Layout contract (mirrors figure1_flirds_concept_prompt.md, 2026-07-27 terminology):
  (a) observed federated round -> round game (Eq. 5) -> closed form (Eq. 6)
  (b) three values: retraining-based Shapley / in-run Shapley / Flirds;
      grey dashed two-way link = "different games, different questions (empirical, §5.2)";
      bold teal one-way link   = "2nd-order Taylor truncation — the only approximation".
Authored at 13.8x6.2 in; at \textwidth (~7 in) the smallest text is ~6.0 pt.
The script prints an OVERFLOW warning if any registered text exceeds its x budget.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "dejavusans"

# palette (prompt: navy / teal / muted orange / grey, white bg)
NAVY = "#1E3A5F"
TEAL = "#0C7B7B"
TEAL_FILL = "#F0F8F7"
ORANGE = "#C05A1E"
INK = "#243244"
GREY = "#7C8593"
LGREY = "#AEB6C1"
CARD = "#44546A"

W, H = 13.8, 6.2
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

_checks = []  # (label, artist, xmax)


def rbox(x, y, w, h, ec, lw=1.2, ls="solid", fc="none", r=0.08, z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                       ec=ec, fc=fc, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def arrow(x0, y0, x1, y1, color, lw=1.2, ls="solid", ms=10, style="-|>", z=3):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, color=color, lw=lw,
                        linestyle=ls, mutation_scale=ms, zorder=z)
    ax.add_patch(a)
    return a


def txt(x, y, s, size, color=INK, ha="left", va="center", weight="normal",
        style="normal", rot=0, z=5, xmax=None):
    t = ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                fontweight=weight, fontstyle=style, rotation=rot, zorder=z)
    if xmax is not None:
        _checks.append((s[:38], t, xmax))
    return t


def person(cx, cy, s, color):
    ax.add_patch(Circle((cx, cy + 0.35 * s), 0.16 * s, ec=color, fc="white", lw=1.3, zorder=6))
    ax.add_patch(Wedge((cx, cy - 0.12 * s), 0.30 * s, 0, 180, ec=color, fc="white", lw=1.3, zorder=6))


def server_icon(cx, cy, color=INK):
    for dy in (0.21, 0.0, -0.21):
        rbox(cx - 0.32, cy + dy - 0.08, 0.64, 0.16, ec=color, lw=1.1, r=0.03, z=6)
        ax.add_patch(Circle((cx - 0.21, cy + dy), 0.019, color=color, zorder=7))
        ax.add_patch(Circle((cx - 0.12, cy + dy), 0.019, color=color, zorder=7))
        ax.plot([cx + 0.02, cx + 0.24], [cy + dy, cy + dy], color=color, lw=0.9, zorder=7)


def zigzag(x0, x1, y, color):
    n = 5
    xs = [x0 + (x1 - x0) * i / n for i in range(n + 1)]
    ys = [y + (0.045 if i % 2 else -0.045) for i in range(n + 1)]
    ax.plot(xs, ys, color=color, lw=1.0, ls=(0, (2.5, 2)), zorder=6)
    arrow(xs[-1], ys[-1], xs[-1] + 0.11, ys[-1] + 0.02, color, lw=1.0, ms=6)


# ============================ panel (a) ============================
txt(0.2, 5.95, "(a)  The federated round game (Eq. 5)", 14, NAVY, weight="bold", xmax=8.7)

# ---- a1: one observed round ----
rbox(0.3, 0.35, 3.7, 5.25, ec=LGREY, lw=1.1, ls=(0, (4, 2.5)), r=0.12)
txt(2.15, 5.36, "one FedAvg round $r$", 12.6, GREY, ha="center", style="italic", xmax=3.95)
txt(2.15, 5.1, "cohort $P_r \\subseteq \\{1,\\ldots,N\\}$", 12, INK, ha="center", xmax=3.95)


def client_box(y, name, selected=True):
    if selected:
        rbox(0.5, y, 1.42, 0.8, ec=TEAL, lw=1.3, fc=TEAL_FILL, r=0.08)
        person(0.78, y + 0.52, 0.5, TEAL)
        txt(1.02, y + 0.52, name, 12, INK)
        zigzag(0.68, 1.6, y + 0.2, GREY)
    else:
        rbox(0.5, y, 1.42, 0.8, ec=LGREY, lw=1.1, ls=(0, (3, 2)), fc="#F4F5F6", r=0.08)
        person(0.78, y + 0.52, 0.5, LGREY)
        txt(1.02, y + 0.56, "not selected", 11.3, GREY)
        txt(1.02, y + 0.34, "this round", 11.3, GREY)


client_box(4.05, "Client 1")
client_box(2.95, "Client 2")
txt(1.21, 2.66, "⋮", 13, GREY, ha="center")
client_box(1.6, "", selected=False)

server_icon(3.28, 3.95)
txt(3.28, 3.5, "Server", 12, INK, ha="center")

# broadcast / submit arrows (participants only)
arrow(2.92, 4.12, 1.97, 4.62, GREY, lw=1.0, ms=8)
txt(2.4, 4.58, "$w^r$", 11.5, GREY, ha="center")
arrow(1.97, 4.38, 2.92, 3.94, TEAL, lw=1.4, ms=9)
txt(2.42, 4.02, "$\\delta_1^r$", 12, TEAL, ha="center")
arrow(2.92, 3.7, 1.97, 3.5, GREY, lw=1.0, ms=8)
txt(2.42, 3.72, "$w^r$", 11.5, GREY, ha="center")
arrow(1.97, 3.32, 2.92, 3.52, TEAL, lw=1.4, ms=9)
txt(2.42, 3.28, "$\\delta_2^r$", 12, TEAL, ha="center")

txt(2.15, 1.3, "dashed path = local steps (not observed)", 11.8, GREY, ha="center", style="italic", xmax=3.95)
txt(2.15, 0.96, "only the accumulated displacement", 11.8, INK, ha="center", xmax=3.95)
txt(2.15, 0.73, "$\\delta_k^r := \\Delta w_k^r$ is sent to the server", 11.8, INK, ha="center", xmax=3.95)
txt(2.15, 0.49, "weights: $p_k^r = n_k\\,/\\,\\Sigma_{j\\in P_r}\\, n_j$", 11.8, INK, ha="center", xmax=3.95)

# ---- a2 top: round game (Eq. 5) ----
rbox(4.25, 2.28, 4.5, 3.32, ec=CARD, lw=1.2, r=0.1)
txt(4.42, 5.36, "Round game $u_r$ (Eq. 5)", 13.8, NAVY, weight="bold", xmax=8.7)

# coalition S chain (label line, chain below)
txt(4.42, 4.94, "coalition $S=\\{1,3\\}$:", 12.2, INK, xmax=6.6)
ax.add_patch(Circle((6.78, 4.94), 0.032, color=INK, zorder=6))
txt(6.68, 4.94, "$w^r$", 11.5, INK, ha="right")
arrow(6.82, 4.94, 7.5, 4.94, TEAL, lw=1.6, ms=10)
txt(7.17, 5.12, "$p_1^r\\delta_1^r$", 11.3, TEAL, ha="center")
arrow(7.54, 4.94, 8.22, 4.94, TEAL, lw=1.6, ms=10)
txt(7.89, 5.12, "$p_3^r\\delta_3^r$", 11.3, TEAL, ha="center")
ax.add_patch(Circle((8.26, 4.94), 0.032, color=INK, zorder=6))
txt(8.26, 4.72, "$w^r{+}\\Delta_S^r$", 11.5, INK, ha="center")

txt(4.42, 4.4, "left out $\\Rightarrow$ $p_2^r\\delta_2^r \\mapsto \\mathbf{0}$ (zero vector);", 12, GREY, xmax=8.7)
txt(4.42, 4.16, "kept clients keep their $p_k^r$ (no renormalization)", 12, GREY, xmax=8.7)

txt(4.42, 3.76, "$u_r(S) \\,=\\, \\ell_{\\mathrm{val}}(w^r) - \\ell_{\\mathrm{val}}(w^r{+}\\Delta_S^r)$", 13, ORANGE, xmax=8.7)
txt(4.42, 3.48, "= validation-loss decrease from applying only $S$", 11.8, ORANGE, style="italic", xmax=8.7)

# grand coalition chain (inline)
txt(4.42, 3.06, "grand $S=P_r$:", 12.2, INK, xmax=5.75)
ax.add_patch(Circle((5.88, 3.06), 0.032, color=INK, zorder=6))
for x0, lab in [(5.92, "$p_1^r\\delta_1^r$"), (6.48, "$p_2^r\\delta_2^r$"), (7.04, "$p_3^r\\delta_3^r$")]:
    arrow(x0, 3.06, x0 + 0.52, 3.06, NAVY, lw=2.2, ms=11)
    txt(x0 + 0.26, 3.24, lab, 10.8, NAVY, ha="center")
ax.add_patch(Circle((7.6, 3.06), 0.032, color=NAVY, zorder=6))
txt(7.7, 3.06, "$w^{r+1}$", 12.4, NAVY, weight="bold", xmax=8.7)
txt(4.42, 2.74, "= the actual server update (Eq. 1)", 11.8, NAVY, xmax=8.7)
txt(4.42, 2.48, "over rounds:  $\\Sigma_r\\, u_r(P_r) \\,=\\, \\ell_{\\mathrm{val}}(w^0) - \\ell_{\\mathrm{val}}(w^R)$", 11.8, NAVY, xmax=8.7)

# taylor arrow between boxes
arrow(5.1, 2.24, 5.1, 1.92, TEAL, lw=2.6, ms=14)
txt(5.3, 2.08, "2nd-order Taylor  $u_r \\to \\hat{u}_r$", 12.8, TEAL, weight="bold", xmax=8.7)

# ---- a2 bottom: closed form (Eq. 6) ----
rbox(4.25, 0.35, 4.5, 1.57, ec=TEAL, lw=1.4, fc=TEAL_FILL, r=0.1)
txt(4.42, 1.68, "Flirds (Eq. 6): exact Shapley value of $\\hat{u}_r$", 13.8, NAVY, weight="bold", xmax=8.7)
txt(6.5, 1.24,
    "$\\hat{\\phi}_k^{(r)} = -\\,p_k^r\\langle g_r,\\, \\delta_k^r\\rangle \\;-\\; \\frac{1}{2}\\, p_k^r\\langle \\delta_k^r,\\, H_r \\Delta W_r\\rangle$",
    13.5, INK, ha="center", xmax=8.7)
txt(6.5, 0.84, "$\\hat{\\phi}_k = \\Sigma_{r:\\,k\\in P_r}\\, \\hat{\\phi}_k^{(r)}$  (accumulated online per round)", 11.8, INK, ha="center", xmax=8.7)
txt(6.5, 0.56, "one JVP/round gives $g_r{=}\\nabla\\ell_{\\mathrm{val}}(w^r)$ and $H_r\\Delta W_r$ ($\\Delta W_r{=}\\Delta_{P_r}^r$)", 11.2, GREY, ha="center", xmax=8.72)

# divider
ax.plot([8.95, 8.95], [0.35, 5.85], color=LGREY, lw=0.9)

# ============================ panel (b) ============================
txt(9.15, 5.95, "(b)  What is exact, what is approximated?", 14, NAVY, weight="bold", xmax=13.75)


def glyph_tree(cx, top):
    ax.add_patch(Circle((cx, top), 0.028, color=INK, zorder=6))
    for dx in (-0.26, 0.0, 0.26):
        ax.plot([cx, cx + dx], [top - 0.02, top - 0.22], color=INK, lw=0.8, zorder=5)
        ax.add_patch(Circle((cx + dx, top - 0.24), 0.022, color=INK, zorder=6))
        for ddx in (-0.08, 0.0, 0.08):
            ax.plot([cx + dx, cx + dx + ddx], [top - 0.26, top - 0.42], color=INK, lw=0.6, zorder=5)
            ax.add_patch(Circle((cx + dx + ddx, top - 0.44), 0.015, color=INK, zorder=6))
    txt(cx, top - 0.58, "⋯", 10.5, GREY, ha="center")


def glyph_traj(cx, cy, fan=True):
    x0 = cx - 0.42
    arrow(x0, cy, cx + 0.46, cy, INK, lw=1.0, ms=7)
    for xn in (x0 + 0.1, x0 + 0.38, x0 + 0.66):
        ax.add_patch(Circle((xn, cy), 0.024, color=INK, zorder=6))
        if fan:
            for ddx in (-0.075, 0.0, 0.075):
                ax.plot([xn, xn + ddx], [cy - 0.03, cy - 0.21], color=INK, lw=0.6, zorder=5)
                ax.add_patch(Circle((xn + ddx, cy - 0.23), 0.015, color=INK, zorder=6))
        else:
            ax.plot([xn, xn], [cy - 0.03, cy - 0.19], color=INK, lw=0.8, zorder=5)
            ax.add_patch(Circle((xn, cy - 0.22), 0.018, color=INK, zorder=6))


def card(y, h, title, lines, cost, ec=CARD, fc="white", lw=1.2, cost_size=11.8):
    rbox(9.15, y, 4.25, h, ec=ec, lw=lw, fc=fc, r=0.1)
    txt(9.32, y + h - 0.2, title, 13.8, INK, weight="bold", xmax=12.5)
    for i, (s, kw) in enumerate(lines):
        txt(9.32, y + h - 0.5 - 0.26 * i, s, 12, xmax=12.5, **kw)
    txt(9.32, y + 0.17, cost, cost_size, INK, xmax=12.5)


# card 1: retraining-based Shapley
card(4.24, 1.42, "Retraining-based Shapley $\\phi^{\\mathrm{re}}$", [
    ("retrain from scratch per subset:", {}),
    ("\u201cwhat if only $S$ had participated?\u201d", {}),
    ("the server cannot run it (no raw data)", {"color": GREY}),
], "cost:  $2^N$ retrainings")
glyph_tree(13.0, 5.38)

# connector 1: different games (two-way, grey dashed)
arrow(9.62, 4.2, 9.62, 3.8, GREY, lw=1.3, ls=(0, (3, 2)), ms=9, style="<|-|>")
txt(9.88, 4.11, "different games, different questions", 12.6, INK, xmax=13.35)
txt(9.88, 3.88, "relation is empirical (§5.2)", 11.8, GREY, style="italic", xmax=13.35)

# card 2: in-run Shapley
card(2.34, 1.42, "In-run Shapley  $\\phi^{\\mathrm{in}}$", [
    ("fix the realized trajectory;", {}),
    ("enumerate all $2^{|P_r|}$ coalitions", {}),
    ("per round; sum over rounds", {}),
], "cost:  $\\Sigma_r\\, 2^{|P_r|}$ validation evaluations", cost_size=11.3)
glyph_traj(13.0, 3.3)

# connector 2: taylor truncation (one-way, bold teal)
arrow(9.62, 2.3, 9.62, 1.9, TEAL, lw=2.6, ms=14)
txt(9.88, 2.21, "2nd-order Taylor truncation", 12.8, TEAL, weight="bold", xmax=13.35)
txt(9.88, 1.98, "the only approximation", 11.8, GREY, style="italic", xmax=13.35)

# card 3: Flirds
card(0.44, 1.42, "Flirds  $\\hat{\\phi}$", [
    ("the exact Shapley value of the", {}),
    ("2nd-order Taylor surrogate $\\hat{u}_r$ (Eq. 6)", {}),
    ("server-side; no extra client work", {"color": GREY}),
], "cost:  1 JVP + $|P_r|$ inner products / round", ec=TEAL, fc=TEAL_FILL, lw=1.5, cost_size=11.3)
glyph_traj(13.0, 1.5, fan=False)

# bracket: cards 2+3 target the same round game
bx = 13.48
ax.plot([bx, bx + 0.06, bx + 0.06, bx], [3.76, 3.76, 0.44, 0.44], color=NAVY, lw=1.1)
txt(13.62, 2.1, "both target the round game (Eq. 5)", 11.5, NAVY, ha="center", rot=90)

# ---------------- overflow report ----------------
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = ax.transData.inverted()
bad = 0
for label, t, xmax in _checks:
    bb = t.get_window_extent(renderer)
    x1 = inv.transform((bb.x1, bb.y1))[0]
    if x1 > xmax + 0.02:
        print(f"OVERFLOW +{x1 - xmax:.2f}in  (limit {xmax}): {label}")
        bad += 1
print(f"{bad} overflow(s)")

fig.savefig("overview.png", dpi=300, facecolor="white")
fig.savefig("overview.pdf", facecolor="white")
print("wrote overview.png / overview.pdf")
