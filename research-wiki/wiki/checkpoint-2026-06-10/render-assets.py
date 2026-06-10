# -*- coding: utf-8 -*-
"""Build a single combined HTML + per-file PDFs from the checkpoint .md files.
HTML/PDF share the same MD->HTML conversion; only CSS differs.
Korean via embedded Nanum fonts (PDF) / system font stack (HTML)."""
import re, os, html as ihtml, fitz
from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

CKPT = "/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/research-wiki/wiki/checkpoint-2026-06-10"
PDFDIR = os.path.join(CKPT, "pdf")
os.makedirs(PDFDIR, exist_ok=True)
ORDER = ["00-overview","01-research-value","02-experimental-setup",
         "03-baselines-and-prior-work","04-plan-vs-implementation-divergences",
         "05-open-issues-and-next","06-closest-competitors-fedif-fedtsv-ripple",
         "07-novelty-limitations-analysis"]
FONTS = {"kf":"/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
         "kb":"/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
         "km":"/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf"}

# ---------- LaTeX inline math -> unicode/HTML ----------
GREEK = {r'\Delta':'Δ',r'\nabla':'∇',r'\ell':'ℓ',r'\eta':'η',r'\sigma':'σ',r'\Pi':'Π',
         r'\Phi':'Φ',r'\phi':'φ',r'\cdots':'⋯',r'\cdot':'·',r'\times':'×',r'\approx':'≈',
         r'\neq':'≠',r'\top':'⊤',r'\Sigma':'Σ',r'\mu':'μ',r'\theta':'θ',r'\lambda':'λ',r'\eta':'η'}
def latex_to_html(s):
    s = s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    s = s.replace(r'\lVert','‖').replace(r'\rVert','‖').replace(r'\|','‖')
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1⁄\2', s)
    for k,v in GREEK.items(): s = s.replace(k,v)
    s = re.sub(r'\^\{([^}]*)\}', lambda m:'<sup>'+m.group(1)+'</sup>', s)
    s = re.sub(r'\^([A-Za-z0-9])', lambda m:'<sup>'+m.group(1)+'</sup>', s)
    s = re.sub(r'_\{([^}]*)\}', lambda m:'<sub>'+m.group(1)+'</sub>', s)
    s = re.sub(r'_([A-Za-z0-9])', lambda m:'<sub>'+m.group(1)+'</sub>', s)
    s = s.replace('\\','')
    return '<span class="math">'+s+'</span>'

# ---------- pipeline diagram (works in browser + PyMuPDF Story) ----------
DIAGRAM = """
<div class="diagram">
<table class="dgr"><tr>
<td class="dbox c-in">데이터 레이어<br><span class="dpath">data/llm.py</span><br>build (silo5 N=5) / build_crossdevice (device100 N=100)</td>
<td class="dplus">＋</td>
<td class="dbox c-in">위협 주입<br><span class="dpath">data/corruptors.py</span><br>answer_swap / free_rider / backdoor<br>+ llm_server scaled_attackers</td>
</tr></table>
<div class="darr">↓</div>
<div class="dbox c-fl">FL 루프 (FedAvg, SGD mom=0) — <span class="dpath">fl/server.py · fl/llm_server.py</span><br>run_fedavg_logs → <b>logs = [(w_r, deltas_map)]</b>, deltas_map[c]=(Δw_c, n_c)</div>
<div class="darr">↓ &nbsp; 얼린 궤적 logs 위에서 분기 &nbsp; ↓</div>
<table class="dgr dfan"><tr>
<td class="dbox c-est">ESTIMATOR<br><span class="dpath">core/flirds_estimator.py</span><br>Flirds / Flirds-1st<br>(1 HVP/round)</td>
<td class="dbox c-orc">(b) in-run oracle<br><span class="dpath">oracle/in_run_sv.py</span><br>exact 2^N · perround</td>
<td class="dbox c-orc">(a) retrain oracle<br><span class="dpath">oracle/exact_sv_llm.py</span><br>2^N 재학습</td>
<td class="dbox c-base">valuation baselines ×7<br>GTG·FedSV·Ripple·Banzhaf<br>ShapleyFL·ComFedSV·loss-heur</td>
<td class="dbox c-det">detectors ×4<br>FLDetector·STD-DAGMM<br>FLTrust·FedDQC</td>
</tr></table>
<div class="darr">↓</div>
<div class="dbox c-eval">평가 — <span class="dpath">eval/metrics.py · eval/generate.py</span><br>Spearman(vs oracle) · AUROC(vs 주입라벨) · ROUGE-L · ASR</div>
<div class="darr">↓</div>
<div class="dbox c-sel">selection / filtering — select_topk → arms {full / flirds_topk / random_k}<br>(detection→성능: noisy+free-rider 드롭 → val_loss↓)</div>
</div>
"""

# ---------- markdown parser ----------
def make_md():
    md = MarkdownIt("gfm-like", {"html": True, "linkify": False, "breaks": False})
    md.use(dollarmath_plugin)
    md.renderer.rules["math_inline"] = lambda toks,i,o,e: latex_to_html(toks[i].content)
    md.renderer.rules["math_block"]  = lambda toks,i,o,e: '<div class="math">'+latex_to_html(toks[i].content)+'</div>'
    return md
MD = make_md()

def split_front(text):
    meta = {}
    if text.startswith("---"):
        m = re.match(r'^---\n(.*?)\n---\n?', text, re.S)
        if m:
            for line in m.group(1).splitlines():
                mm = re.match(r'^(\w+):\s*(.+)$', line.strip())
                if mm: meta[mm.group(1)] = mm.group(2).strip().strip('"').strip("'")
            text = text[m.end():]
    return meta, text

def preprocess(md_text, for_html):
    # mermaid fence -> diagram HTML block
    md_text = re.sub(r'```mermaid.*?```', "\n\n"+DIAGRAM+"\n\n", md_text, flags=re.S)
    # wikilinks [[a|b]] -> b ; [[a]] -> a  (as styled span)
    md_text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', lambda m:'<span class="wl">'+m.group(2)+'</span>', md_text)
    md_text = re.sub(r'\[\[([^\]]+)\]\]', lambda m:'<span class="wl">'+m.group(1).split('/')[-1]+'</span>', md_text)
    # markdown links to *.md
    def mdlink(m):
        label, target = m.group(1), m.group(2)
        if for_html:
            stem = re.sub(r'\.md(#.*)?$', '', target)
            return f'[{label}](#doc-{stem})'
        return f'<span class="wl">{label}</span>'
    md_text = re.sub(r'\[([^\]]+)\]\((\d[\w\-]*\.md(?:#[^)]*)?)\)', mdlink, md_text)
    md_text = re.sub(r'\[([^\]]+)\]\((README\.md(?:#[^)]*)?)\)', mdlink, md_text)
    return md_text

CHIP = re.compile(r'(\[CODE-VERIFIED\]|\[CODE\]|\[PDF-VERIFIED\]|\[PDF\]|\[WEB-EXTRACT-ONLY\]|\[WEB\]|\[DOC-CLAIMED\]|\[DOC\]|\[ⓑ[^\]]*\]|\[ⓐ[^\]]*\]|\[ⓒ[^\]]*\])')
def chip_sub(m):
    t = m.group(1); inner = t[1:-1]
    if t.startswith('[CODE') or t.startswith('[PDF'): cls='v'
    elif t.startswith('[DOC') or t.startswith('[WEB'): cls='c'
    elif t.startswith('[ⓐ'): cls='a'
    elif t.startswith('[ⓑ'): cls='b'
    else: cls='g'
    return f'<span class="chip chip-{cls}">{inner}</span>'
STATE = re.compile(r'(ⓐ|ⓑ|ⓒ)')
def state_sub(m):
    return f'<span class="chip chip-{ {"ⓐ":"a","ⓑ":"b","ⓒ":"g"}[m.group(1)] }">{m.group(1)}</span>'

def decorate(html_str):
    parts = re.split(r'(<pre[\s\S]*?</pre>|<code>[\s\S]*?</code>)', html_str)
    out=[]
    for i,p in enumerate(parts):
        if i%2==1: out.append(p); continue   # code segment, leave alone
        p = CHIP.sub(chip_sub, p)
        p = STATE.sub(state_sub, p)
        out.append(p)
    return "".join(out)

def render_body(md_text, for_html):
    body = MD.render(preprocess(md_text, for_html))
    body = decorate(body)
    # wrap wide tables for horizontal scroll (html only)
    if for_html:
        body = body.replace("<table>", '<div class="tw"><table>').replace("</table>","</table></div>")
    return body

def doc_html(stem, for_html):
    raw = open(os.path.join(CKPT, stem+".md"), encoding="utf-8").read()
    meta, text = split_front(raw)
    title = meta.get("title", stem)
    num = str(int(stem.split("-")[0]))
    head = f'<div class="dochead"><span class="docnum">{num}</span><h1 class="doctitle">{ihtml.escape(title)}</h1></div>'
    note = ''
    if meta.get("note"): note = f'<p class="docnote">{ihtml.escape(meta["note"])}</p>'
    return title, head + note + render_body(text, for_html)

# ================= COMBINED HTML =================
HTML_CSS = r"""
:root{--ink:#1f2733;--mut:#5b6675;--line:#e3e8ef;--bg:#f6f8fb;--card:#fff;
--accent:#3949ab;--accent2:#00897b;--code:#0b1324;--codebg:#f3f5f9;}
*{box-sizing:border-box;} html{scroll-behavior:smooth;}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:"Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic","NanumGothic","Segoe UI",sans-serif;
font-size:15px;line-height:1.7;-webkit-font-smoothing:antialiased;}
a{color:var(--accent);text-decoration:none;} a:hover{text-decoration:underline;}
.layout{display:flex;align-items:flex-start;max-width:1680px;margin:0 auto;}
/* sidebar */
.side{position:sticky;top:0;height:100vh;overflow-y:auto;overscroll-behavior:contain;width:288px;flex:0 0 288px;
background:#0f1830;color:#cdd6e6;padding:22px 16px 60px;}
.side .brand{font-weight:800;font-size:18px;color:#fff;letter-spacing:.3px;}
.side .brand small{display:block;font-weight:500;color:#8c9bb8;font-size:12px;margin-top:3px;}
.side nav{margin-top:18px;}
.side .navdoc{display:block;color:#e7ecf7;font-weight:700;padding:7px 10px;border-radius:8px;margin-top:6px;font-size:13.5px;}
.side .navdoc:hover{background:#1c2a4a;text-decoration:none;}
.side .navdoc .n{display:inline-block;width:26px;color:#6fc3b6;font-weight:800;}
.side .navsub{display:block;color:#9fb0cc;padding:3px 10px 3px 38px;font-size:12px;border-radius:6px;}
.side .navsub:hover{background:#172642;color:#fff;text-decoration:none;}
/* main */
.main{flex:1 1 auto;min-width:0;padding:0 38px 90px;overflow-wrap:break-word;}
.hero{padding:48px 0 26px;border-bottom:2px solid var(--line);margin-bottom:10px;}
.hero h1{font-size:30px;margin:0 0 8px;letter-spacing:-.3px;}
.hero .sub{color:var(--mut);font-size:15px;max-width:760px;}
.hero .meta{margin-top:14px;font-size:12.5px;color:var(--mut);}
.legend{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px;}
.legend .chip{cursor:default;}
.section{padding:30px 0;border-bottom:1px solid var(--line);}
.dochead{display:flex;align-items:center;gap:14px;margin:8px 0 6px;}
.docnum{flex:0 0 auto;background:linear-gradient(135deg,var(--accent),#5c6bc0);color:#fff;
font-weight:800;font-size:15px;width:46px;height:46px;border-radius:12px;display:flex;align-items:center;justify-content:center;
box-shadow:0 6px 16px rgba(57,73,171,.28);}
.doctitle{font-size:21px;margin:0;line-height:1.3;}
.docnote{color:var(--mut);font-size:13px;background:#eef2f8;border-left:3px solid var(--accent2);padding:8px 12px;border-radius:0 8px 8px 0;margin:8px 0 16px;}
h2{font-size:18px;margin:26px 0 10px;padding-left:11px;border-left:4px solid var(--accent);}
h3{font-size:15.5px;margin:18px 0 7px;color:#28324a;}
p{margin:9px 0;} ul,ol{margin:9px 0;padding-left:22px;} li{margin:3px 0;}
strong,b{font-weight:700;color:#11192a;}
/* code */
code{font-family:"SFMono-Regular","NanumGothicCoding",Consolas,monospace;font-size:12.6px;
background:var(--codebg);color:#143;padding:1.5px 5px;border-radius:5px;border:1px solid #e6ebf2;}
pre{background:var(--code);color:#e7edf6;padding:14px 16px;border-radius:10px;overflow-x:auto;
border:1px solid #0a1020;box-shadow:inset 0 0 0 1px #1b2740;font-size:12.6px;line-height:1.55;}
pre code{background:none;border:none;color:inherit;padding:0;}
/* tables */
.tw{overflow-x:auto;margin:12px 0;border:1px solid var(--line);border-radius:10px;}
table{border-collapse:collapse;width:100%;font-size:12.9px;background:var(--card);}
th{background:#eef1f8;color:#27314a;text-align:left;font-weight:700;}
th,td{border:1px solid #e6ebf2;padding:7px 10px;vertical-align:top;}
tbody tr:nth-child(even){background:#fafbfe;}
/* blockquote callouts */
blockquote{margin:13px 0;background:#fff8e6;border:1px solid #f3e2b3;border-left:5px solid #f0ad2e;
padding:10px 14px;border-radius:0 10px 10px 0;color:#5b4a16;}
blockquote p{margin:5px 0;} blockquote strong{color:#7a5b00;}
/* chips */
.chip{display:inline-block;font-size:11px;font-weight:700;padding:1px 8px;border-radius:20px;
line-height:1.7;white-space:nowrap;vertical-align:baseline;border:1px solid transparent;}
.chip-a{background:#fff1e0;color:#b25f00;border-color:#ffd8a8;}
.chip-b{background:#e3f6ec;color:#1b7a44;border-color:#aee3c4;}
.chip-g{background:#eef0f4;color:#54607a;border-color:#d7dce6;}
.chip-v{background:#e6efff;color:#2b56b8;border-color:#bcd2ff;}
.chip-c{background:#fdeede;color:#a3650a;border-color:#f3d6a8;}
.wl{color:#5560a8;border-bottom:1px dotted #9aa4d6;font-style:italic;}
.math{font-family:"Cambria Math","Times New Roman",serif;font-style:italic;background:#f4f1fb;padding:0 3px;border-radius:4px;}
/* diagram */
.diagram{margin:18px 0;padding:18px;background:linear-gradient(180deg,#f8fafc,#eef2f8);border:1px solid var(--line);border-radius:14px;overflow-x:auto;}
.dgr{border-collapse:separate;border-spacing:10px 0;margin:0 auto;}
.dfan{table-layout:fixed;width:100%;}
.dbox{border-radius:10px;padding:10px 12px;text-align:center;font-size:12px;line-height:1.45;
border:1.5px solid #c7d0e0;background:#fff;box-shadow:0 2px 6px rgba(20,30,60,.06);}
.dpath{font-family:"NanumGothicCoding",monospace;font-size:10.6px;color:#5b6675;}
.darr{text-align:center;color:#7a8aa8;font-weight:800;font-size:15px;margin:7px 0;}
.dplus{font-weight:800;color:#8a97b3;padding:0 4px;vertical-align:middle;}
.c-in{background:#eef4ff;border-color:#b7ccf6;}
.c-fl{background:#fff4e6;border-color:#ffd8a8;}
.c-est{background:#e7f6ef;border-color:#9ad8b6;font-weight:700;}
.c-orc{background:#fdeede;border-color:#f3cf9a;}
.c-base{background:#eef0f4;border-color:#d2d8e4;}
.c-det{background:#fdeaf0;border-color:#f3b9cd;}
.c-eval{background:#eaf3ff;border-color:#aeccf6;}
.c-sel{background:#ecfdf5;border-color:#9fe0c4;}
@media print{.side{display:none;} .main{padding:0;} body{font-size:11px;} .section{break-inside:avoid;}}
@media(max-width:980px){.side{display:none;} .main{padding:0 16px 60px;}}
"""

def h2_anchors(stem):
    raw = open(os.path.join(CKPT, stem+".md"), encoding="utf-8").read()
    _, text = split_front(raw)
    out=[]
    for line in text.splitlines():
        m = re.match(r'^##\s+(.+)$', line)
        if m:
            t = re.sub(r'[`*]','', m.group(1)).strip()
            out.append(t)
    return out[:8]

def build_combined():
    sections=[]; nav=[]
    for stem in ORDER:
        title, body = doc_html(stem, for_html=True)
        sid = "doc-"+stem
        sections.append(f'<section class="section" id="{sid}">{body}</section>')
        num = str(int(stem.split("-")[0]))
        short = title.split("—")[-1].strip() if "—" in title else title
        nav.append(f'<a class="navdoc" href="#{sid}"><span class="n">{num}</span>{ihtml.escape(short)}</a>')
    legend = ('<span class="chip chip-b">ⓑ 실측 결과</span>'
              '<span class="chip chip-a">ⓐ 코드+smoke</span>'
              '<span class="chip chip-g">ⓒ 미실행/설계</span>'
              '<span class="chip chip-v">[CODE/PDF] 원본대조</span>'
              '<span class="chip chip-c">[DOC/WEB] 문서주장</span>')
    hero = f'''<div class="hero"><h1>Flirds 연구 체크포인트 — 2026-06-10</h1>
<div class="sub">Phase 0–2 구현 + step5 matrix orchestrator 종료 시점의 전체 재오리엔테이션.
코드·raw 로그·논문 PDF를 직접 대조해 정리한 8개 문서(00–07)를 한 곳에 모았습니다.</div>
<div class="legend">{legend}</div>
<div class="meta">근거 추적: 코드 <code>path:line</code> · 실측 <code>codes/runs/*/metrics.json</code> · 논문 <code>raw/papers/flirds/*.pdf</code>. 3-state(ⓐ/ⓑ/ⓒ) 규율 적용.</div></div>'''
    side = f'''<aside class="side"><div class="brand">Flirds Checkpoint<small>2026-06-10 · 재오리엔테이션</small></div>
<nav>{"".join(nav)}</nav></aside>'''
    doc = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flirds 체크포인트 2026-06-10</title><style>{HTML_CSS}</style></head>
<body><div class="layout">{side}<main class="main">{hero}{"".join(sections)}</main></div></body></html>'''
    outp = os.path.join(CKPT, "flirds-checkpoint-2026-06-10.html")
    open(outp,"w",encoding="utf-8").write(doc)
    return outp, len(doc)

# ================= PER-FILE PDFs =================
PDF_CSS = """
* { font-family: kf, sans-serif; font-size: 9.4pt; line-height: 1.5; color:#1f2733; }
h1.doctitle, .doctitle { font-family: kb; }
b, strong, th, h1, h2, h3 { font-family: kb; }
code, pre, .dpath { font-family: km; }
.dochead { padding: 4px 0 2px; }
.docnum { background:#3949ab; color:#fff; font-family:kb; padding:2px 7px; border-radius:5px; font-size:10pt; }
.doctitle { color:#26315a; font-size:15pt; margin:2px 0; }
.docnote { color:#5b6675; background:#eef2f8; padding:5px 9px; font-size:8.4pt; }
h2 { color:#26315a; font-size:12.5pt; margin:12px 0 5px; border-left:3px solid #3949ab; padding-left:7px; background:#f4f6fb; }
h3 { color:#2b3550; font-size:10.4pt; margin:9px 0 3px; }
p { margin:5px 0; }
a { color:#3949ab; }
code { background:#f1f4f9; color:#10502f; padding:0 2px; }
pre { background:#11192a; color:#e7edf6; padding:8px 10px; font-size:8.4pt; }
pre code { background:none; color:#e7edf6; }
table { border-collapse: collapse; width:100%; font-size:8.2pt; }
th { background:#e9edf6; color:#26315a; font-family:kb; }
th, td { border:1px solid #ccd3e0; padding:3px 5px; text-align:left; vertical-align:top; }
blockquote { background:#fff8e6; border-left:4px solid #f0ad2e; padding:5px 10px; color:#5b4a16; margin:7px 0; }
.chip { font-family:kb; font-size:7.6pt; padding:0 5px; border-radius:8px; }
.chip-a{background:#fff1e0;color:#b25f00;} .chip-b{background:#e3f6ec;color:#1b7a44;}
.chip-g{background:#eef0f4;color:#54607a;} .chip-v{background:#e6efff;color:#2b56b8;} .chip-c{background:#fdeede;color:#a3650a;}
.wl{ color:#5560a8; font-style:italic; }
.math{ font-style:italic; background:#f4f1fb; }
.diagram{ background:#f6f8fc; border:1px solid #dde3ee; padding:8px; }
.dbox{ border:1px solid #c7d0e0; background:#fff; padding:5px 6px; text-align:center; font-size:7.8pt; }
.dpath{ color:#5b6675; font-size:7pt; }
.darr{ text-align:center; color:#7a8aa8; font-family:kb; }
.dgr{ border-spacing:5px 0; }
.c-in{background:#eef4ff;} .c-fl{background:#fff4e6;} .c-est{background:#e7f6ef;}
.c-orc{background:#fdeede;} .c-base{background:#eef0f4;} .c-det{background:#fdeaf0;}
.c-eval{background:#eaf3ff;} .c-sel{background:#ecfdf5;}
"""
def make_arch():
    a = fitz.Archive()
    for k,p in FONTS.items(): a.add(open(p,"rb").read(), k+".ttf")
    return a
ARCH_CSS_FONTS = "".join(
    f"@font-face{{font-family:{k};src:url({k}.ttf);}}" for k in FONTS)

def build_pdf(stem):
    title, body = doc_html(stem, for_html=False)
    full = f'<html><head></head><body>{body}</body></html>'
    css = ARCH_CSS_FONTS + PDF_CSS
    story = fitz.Story(html=full, user_css=css, archive=make_arch())
    out = os.path.join(PDFDIR, stem+".pdf")
    tmp = os.path.join(PDFDIR, stem+".raw.pdf")
    writer = fitz.DocumentWriter(tmp)
    MARG=42; A4=fitz.paper_rect("a4"); more=1
    area = fitz.Rect(MARG, MARG, A4.x1-MARG, A4.y1-MARG-16)
    while more:
        dev = writer.begin_page(A4)
        more,_ = story.place(area)
        story.draw(dev)
        writer.end_page()
    writer.close()
    # subset fonts + footer page numbers (Latin, helv)
    d = fitz.open(tmp)
    foot = f"Flirds Checkpoint 2026-06-10  ·  {stem}"
    for i,pg in enumerate(d):
        r = pg.rect
        pg.insert_text((MARG, r.y1-22), foot, fontname="helv", fontsize=7, color=(.42,.46,.54))
        pg.insert_text((r.x1-MARG-46, r.y1-22), f"p. {i+1} / {d.page_count}", fontname="helv", fontsize=7, color=(.42,.46,.54))
    try: d.subset_fonts()
    except Exception as e: print("  subset skip:", e)
    pc = d.page_count
    d.save(out, deflate=True, garbage=4)
    d.close(); os.remove(tmp)
    return out, pc, os.path.getsize(out)

if __name__ == "__main__":
    hp, n = build_combined()
    print(f"[HTML] {hp}  ({n//1024} KB)")
    tot=0
    for stem in ORDER:
        o,pg,sz = build_pdf(stem); tot+=sz
        print(f"[PDF ] {os.path.basename(o):46} {pg:2}p  {sz//1024:4} KB")
    print(f"PDF total: {tot//1024} KB")
