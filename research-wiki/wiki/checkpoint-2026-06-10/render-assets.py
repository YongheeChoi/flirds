# -*- coding: utf-8 -*-
"""Build the combined HTML (app-shell layout: fixed sidebar + scrolling main pane,
MathJax-rendered LaTeX) + one PDF per source .md (Chromium print-to-PDF, real math).
markdown-it-py for MD->HTML; MathJax (CDN) for LaTeX; Playwright/Chromium for PDFs.
Math needs internet at view time (MathJax CDN) / at build time (PDF render)."""
import re, os, html as ihtml, urllib.request
from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
from playwright.sync_api import sync_playwright

CKPT = "/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/research-wiki/wiki/checkpoint-2026-06-10"
PDFDIR = os.path.join(CKPT, "pdf")
os.makedirs(PDFDIR, exist_ok=True)
ORDER = ["00-overview","01-research-value","02-experimental-setup",
         "03-baselines-and-prior-work","04-plan-vs-implementation-divergences",
         "05-open-issues-and-next","06-closest-competitors-fedif-fedtsv-ripple",
         "07-novelty-limitations-analysis"]

def esc(s):
    return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

# ASCII math notation (w_r, p_k^r, g^r, 2^N, Σ_{j∈P_r}, ΔW^r) -> real <sub>/<sup>.
# Strict single latin/greek/digit base, NOT inside an identifier -> answer_swap,
# flirds_estimator.py, n_clients, second_order=False, per_client=300 are left intact.
_SCRIPT = re.compile(
    r'(?<![A-Za-z0-9_\\$])'
    r'([0-9A-Za-zΑ-ω])'
    r'((?:[_^](?:\{[^}]{1,18}\}|[A-Za-z0-9]))+)'
    r'(?![A-Za-z0-9_])')
def _scripts(base, rest):
    out = base
    for mm in re.finditer(r'([_^])(?:\{([^}]+)\}|([A-Za-z0-9]))', rest):
        tag = 'sub' if mm.group(1) == '_' else 'sup'
        out += f'<{tag}>{mm.group(2) if mm.group(2) is not None else mm.group(3)}</{tag}>'
    return out
def mathify(s):
    return _SCRIPT.sub(lambda m: _scripts(m.group(1), m.group(2)), s)

# MathJax: load tex-svg; output inline math as \(...\) and block as \[...\] (MathJax v3 defaults).
MATHJAX = ('<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" id="MJ" async></script>')

def make_md():
    md = MarkdownIt("gfm-like", {"html": True, "linkify": False, "breaks": False})
    md.use(dollarmath_plugin)
    # math is authored as $...$ / $$...$$ in the .md -> emit MathJax delimiters; MathJax typesets.
    md.renderer.rules["math_inline"] = lambda t,i,o,e: "\\(" + t[i].content + "\\)"
    md.renderer.rules["math_block"]  = lambda t,i,o,e: '<div class="mathblk">\\[' + t[i].content + "\\]</div>"
    return md
MD = make_md()

# ---------- pipeline diagram (HTML boxes; renders in browser + chromium-print) ----------
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
<td class="dbox c-orc">in-run oracle<br><span class="dpath">oracle/in_run_sv.py</span><br>exact 2^N · perround</td>
<td class="dbox c-orc">retrain oracle<br><span class="dpath">oracle/exact_sv_llm.py</span><br>2^N 재학습</td>
<td class="dbox c-base">valuation baselines ×7<br>GTG·FedSV·Ripple·Banzhaf<br>ShapleyFL·ComFedSV·loss-heur</td>
<td class="dbox c-det">detectors ×4<br>FLDetector·STD-DAGMM<br>FLTrust·FedDQC</td>
</tr></table>
<div class="darr">↓</div>
<div class="dbox c-eval">평가 — <span class="dpath">eval/metrics.py · eval/generate.py</span><br>Spearman(vs oracle) · AUROC(vs 주입라벨) · ROUGE-L · ASR</div>
<div class="darr">↓</div>
<div class="dbox c-sel">selection / filtering — select_topk → arms {full / flirds_topk / random_k}<br>(detection→성능: noisy+free-rider 드롭 → val_loss↓)</div>
</div>
"""

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
    md_text = re.sub(r'```mermaid.*?```', "\n\n"+mathify(DIAGRAM)+"\n\n", md_text, flags=re.S)
    md_text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', lambda m:'<span class="wl">'+m.group(2)+'</span>', md_text)
    md_text = re.sub(r'\[\[([^\]]+)\]\]', lambda m:'<span class="wl">'+m.group(1).split('/')[-1]+'</span>', md_text)
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
        if i%2==1: out.append(p); continue
        p = CHIP.sub(chip_sub, p); p = STATE.sub(state_sub, p)
        out.append(p)
    return "".join(out)

def render_body(md_text, for_html):
    body = MD.render(preprocess(md_text, for_html))
    body = decorate(body)
    # wrap ONLY markdown tables (bare <table>) for horizontal scroll; leave the
    # diagram tables (<table class="dgr">) alone -- a blunt </table> replace would add
    # stray </div> for them and prematurely close .section/.main (layout break).
    body = re.sub(r'<table>(.*?)</table>', r'<div class="tw"><table>\1</table></div>', body, flags=re.S)
    return body

def doc_html(stem, for_html):
    raw = open(os.path.join(CKPT, stem+".md"), encoding="utf-8").read()
    meta, text = split_front(raw)
    title = meta.get("title", stem)
    num = str(int(stem.split("-")[0]))
    head = f'<div class="dochead"><span class="docnum">{num}</span><h1 class="doctitle">{ihtml.escape(title)}</h1></div>'
    note = f'<p class="docnote">{ihtml.escape(meta["note"])}</p>' if meta.get("note") else ''
    return title, head + note + render_body(text, for_html)

# ---------- shared content styling (used by both HTML and PDF) ----------
CONTENT_CSS = r"""
.dochead{display:flex;align-items:center;gap:13px;margin:6px 0 4px;}
.docnum{flex:0 0 auto;background:linear-gradient(135deg,#3949ab,#5c6bc0);color:#fff;font-weight:800;
font-size:15px;width:42px;height:42px;border-radius:11px;display:flex;align-items:center;justify-content:center;}
.doctitle{font-size:20px;margin:0;line-height:1.3;color:#1d2540;}
.docnote{color:#5b6675;font-size:12.5px;background:#eef2f8;border-left:3px solid #00897b;padding:7px 12px;border-radius:0 8px 8px 0;margin:8px 0 14px;}
h2{font-size:17px;margin:24px 0 9px;padding-left:11px;border-left:4px solid #3949ab;color:#26315a;}
h3{font-size:15px;margin:16px 0 6px;color:#28324a;}
p{margin:8px 0;} ul,ol{margin:8px 0;padding-left:22px;} li{margin:3px 0;}
strong,b{font-weight:700;color:#11192a;}
a{color:#3949ab;text-decoration:none;} a:hover{text-decoration:underline;}
code{font-family:"SFMono-Regular","NanumGothicCoding",Consolas,monospace;font-size:12.4px;
background:#f3f5f9;color:#0f5132;padding:1.5px 5px;border-radius:5px;border:1px solid #e6ebf2;overflow-wrap:anywhere;}
pre{background:#11192a;color:#e7edf6;padding:13px 15px;border-radius:10px;overflow-x:auto;font-size:12.4px;line-height:1.55;break-inside:avoid;}
pre code{background:none;border:none;color:inherit;padding:0;}
.mathblk{margin:14px 0;padding:7px 16px;background:#f5f7fc;border:1px solid #dde4f1;border-left:4px solid #5c6bc0;border-radius:8px;break-inside:avoid;}
mjx-container{overflow-x:auto;overflow-y:hidden;max-width:100%;}
sub,sup{line-height:0;font-size:.72em;}
.dpath sub,.dpath sup{font-size:.78em;}
.tw{overflow-x:auto;margin:12px 0;border:1px solid #e3e8ef;border-radius:10px;}
table{border-collapse:collapse;width:100%;font-size:12.6px;background:#fff;}
th{background:#eef1f8;color:#27314a;text-align:left;font-weight:700;}
th,td{border:1px solid #e6ebf2;padding:7px 10px;vertical-align:top;overflow-wrap:anywhere;}
tbody tr:nth-child(even){background:#fafbfe;} tr{break-inside:avoid;}
blockquote{margin:13px 0;background:#fff8e6;border:1px solid #f3e2b3;border-left:5px solid #f0ad2e;
padding:9px 14px;border-radius:0 10px 10px 0;color:#5b4a16;break-inside:avoid;}
blockquote p{margin:5px 0;} blockquote strong{color:#7a5b00;}
.chip{display:inline-block;font-size:11px;font-weight:700;padding:1px 8px;border-radius:20px;line-height:1.7;white-space:nowrap;border:1px solid transparent;}
.chip-a{background:#fff1e0;color:#b25f00;border-color:#ffd8a8;}
.chip-b{background:#e3f6ec;color:#1b7a44;border-color:#aee3c4;}
.chip-g{background:#eef0f4;color:#54607a;border-color:#d7dce6;}
.chip-v{background:#e6efff;color:#2b56b8;border-color:#bcd2ff;}
.chip-c{background:#fdeede;color:#a3650a;border-color:#f3d6a8;}
.wl{color:#5560a8;border-bottom:1px dotted #9aa4d6;font-style:italic;}
mjx-container{margin:0 2px;}
.diagram{margin:16px 0;padding:16px;background:linear-gradient(180deg,#f8fafc,#eef2f8);border:1px solid #e3e8ef;border-radius:14px;overflow-x:auto;break-inside:avoid;}
.dgr{border-collapse:separate;border-spacing:9px 0;margin:0 auto;} .dfan{table-layout:fixed;width:100%;}
.dbox{border-radius:10px;padding:10px 12px;text-align:center;font-size:12.5px;line-height:1.5;border:1.5px solid #c7d0e0;background:#fff;}
.dpath{font-family:"NanumGothicCoding",monospace;font-size:11.2px;color:#5b6675;}
.darr{text-align:center;color:#7a8aa8;font-weight:800;font-size:15px;margin:6px 0;}
.dplus{font-weight:800;color:#8a97b3;padding:0 4px;vertical-align:middle;}
.c-in{background:#eef4ff;border-color:#b7ccf6;} .c-fl{background:#fff4e6;border-color:#ffd8a8;}
.c-est{background:#e7f6ef;border-color:#9ad8b6;font-weight:700;} .c-orc{background:#fdeede;border-color:#f3cf9a;}
.c-base{background:#eef0f4;border-color:#d2d8e4;} .c-det{background:#fdeaf0;border-color:#f3b9cd;}
.c-eval{background:#eaf3ff;border-color:#aeccf6;} .c-sel{background:#ecfdf5;border-color:#9fe0c4;}
"""

# ================= COMBINED HTML (app-shell) =================
HTML_CSS = r"""
:root{--ink:#1f2733;--mut:#5b6675;--line:#e3e8ef;--bg:#eef1f6;--accent:#3949ab;}
*{box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{margin:0;background:#fff;color:var(--ink);
font-family:"Apple SD Gothic Neo","Noto Sans KR","Noto Sans CJK KR","Malgun Gothic","NanumGothic","Segoe UI",sans-serif;
font-size:16.5px;line-height:1.75;-webkit-font-smoothing:antialiased;}
.layout{display:block;}
/* truly fixed sidebar (position:fixed) — always visible; the BODY scrolls the content (robust) */
.side{position:fixed;top:0;left:0;width:288px;height:100vh;overflow-y:auto;overscroll-behavior:contain;background:#0f1830;color:#cdd6e6;padding:22px 16px 40px;z-index:10;}
.side .brand{font-weight:800;font-size:18px;color:#fff;}
.side .brand small{display:block;font-weight:500;color:#8c9bb8;font-size:12px;margin-top:3px;}
.side nav{margin-top:18px;}
.side .navdoc{display:flex;gap:9px;color:#e7ecf7;font-weight:700;padding:9px 11px;border-radius:9px;margin-top:5px;font-size:14.5px;line-height:1.35;}
.side .navdoc:hover{background:#1c2a4a;text-decoration:none;} .side .navdoc.active{background:#21345c;}
.side .navdoc .n{flex:0 0 auto;color:#6fc3b6;font-weight:800;}
.main{margin-left:288px;max-width:1480px;padding:0 48px 120px;overflow-wrap:break-word;}
.hero{padding:42px 0 24px;border-bottom:2px solid var(--line);}
.hero h1{font-size:30px;margin:0 0 8px;letter-spacing:-.3px;}
.hero .sub{color:var(--mut);font-size:16px;max-width:860px;}
.hero .meta{margin-top:14px;font-size:13px;color:var(--mut);}
.legend{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px;}
.section{padding:30px 0;border-bottom:1px solid var(--line);scroll-margin-top:18px;}
""" + CONTENT_CSS + r"""
.side::-webkit-scrollbar{width:9px;} .side::-webkit-scrollbar-thumb{background:#2b3c5e;border-radius:6px;}
/* HTML font sizes (PDF keeps PRINT_CSS) */
.main p,.main li{font-size:16.5px;}
.main table{font-size:14.4px;} .main th,.main td{padding:8px 11px;}
.main code{font-size:13.9px;} .main pre{font-size:13.5px;}
.main .chip{font-size:12px;} .main .dpath{font-size:13px;} .main .docnote{font-size:14.5px;}
.main .dbox{font-size:14.5px;line-height:1.55;} .main .darr{font-size:18px;} .main .dplus{font-size:16px;}
.main h2{font-size:20px;} .main h3{font-size:17px;} .main .doctitle{font-size:22px;}
@media(max-width:980px){.side{position:static;width:auto;height:auto;} .main{margin-left:0;padding:0 16px 60px;}}
"""

def build_combined():
    sections=[]; nav=[]
    for stem in ORDER:
        title, body = doc_html(stem, for_html=True)
        sid = "doc-"+stem
        sections.append(f'<section class="section" id="{sid}">{body}</section>')
        num = str(int(stem.split("-")[0]))
        short = title.split("—")[-1].strip() if "—" in title else title
        nav.append(f'<a class="navdoc" href="#{sid}"><span class="n">{num}</span><span>{ihtml.escape(short)}</span></a>')
    legend = ('<span class="chip chip-b">ⓑ 실측 결과</span><span class="chip chip-a">ⓐ 코드+smoke</span>'
              '<span class="chip chip-g">ⓒ 미실행/설계</span><span class="chip chip-v">[CODE/PDF] 원본대조</span>'
              '<span class="chip chip-c">[DOC/WEB] 문서주장</span>')
    hero = f'''<div class="hero"><h1>Flirds 연구 체크포인트 — 2026-06-10</h1>
<div class="sub">Phase 0–2 구현 + step5 matrix orchestrator 종료 시점의 전체 재오리엔테이션.
코드·raw 로그·논문 PDF를 직접 대조해 정리한 8개 문서(00–07)를 한 곳에 모았습니다.</div>
<div class="legend">{legend}</div>
<div class="meta">근거 추적: 코드 <code>path:line</code> · 실측 <code>codes/runs/*/metrics.json</code> · 논문 <code>raw/papers/flirds/*.pdf</code>. 3-state(ⓐ/ⓑ/ⓒ) 규율 적용. 좌측 목차로 섹션 이동.</div></div>'''
    side = f'<aside class="side"><div class="brand">Flirds Checkpoint<small>2026-06-10 · 재오리엔테이션</small></div><nav>{"".join(nav)}</nav></aside>'
    doc = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flirds 체크포인트 2026-06-10</title>{MATHJAX}<style>{HTML_CSS}</style></head>
<body><div class="layout">{side}<main class="main" id="scroller">{hero}{"".join(sections)}</main></div>
<script>
// highlight active nav as the main pane scrolls
// fixed sidebar + native body anchor-scroll (href="#doc-..") -> robust, repeatable.
const secs=[...document.querySelectorAll('.section')],links=[...document.querySelectorAll('.navdoc')];
window.addEventListener('scroll',()=>{{let a=0;secs.forEach((s,i)=>{{if(s.getBoundingClientRect().top<=130)a=i;}});links.forEach((l,i)=>l.classList.toggle('active',i===a));}},{{passive:true}});
</script></body></html>'''
    outp = os.path.join(CKPT, "flirds-checkpoint-2026-06-10.html")
    open(outp,"w",encoding="utf-8").write(doc)
    return outp, len(doc.encode())

# ================= PER-FILE PDFs (Chromium print-to-PDF) =================
PRINT_CSS = r"""
*{box-sizing:border-box;}
body{margin:0;color:#1f2733;font-family:"Noto Sans CJK KR","NanumGothic","Apple SD Gothic Neo",sans-serif;
font-size:10.2px;line-height:1.55;}
.doctitle{font-size:18px;} .docnum{font-size:14px;width:38px;height:38px;}
h2{font-size:14.5px;} h3{font-size:12.5px;}
""" + CONTENT_CSS

MJ_LOCAL = "/tmp/flirds_texsvg.js"   # local MathJax (fast, offline at build time)
def ensure_mathjax():
    if not os.path.exists(MJ_LOCAL) or os.path.getsize(MJ_LOCAL) < 500000:
        urllib.request.urlretrieve("https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js", MJ_LOCAL)
    return MJ_LOCAL

def build_pdfs():
    mj = ensure_mathjax()
    mj_tag = f'<script src="file://{mj}"></script>'
    tmphtml = "/tmp/_flirds_doc.html"
    sizes=[]
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--no-sandbox","--allow-file-access-from-files"])
        page = b.new_page()
        for stem in ORDER:
            _, body = doc_html(stem, for_html=False)
            html = f'<!doctype html><html lang="ko"><head><meta charset="utf-8">{mj_tag}<style>{PRINT_CSS}</style></head><body>{body}</body></html>'
            open(tmphtml, "w", encoding="utf-8").write(html)
            page.goto("file://"+tmphtml, wait_until="load")
            page.evaluate("""async () => {
                for(let i=0;i<60 && !(window.MathJax&&MathJax.startup&&MathJax.startup.promise);i++) await new Promise(r=>setTimeout(r,100));
                if(window.MathJax&&MathJax.startup) await MathJax.startup.promise;
            }""")
            out = os.path.join(PDFDIR, stem+".pdf")
            page.pdf(path=out, format="A4", print_background=True,
                     margin={"top":"13mm","bottom":"15mm","left":"13mm","right":"13mm"},
                     display_header_footer=True, header_template="<div></div>",
                     footer_template=("<div style='font-size:8px;width:100%;padding:0 13mm;color:#8a8f99;"
                                      "display:flex;justify-content:space-between;'>"
                                      f"<span>Flirds Checkpoint 2026-06-10 &middot; {stem}</span>"
                                      "<span><span class='pageNumber'></span> / <span class='totalPages'></span></span></div>"))
            sizes.append((stem, os.path.getsize(out)))
        b.close()
    return sizes

if __name__ == "__main__":
    hp, n = build_combined(); print(f"[HTML] {hp}  ({n//1024} KB)")
    for stem, sz in build_pdfs(): print(f"[PDF ] {stem:46} {sz//1024:4} KB")
