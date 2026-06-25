# -*- coding: utf-8 -*-
"""Render 08-baselines-paper-summaries.md -> pdf/08-baselines-paper-summaries.pdf.
Re-uses the render-assets.py pipeline (markdown-it -> HTML -> Chromium print-to-PDF,
local MathJax) so the baseline-summary PDF matches the 00-07 checkpoint PDFs."""
import importlib.util, os

CKPT = "/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/research-wiki/wiki/checkpoint-2026-06-10"
spec = importlib.util.spec_from_file_location("ra", os.path.join(CKPT, "render-assets.py"))
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)

from playwright.sync_api import sync_playwright

STEM = "08-baselines-paper-summaries"

mj = ra.ensure_mathjax()
_, body = ra.doc_html(STEM, for_html=False)
html = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        f'<script src="file://{mj}"></script><style>{ra.PRINT_CSS}</style></head>'
        f'<body>{body}</body></html>')
tmp = "/tmp/_flirds_baselines.html"
open(tmp, "w", encoding="utf-8").write(html)

with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox", "--allow-file-access-from-files"])
    page = b.new_page()
    page.goto("file://" + tmp, wait_until="load")
    page.evaluate("""async () => {
        for(let i=0;i<60 && !(window.MathJax&&MathJax.startup&&MathJax.startup.promise);i++) await new Promise(r=>setTimeout(r,100));
        if(window.MathJax&&MathJax.startup) await MathJax.startup.promise;
    }""")
    out = os.path.join(ra.PDFDIR, STEM + ".pdf")
    page.pdf(path=out, format="A4", print_background=True,
             margin={"top": "13mm", "bottom": "15mm", "left": "13mm", "right": "13mm"},
             display_header_footer=True, header_template="<div></div>",
             footer_template=("<div style='font-size:8px;width:100%;padding:0 13mm;color:#8a8f99;"
                              "display:flex;justify-content:space-between;'>"
                              f"<span>Flirds baseline papers &middot; {STEM}</span>"
                              "<span><span class='pageNumber'></span> / <span class='totalPages'></span></span></div>"))
    b.close()
print("[PDF]", out, os.path.getsize(out) // 1024, "KB")
