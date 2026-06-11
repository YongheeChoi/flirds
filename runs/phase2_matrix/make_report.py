#!/usr/bin/env python
"""Aggregate phase2_matrix cell logs into ONE organized markdown report (RESULTS.md).

Re-runnable: scans runs/phase2_matrix/{tier1,tier2,tier3}/*.log, parses each cell's
header config + per-seed metadata (selected / corrupt / seen / deployed-ASR) + metrics
(3-seed aggregate table if present, else the single-seed table), and writes RESULTS.md:
  - a status overview of every cell (done / running / FAILED)
  - silo5 (Tier 1) + scale (3B/7B): per-threat method tables (AUROC, Spearman, runtime)
  - device100 (Tier 2): per-threat ALPHA-SWEEP pivots (method x alpha) for AUROC + Spearman,
    with the deployed-ASR row for poison
Pure stdlib (no torch/numpy) -> runs anywhere.  Usage: python make_report.py
"""
import os
import re
import glob

BASE = "/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/phase2_matrix"
OUT = os.path.join(BASE, "RESULTS.md")
LOG_DIRS = ["tier1", "tier2", "tier3"]

HEADER_RE = re.compile(
    r"=== step5 MATRIX \| (\S+) (\w+)(?: alpha=([\d.]+))? \| R=(\d+) K_frac=(\S+) "
    r"lr=(\S+) batch=(\d+) val_chunk=(\d+) \| ORACLE_B=(\w+) COALITION=(\w+) \| "
    r"threats=(\[[^\]]*\]) seeds=(\[[^\]]*\])")
META_RE = re.compile(
    r"^\[(\w+)\] selected=(\S+) corrupt=(\[[^\]]*\]) seen_corrupt=(\[[^\]]*\])"
    r"(?: deployed-ASR=([\d.]+))?", re.M)
AGG_ROW_RE = re.compile(
    r"\s+(\S+)\s+AUROC=([\d.naN]+)\+/-([\d.naN]+)"
    r"(?:\s+Spearman=([+\-\d.naN]+)\+/-([\d.naN]+))?\s+runtime=([\d.]+)s")
FLOATS = re.compile(r"[\d.]+|nan")
RT_TOK = re.compile(r"[\d.]+s")

# canonical column / row order
METHOD_ORDER = ["(b)oracle", "Flirds", "Flirds1st", "FedIF", "GTG", "FedSV", "ShapleyFL", "Banzhaf",
                "ComFedSV", "loss-heur", "FLDetector", "STD-DAGMM", "FLTrust", "FedDQC"]
THREAT_ORDER = ["noisy", "freerider_random", "freerider_zero", "poison"]


def _mkey(name):
    return METHOD_ORDER.index(name) if name in METHOD_ORDER else len(METHOD_ORDER)


def parse_log(path):
    txt = open(path, errors="replace").read()
    d = {"name": os.path.basename(path)[:-4], "scale": "?", "regime": "?", "alpha": None,
         "R": None, "lr": None, "batch": None, "oracle_b": None, "coalition": None,
         "seeds": "?", "status": "running", "threats": {}}
    h = HEADER_RE.search(txt)
    d["hdr_threats"] = []
    if h:
        d.update(scale=h[1], regime=h[2], alpha=h[3], R=h[4], lr=h[6], batch=h[7],
                 oracle_b=h[9], coalition=h[10], seeds=h[12])
        d["hdr_threats"] = re.findall(r"'(\w+)'", h[11])
    if "label pool exhausted" in txt:
        d["status"] = "FAILED(pool)"
    elif re.search(r"Traceback|CUDA out of memory|OutOfMemoryError", txt):
        d["status"] = "FAILED(err)"
    elif "MATRIX DONE" in txt:
        d["status"] = "done"

    # per-threat metadata (collect ASR over all per-seed lines)
    meta = {}
    for m in META_RE.finditer(txt):
        th = m[1]
        e = meta.setdefault(th, {"selected": m[2], "corrupt": m[3], "seen": m[4], "asr": []})
        e["selected"], e["corrupt"], e["seen"] = m[2], m[3], m[4]
        if m[5]:
            e["asr"].append(float(m[5]))

    # metrics: prefer the multi-seed aggregate; else the single-seed table
    threats = {}
    if "=== aggregate" in txt:
        cur = None
        for line in txt.split("=== aggregate", 1)[1].splitlines():
            tm = re.match(r"\[(\w+)\]\s*$", line.strip())
            if tm:
                cur = tm[1]
                threats[cur] = []
                continue
            rm = AGG_ROW_RE.match(line)
            if rm and cur:
                threats[cur].append({"name": rm[1], "auroc": rm[2], "auroc_sd": rm[3],
                                     "sp": rm[4], "sp_sd": rm[5], "rt": rm[6]})
    else:  # single-seed: parse the per-seed table rows
        cur = None
        for line in txt.splitlines():
            hm = re.match(r"\[(\w+)\] selected=", line)
            if hm:
                cur = hm[1]
                threats.setdefault(cur, [])
                continue
            if cur is None:
                continue
            toks = line.split()
            rts = [i for i, t in enumerate(toks) if RT_TOK.fullmatch(t)]
            if len(toks) >= 3 and FLOATS.fullmatch(toks[1]) and rts:
                ri = rts[0]
                mid = toks[2:ri]
                sp = mid[0] if mid else None
                truth = sp == "(truth)"
                threats[cur].append({"name": toks[0], "auroc": toks[1], "auroc_sd": None,
                                     "sp": (None if (sp is None or truth) else sp),
                                     "sp_sd": None, "rt": toks[ri][:-1], "truth": truth})
            elif line.strip() == "" or line[:1] in ("-", "="):
                cur = None

    for th, rows in threats.items():
        rows.sort(key=lambda r: _mkey(r["name"]))
        m = meta.get(th, {})
        asr = m.get("asr") or []
        d["threats"][th] = {"rows": rows, "selected": m.get("selected", "?"),
                            "corrupt": m.get("corrupt", "?"), "seen": m.get("seen", "?"),
                            "asr": (sum(asr) / len(asr)) if asr else None, "n_asr": len(asr)}
    return d


def fmt(mean, sd):
    if mean is None:
        return ""
    try:
        v = f"{float(mean):+.3f}" if str(mean).lstrip("+-").startswith(("0", "1", "n")) else str(mean)
    except ValueError:
        v = str(mean)
    if sd and float(sd) > 0:
        return f"{v}±{float(sd):.2f}"
    return v


def auroc_cell(mean, sd):
    if mean is None:
        return ""
    return f"{float(mean):.3f}" + (f"±{float(sd):.2f}" if sd and float(sd) > 0 else "")


# ---------------------------------------------------------------- writers
def write_detail(o, cells):
    """silo5 / scale: one method-table per (cell, threat)."""
    for c in sorted(cells, key=lambda x: x["name"]):
        for th in sorted(c["threats"], key=lambda t: THREAT_ORDER.index(t) if t in THREAT_ORDER else 9):
            t = c["threats"][th]
            asr = f" · deployed-ASR≈**{t['asr']:.2f}**" if t["asr"] is not None else ""
            o.append(f"\n**`{c['name']}`** — {c['scale']} {c['regime']} · {th} · "
                     f"R={c['R']} lr={c['lr']} · seeds={c['seeds']} · _{c['status']}_{asr}  ")
            o.append(f"<sub>selected={t['selected']} corrupt={t['corrupt']} seen={t['seen']}</sub>\n")
            o.append("| method | AUROC | Spearman | runtime |")
            o.append("|---|---|---|---|")
            for r in t["rows"]:
                sp = "(truth)" if r.get("truth") else fmt(r["sp"], r["sp_sd"])
                o.append(f"| {r['name']} | {auroc_cell(r['auroc'], r['auroc_sd'])} | {sp} | {float(r['rt']):.0f}s |")


def write_pivots(o, cells):
    """device100: per-threat alpha-sweep pivots (method x alpha)."""
    by_threat = {}
    for c in cells:
        a = c["alpha"]
        for th, t in c["threats"].items():
            by_threat.setdefault(th, {})[a] = (t, c)
    for th in sorted(by_threat, key=lambda t: THREAT_ORDER.index(t) if t in THREAT_ORDER else 9):
        amap = by_threat[th]
        alphas = sorted(amap, key=lambda x: float(x) if x is not None else 0.0)
        methods = sorted({r["name"] for a in amap for r in amap[a][0]["rows"]}, key=_mkey)
        o.append(f"\n### device100 · {th}\n")
        # status / ASR / truth line per alpha
        stat = []
        for a in alphas:
            t, c = amap[a]
            s = f"α={a}: {c['status']}"
            if t["asr"] is not None:
                s += f" ASR≈{t['asr']:.2f}"
            stat.append(s)
        o.append("<sub>" + " · ".join(stat) + "</sub>\n")
        # AUROC pivot
        o.append("**detection AUROC**\n")
        o.append("| method | " + " | ".join(f"α={a}" for a in alphas) + " |")
        o.append("|---|" + "---|" * len(alphas))
        for mname in methods:
            cells_row = []
            for a in alphas:
                rd = {r["name"]: r for r in amap[a][0]["rows"]}
                r = rd.get(mname)
                cells_row.append(auroc_cell(r["auroc"], r["auroc_sd"]) if r else "")
            o.append(f"| {mname} | " + " | ".join(cells_row) + " |")
        # Spearman pivot (valuation methods only)
        o.append("\n**Spearman vs truth** <sub>(α=0.5=anchor→vs (b); others→vs Flirds\\*)</sub>\n")
        o.append("| method | " + " | ".join(f"α={a}" for a in alphas) + " |")
        o.append("|---|" + "---|" * len(alphas))
        for mname in methods:
            row, any_sp = [], False
            for a in alphas:
                rd = {r["name"]: r for r in amap[a][0]["rows"]}
                r = rd.get(mname)
                if r and (r.get("truth") or r.get("sp") is not None):
                    any_sp = True
                    row.append("(truth)" if r.get("truth") else fmt(r["sp"], r["sp_sd"]))
                else:
                    row.append("")
            if any_sp:
                o.append(f"| {mname} | " + " | ".join(row) + " |")


def main():
    cells = []
    for sub in LOG_DIRS:
        for p in sorted(glob.glob(os.path.join(BASE, sub, "*.log"))):
            if os.path.basename(p).startswith("_"):
                continue
            try:
                cells.append(parse_log(p))
            except Exception as e:  # never let one bad log break the report
                cells.append({"name": os.path.basename(p)[:-4], "status": f"parse-error: {e}",
                              "scale": "?", "regime": "?", "alpha": None, "threats": {}})

    n_done = sum(c["status"] == "done" for c in cells)
    n_fail = sum(str(c["status"]).startswith("FAIL") for c in cells)
    n_run = sum(c["status"] == "running" for c in cells)
    o = ["# Phase 2 step5 — matrix results",
         f"\n_auto-generated by `make_report.py` · {len(cells)} cells: "
         f"{n_done} done · {n_run} running · {n_fail} failed_\n",
         "Detection AUROC: corrupt=high-φ (1.0 = corrupt ranked most-suspicious). "
         "Spearman: valuation methods vs the (b) oracle (silo5/anchor) or Flirds proxy-truth "
         "(off-anchor). Values are 3-seed mean±std unless seeds=[0].\n"]

    # status overview
    o.append("## Status overview\n")
    o.append("| cell | scale | regime | α | threat(s) | seeds | status | ASR |")
    o.append("|---|---|---|---|---|---|---|---|")
    for c in sorted(cells, key=lambda x: (x["regime"], str(x["alpha"]), x["name"])):
        ths = ",".join(c["threats"]) or ",".join(c.get("hdr_threats", [])) or "—"
        asr = next((f"{t['asr']:.2f}" for t in c["threats"].values() if t["asr"] is not None), "")
        o.append(f"| `{c['name']}` | {c['scale']} | {c['regime']} | {c['alpha'] or ''} | {ths} "
                 f"| {c.get('seeds','?')} | {c['status']} | {asr} |")

    silo = [c for c in cells if c["regime"] == "silo5" and c["scale"] == "1B"]
    dev = [c for c in cells if c["regime"] == "device100"]
    scale = [c for c in cells if c["scale"] in ("3B", "7B")]

    if silo:
        o.append("\n## Tier 1 — silo5 N=5 (1B, all methods + (b) oracle)\n")
        write_detail(o, silo)
    if dev:
        o.append("\n## Tier 2 — device100 N=100 (α-sweep; Flirds proxy-truth off-anchor)\n")
        write_pivots(o, dev)
    if scale:
        o.append("\n## Tier 3 — scale (silo5 N=5)\n")
        write_detail(o, scale)

    with open(OUT, "w") as f:
        f.write("\n".join(o) + "\n")
    print(f"wrote {OUT} | {len(cells)} cells: {n_done} done, {n_run} running, {n_fail} failed")


if __name__ == "__main__":
    main()
