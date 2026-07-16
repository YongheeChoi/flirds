#!/usr/bin/env python
"""runs/rerun_beta03/make_figures.py -- ShapleyFL beta 0.5->0.3 unification: contrast + provenance.

This campaign folder owns no rundirs; it re-runs cells in place elsewhere.  Two
figure sources, both artifact-only:

  1) before/after contrast: the 6 track_d 3B cells were re-run at beta=0.3 and
     committed as b1b95d0; the beta=0.5 originals live one commit earlier
     (b1b95d0~1) in git history.  Both versions of {metrics.json, phi.parquet}
     are read (git show) and compared.  Control methods give the re-run noise
     floor, isolating the beta effect on ShapleyFL.
  2) beta-era provenance map: every rundir's meta.json git_sha is classified by
     git ancestry against e89af94 (the beta 0.5->0.3 commit).  A sha that does
     not contain e89af94 ran beta=0.5-era code.  git_dirty shas are flagged --
     ancestry then bounds, not proves, the beta value.

The pending-cell coverage report is DERIVED from the provenance scan (not copied
from RESUME_AFTER_MIGRATION.md) -- mismatches with that doc are printed, not hidden.

  python runs/rerun_beta03/make_figures.py
"""
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent            # runs/rerun_beta03
RUNS = HERE.parent
REPO = RUNS.parent
FIG = HERE / "figures"

BETA_COMMIT = "e89af94"                           # ShapleyFL beta 0.5 -> 0.3
RERUN_COMMIT = "b1b95d0"                          # 3B track_d re-run at beta=0.3
BEFORE_REF = f"{RERUN_COMMIT}~1"                  # last beta=0.5 3B artifacts
CELLS_3B = [f"3B_{r}_seed{s}" for r in ("std20", "anchor5") for s in (0, 1, 2)]

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
                     "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})

METHOD_ORDER = ["Flirds", "Flirds1st", "FedIF", "GTG", "FedSV", "ShapleyFL",
                "Banzhaf", "ComFedSV", "loss-heur"]


def footnote(fig, text):
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.text(0.005, bb.y0 / fig.get_figheight() - 0.012, text,
             fontsize=6.5, color="#666666", va="top", ha="left")


def save(fig, path):
    FIG.mkdir(exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(HERE)}")


def git(*args, binary=False):
    r = subprocess.run(["git", "-C", str(REPO)] + list(args), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[:200])
    return r.stdout if binary else r.stdout.decode()


def is_beta03_era(sha):
    """True iff sha contains the beta commit (beta=0.3 code)."""
    r = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor",
                        BETA_COMMIT, sha], capture_output=True)
    return r.returncode == 0


# ------------------------------------------------------------------ before/after
def load_pair(cell):
    rel = f"runs/track_d/rundirs/{cell}"
    before = dict(
        metrics=json.loads(git("show", f"{BEFORE_REF}:{rel}/metrics.json")),
        meta=json.loads(git("show", f"{BEFORE_REF}:{rel}/meta.json")),
        phi=pd.read_parquet(io.BytesIO(git("show", f"{BEFORE_REF}:{rel}/phi.parquet",
                                           binary=True))))
    d = RUNS / "track_d" / "rundirs" / cell
    after = dict(metrics=json.loads((d / "metrics.json").read_text()),
                 meta=json.loads((d / "meta.json").read_text()),
                 phi=pd.read_parquet(d / "phi.parquet"))
    return before, after


def fig_contrast(pairs, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), gridspec_kw={"width_ratios": [1, 1.35]})
    # (left) ShapleyFL fidelity before vs after
    ax = axes[0]
    xs = np.arange(len(CELLS_3B))
    key = lambda m: next(iter(m))                              # 'seedN' block
    b_vals = [pairs[c][0]["metrics"][key(pairs[c][0]["metrics"])]["spearman"]["ShapleyFL"]
              for c in CELLS_3B]
    a_vals = [pairs[c][1]["metrics"][key(pairs[c][1]["metrics"])]["spearman"]["ShapleyFL"]
              for c in CELLS_3B]
    ax.scatter(xs - 0.1, b_vals, s=30, color="#D55E00", label="beta=0.5 (git b1b95d0~1)")
    ax.scatter(xs + 0.1, a_vals, s=30, marker="s", color="#0072B2", label="beta=0.3 (current)")
    for x, b, a in zip(xs, b_vals, a_vals):
        ax.plot([x - 0.1, x + 0.1], [b, a], color="#999999", lw=0.8)
    ax.set_xticks(xs, [c.replace("3B_", "") for c in CELLS_3B], rotation=30, ha="right",
                  fontsize=7.5)
    ax.set_ylim(-1.05, 1.1)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_ylabel("ShapleyFL Spearman vs (b) oracle")
    ax.set_title("ShapleyFL fidelity, before vs after")
    ax.legend(fontsize=7)
    # (right) rank agreement of each method's phi across the two runs
    ax = axes[1]
    methods = [m for m in METHOD_ORDER
               if all(m in set(pairs[c][0]["phi"]["method"]) and
                      m in set(pairs[c][1]["phi"]["method"]) for c in CELLS_3B)]
    rows = []
    for c in CELLS_3B:
        b, a = pairs[c]
        for m in methods:
            vb = b["phi"][b["phi"]["method"] == m].set_index("client")["phi"].sort_index()
            va = a["phi"][a["phi"]["method"] == m].set_index("client")["phi"].sort_index()
            vb, va = vb.align(va, join="inner")
            rows.append(dict(cell=c, method=m,
                             rho=float(spearmanr(vb, va).statistic)))
    df = pd.DataFrame(rows)
    xs = np.arange(len(methods))
    for i, m in enumerate(methods):
        v = df[df["method"] == m]["rho"]
        col = "#D55E00" if m == "ShapleyFL" else "#0072B2"
        ax.scatter(np.full(len(v), i) + np.linspace(-0.14, 0.14, len(v)), v, s=18,
                   color=col, alpha=0.8, linewidths=0)
        ax.plot([i - 0.2, i + 0.2], [v.mean()] * 2, color=col, lw=2)
    ax.set_xticks(xs, methods, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylim(-1.05, 1.1)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_ylabel("Spearman(phi before, phi after)")
    ax.set_title("Per-method rank agreement across the re-run\n"
                 "(non-ShapleyFL methods = trajectory re-run noise floor; "
                 "ShapleyFL adds the beta change)")
    fig.suptitle("beta 0.5 -> 0.3 contrast on the re-run track_d 3B cells (std20 / anchor5 x 3 seeds)",
                 fontsize=10.5, y=1.03)
    footnote(fig, f"source: git {BEFORE_REF}:runs/track_d/rundirs/3B_* (beta0.5) vs working tree (beta0.3, commit {RERUN_COMMIT}) "
                  "| dots=cells/seeds, bar=mean")
    save(fig, path)
    return df


# ------------------------------------------------------------------ provenance map
GROUPS = [
    ("track_d 1B/3B/7B", "track_d/rundirs", "*"),
    ("phase2 June grid", "phase2_matrix/rundirs", "*"),
    ("phase2 July re-run", "phase2_matrix/rundirs_2026-07", "*"),
    ("track_c c1 (CNN)", "track_c/c1", "*"),
    ("track_c c2 (CNN)", "track_c/c2", "*"),
    ("probe_signal LLM", "probe_signal/rundirs", "*"),
    ("probe_signal CNN", "probe_signal/cnn_c*", "*"),
]


def scan_provenance():
    rows = []
    for label, rel, pat in GROUPS:
        for base in sorted(RUNS.glob(rel)):
            for d in sorted(base.glob(pat)):
                mp = d / "meta.json"
                if not mp.exists():
                    continue
                meta = json.loads(mp.read_text())
                sha, dirty = meta.get("git_sha", ""), bool(meta.get("git_dirty"))
                rows.append(dict(group=label, cell=d.name, sha=sha[:9], dirty=dirty,
                                 era="beta0.3-era" if sha and is_beta03_era(sha)
                                     else "beta0.5-era"))
    return pd.DataFrame(rows)


def fig_provenance(df, path):
    order = [g for g, _, _ in GROUPS]
    agg = (df.groupby(["group", "era"]).size().unstack(fill_value=0)
             .reindex(order).fillna(0))
    for col in ("beta0.3-era", "beta0.5-era"):
        if col not in agg:
            agg[col] = 0
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    ys = np.arange(len(agg))
    ax.barh(ys, agg["beta0.3-era"], 0.6, color="#0072B2", label="beta=0.3-era code")
    ax.barh(ys, agg["beta0.5-era"], 0.6, left=agg["beta0.3-era"], color="#D55E00",
            label="beta=0.5-era code (ShapleyFL numbers pre-unification)")
    for y, (g, r) in zip(ys, agg.iterrows()):
        n03, n05 = int(r["beta0.3-era"]), int(r["beta0.5-era"])
        ndirty = int(df[(df["group"] == g)]["dirty"].sum())
        ax.text(n03 + n05 + 0.6, y, f"{n03}+{n05}" + (f" ({ndirty} dirty)" if ndirty else ""),
                va="center", fontsize=7)
    ax.set_yticks(ys, agg.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("rundirs")
    ax.set_title("ShapleyFL beta provenance of every persisted rundir\n"
                 "(meta.json git_sha ancestry vs the beta commit; 'dirty' shas bound, not prove, the era)")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False)
    footnote(fig, f"source: <group>/*/meta.json git_sha, classified by `git merge-base --is-ancestor {BETA_COMMIT} <sha>` "
                  "| beta affects ONLY the ShapleyFL method/arm rows of those cells")
    save(fig, path)
    return agg


# ------------------------------------------------------------------ main
def main():
    print("== before/after contrast (3B track_d) ==")
    pairs, missing = {}, []
    for c in CELLS_3B:
        try:
            pairs[c] = load_pair(c)
        except Exception as e:
            missing.append(f"{c}: {e}")
    print(f"  pairs loaded: {len(pairs)}/6" + (f" | MISSING {missing}" if missing else ""))
    for c, (b, a) in pairs.items():
        okb = not is_beta03_era(b["meta"]["git_sha"])
        oka = is_beta03_era(a["meta"]["git_sha"])
        flag = "" if (okb and oka) else "  <-- UNEXPECTED ERA"
        print(f"    {c}: before {b['meta']['git_sha'][:9]} (beta0.5-era={okb}), "
              f"after {a['meta']['git_sha'][:9]} (beta0.3-era={oka}){flag}")

    print("== figures ==")
    if len(pairs) == len(CELLS_3B):
        contrast = fig_contrast(pairs, FIG / "01_beta_contrast_3b_before_after.png")
        FIG.mkdir(exist_ok=True)
        contrast.to_csv(FIG / "beta_contrast_3b.csv", index=False)
    else:
        print("  [skip] contrast figure: incomplete pairs")

    df = scan_provenance()
    agg = fig_provenance(df, FIG / "02_beta_provenance_map.png")
    df.to_csv(FIG / "beta_provenance.csv", index=False)
    print(f"  wrote figures/beta_provenance.csv ({len(df)} rundirs scanned)")

    pend = df[df["era"] == "beta0.5-era"]
    print(f"== coverage: beta0.5-era rundirs remaining = {len(pend)} ==")
    for g, sub in pend.groupby("group"):
        cells = sorted(sub["cell"])
        print(f"    {g}: {len(cells)}" + (f" -> {', '.join(cells[:6])}"
                                          + (" ..." if len(cells) > 6 else "")))
    # resume-doc says 31 pending (7B x6 + phase2 June x25); report agreement
    n_doc = 31
    n_scan_pending_scope = len(pend[pend["group"].isin(["track_d 1B/3B/7B", "phase2 June grid"])])
    print(f"  RESUME doc claims 31 pending (7B x6 + June phase2 x25); "
          f"scan finds {n_scan_pending_scope} beta0.5-era in that scope "
          f"(+{len(pend) - n_scan_pending_scope} outside it, incl. the '1B/CNN completed' "
          f"claims the scan cannot confirm -- see MANIFEST)")


if __name__ == "__main__":
    main()
