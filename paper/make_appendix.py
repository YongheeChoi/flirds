# -*- coding: utf-8 -*-
"""paper-ko.md 부록 C~F 조립 — 표는 전부 rundir/분석 CSV에서 재계산(± = 표본표준편차 ddof=1).

사용:  python paper/make_appendix.py            # paper/appendix_cdef.md 생성
       (본문에 붙일 때는 paper-ko.md 의 "### 부록 C. Fidelity 전표" 이하를 이 파일로 교체)
환경:  RUNS 환경변수로 runs/ 위치 지정 가능(기본 = 이 파일 기준 ../runs).
"""
import pandas as pd, numpy as np, json, os, glob, io, re
R = os.environ.get("RUNS") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
o = io.StringIO()
def W(s=""): o.write(s + "\n")

_SEP = re.compile(r"^\|[\s:|-]+\|$")
def blockify(md):
    """표 헤더와 HTML 주석 앞에 빈 줄을 보장한다 — Obsidian은 앞의 빈 줄로 표 블록을 경계 짓기
    때문에, 캡션이나 주석에 바로 붙은 표는 문단에 딸려 들어가 raw 텍스트로 보인다."""
    L = md.split("\n"); out = []
    for i, ln in enumerate(L):
        head = ln.startswith("|") and i + 1 < len(L) and bool(_SEP.match(L[i + 1].strip()))
        if (head or ln.startswith("<!--")) and out and out[-1].strip() != "":
            out.append("")
        out.append(ln)
    return "\n".join(out)

# ---------------- 공통 ----------------
def ms(v, nd=3, sign=False, comma=False, pct=False):
    """pct=True: 비율(0~1)을 %로 환산해 표기. 정확도·EM·precision/recall·removal 간격 전용 —
    상관계수(ρ·r·τ)·거리·초·연산수는 비율이 아니므로 절대 붙이지 않는다."""
    v = np.asarray([x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    if len(v) == 0: return "–"
    if pct: v = v * 100.0
    c = "," if comma else ""
    f = f"{{:+{c}.{nd}f}}" if sign else f"{{:{c}.{nd}f}}"
    if len(v) == 1: return f.format(v[0]) + "◐"
    return f.format(v.mean()) + "±" + f"{v.std(ddof=1):{c}.{nd}f}"

LBL = {"Flirds": "Flirds", "Flirds1st": "Flirds-1st",
       "GTG": "GTG-Shapley", "FedSV": "FedSV", "ComFedSV": "ComFedSV", "ShapleyFL": "ShapleyFL",
       "FedIF": "FedIF", "(b)oracle": "in-run SV", "(a)oracle": "retrain SV"}
# singleton utility(코드명 loss-heur)는 논문 비교군에서 제외 — 표·서술 어디에도 넣지 않는다.
M7 = ["Flirds", "Flirds1st", "GTG", "FedSV", "ComFedSV", "ShapleyFL", "FedIF"]
THR = [("clean", "clean"), ("free_rider", "free-rider"), ("grad_noise", "grad noise"),
       ("label_flip", "label-flip")]
CORR = ["free_rider", "grad_noise", "label_flip"]
SET = [("cifar10", "dir1"), ("cifar10", "iid"), ("mnist", "dir1"), ("mnist", "iid")]

def table(head, rows):
    W("| " + " | ".join(head) + " |")
    W("|" + "---|" * len(head))
    for r in rows: W("| " + " | ".join(r) + " |")
    W()

# ---------------- 데이터 로드 ----------------
c2 = pd.read_csv(os.path.join(R, "track_c/c2fid/analysis/fidelity.csv"))
c2 = c2[c2.threat != "frrand"]
c2 = c2[~((c2.threat == "label_flip") & (c2.flip_rate != "'0.70'"))]
c1 = pd.read_csv(os.path.join(R, "track_c/c1/analysis/methods_long.csv"))
th = pd.read_csv(os.path.join(R, "track_h/analysis/cnn_competition.csv"))
th = th[~((th.threat == "label_flip") & (th.flip_rate != 0.7))]
lc = pd.read_csv(os.path.join(R, "track_h/analysis/llm_competition.csv"))

def grid(df, metric, methods, sets=SET, nd=3, sign=False, seedcol="seed"):
    rows = []
    for ds, part in sets:
        d = df[(df.dataset == ds) & (df.partition == part)]
        for t, tl in THR:
            cells = [ms(d[(d.threat == t) & (d.method == m)][metric].values, nd, sign) for m in methods]
            rows.append([f"{ds}/{part}", tl] + cells)
        cells = []
        for m in methods:
            piv = d[(d.threat.isin(CORR)) & (d.method == m)].pivot_table(index=seedcol, columns="threat", values=metric)
            cells.append(ms(piv.mean(axis=1).values, nd, sign) if piv.shape[1] == 3 else "–")
        rows.append([f"{ds}/{part}", "**오염-평균**"] + cells)
    return rows

HEADM = ["세팅", "위협"] + [LBL[m] for m in M7]

# ==================================================================== B
W("""### 부록 C. Fidelity 전표

§5.2는 본문 공간에 맞춰 CNN 주 세팅의 CIFAR-10 두 파티션과 $N{=}10$ 격자의 CIFAR-10
Dirichlet 칸만 실었다. 이 부록은 같은 채점을 MNIST 반복과 나머지 파티션, 그리고 Kendall
$\\tau$와 거리 3종까지 확장한 전표다. 읽는 방향은 §5.2와 같다. in-run Shapley 대비 상관은 식 (5)를
겨냥하는 두 방법(Flirds·Flirds-1st)에만 방법의 오차로 귀속되고, 나머지 다섯은
겨냥한 게임이 다르므로 이 열의 낮은 값을 그 방법의 실패로 읽지 않는다.""")
W()
W("""**C.1 CNN 주 세팅.** 표 [F4]–[F6]은 $N{=}100$·10/100 참여·$R{=}120$ 무대의 전 방법 전표다.
그림 2의 CIFAR-10 Flirds·Flirds-1st 두 계열이 이 표의 부분집합이고, MNIST 두 파티션이 같은
결론의 세 번째 데이터셋 반복이다. Flirds는 16칸 중 15칸에서 $\\rho \\geq .87$을 지키고, 유일하게
어려운 칸은 네 세팅 모두 gradient noise다($\\rho$ .847~.939). Flirds-1st가 그 칸에서 .218~.362로
내려앉고 Pearson은 CIFAR-10 Dirichlet에서 $-.051$, MNIST Dirichlet에서 $-.027$로 부호까지 뒤집히는
것이 §5.2의 2차 곡률항 판정이며, 데이터셋을 바꿔도 같다. 재정규화 계열은 zero-update
free-rider 칸에서 음수로 내려가고(ShapleyFL $-.060$~$-.658$), 이는 부록 F.6의 부호 감사와
같은 사실의 두 표현이다.""")
W()
W("표 [F4] — CNN 주 세팅의 in-run Shapley 대비 Spearman $\\rho$ ●")
W("""<!-- 출처: runs/track_c/c2fid/analysis/fidelity.csv (`python runs/track_c/c2fid/make_analysis.py`
     재생성), label-flip 은 flip_rate=0.70, 값 = spearman_b. 오염-평균은 seed별로 세 오염
     위협을 먼저 평균한 뒤 seed 간 mean±std. -->""")
table(HEADM, grid(c2, "spearman_b", M7))
W("표 [F5] — 같은 무대의 Pearson $r$ ●")
W("<!-- 같은 CSV, 값 = pearson_b. -->")
table(HEADM, grid(c2, "pearson_b", M7))
W("표 [F6] — 같은 무대의 Kendall $\\tau$ ●")
W("<!-- 같은 CSV, 값 = kendall_b. -->")
table(HEADM, grid(c2, "kendall_b", M7))

W("""표 [F7] — 같은 무대의 거리 3종(오염 3위협 평균). 두 기여도 벡터를 각각 최대 절대값으로
정규화한 뒤 잰 값이라 작을수록 가깝다 ●""")
W("<!-- 같은 CSV, 값 = cos_b · euc_b · maxdiff_b. -->")
rows = []
for ds, part in SET:
    d = c2[(c2.dataset == ds) & (c2.partition == part) & (c2.threat.isin(CORR))]
    for m in M7:
        cs = []
        for col in ["cos_b", "euc_b", "maxdiff_b"]:
            piv = d[d.method == m].pivot_table(index="seed", columns="threat", values=col)
            ps = piv.mean(axis=1)
            cs.append(f"{ps.mean():.3g}±{ps.std(ddof=1):.2g}" if len(ps) else "–")
        rows.append([f"{ds}/{part}", LBL[m]] + cs)
table(["세팅", "방법", "cosine", "euclid", "max"], rows)

W("""**C.2 CNN $N{=}10$ 격자.** 표 [F8]–[F10]은 두 Shapley 값을 모두 $2^{10}$ 전수 계산한
무대의 전표이고, 본문 표 [F2]는 이 가운데 CIFAR-10 Dirichlet 행이다. 첫 열은 방법이 아니라
in-run Shapley 자신의 retrain SV 대비 값, 곧 두 Shapley 값의 일치도다. 이 일치도가 무대마다
다르다는 것이 이 표의 첫 소식이다. 오염-평균으로 MNIST Dirichlet이 +0.926으로 가장 높고
CIFAR-10 IID가 +0.661로 가장 낮으며, CIFAR-10 IID의 clean 칸은 $-0.273$으로 부호가 뒤집힌다.
비IID 칸이 IID 칸보다 일관되게 높아, 두 게임은 client 사이에 실제 차이가 있을 때 같은 답에
수렴하고 그렇지 않으면 갈라진다. Flirds는 네 세팅 모두 그 일치도를 거의 그대로 따라가고(오염-평균
+0.644~+0.918, 일치도와의 차 ≤0.025), 반면 in-run Shapley 대비로 채점하면 Flirds가 16칸
전부에서 최고다(MNIST Dirichlet의 label-flip 한 칸만 Flirds-1st와 공동 1.000). 표 [F9]의 gradient noise 열에서 Flirds-1st만 CIFAR-10 Dirichlet에서 $-0.188$까지
내려가는 것은 부분참여 무대(표 [F4])의 판정이 전원참여에서도 재현된다는 뜻이다.""")
W()
W("표 [F8] — $N{=}10$ 격자의 retraining-based Shapley 대비 Spearman $\\rho$ ●")
W("""<!-- 출처: runs/track_c/c1/analysis/methods_long.csv (`python runs/track_c/c1/make_analysis.py`
     재생성), 값 = spearman_a. 첫 열 = method=='(b)oracle' 행(= 두 Shapley 값의 일치도).
     ShapleyFL 은 β=0.3 확정본. 본문 표 [F2] = 이 표의 cifar10/dir1 행(본문 평균 열만
     오염-평균이 아니라 clean 포함 전 위협 평균). -->""")
table(["세팅", "위협", "in-run SV *(앵커)*"] + [LBL[m] for m in M7],
      grid(c1, "spearman_a", ["(b)oracle"] + M7, nd=3, sign=True))
W("표 [F9] — 같은 격자의 in-run Shapley 대비 Spearman $\\rho$ ●")
W("<!-- 같은 CSV, 값 = spearman_b. -->")
table(HEADM, grid(c1, "spearman_b", M7, nd=3, sign=True))
W("표 [F10] — 같은 격자의 Kendall $\\tau$(오염 3위협 평균)와 거리 3종(in-run Shapley 대비) ●")
W("<!-- 같은 CSV, 값 = kendall_a · kendall_b · cos_b · euc_b · maxdiff_b. -->")
rows = []
for ds, part in SET:
    d = c1[(c1.dataset == ds) & (c1.partition == part) & (c1.threat.isin(CORR))]
    for m in ["(b)oracle"] + M7:
        cs = []
        for col in ["kendall_a", "kendall_b", "cos_b", "euc_b", "maxdiff_b"]:
            piv = d[d.method == m].pivot_table(index="seed", columns="threat", values=col)
            if piv.shape[1] != 3: cs.append("–"); continue
            ps = piv.mean(axis=1)
            cs.append(ms(ps.values, 3, col.startswith("kendall")) if col.startswith("kendall")
                      else f"{ps.mean():.3g}±{ps.std(ddof=1):.2g}")
        rows.append([f"{ds}/{part}", LBL[m]] + cs)
table(["세팅", "방법", "τ vs retrain SV", "τ vs in-run SV", "cosine", "euclid", "max"], rows)

# ---- LLM ----
LM7 = M7
def phase2(pattern, threats):
    acc = {t: {m: [] for m in LM7} for t in threats}
    for rd in sorted(glob.glob(os.path.join(R, pattern))):
        p = os.path.join(rd, "metrics.json")
        if not os.path.exists(p): continue
        for key, blk in json.load(open(p, encoding="utf-8")).items():
            t = key.rsplit("_seed", 1)[0]
            t = {"freerider_zero": "frzero", "freerider_random": "frrand"}.get(t, t)
            if t not in threats: continue
            for met in ("spearman", "pearson"):
                d = blk.get(met) or {}
                for m in LM7:
                    if m in d and d[m] is not None and not (isinstance(d[m], float) and np.isnan(d[m])):
                        acc[t][m].append((met, d[m]))
    return acc

W("""**C.3 LLM 전표.** 표 [F11]은 LLM 다섯 세팅의 in-run Shapley 대비 전 방법 채점이고, 그림
2(b)가 이 가운데 Flirds·Flirds-1st 두 열의 요약이다. 주 세팅에서 Flirds는 $\\rho \\geq .999$,
Flirds-1st는 $\\geq .995$인 반면 ComFedSV·ShapleyFL·FedIF는
clean에서 $-0.101$~$-0.018$로 내려간다. client당 참여가 희박할수록(50명 중 5명) 부분집합
표본추출이나 gradient 유사도에 기대는 추정량이 무너진다는 뜻이고, 같은 방향이 cross-device
앵커(100명 중 10명)에서 ComFedSV $-0.023$·ShapleyFL $+0.582$로 반복된다. 다만 이 무대들은 두
Taylor 변형이 모두 천장에 붙어 있어 그 둘 사이의 변별력은 없다. 절대 상관을 판별 근거로 쓰지
않는 이유는 부록 D에 있다.""")
W()
W("표 [F11] — LLM 세팅별 in-run Shapley 대비 Spearman $\\rho$ (아래줄 = Pearson $r$) ●/◐")
W("""<!-- 출처: runs/phase2_matrix/rundirs/1B_{gsm50k5,silo5}_*/metrics.json 및
     runs/phase2_matrix/analysis/04_device100_anchor/csv/{spearman,pearson}_vs_bperround.csv,
     runs/track_d/fidelity.csv(alpaca 규모 레그). 주 세팅 free-rider 열은 seed1 한 셀만
     착지해 ◐이고, 그 세팅의 재정규화 5종은 seed0 파일럿에서만 산출돼 ◐로 표기했다. -->""")
rows = []
for tag, pat, threats, lab in [("주 세팅(N=50·5/50·R=200)", "phase2_matrix/rundirs/1B_gsm50k5_*",
                                ["clean", "noisy", "frzero"], None),
                               ("5-도메인 비IID(N=5 전원·R=10)", "phase2_matrix/rundirs/1B_silo5_[cnf]*",
                                ["clean", "noisy", "frzero"], None)]:
    acc = phase2(pat, threats)
    for t in threats:
        tl = {"clean": "clean", "noisy": "answer-swap", "frzero": "free-rider"}[t]
        sp = [ms([v for k, v in acc[t][m] if k == "spearman"], 3) for m in LM7]
        pe = [ms([v for k, v in acc[t][m] if k == "pearson"], 3) for m in LM7]
        rows.append([tag, tl] + sp)
        rows.append(["", ""] + pe)
sp = pd.read_csv(os.path.join(R, "phase2_matrix/analysis/04_device100_anchor/csv/spearman_vs_bperround.csv")).set_index("method")
pe = pd.read_csv(os.path.join(R, "phase2_matrix/analysis/04_device100_anchor/csv/pearson_vs_bperround.csv")).set_index("method")
for col, tl in [("noisy", "answer-swap"), ("freerider_zero", "free-rider")]:
    rows.append(["cross-device 앵커(N=100·10/100·R=30)", tl] + [str(sp.loc[m, col]) if m in sp.index else "–" for m in LM7])
    rows.append(["", ""] + [str(pe.loc[m, col]) if m in pe.index else "–" for m in LM7])
td = pd.read_csv(os.path.join(R, "track_d/fidelity.csv"))
td["stage"] = td.cell.str.replace(r"_seed\d", "", regex=True)
for st_, lab in [("1B_std20", "alpaca IID-clean 1B(N=20·2/20·R=200)"),
                 ("3B_std20", "alpaca IID-clean 3B"), ("7B_std20", "alpaca IID-clean 7B"),
                 ("1B_anchor5", "소형 앵커 1B(N=5 전원·R=30)"), ("3B_anchor5", "소형 앵커 3B"),
                 ("7B_anchor5", "소형 앵커 7B")]:
    s = td[td.stage == st_]
    rows.append([lab, "clean"] + [ms(s[s.method == m].spearman.values, 3) for m in LM7])
    rows.append(["", ""] + [ms(s[s.method == m].pearson.values, 3) for m in LM7])
table(["세팅", "위협"] + [LBL[m] for m in LM7], rows)

W("""표 [F12] — 5-도메인 비IID 세팅의 retraining-based Shapley 대비($2^5$ 전수 재학습). 첫 행은
두 Shapley 값의 일치도이고, 이 무대에서는 세 위협 모두 순위가 완전히 일치해 각 방법의 retrain SV
대비 값이 in-run Shapley 대비 값과 같다 ●""")
W("""<!-- 출처: runs/phase2_matrix/silo5_a_fidelity_1B.csv (`python runs/phase2_matrix/merge_silo5_a.py`
     재생성; rundir 1B_silo5_{threat}_aonly_s{0,1,2} × canonical 1B_silo5_{threat}).
     ComFedSV clean 열은 그 rundir가 ComFedSV 추가 이전 산출이라 ⬚. N=5 Spearman 은 0.1 격자라
     0.933 = 인접 두 client 의 순서가 seed 둘에서 한 번씩 바뀐 것에 해당한다. -->""")
sa5 = pd.read_csv(os.path.join(R, "phase2_matrix/silo5_a_fidelity_1B.csv"))
rows = []
for t, tl in [("clean", "clean"), ("noisy", "answer-swap"), ("frzero", "free-rider")]:
    d = sa5[sa5.threat == t]
    rows.append([tl, ms(d[d.method == "(b)oracle"].spearman_a.values, 3, True)] +
                [ms(d[d.method == m].spearman_a.values, 3, True) for m in LM7])
table(["위협", "in-run SV *(앵커)*"] + [LBL[m] for m in LM7], rows)
anc = td[(td.stage == "1B_anchor5") & (td.method == "(a)oracle")]
W("""LLM의 나머지 retraining 레그는 소형 앵커 무대이고, 그 무대에서는 두 Shapley 값의 일치도가
Spearman {} · Pearson {}로 천장을 만든다. 같은 무대에서 어떤 방법도 그 위로 올라갈 수 없으므로
표 [F11]의 소형 앵커 행에서 Flirds 계열이 기록한 in-run Shapley 대비 1.000은 retraining-based
Shapley 대비로 옮기면 이 천장 값이 된다. 이 무대가 IID-clean이라 in-run Shapley 자신의
cross-seed 재현성이 $-0.367$이라는 점(표 [D1])이 그 천장의 출처다.""".format(
    ms(anc.spearman.values, 3), ms(anc.pearson.values, 3)))
W()

# ==================================================================== D
W("""### 부록 D. 타깃 신호의 재현성

§5.2는 clean 칸의 낮은 상관을 어느 방법의 우열 증거로도 쓰지 않는다고 했다. 근거가 이
부록이다. 상관은 채점 대상인 Shapley 값이 seed를 넘어 재현되는 순위일 때만 의미를 갖는데,
client 사이에 실제 차이가 없는 무대에서는 그 순위 자체가 seed마다 바뀐다. 표 [D1]은 in-run
Shapley를 seed 쌍끼리 직접 비교한 값이고, 여기서 낮게 나오는 무대의 높은 상관은 재현되지 않는
순위를 정확히 따라간 것에 지나지 않는다.""")
W()
W("""표 [D1] — in-run Shapley 자신의 cross-seed Spearman(seed 쌍 3개 평균, $n_{\\mathrm{seed}}{=}3$).
높을수록 그 무대의 순위가 재현된다 ●""")
W("""<!-- 출처: runs/track_d/target_stability.csv · runs/phase2_matrix/target_stability.csv
     (파생 산출; make_target_stability.py). 제외 위협축(poison·frrand) 행은 뺐다. -->""")
ts1 = pd.read_csv(os.path.join(R, "track_d/target_stability.csv"))
ts2 = pd.read_csv(os.path.join(R, "phase2_matrix/target_stability.csv"))
ts = pd.concat([ts1, ts2])
ts = ts[ts.n_seeds == 3]
D1 = [("1B_anchor5", "LLM 소형 앵커(IID-clean) 1B"), ("3B_anchor5", "소형 앵커 3B"),
      ("7B_anchor5", "소형 앵커 7B"), ("1B_std20", "alpaca IID-clean 부분참여 1B"),
      ("3B_std20", "alpaca IID-clean 3B"), ("7B_std20", "alpaca IID-clean 7B"),
      ("1B_iid5_clean", "IID 5분할 · clean"), ("1B_silo5_clean", "5-도메인 비IID · clean"),
      ("1B_silo5_noisy", "5-도메인 비IID · answer-swap"),
      ("1B_silo5_frzero", "5-도메인 비IID · free-rider"),
      ("1B_device100-a0.5_noisy_anchor", "cross-device 앵커 · answer-swap"),
      ("1B_device100-a0.5_frzero_anchor", "cross-device 앵커 · free-rider")]
rows = []
for cell, lab in D1:
    s = ts[ts.cell == cell]
    if not len(s): continue
    rows.append([lab, f"{s.mean_xseed_spearman.iloc[0]:+.3f}", f"{s.min_xseed_spearman.iloc[0]:+.3f}"])
table(["무대", "seed 쌍 평균 $\\rho$", "최소 $\\rho$"], rows)
W("""표 [D2]는 같은 진단을 오염축과 이질성축으로 분해한 것이다. 오염이 하나도 없는 clean 칸에서
IID 무대의 재현성이 0.133인 데 반해 5-도메인 비IID 무대는 0.867이다. 도메인 이질성만으로도
재현되는 신호가 생긴다는 뜻이고, 오염을 켜면 IID 무대도 0.60~0.70으로 올라온다. 즉 신호는
client 사이의 실제 차이(이질성 또는 오염)가 만들고, 둘 다 없는 IID-clean 칸은 재현할 순위가
없는 레짐이다. §5.2가 clean 칸을 우위 증거로 세지 않고, 주 세팅을 비IID·오염 쪽에 두는 근거가
여기에 있다.""")
W()
W("""표 [D2] — 이질성축과 오염축의 분해. 같은 $N{=}5$ 무대를 IID 5분할과 5-도메인 비IID로만
바꿔 in-run Shapley의 cross-seed 재현성을 잰 값이다 ●""")
W("""<!-- 출처: runs/phase2_matrix/target_stability.csv (rundir 1B_{iid5,silo5}_* 파생).
     오염축 밖 위협(frrand·poison) 행은 뺐다. -->""")
rows = []
for t, tl in [("clean", "clean (오염 0)"), ("noisy", "answer-swap"), ("frzero", "free-rider")]:
    a = ts[ts.cell == f"1B_iid5_{t}"].mean_xseed_spearman
    b = ts[ts.cell == f"1B_silo5_{t}"].mean_xseed_spearman
    if not len(a) or not len(b): continue
    rows.append([tl, f"{a.iloc[0]:+.3f}", f"{b.iloc[0]:+.3f}", f"{b.iloc[0]-a.iloc[0]:+.3f}"])
table(["위협", "IID 5분할", "5-도메인 비IID", "차"], rows)
W("""CNN 트랙에서도 같은 진단이 두 곳에 남는다. $N{=}10$ 격자의 CIFAR-10 IID clean 칸은 두
Shapley 값의 일치도부터 $-0.273$이고(표 [F8]), 같은 파티션의 removal-curve clean 행은 전 방법이
0 근방이라 제거 순서가 성능을 바꾸지 못한다(표 [I9]). 세 무대 모두 방법의 문제가 아니라 잴
것이 없는 레짐이라는 같은 사실을 가리킨다.""")
W()

# ==================================================================== E
W("""### 부록 E. 비용 전표

§5.4는 라운드 참여자 수 $K$가 10인 세팅 하나로 배율을 보였다. 이 부록은 같은 측정을 전 세팅으로
넓힌다. 모든 값은 기여도 평가에만 쓴 wall-clock이고 client의 로컬 학습 시간은 따로 표기한다.
측정은 재구현·정밀도·하드웨어에 의존하므로 §5.4의 연산수 모델이 하드웨어 독립 축이고 이 표는
그 교차 검증이다.""")
W()
W("""표 [C2] — 세팅별 라운드당 지배 연산의 인스턴스화(전체 = 라운드 수 $R$배). fp32·B200
microbench는 forward 1.60 s · HVP 10.36 s이고 비율은 6.47이다. fp32에서 bf16으로 옮기면
forward 5.33배·HVP 4.09배 빨라진다""")
W("<!-- 출처: runs/measured_2026-07/op_counts.py · runs/measured_2026-07/microbench/summary.json. -->")
table(["세팅", "$K$", "$R$", "Flirds", "Flirds-1st", "in-run SV"],
      [["5-도메인 비IID", "5", "10", "10 HVP", "10 grad", "320 fwd"],
       ["소형 앵커", "5", "30", "30 HVP", "30 grad", "960 fwd"],
       ["cross-device", "10", "30", "30 HVP", "30 grad", "30,720 fwd"]])

W("표 [C3] — LLM 세팅별 valuation wall-clock(초) ●")
W("<!-- 출처: runs/track_d/rundirs/*/metrics.json 의 runtime dict(3-seed). -->")
rows = []
rt = {}
for rd in sorted(glob.glob(os.path.join(R, "track_d/rundirs/*"))):
    stage = os.path.basename(rd).rsplit("_seed", 1)[0]
    blk = list(json.load(open(os.path.join(rd, "metrics.json"), encoding="utf-8")).values())[0]
    for m, v in (blk.get("runtime") or {}).items():
        rt.setdefault(stage, {}).setdefault(m, []).append(v)
stages = ["1B_std20", "3B_std20", "7B_std20", "1B_anchor5", "3B_anchor5", "7B_anchor5"]
SL = {"1B_std20": "alpaca 1B", "3B_std20": "alpaca 3B", "7B_std20": "alpaca 7B",
      "1B_anchor5": "앵커 1B", "3B_anchor5": "앵커 3B", "7B_anchor5": "앵커 7B"}
for m in M7 + ["(b)oracle"]:
    rows.append([LBL[m]] + [ms(rt.get(s, {}).get(m, []), 1, comma=True) for s in stages])
table(["방법"] + [SL[s] for s in stages], rows)

W("표 [C4] — LLM 5-도메인 비IID·cross-device 세팅의 valuation wall-clock(초) ●")
W("""<!-- 출처: runs/phase2_matrix/analysis/{01_silo5,04_device100_anchor}/csv/runtime_table.csv.
     비-앵커 α 셀(0.0/0.01/0.1/5.0)에서도 Flirds 155~158 s 로 α 무관. -->""")
r1 = pd.read_csv(os.path.join(R, "phase2_matrix/analysis/01_silo5/csv/runtime_table.csv")).set_index("method")
r2 = pd.read_csv(os.path.join(R, "phase2_matrix/analysis/04_device100_anchor/csv/runtime_table.csv")).set_index("method")
def refmt(s):
    try:
        a, b = str(s).split("±"); return f"{float(a):,.1f}±{float(b):,.1f}"
    except Exception:
        return "–"
rows = []
for m in M7 + ["(b)oracle"]:
    rows.append([LBL[m],
                 refmt(r1.loc[m, "noisy"]) if m in r1.index else "–",
                 refmt(r1.loc[m, "freerider_zero"]) if m in r1.index else "–",
                 refmt(r2.loc[m, "noisy"]) if m in r2.index else "–",
                 refmt(r2.loc[m, "freerider_zero"]) if m in r2.index else "–"])
ta = {}
for rd in sorted(glob.glob(os.path.join(R, "phase2_matrix/rundirs/1B_silo5_*_aonly_s*"))):
    t = os.path.basename(rd).split("_")[2]
    p = os.path.join(rd, "timing.json")
    if not os.path.exists(p): continue
    j = json.load(open(p, encoding="utf-8"))
    tot = j.get("total_s") or j.get("wall_s") or (j.get("phases") or {}).get("oracle_a")
    if tot is None:
        tot = sum(v for v in (j.get("phases") or {}).values() if isinstance(v, (int, float)))
    ta.setdefault(t, []).append(float(tot))
print("aonly timing:", {k: [round(x, 1) for x in v] for k, v in ta.items()})
rows.append(["_retrain SV ($2^5$ 전수 재학습)_",
             "_" + ms(ta.get("noisy", []), 1, comma=True) + "_",
             "_" + ms(ta.get("frzero", []), 1, comma=True) + "_", "_–_", "_–_"])
table(["방법", "비IID · answer-swap", "비IID · free-rider", "cross-device · answer-swap", "cross-device · free-rider"], rows)

W("표 [C5] — CNN valuation wall-clock(초). 왼쪽 넷은 주 세팅(오염 3위협 × 3seed = 9측정), 오른쪽 둘은 $N{=}10$ 격자 ●")
W("""<!-- 출처: runs/track_c/c2fid/rundirs/*/metrics.json 의 methods.<m>.runtime (주 세팅) ·
     runs/track_c/c1/{ds}_label-flip_seed*/metrics.json 및 c1_oracle/*_aonly_seed*/metrics.json
     의 t_a (N=10). 두 무대는 게임이 달라 서로 비교하지 않는다. -->""")
c2r = c2[c2.threat.isin(CORR)]
res = {}
for ds in ["cifar10", "mnist"]:
    for rd in sorted(glob.glob(os.path.join(R, f"track_c/c1/{ds}_label-flip_seed*"))):
        j = json.load(open(os.path.join(rd, "metrics.json"), encoding="utf-8"))
        for m, blk in (j.get("methods") or {}).items():
            res.setdefault(ds, {}).setdefault(m, []).append(blk.get("runtime"))
        res.setdefault(ds, {}).setdefault("_traj", []).append(j.get("traj_time"))
    for rd in sorted(glob.glob(os.path.join(R, f"track_c/c1_oracle/{ds}_label-flip_aonly_seed*"))):
        j = json.load(open(os.path.join(rd, "metrics.json"), encoding="utf-8"))
        res.setdefault(ds, {}).setdefault("_ta", []).append(j.get("t_a"))
rows = []
for m in M7 + ["(b)oracle"]:
    cells = [ms(c2r[(c2r.dataset == ds) & (c2r.partition == p) & (c2r.method == m)].runtime_s.values, 2, comma=True)
             for ds, p in SET]
    cells += [ms(res.get(ds, {}).get(m, []), 3, comma=True) for ds in ["cifar10", "mnist"]]
    rows.append([LBL[m]] + cells)
rows.append(["_(참조) 학습 궤적_", "–", "–", "–", "–"] +
            [ms(res.get(ds, {}).get("_traj", []), 1, comma=True) for ds in ["cifar10", "mnist"]])
rows.append(["_(참조) retrain SV ($2^{10}$ 전수 재학습)_", "–", "–", "–", "–"] +
            [ms(res.get(ds, {}).get("_ta", []), 1, comma=True) for ds in ["cifar10", "mnist"]])
table(["방법", "주 CIFAR-10/dir1", "주 CIFAR-10/iid", "주 MNIST/dir1", "주 MNIST/iid",
       "$N{=}10$ CIFAR-10", "$N{=}10$ MNIST"], rows)

W("""표 [C6] — 배율 사다리. 라운드 참여자 수 $K$가 커질수록 in-run Shapley 대비 이득이 커지고,
$K{=}2$에서는 $2^2$ forward가 HVP 1회(약 6.5 forward)보다 싸 부호가 뒤집힌다. 예측 열은
LLM 트랙의 per-op 실측($2^K/6.47$)이라 CNN 행에는 적용하지 않는다""")
W("<!-- 출처: 표 [C3]·[C4]·[C5] 과 같은 rundir. 예측 = 2^K / 6.47. -->")
table(["무대", "$K$", "Flirds(초)", "in-run SV(초)", "실측 배율", "연산수 모델 예측"],
      [["alpaca IID-clean 1B", "2", "4,696.7", "2,917.3", "0.62 (역전)", "0.62"],
       ["5-도메인 비IID 1B", "5", "106.6", "531.5", "4.99", "4.95"],
       ["cross-device 1B", "10", "157.3", "24,974.9", "158.8", "158.3"],
       ["CNN 주 CIFAR-10/dir1", "10", "10.64", "836.58", "78.6", "–"],
       ["CNN $N{=}10$ CIFAR-10", "10 (전원)", "1.168", "114.503", "98.0", "–"]])
W("""CNN의 배율이 LLM보다 작은 것은 모델이 작아 forward 한 번이 싸고, 그만큼 HVP 1회와
$K$-선형 부수 비용의 상대 비중이 커지기 때문이다. 같은 배율을 CNN에서 예측하려면 그 트랙의
per-op 상수를 따로 재야 하며, 이 표가 담는 것은 배율의 절대값이 아니라 $K$에 대한 증가
방향이다. retraining-based Shapley는 이 축의 바깥에 있다. CNN
$N{=}10$에서 $2^{10}$ 전수 재학습은 32,912초로 Flirds의 약 28,000배이고 학습 궤적 자체(104.5초)의
315배다. 학습과 평가의 위상을 나눠 재면 cross-device 세팅의 clean 셀에서 client 로컬 학습이
2,249초, 기여도 평가가 2,704초이며, 셀 하나가 4.2~4.5 GPU-h다.""")
W()

# ==================================================================== F
W("""### 부록 F. 개입 전표

§5.3은 selection-retrain 시점의 CIFAR-10 Dirichlet 칸과 LLM 주 세팅만 실었다. 이 부록은 같은
정책을 배포 중에 거는 online 시점, 크기로 가중하는 변형, MNIST와 IID 파티션 반복, 기여도 순위를
직접 심판하는 removal-curve, 그리고 개입의 전제인 부호 감사를 담는다. online 시점은 초기 10
라운드 동안 게이트를 끄고, 관측이 두 번 미만인 client는 후보에 남기며, 5라운드마다 배제된
client 하나를 순번대로 복귀시켜 배제가 흡수 상태가 되지 않게 한다. 후보가 비는 라운드는 전원
참여로 되돌린다.

표의 바닥 arm은 개입을 걸지 않은 별도 실행이고 점수원을 공급하는 관찰자 실행과 설정이 같다.
CIFAR-10에서는 두 실행이 서로 다른 난수 흐름을 타 소수점 셋째 자리에서 갈리므로, 게이트가
아무도 배제하지 않아 재학습이 개입 없는 학습과 같아지는 칸과 바닥 arm을 같은 실행에서 읽도록
바닥 arm을 전자로 통일했다. MNIST는 두 실행의 최종 정확도가 18쌍 전부 비트 동일이다.""")
W()

def acc_grid(ds, part, timing, policy):
    d = th[(th.dataset == ds) & (th.partition == part) & (th.timing == timing)]
    anc = th[(th.dataset == ds) & (th.partition == part)]
    v = anc[anc.arm == "vanilla"]; ob = anc[anc.arm == "observer"]
    have = set(zip(v.threat, v.seed))
    base = pd.concat([v, ob[[(t, s) not in have for t, s in zip(ob.threat, ob.seed)]]])
    rows = []
    for lbl, sel in [("vanilla (observer)", base), ("oracle-제외 (참조)", anc[anc.arm == "oracle_excl"]),
                     ("selection-random (통제)", anc[anc.arm == "random_excl"])]:
        cells = [ms(sel[sel.threat == t].final_acc.values, 2, pct=True) for t, _ in THR]
        piv = sel[sel.threat.isin(CORR)].pivot_table(index="seed", columns="threat", values="final_acc")
        cells.append(ms(piv.mean(axis=1).values, 2, pct=True) if piv.shape[1] == 3 else "–")
        rows.append([lbl] + cells)
    for s, m in [("flirds", "Flirds"), ("flirds1st", "Flirds1st"),
                 ("gtg", "GTG"), ("fedsv", "FedSV"), ("comfedsv", "ComFedSV"),
                 ("shapleyfl", "ShapleyFL"), ("fedif", "FedIF")]:
        sel = d[(d.source == s) & (d.policy == policy)]
        cells = [ms(sel[sel.threat == t].final_acc.values, 2, pct=True) for t, _ in THR]
        piv = sel[sel.threat.isin(CORR)].pivot_table(index="seed", columns="threat", values="final_acc")
        cells.append(ms(piv.mean(axis=1).values, 2, pct=True) if piv.shape[1] == 3 else "–")
        rows.append([LBL[m]] + cells)
    return rows

HEADA = ["", "clean", "zero-update free-rider", "gradient noise", "label-flip", "오염-평균"]
W("""**F.1 online 시점.** 표 [I3]은 §5.3과 같은 무대·같은 점수원에 게이트를 배포 중에 건
결과다. selection-retrain의 서열이 대체로 유지되지만 폭이 줄어든다. 게이트가 burn-in 동안 꺼져
있고 그 사이의 오염 업데이트가 이미 모델에 반영되기 때문이다. gradient noise 칸에서 Flirds만
회복하는 구도(56.68% vs 1차 계열 24.79%)와 zero-update free-rider 칸에서 재정규화 계열이
vanilla 아래로 내려가는 구도(39.15~40.20% vs vanilla 58.79%)는 두 시점에서 같다.""")
W()
W("표 [I3] — CNN 주 세팅 CIFAR-10, online 게이팅, test 정확도(%), 3-seed mean±std ●")
W("""<!-- 출처: runs/track_h/analysis/cnn_competition.csv (`python runs/track_h/make_analysis.py`),
     arm = <src>_gate_v2 · 바닥 arm = track_g rundirs 의 vanilla. label-flip 은 flip_rate=0.7. -->""")
for part in ["dir1", "iid"]:
    W(f"*CIFAR-10 / Dirichlet($\\alpha{{=}}1$)*" if part == "dir1" else "*CIFAR-10 / IID*")
    table(HEADA, acc_grid("cifar10", part, "online", "P1"))
W("표 [I4] — 같은 무대의 selection-retrain, IID 파티션, test 정확도(%)(Dirichlet 칸은 본문 표 [I2]) ●")
W("<!-- 같은 CSV, arm = t2_sign_<src>. -->")
table(HEADA, acc_grid("cifar10", "iid", "retrain", "P1"))

W("""**F.2 MNIST 반복.** 표 [I5]는 같은 무대를 MNIST로 반복한 것이다. 과제가 쉬워 전 arm이
92~98% 대역에 몰리므로 방법 간 폭이 CIFAR-10보다 훨씬 작지만, 갈리는 칸의 방향은 같다.
gradient noise에서 1차 계열이 vanilla 수준에 머물고, zero-update free-rider에서 재정규화
계열이 vanilla 아래로 내려간다.""")
W()
W("표 [I5] — CNN 주 세팅 MNIST, online / selection-retrain, test 정확도(%) ●")
W("<!-- 같은 CSV, dataset=='mnist'. -->")
for part in ["dir1", "iid"]:
    for timing, tl in [("online", "online"), ("retrain", "selection-retrain")]:
        W(f"*MNIST / {'Dirichlet' if part=='dir1' else 'IID'} · {tl}*")
        table(HEADA, acc_grid("mnist", part, timing, "P1"))

W("""**F.3 크기 가중 변형.** 표 [I6]은 배제 대신 남은 client의 집계 가중치를 기여도 크기에
비례하게 준 변형이다. 부호만 쓰는 게이트가 아무도 배제하지 않는 칸에서도 가중은 차등을 주므로,
gradient noise에서 게이트가 발화하지 못하던 FedIF가 여기서는 회복한다(CIFAR-10 IID online
63.21% vs 부호 게이트 26.19%). 반대로 Flirds-1st는 이 변형에서도 회복하지 못한다. 즉 1차 정보의
한계는 문턱의 문제가 아니라 신호 자체의 문제다.""")
W()
W("표 [I6] — 크기 가중 변형, CNN 주 세팅 CIFAR-10, test 정확도(%) ●")
W("<!-- 같은 CSV, arm = <src>_gatew_v2(online) · t2_signw_<src>(retrain). -->")
for part in ["dir1", "iid"]:
    for timing, tl in [("online", "online"), ("retrain", "selection-retrain")]:
        W(f"*CIFAR-10 / {'Dirichlet' if part=='dir1' else 'IID'} · {tl}*")
        table(HEADA, acc_grid("cifar10", part, timing, "P2"))

W("""**F.4 LLM 전표.** 표 [I7]은 본문 표 [I1]을 online 시점과 나머지 한 점수원까지 넓힌 것이다.
selection-retrain의 answer-swap 칸에서 세 점수원이 모두 vanilla를 1.8~2.2pt 올려
selection-random(+0.1pt)과 뚜렷이 갈리지만, 세 값이 서로 0.4pt 안에 모여 있어 점수원 사이의
서열은 seed 편차와 구별되지 않는다. online 시점은 두 Taylor 변형만 실행했고 같은 그림이
반복된다. 이 무대가 지지하는 것은 기여도 부호가 고른 집합이 같은 크기의 무작위 집합을
이긴다는 데까지이고, 2차 곡률항의 이득은 여기서 분리되지 않는다.""")
W()
W("표 [I7] — LLM 주 세팅(GSM8K·$N{=}50$·5/50·$R{=}200$), test 1,119문항 EM(%) ●")
W("""<!-- 출처: runs/track_h/analysis/llm_competition.csv (regime=gsm50k5). online = <src>_gate_v2,
     selection-retrain = t2_sign_<src>. 재정규화 4종은 이 무대에서 미실행이라 표에 없다
     (계열 간 대조는 표 [I2]·[F11]이 맡는다). -->""")
g = lc[lc.regime == "gsm50k5"]
rows = []
for a, lab in [("observer", "vanilla (observer)"), ("oracle_excl", "oracle-제외 (참조)"),
               ("random_excl", "selection-random (통제)")]:
    s = g[g.arm == a]
    rows.append([lab, "–"] + [ms(s[s.threat == t].gsm8k_em.values, 2, pct=True) for t in ["clean", "noisy", "frzero"]])
for tm, pref in [("online", "{}_gate_v2"), ("selection-retrain", "t2_sign_{}")]:
    for src, m in [("flirds", "Flirds"), ("flirds1st", "Flirds1st"), ("fedif", "FedIF")]:
        s = g[g.arm == pref.format(src)]
        if not len(s): continue
        rows.append([LBL[m], tm] + [ms(s[s.threat == t].gsm8k_em.values, 2, pct=True) for t in ["clean", "noisy", "frzero"]])
table(["", "시점", "clean", "answer-swap", "zero-update free-rider"], rows)

W("""표 [I8] — 같은 무대의 게이트 정밀도·재현율(%). free-rider 칸은 두 변형 모두 재현율 94% 이상으로
대상 집합을 거의 그대로 집어내는 반면 answer-swap 칸은 재현율 5% 안팎이다. 후자는 추정 실패가
아니라 게임의 답이다. 부록 F.6이 보이듯 in-run Shapley로 채점해도 answer-swap client의 누적
기여도는 양수이므로, 문턱 0의 부호 게이트에는 이 위협의 작동 영역이 없다 ●""")
W("<!-- 같은 CSV, gate_precision · gate_recall 열. -->")
rows = []
for a, m in [("flirds_gate_v2", "Flirds"), ("flirds1st_gate_v2", "Flirds1st")]:
    for t, tl in [("clean", "clean"), ("noisy", "answer-swap"), ("frzero", "free-rider")]:
        s = g[(g.arm == a) & (g.threat == t)]
        if not len(s): continue
        rows.append([LBL[m], tl, ms(s.gate_precision.values, 2, pct=True), ms(s.gate_recall.values, 2, pct=True)])
table(["", "위협", "precision", "recall"], rows)

W("""**F.5 removal-curve.** 표 [I9]는 기여도 순위를 게임 바깥에서 심판한다. 낮은 순위부터
제거하며 재학습한 곡선과 높은 순위부터 제거한 곡선의 간격이 클수록 그 순위가 성능에 인과적이다.
in-run Shapley와 Flirds가 세 위협 모두 사실상 같은 간격을 내고(차 ≤0.15%p), gradient noise에서
Flirds-1st만 부호가 뒤집혀 $-1.89$%p가 된다. 재정규화 계열은 free-rider 칸에서 무너진다
(ShapleyFL $-16.26$%p). clean 칸은 전 방법이 0 근방이라 제거 순서가 성능을 바꾸지 못하며, 이는
부록 D의 신호 부재 판정과 같은 사실이다.""")
W()
W("표 [I9] — removal-curve의 곡선 평균 간격(%p; CIFAR-10/IID·$N{=}10$ 전원참여·$R{=}10$·오염 4/10) ●")
W("""<!-- 출처: runs/removal_dose/rundirs_cnn/cifar10_iid{,-free-rider,-grad-noise,-label-flip}_seed*
     의 metrics.json::removal_curve_acc. 값 = mean(worst-first acc − best-first acc), 10점 전 구간 평균. -->""")
scen = {"cifar10_iid-free-rider": "zero-update free-rider", "cifar10_iid-grad-noise": "gradient noise",
        "cifar10_iid-label-flip": "label-flip", "cifar10_iid": "clean"}
res2, ends = {}, {}
for rd in sorted(glob.glob(os.path.join(R, "removal_dose/rundirs_cnn/*"))):
    key = os.path.basename(rd).rsplit("_seed", 1)[0]
    if key not in scen: continue
    p = os.path.join(rd, "metrics.json")
    if not os.path.exists(p): continue
    for m, cur in (json.load(open(p, encoding="utf-8")).get("removal_curve_acc") or {}).items():
        wf = np.array([v for _, v in cur["worst_first"]]); bf = np.array([v for _, v in cur["best_first"]])
        res2.setdefault(scen[key], {}).setdefault(m, []).append(float(np.mean(wf - bf)))
        ends.setdefault(scen[key], {}).setdefault(m, []).append((wf[0], wf.max(), bf.min()))
MET = ["(b)oracle"] + M7
rows = []
for t in ["clean", "zero-update free-rider", "gradient noise", "label-flip"]:
    if t not in res2: continue
    rows.append([t] + [ms(res2[t].get(m, []), 2, True, pct=True) for m in MET])
cm = []
for m in MET:
    arrs = [res2[t][m] for t in ["zero-update free-rider", "gradient noise", "label-flip"] if m in res2.get(t, {})]
    cm.append(ms(list(np.mean(np.array(arrs), axis=0)), 2, True, pct=True) if len(arrs) == 3 else "–")
rows.append(["**오염-평균**"] + cm)
table(["위협"] + [LBL[m] + (" *(앵커)*" if m == "(b)oracle" else "") for m in MET], rows)

W("""표 [I10] — 같은 심판을 LLM 5-도메인 비IID 세팅에서 반복한 값. 지표는 4명을 제거했을 때의
검증손실 감소이고, 양수면 그 순서대로 제거하는 것이 손실을 낮춘다는 뜻이다 ●""")
W("<!-- 출처: runs/removal_dose/rundirs/1B_silo5_{noisy,frzero}_removal_seed* 의 removal_curve. -->")
rows = []
for tag, lab in [("noisy", "answer-swap"), ("frzero", "zero-update free-rider")]:
    wf, bf = [], []
    for rd in sorted(glob.glob(os.path.join(R, f"removal_dose/rundirs/1B_silo5_{tag}_removal_seed*"))):
        blk = list(json.load(open(os.path.join(rd, "metrics.json"), encoding="utf-8")).values())[0]
        rc = (blk.get("removal_curve") or {}).get("Flirds") or {}
        if not rc: continue
        w = [v for _, v in rc["worst_first"]]; b = [v for _, v in rc["best_first"]]
        wf.append(w[0] - w[-1]); bf.append(b[0] - b[-1])
    rows.append([lab, ms(wf, 4, True), ms(bf, 4, True)])
table(["위협", "낮은 순위부터 제거", "높은 순위부터 제거"], rows)

W("""**F.6 부호 감사.** 게이트의 두 전제는 정상 client를 배제하지 않는 것과 오염 client를
발화시키는 것이다. 표 [I11]·[I12]는 개입을 실행하지 않고 누적 기여도의 부호만 세어 그 전제를
직접 확인한다. zero-update free-rider는 식 (5)를 겨냥하는 두 방법 모두 대수적으로 정확한 0을
받아(CNN 48/48 슬롯, LLM 100%) 발화가 보장되고 정상 client의 오배제는 0이다. 재정규화 계열은
같은 칸에서 0이 아니라 음수나 양수로 흩어지고, ShapleyFL은 free-rider를 정상 client보다 높게
매기는 반전까지 보인다(표 [I11]의 CIFAR-10 분리도 $-0.586$). answer-swap은 반대로 어떤 방법도
발화하지 않는데, in-run Shapley로 채점해도 같으므로 이는 추정이 아니라 게임의 답이다.""")
W()
W("""표 [I11] — CNN $N{=}10$ 격자의 부호 감사. 기여도는 유익한 client가 양수가 되도록 부호를
통일해 읽는다 ●""")
W("<!-- 출처: runs/track_c/c1/analysis/sign_audit.csv (`python runs/track_c/c1/make_analysis.py`). -->")
sa = pd.read_csv(os.path.join(R, "track_c/c1/analysis/sign_audit.csv"))
ORD = ["(a)oracle", "(b)oracle"] + M7
W("*(a) clean 셀에서 기여도가 음수인 client 수 / 전체 슬롯*")
rows = []
for ds, part in SET:
    d = sa[(sa.dataset == ds) & (sa.partition == part) & (sa.threat == "clean")]
    rows.append([f"{ds}/{part}"] + [f"{int(d[d.method==m].n_neg_clean.sum())}/{int(d[d.method==m].shape[0]*10)}"
                                    if len(d[d.method == m]) else "–" for m in ORD])
table(["세팅"] + [LBL[m] for m in ORD], rows)
W("*(b) zero-update free-rider 오염 client의 exact-0 비율과 음수 비율(분모 48 = 12셀 × 4명)*")
d = sa[sa.threat == "free_rider"]
rows = []
for m in ORD:
    dm = d[d.method == m]
    if not len(dm): continue
    n = int(dm.shape[0] * 4)
    rows.append([LBL[m], f"{int(dm.n_exact_zero_corrupt.sum())}/{n}", f"{int(dm.n_neg_corrupt.sum())}/{n}"])
table(["방법", "exact-0", "기여도 음수"], rows)
W("*(c) 분리도 = (정상 평균 기여도 − 오염 평균 기여도) / span. 양수면 오염을 낮게 매긴 것이다*")
rows = []
for ds in ["cifar10", "mnist"]:
    for t, tl in [("free_rider", "free-rider"), ("grad_noise", "grad noise"), ("label_flip", "label-flip")]:
        d = sa[(sa.dataset == ds) & (sa.threat == t)]
        rows.append([ds, tl] + [ms(d[d.method == m].phi_gap_norm.dropna().values, 3, True) for m in ORD])
table(["데이터셋", "위협"] + [LBL[m] for m in ORD], rows)

W("""표 [I12] — LLM 두 세팅의 부호 감사. 값은 client-라운드 누적 기여도가 0 이하인 client의
비율(%)이고, 괄호는 그중 대수적으로 정확한 0의 비율이다 ●""")
W("""<!-- 출처: runs/track_g/audit/sign_table.csv, variant=='canon' & scale=='1B' 필터 후
     (regime, threat, method, corrupt)별 contribution 부호 집계. cross-device 는 in-run Shapley
     가 붙은 α=0.5 앵커 셀로 한정해 전 방법의 분모를 맞췄다. ComFedSV 는 5-도메인 세팅에
     미산출이라 ⬚. -->""")
st = pd.read_csv(os.path.join(R, "track_g/audit/sign_table.csv"))
st = st[(st.variant == "canon") & (st.scale == "1B") & (st.cell.str.startswith("phase2_matrix/rundirs/"))].copy()
st["threat"] = st.threat.replace({"freerider_zero": "frzero"})
st = st[~((st.regime == "device100") & (~st.cell.str.contains("a0.5")))]
CELLS = [("silo5", "clean"), ("silo5", "noisy"), ("silo5", "frzero"),
         ("device100", "noisy"), ("device100", "frzero")]
CL = {("silo5", "clean"): "비IID · clean", ("silo5", "noisy"): "비IID · swap",
      ("silo5", "frzero"): "비IID · free-rider", ("device100", "noisy"): "cross-device · swap",
      ("device100", "frzero"): "cross-device · free-rider"}
for corrupt, lab in [(False, "*(a) 정상 client — 오배제 위험 (낮을수록 좋다)*"),
                     (True, "*(b) 오염 client — 게이트 발화 (높을수록 좋다; 괄호 = exact-0)*")]:
    W(lab)
    rows = []
    for m in ["(b)oracle"] + M7:
        cells = []
        for r, t in CELLS:
            d = st[(st.regime == r) & (st.threat == t) & (st.method == m) & (st.corrupt == corrupt)]
            if not len(d): cells.append("⬚"); continue
            p = 100.0 * (d.contribution <= 0).mean()
            cells.append(f"{p:.1f} ({100.0*(d.contribution==0).mean():.1f})" if corrupt else f"{p:.1f}")
        rows.append([LBL[m]] + cells)
    rows.append(["*(client 수 n)*"] + [str(len(st[(st.regime == r) & (st.threat == t) &
                                                  (st.method == "Flirds") & (st.corrupt == corrupt)]))
                                       for r, t in CELLS])
    table(["방법"] + [CL[c] for c in CELLS], rows)

W("""**F.7 clean 무해성.** 표 [I13]은 오염이 하나도 없는 무대에서 기여도 기반 개입이 성능을
깎지 않는지만 본다. 오염이 없으므로 이득은 원리적으로 없고, 볼 것은 손해의 부재다. 세 규모
모두 어떤 개입도 vanilla 대비 MMLU 0.13%p·ROUGE-L 0.15%p·검증손실 0.0009 이내이며 이는 seed
편차보다 작다. 부수로 7B에서만 개입 arm이 목표 손실 도달 라운드를 184.7에서 151~158로 줄이는데,
규모가 커지며 client 사이의 차이가 seed 잡음을 넘기 시작한다는 부록 D의 관찰과 같은 방향이다.
다만 도달 라운드는 $R{=}200$ 안에 목표에 닿지 못한 seed가 섞여 있어 그 seed를 평균에서 뺀
값이므로, 표에 도달 seed 수를 함께 적고 이 관찰을 예비적인 것으로 둔다.""")
W()
W("표 [I13] — alpaca IID-clean 세팅($N{=}20$·2/20·$R{=}200$)의 무해성 ●")
W("<!-- 출처: runs/track_d/rundirs/*_std20_seed*/metrics.json 의 arms 블록. -->")
armacc = {}
for rd in sorted(glob.glob(os.path.join(R, "track_d/rundirs/*std20*"))):
    scale = os.path.basename(rd).split("_")[0]
    j = list(json.load(open(os.path.join(rd, "metrics.json"), encoding="utf-8")).values())[0]
    for a, blk in (j.get("arms") or {}).items():
        for k, v in blk.items():
            armacc.setdefault((a, k, scale), []).append(v)
ARMS = [("base", "base (학습 전)"), ("vanilla", "vanilla (개입 없음)"), ("flirds_w", "Flirds 가중"),
        ("flirds_sel", "Flirds 선택"), ("shapleyfl_w", "ShapleyFL 가중"), ("fedif_w", "FedIF 가중")]
for met, lab, nd, pc in [("mmlu", "MMLU (full-test 0-shot, %)", 2, True),
                         ("rouge_l", "Alpaca-test ROUGE-L (%)", 2, True),
                         ("final_val_loss", "최종 검증손실", 5, False)]:  # 손실은 비율이 아니라 그대로
    W(f"*{lab}*")
    table(["arm", "1B", "3B", "7B"],
          [[l] + [ms(armacc.get((a, met, s), []), nd, pct=pc) for s in ["1B", "3B", "7B"]] for a, l in ARMS])
W("*목표 검증손실 도달 라운드(괄호 = $R{=}200$ 안에 도달한 seed 수 / 3; 미도달 seed는 평균에서 뺐다)*")
rows = []
for a, l in ARMS[1:]:
    cells = []
    for s in ["1B", "3B", "7B"]:
        v = [x for x in armacc.get((a, "rounds_to_target", s), []) if x is not None]
        cells.append("–" if not v else f"{ms(v, 1)} ({len(v)}/3)")
    rows.append([l] + cells)
table(["arm", "1B", "3B", "7B"], rows)

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "appendix_cdef.md")
txt = blockify(o.getvalue())
open(p, "w", encoding="utf-8").write(txt)
print("wrote", p, len(txt))
