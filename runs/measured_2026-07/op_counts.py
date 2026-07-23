"""Analytic op-count cost model for the valuation methods + cross-validation
against the B200 microbench per-op times (microbench/summary.json).

Why this exists (cost-comparison-methodology 2026-07 sec 5.1 #3): a method's
per-round #forward / #val-grad / #HVP is fixed by its ALGORITHM and config, not by
fp32/bf16, CPU/GPU, or our re-implementation choices.  So the op-count is a
hardware- and precision-INDEPENDENT cost axis -- the one axis on which our numbers
are directly comparable to the field's asymptotic reporting (GTG/FedSV/ComFedSV/
FLDetector all report cost in utility-eval counts, never wall-clock; see sec 2.8).

Cross-check: for the pure-forward methods (coalition family, loss-heuristic) and
the pure-HVP method (Flirds), (op-count x microbench per-op time) reproduces the
measured wall-clock to within a few percent -> the op-count IS the cost claim, and
it ports to any hardware/precision by swapping the per-op seconds.

Counts are read straight off the implementations (all on the frozen shared logs):
  Flirds        core/flirds_estimator.py   1 HVP/round (jvp-of-grad yields g and
                                            u=H dW together) + |P_r| dot products
  Flirds-1st    (second_order=False)        1 val-grad/round + |P_r| dots
  FedIF         baselines/fedif.py          1 val-grad/round
  loss-heur     oracle/in_run_sv.py         in_run_singletons: 1 base + |P_r| forwards
                (in_run_singletons)          /round  [was 2|P_r| before the C6 fix]
  Fed-LOO       oracle/in_run_sv.py         in_run_loo: 2 + |P_r| forwards/round
  ShapleyFL     baselines/shapleyfl.py      exact_shapley over P_r: 2^|P_r| forwards
  (b)/Banzhaf   oracle/in_run_sv.py         exact per-round: 2^|P_r| forwards/round
  FedSV         baselines/fedsv.py          perm-MC, n_perm=max(30,2|P_r|); distinct
                                            forwards <= min(2^|P_r|, n_perm*|P_r|),
                                            cut by subset cache + TMC trunc (eps 1e-3)
  GTG           baselines/gtg.py            guided-trunc MC, <=max(30,0.8*2^|P_r|)
                                            perms; distinct forwards <= 2^|P_r|
  ComFedSV      baselines/comfedsv.py       M=max(10,ceil(N ln N)) perms; forwards =
                                            R*(1 + #coalitions subset of cohort), + CPU ALS

Run: python runs/measured_2026-07/op_counts.py   (no deps beyond the json)
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, R rounds, K participants/round, N clients).  Full participation => K == N.
REGIMES = [
    ("silo   (N=5,  R=10, full)", 10, 5, 5),
    ("anchor (N=5,  R=30, full)", 30, 5, 5),
    ("device (N=100,R=30, K=10)", 30, 10, 100),
]

# Measured valuation wall-clock (s) from Table tab:cost / methodology sec 1.4, for the
# cross-check.  loss-heur silo/anchor are the pre-fix (inflated) entries; "fixed" is the
# op-count prediction the re-measurement should land on.
MEASURED = {
    "silo   (N=5,  R=10, full)": {"Flirds": 107, "Flirds-1st": 35, "loss-heur": 170,
                                  "(b)/Banzhaf": 530, "ShapleyFL": 530},
    "anchor (N=5,  R=30, full)": {"Flirds": 707, "Flirds-1st": 231, "loss-heur": 1093,
                                  "(b)/Banzhaf": 3528},
    "device (N=100,R=30, K=10)": {"Flirds": 157, "Flirds-1st": 53, "(b)/Banzhaf": 24975},
}


def per_round(method, K, N):
    """Return (counts dict over fwd/grad/hvp/dots, is_upper_bound)."""
    two_K = 2 ** K
    if method == "Flirds":        return {"hvp": 1, "dots": K}, False
    if method == "Flirds-1st":    return {"grad": 1, "dots": K}, False
    if method == "FedIF":         return {"grad": 1}, False
    if method == "loss-heur":     return {"fwd": 1 + K}, False           # C6-fixed
    if method == "loss-heur(pre-fix)": return {"fwd": 2 * K}, False
    if method == "Fed-LOO":       return {"fwd": 2 + K}, False
    if method == "ShapleyFL":     return {"fwd": two_K}, False
    if method == "(b)/Banzhaf":   return {"fwd": two_K}, False
    if method == "FedSV":
        n_perm = max(30, 2 * K)
        return {"fwd": min(two_K, n_perm * K)}, True                     # cap; cache+trunc lower
    if method == "GTG":
        n_perm = min(two_K, max(30, math.ceil(0.8 * two_K)))
        return {"fwd": min(two_K, n_perm * K)}, True
    raise ValueError(method)


METHODS = ["Flirds", "Flirds-1st", "FedIF", "loss-heur", "loss-heur(pre-fix)",
           "ShapleyFL", "(b)/Banzhaf", "FedSV", "GTG"]   # Fed-LOO dropped 07-23
# NOTE: this list drives tab:opcount in the paper -- dropping the row here removes
# Fed-LOO from that table too.  The per-op formula below is kept for replay.


def comfedsv_forwards(R, K, N):
    """R*(1 + #coalitions subset of cohort); upper-bounded by the full coalition set
    C <= M*N.  Report the budget M and the full-obs upper bound R*(1+C_max)."""
    M = max(10, math.ceil(N * math.log(N))) if N > 1 else 1
    C_max = min(2 ** K, M * K)          # distinct prefixes of length<=K observable in a K-cohort
    return M, R * (1 + C_max)


def main():
    with open(os.path.join(HERE, "microbench", "summary.json")) as f:
        mb = json.load(f)
    t_fwd = mb["forward_fp32"]["s_per_pass"]      # 1.601 s  (val=100)
    t_hvp = mb["hvp_fp32"]["s_per_pass"]          # 10.360 s (val=100)
    print(f"# microbench per-op (fp32, val=100, B200): forward {t_fwd:.3f}s, HVP {t_hvp:.3f}s")
    print(f"#   HVP/forward = {t_hvp / t_fwd:.2f}  (1 Flirds HVP costs ~{t_hvp / t_fwd:.1f} coalition forwards)\n")

    for label, R, K, N in REGIMES:
        print(f"## {label}")
        print(f"{'method':<20}{'#fwd':>9}{'#grad':>7}{'#HVP':>6}{'#dots':>8}"
              f"{'pred_s':>9}{'meas_s':>8}  note")
        for m in METHODS:
            c, ub = per_round(m, K, N)
            fwd, grad, hvp, dots = (R * c.get(k, 0) for k in ("fwd", "grad", "hvp", "dots"))
            # prediction only where op-cost is unambiguous (pure fwd or pure HVP; val=100 => silo)
            pred = ""
            if K == N == 5 and R == 10:                       # silo matches microbench val=100
                if hvp and not fwd and not grad:
                    pred = f"{hvp * t_hvp:6.0f}"
                elif fwd and not hvp and not grad:
                    pred = f"{fwd * t_fwd:6.0f}"
            meas = MEASURED.get(label, {}).get(m, "")
            pre = "<=" if ub else ""
            note = "cache+TMC-trunc lowers" if ub else ""
            print(f"{m:<20}{pre + format(fwd, ',') :>9}{grad or '':>7}{hvp or '':>6}"
                  f"{dots or '':>8}{pred:>9}{str(meas):>8}  {note}")
        M, cf = comfedsv_forwards(R, K, N)
        print(f"{'ComFedSV':<20}{'<=' + format(cf, ','):>9}{'':>7}{'':>6}{'':>8}{'':>9}{'':>8}"
              f"  M={M} perms; +CPU ALS")
        print()

    print("# cross-check (silo, val=100): pure-forward and pure-HVP predictions vs measured")
    print("#   Flirds       10 HVP  x 10.36 = 104 s   (measured 107)")
    print("#   (b)/Banzhaf  320 fwd x 1.60  = 512 s   (measured ~530)")
    print("#   loss-heur    60 fwd  x 1.60  =  96 s   (measured 170 pre-fix -> ~102 fixed)")


if __name__ == "__main__":
    main()
