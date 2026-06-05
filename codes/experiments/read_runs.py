"""Read phase1_clean_run run-dirs and print a per-lr comparison (the #7 question:
does Flirds-top-K beat random-K and approach full?).

Usage from codes/:
  PYTHONPATH=. python experiments/read_runs.py runs/full_lr1e-3 runs/full_lr3e-3
  (no args -> auto-globs runs/full_lr*)
"""
import glob
import json
import os
import sys
from statistics import mean, pstdev

ARMS = ("full", "flirds_topk", "random_k")


def _runs(root):
    out = []
    for mj in sorted(glob.glob(os.path.join(root, "*", "metrics.json"))):
        out.append((os.path.basename(os.path.dirname(mj)), json.load(open(mj))))
    return out


def _rouge(m, arm):                       # mean per-domain ROUGE-L for an arm
    ta = m["arms"][arm]["task_acc"]
    return mean(ta[d]["rouge_l"] for d in ta)


def _vfin(m, arm):                        # final val-loss of an arm
    return m["arms"][arm]["val_loss_curve"][-1]


def main():
    roots = sys.argv[1:] or sorted(glob.glob("runs/full_lr*"))
    for root in roots:
        runs = _runs(root)
        if not runs:
            print(f"\n=== {root} === (no completed seeds yet)")
            continue
        print(f"\n=== {root} ({len(runs)} seed(s)) ===")
        for name, m in runs:
            keep = m.get("selection", {})
            print(f"  {name}: AUROC noisy={m['auroc_noisy']:.3f} fr={m['auroc_freerider']:.3f} "
                  f"| flirds_keep={keep.get('flirds_keep')} random_keep={keep.get('random_keep')}")
            for a in ARMS:
                print(f"      {a:11s} val_loss->{_vfin(m, a):.4f}  ROUGE-L={_rouge(m, a):.4f}")
        # per-arm mean+/-std across seeds (the headline comparison)
        print(f"  -- mean+/-std over {len(runs)} seed(s) --")
        for a in ARMS:
            v = [_vfin(m, a) for _, m in runs]
            r = [_rouge(m, a) for _, m in runs]
            print(f"      {a:11s} val_loss={mean(v):.4f}+/-{pstdev(v):.4f}  ROUGE-L={mean(r):.4f}+/-{pstdev(r):.4f}")
        print("  (Flirds works if flirds_topk val_loss <= random_k and ROUGE-L >= random_k.)")


if __name__ == "__main__":
    main()
