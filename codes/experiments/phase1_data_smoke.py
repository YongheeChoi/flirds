"""Phase 1 data-layer smoke: 5-domain loader counts / 3-way disjointness / format.

No model/GPU.  Validates flirds.data.llm.build: cross-silo partition (N=5 and
N=10), §3.4 validation sizes, the held-out test set (counts + per-domain `domain`
/ math gold `answer` tags), and train/val/test mutual disjointness per domain.
Downloads the 5 HF datasets on first run (cached).
"""
from flirds.data.llm import ORDER, build
from flirds.repro import seed_everything

PDT, PDV, PDTEST = 200, 50, 50


def _pairs(records):
    return {(r["prompt"], r["completion"]) for r in records}


def main():
    seed_everything(0)
    for n in (5, 10):
        clients, val, test = build(n_clients=n, per_domain_train=PDT, per_domain_val=PDV,
                                   per_domain_test=PDTEST, seed=0)
        sizes = [len(c) for c in clients]
        print(f"N={n}: clients={len(clients)} sizes={sizes} val={len(val)} test={len(test)}")
        assert len(clients) == n
        assert len(val) == 5 * PDV and len(test) == 5 * PDTEST
        assert all(s == PDT // (n // 5) for s in sizes)

    # train/val/test 3-way disjointness per domain (N=5 -> client i == domain i)
    clients, val, test = build(5, PDT, PDV, per_domain_test=PDTEST, seed=0)
    for i, dom in enumerate(ORDER):
        tr = _pairs(clients[i])
        va = _pairs(val[i * PDV:(i + 1) * PDV])
        te = _pairs(test[i * PDTEST:(i + 1) * PDTEST])
        d_tv, d_tt, d_vt = tr & va, tr & te, va & te
        print(f"  {dom:8s} train={len(tr)} val={len(va)} test={len(te)} | "
              f"train∩val={len(d_tv)} train∩test={len(d_tt)} val∩test={len(d_vt)}")
        assert not (d_tv or d_tt or d_vt), f"{dom} splits overlap"

    # test records carry domain (+ math gold answer for exact-match)
    assert all("domain" in r for r in test)
    math = [r for r in test if r["domain"] == "math"]
    assert math and all(r.get("answer") in list("ABCDE") for r in math)
    print(f"  test tags: domains={sorted({r['domain'] for r in test})} | math gold e.g. {math[0]['answer']!r}")

    # one validation sample per domain (eyeball the prompt/completion mapping)
    for i, dom in enumerate(ORDER):
        r = val[i * PDV]
        print(f"\n--- {dom} ---\nPROMPT: {r['prompt'][:200]!r}\nCOMPL : {r['completion'][:140]!r}")
    print("\nDATA SMOKE OK")


if __name__ == "__main__":
    main()
