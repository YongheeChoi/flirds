"""Phase 1 data-layer smoke: 5-domain loader counts / disjointness / format.

No model/GPU.  Validates flirds.data.llm.build: cross-silo partition (N=5 and
N=10), §3.4 validation sizes, Dolly train/val disjointness, and the per-domain
prompt/completion mapping.  Downloads the 5 HF datasets on first run (cached).
"""
from flirds.data.llm import ORDER, build
from flirds.repro import seed_everything

PDT, PDV = 200, 200


def main():
    seed_everything(0)
    for n in (5, 10):
        clients, val = build(n_clients=n, per_domain_train=PDT, per_domain_val=PDV, seed=0)
        sizes = [len(c) for c in clients]
        print(f"N={n}: clients={len(clients)} sizes={sizes} val={len(val)}")
        assert len(clients) == n
        assert len(val) == 5 * PDV
        assert all(s == PDT // (n // 5) for s in sizes)

    # Dolly (general) is the only domain carved from a single split -> check disjoint
    clients, val = build(10, PDT, PDV, seed=0)
    gen_train = set()
    for c in clients[8:10]:                       # general = ORDER[4] -> clients 8,9 at N=10
        gen_train |= {r["prompt"] for r in c}
    gen_val = {r["prompt"] for r in val[4 * PDV:5 * PDV]}
    overlap = gen_train & gen_val
    print(f"Dolly train n val overlap = {len(overlap)} (expect 0)")
    assert not overlap

    # one validation sample per domain (eyeball the mapping)
    clients, val = build(5, PDT, PDV, seed=0)
    for i, dom in enumerate(ORDER):
        r = val[i * PDV]
        print(f"\n--- {dom} ---\nPROMPT: {r['prompt'][:240]!r}\nCOMPL : {r['completion'][:160]!r}")
    print("\nDATA SMOKE OK")


if __name__ == "__main__":
    main()
