"""Phase 2 task 7a cross-device data smoke: per-client Dirichlet partition math
+ build_crossdevice wiring.

Part 1 (synthetic, no HF): client_dirichlet_partition over 5 domains -> 100 clients
across the alpha-sweep {0, 0.01, 0.1, 0.5, 5.0} -- asserts all-non-empty, exact
per-client size, cross-client disjointness, alpha=0 balanced single-domain, and
purity monotone-decreasing in alpha (the heterogeneity knob).
Part 2 (HF, cached): build_crossdevice small-scale -- client counts/sizes, §3.4
val/test held-out parity, noisy answer-swap (prompts kept, pairings broken).
"""
import numpy as np

from flirds.data.llm import ORDER, build_crossdevice
from flirds.fl.partition import client_dirichlet_partition
from flirds.repro import seed_everything

ALPHAS = [0.0, 0.01, 0.1, 0.5, 5.0]


def _purity(idx, labels, n_cls):
    cnt = np.bincount(labels[np.asarray(idx)], minlength=n_cls)
    return cnt.max() / cnt.sum(), int((cnt > 0).sum())


def synthetic():
    D, PER_DOMAIN, N, PC = 5, 12000, 100, 200
    labels = np.repeat(np.arange(D), PER_DOMAIN)
    print(f"=== Part 1: client_dirichlet_partition ({D} domains x {PER_DOMAIN} -> {N} clients, per_client={PC}) ===")
    print(f"{'alpha':>6} {'nonempty':>9} {'size(min/max)':>14} {'purity':>7} {'doms/cli':>9} {'disjoint':>9}")
    prev = 2.0
    for a in ALPHAS:
        parts = client_dirichlet_partition(labels, N, a, PC, seed=0)
        sizes = np.array([len(p) for p in parts])
        flat = np.concatenate([np.asarray(p) for p in parts])
        disjoint = len(flat) == len(set(flat.tolist()))
        purs, dpc = zip(*[_purity(p, labels, D) for p in parts])
        mp, mdpc = float(np.mean(purs)), float(np.mean(dpc))
        print(f"{a:>6} {(sizes > 0).sum():>9} {str((int(sizes.min()), int(sizes.max()))):>14} "
              f"{mp:>7.3f} {mdpc:>9.2f} {str(disjoint):>9}")
        assert (sizes > 0).sum() == N, "empty clients"
        assert sizes.min() == PC and sizes.max() == PC, "client size not fixed"
        assert disjoint, "indices reused across clients"
        assert mp <= prev + 1e-9, "purity not monotone-decreasing in alpha"
        prev = mp
        if a == 0.0:                                  # domain-disjoint: 1 domain/client, balanced N/D
            assert mp == 1.0 and mdpc == 1.0
            counts = np.bincount([int(labels[p[0]]) for p in parts], minlength=D)
            assert counts.min() == counts.max() == N // D, f"alpha=0 imbalance {counts}"
    print("  Part 1 OK (non-empty, fixed-size, disjoint, alpha=0 balanced single-domain, purity monotone)\n")


def integration():
    N, PC, POOL, PDV, PDTEST = 100, 50, 2000, 20, 10
    print(f"=== Part 2: build_crossdevice(N={N}, alpha=0.5, per_client={PC}, pool={POOL}) [loads 5 HF datasets] ===")
    clients, val, test = build_crossdevice(N, alpha=0.5, per_client_train=PC, per_domain_pool=POOL,
                                           per_domain_val=PDV, per_domain_test=PDTEST, seed=0, noisy={7})
    sizes = [len(c) for c in clients]
    print(f"  clients={len(clients)} size(min/max)=({min(sizes)},{max(sizes)}) val={len(val)} test={len(test)}")
    assert len(clients) == N
    assert all(s == PC for s in sizes), "client size != per_client_train"
    assert len(val) == 5 * PDV and len(test) == 5 * PDTEST
    assert all(("prompt" in r and "completion" in r) for c in clients for r in c)
    assert all("domain" in r for r in test) and {r["domain"] for r in test} == set(ORDER)

    # noisy client 7: answer_swap keeps the prompt set, breaks prompt->completion pairing
    clean, _, _ = build_crossdevice(N, alpha=0.5, per_client_train=PC, per_domain_pool=POOL,
                                    per_domain_val=PDV, per_domain_test=PDTEST, seed=0)
    p7n = {r["prompt"] for r in clients[7]}
    assert p7n == {r["prompt"] for r in clean[7]}, "noisy changed the prompt set"
    pn = {(r["prompt"], r["completion"]) for r in clients[7]}
    pc = {(r["prompt"], r["completion"]) for r in clean[7]}
    assert pn != pc, "answer_swap did not change pairings"
    print(f"  noisy client 7: prompts preserved, {len(pn - pc)}/{PC} pairings swapped")
    print("  Part 2 OK\n\nCROSS-DEVICE DATA SMOKE OK")


if __name__ == "__main__":
    seed_everything(0)
    synthetic()
    integration()
