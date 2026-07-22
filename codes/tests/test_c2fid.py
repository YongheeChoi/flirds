"""Track C2-FID unit tests (2026-07-23).

Top risks: (1) the per-round-chunked (b) oracle must equal the 2^N enumeration
(the fidelity ground truth -- any drift silently corrupts every correlation),
(2) round sharding must merge back to the exact full value with complete
coverage, (3) the rate-capture wrapper must be BIT-NEUTRAL (the fidelity stage
must realize the same corruption as its downstream twin), (4) the logged
trajectory must be bit-identical to the unlogged one (the §4.9 join property).

Synthetic tests need no data/GPU; the last two build the real fmnist smoke
stage (CPU, ~1 min).  From codes/:  PYTHONPATH=. python tests/test_c2fid.py
"""
import os

os.environ["C2_DATASET"] = "fmnist"                # before the track_c2 import
os.environ["C2_MODE"] = "smoke"
os.environ["C2_THREAT"] = "label_flip"             # FedCorr rate~U(0.5,1) = strmain
os.environ["C2_SEED"] = "0"
os.environ["C2_FRAC"] = "0.2"                      # k=4 -> 2^4 oracle (CPU-fast)

import numpy as np
import torch

import experiments.track_c2 as c2
import experiments.track_c2_fid as fid
from flirds.data.corruptors import CNN_CORRUPTORS
from flirds.fl.server import fedavg
from flirds.oracle.in_run_sv import (in_run_shapley, in_run_shapley_perround,
                                     in_run_utility)


def _synth(n_clients=4, rounds=4, dim=6):
    """Partial-participation logs over a quadratic game (no data, no GPU)."""
    torch.manual_seed(0)
    tgt = torch.randn(dim)

    def loss_fn(params, buffers):
        return ((params["w"] - tgt) ** 2).sum()

    rng = np.random.default_rng(7)
    logs = []
    for _ in range(rounds):
        w_r = {"w": torch.randn(dim)}
        players = rng.choice(n_clients, size=int(rng.integers(2, n_clients)),
                             replace=False)
        dm = {int(c): ({"w": torch.randn(dim) * 0.3}, int(rng.integers(50, 200)))
              for c in players}
        logs.append((w_r, dm))
    return logs, loss_fn, ["w"]


def test_chunked_oracle_equals_2N_enumeration():
    logs, loss_fn, pkeys = _synth()
    phi_chunk, rows = fid.oracle_b_rounds(logs, 4, loss_fn, pkeys, "cpu")
    phi_full, _ = in_run_shapley(logs, 4, loss_fn, pkeys, "cpu")        # 2^N
    phi_pr, _ = in_run_shapley_perround(logs, 4, loss_fn, pkeys, "cpu")  # one call
    assert np.allclose(phi_chunk, phi_full, atol=1e-4), (phi_chunk, phi_full)
    assert np.allclose(phi_chunk, phi_pr, atol=1e-5)
    # per-round rows: one per (round, participant), reconstructing phi by groupby-sum
    by_client = {}
    for x in rows:
        by_client[x["client"]] = by_client.get(x["client"], 0.0) + x["phi_b"]
    assert np.allclose([by_client.get(i, 0.0) for i in range(4)], phi_chunk, atol=1e-6)


def test_round_shard_merge_and_coverage():
    logs, loss_fn, pkeys = _synth()
    full, rows_f = fid.oracle_b_rounds(logs, 4, loss_fn, pkeys, "cpu")
    a, rows_a = fid.oracle_b_rounds(logs, 4, loss_fn, pkeys, "cpu", rounds=range(0, 2))
    b, rows_b = fid.oracle_b_rounds(logs, 4, loss_fn, pkeys, "cpu", rounds=range(2, 4))
    assert np.allclose(a + b, full, atol=1e-6)         # merge = plain sum
    cov = sorted({x["round"] for x in rows_a} | {x["round"] for x in rows_b})
    assert cov == sorted({x["round"] for x in rows_f}) == [0, 1, 2, 3]  # exact coverage
    assert not ({x["round"] for x in rows_a} & {x["round"] for x in rows_b})  # disjoint


def test_efficiency_vs_independent_grand_utility():
    logs, loss_fn, pkeys = _synth()
    phi, _ = fid.oracle_b_rounds(logs, 4, loss_fn, pkeys, "cpu")
    u = in_run_utility(logs, (0, 1, 2, 3), loss_fn, pkeys, "cpu")
    assert abs(float(phi.sum()) - u) < 1e-4, (phi.sum(), u)


def test_label_flip_rate_capture_is_bit_neutral():
    """build_with_rates() must realize EXACTLY the plain build() (same corruption
    as the downstream twin), only adding the rate record."""
    l1, c1_, d1, vx1, vy1, _, _ = c2.build()
    (l2, c2_, d2, vx2, vy2, _, _), rates = fid.build_with_rates()
    assert CNN_CORRUPTORS["label_flip"].__name__ == "label_flip"   # wrapper restored
    assert np.array_equal(c1_, c2_) and d1 is None and d2 is None
    for a, b in zip(l1, l2):
        xa, ya = a.dataset.tensors
        xb, yb = b.dataset.tensors
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
    assert torch.equal(vx1, vx2) and torch.equal(vy1, vy2)
    n = c2.CFG["n"]
    assert len(rates) == n
    for i in range(n):                                 # FedCorr: corrupt -> U(0.5,1)
        if c2_[i]:
            assert 0.5 <= rates[i] <= 1.0, (i, rates[i])
        else:
            assert rates[i] == 0.0


def test_trajectory_join_logged_equals_unlogged():
    """on_round must only OBSERVE: the logged trajectory (fid) is bit-identical
    to the downstream twin's vanilla arm (fedavg re-seeds at entry -- §4.9)."""
    loaders, _, dtf, _, _, _, tl = c2.build()
    R, E, lr, frac = 2, c2.CFG["epochs"], c2.CFG["lr"], c2.CFG["frac"]
    logs = []
    s1, h1 = fedavg(c2.MODEL_FN, loaders, tl, R, E, lr, sample_frac=frac,
                    device="cpu", seed=0, delta_transform=dtf,
                    on_round=lambda r, gb, dm: logs.append((fid._cpu_state(gb),
                        {c: (fid._cpu_state(d), n) for c, (d, n) in dm.items()})))
    s2, h2 = fedavg(c2.MODEL_FN, loaders, tl, R, E, lr, sample_frac=frac,
                    device="cpu", seed=0, delta_transform=dtf)
    assert h1 == h2, (h1, h2)                          # eval history bit-equal
    assert all(torch.equal(s1[k], s2[k]) for k in s1)  # final state bit-equal
    assert len(logs) == R and all(len(dm) == max(1, round(frac * c2.CFG["n"]))
                                  for _, dm in logs)
    assert all(v.device.type == "cpu" for _, dm in logs
               for d, _ in dm.values() for v in d.values())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASS")
