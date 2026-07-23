"""L8/T5 retrain-(a) suite unit tests -- the pure pieces (no GPU, no network):
  1. build_gsm8k_iid per_client subsample (149/client math + determinism), via a
     monkeypatched load_dataset so it needs no download.
  2. phase2_matrix.report_vs_a (vs-(a) fidelity: keys, (a)/det exclusion, values).
  3. the (a) retrain-game orientation + null-player convention, on a synthetic
     additive utility through exact_shapley (the frzero (a)-side null check, T5 §2).

The model/data WIRING (subset_valloss_utility retrain, gsm5 dual-oracle end to end)
is covered by the tiny-model smokes -- not unit-testable offline.

From codes/:  PYTHONPATH=. python tests/test_gsm5_a.py
"""
import random

import numpy as np
from scipy.stats import spearmanr

import flirds.data.llm as dl
from flirds.oracle.exact_sv import exact_shapley


# --------------------------------------------------------------------------- #
# 1. build_gsm8k_iid per_client subsample                                     #
# --------------------------------------------------------------------------- #
class _FakeDS:
    """Minimal HF-Dataset stand-in: shuffle(seed) + to_list, deterministic."""
    def __init__(self, rows):
        self.rows = rows

    def shuffle(self, seed=0):
        r = self.rows[:]
        random.Random(seed).shuffle(r)
        return _FakeDS(r)

    def to_list(self):
        return self.rows


def _install_fake_gsm8k(monkey_rows_train=400, monkey_rows_test=20):
    """Patch dl.load_dataset/rev to return synthetic gsm8k rows (no download)."""
    tr = [{"question": f"q{i}?", "answer": f"reason {i}\n#### {i}"} for i in range(monkey_rows_train)]
    te = [{"question": f"tq{i}?", "answer": f"r\n#### {i}"} for i in range(monkey_rows_test)]

    def fake_load(_id, _cfg=None, split=None, revision=None):
        return _FakeDS(tr if split == "train" else te)

    dl.load_dataset = fake_load           # module-local rebind (restored below)
    dl.rev = lambda _id: "main"


def test_gsm8k_per_client_subsample():
    orig_load, orig_rev = dl.load_dataset, dl.rev
    try:
        _install_fake_gsm8k(monkey_rows_train=400)
        # per_client=149 at N=5 -> exactly 149/client, 745 of 400? 400<745 -> clamps to
        # len//5; use a pool >= 745 to exercise the real drop.  Re-install with 800 rows.
        _install_fake_gsm8k(monkey_rows_train=800)
        clients, val, test = dl.build_gsm8k_iid(5, n_val=10, n_test=0, seed=0, per_client=149)
        sizes = [len(c) for c in clients]
        assert sizes == [149] * 5, f"per_client=149 -> {sizes}"            # 745 of 800 used
        assert len(val) == 10
        # determinism: same seed -> identical client 0 records
        clients2, _, _ = dl.build_gsm8k_iid(5, n_val=10, n_test=0, seed=0, per_client=149)
        assert clients[0].to_list() == clients2[0].to_list()
        # different seed -> different shuffle (so different partition)
        clients3, _, _ = dl.build_gsm8k_iid(5, n_val=10, n_test=0, seed=1, per_client=149)
        assert clients[0].to_list() != clients3[0].to_list()
        # per_client=None -> uses the whole pool (800 // 5 = 160/client)
        cfull, _, _ = dl.build_gsm8k_iid(5, n_val=10, n_test=0, seed=0, per_client=None)
        assert [len(c) for c in cfull] == [160] * 5
    finally:
        dl.load_dataset, dl.rev = orig_load, orig_rev


# --------------------------------------------------------------------------- #
# 2. report_vs_a                                                              #
# --------------------------------------------------------------------------- #
def test_report_vs_a():
    from experiments.phase2_matrix import report_vs_a
    a = np.array([0.5, -0.2, 0.1, -0.4])                  # (a)oracle suspicion vector
    flirds = a + 0.01 * np.array([1, -1, 1, -1])          # near-perfect tracker
    noise = np.array([-0.4, 0.1, -0.2, 0.5])              # anti-correlated
    methods = [("(b)oracle", "val", a * 0.9, 0.0),        # (b) tracks (a) here
               ("Flirds", "val", flirds, 0.0),
               ("noise", "val", noise, 0.0),
               ("FLDetector", "det", np.array([1.0, 2, 3, 4]), 0.0),   # det -> excluded
               ("(a)oracle", "val", a, 0.0)]
    out = report_vs_a(methods, [0, 1, 2, 3])
    assert set(out) == {"spearman_a", "pearson_a"}
    assert "(a)oracle" not in out["spearman_a"]           # (a) is the truth, not scored vs itself
    assert "FLDetector" not in out["spearman_a"]          # detectors are not val-fidelity
    assert "(b)oracle" in out["spearman_a"]               # the two-oracle agreement IS reported
    assert out["spearman_a"]["Flirds"] > 0.99             # near-perfect rank tracker
    assert out["spearman_a"]["noise"] < -0.9              # anti-correlated
    # matches a direct spearman
    exp = float(spearmanr(flirds, a).correlation)
    assert abs(out["spearman_a"]["Flirds"] - exp) < 1e-9
    # empty when (a) absent
    assert report_vs_a([("Flirds", "val", flirds, 0.0)], [0, 1, 2, 3]) == {}


# --------------------------------------------------------------------------- #
# 3. (a) retrain-game orientation + null player (frzero (a)-side check)        #
# --------------------------------------------------------------------------- #
def test_a_game_orientation_and_null_player():
    # Synthetic additive val-loss game: val_loss(S) = L0 + sum_{i in S} d_i, so the
    # (a) utility U(S) = -val_loss(S).  d<0 = helpful (lowers loss), d=0 = null player
    # (a zero-delta free-rider -- the frzero (a)-side case), d>0 = harmful.
    L0 = 10.0
    d = {0: -2.0, 1: 0.0, 2: 3.0}                         # helpful / null / harmful

    def utility(S):                                        # exact_shapley passes sorted tuples
        return -(L0 + sum(d[i] for i in S))

    phi_a = exact_shapley(3, utility)                     # good->HIGH (utility = -val_loss)
    # additive game -> Shapley value = the client's own marginal = -d_i
    assert np.allclose(phi_a, [2.0, 0.0, -3.0], atol=1e-9)
    assert abs(phi_a[1]) < 1e-9                           # NULL PLAYER: zero-delta -> exactly 0 (T5 §2)
    assert phi_a[0] > 0 and phi_a[2] < 0                  # helpful high, harmful low
    # the runner stores suspicion = -phi_a (good->low), matching the repo/canonical convention
    susp = -phi_a
    assert susp[2] == max(susp) and susp[0] == min(susp)  # harmful = most suspicious, helpful = least


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASS")
