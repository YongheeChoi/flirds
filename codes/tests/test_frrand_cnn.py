"""CNN `frrand` threat (pure random update) unit tests -- 2026-07-22 port of the
LLM leg's frrand into track_c2's update-level threat set.

Top risks: (1) the honest signal must be GONE (not merely perturbed -- that is
grad_noise), (2) the fabricated update must be norm-indistinguishable from the
benign one (Lin et al. scale tuning; otherwise any norm filter trivially wins and
the gate result is uninteresting), (3) honest clients and the two legacy threats
must be bit-unchanged.

No GPU, no dataset.  From codes/:  PYTHONPATH=. python tests/test_frrand_cnn.py
"""
import torch

from flirds.fl.intervene import _benign_matched_scale, make_delta_transform


def _delta(seed=0, scale=0.03):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(400, 20, generator=g) * scale,
            "b": torch.randn(50, generator=g) * scale}


def _flat(d):
    return torch.cat([t.reshape(-1) for t in d.values()])


def test_frrand_destroys_the_signal():
    tf = make_delta_transform([1], "frrand", std=1.0, seed=0)
    honest = _delta()
    out = tf(1, 0, honest)
    a, b = _flat(honest), _flat(out)
    assert out.keys() == honest.keys()
    assert all(out[k].shape == honest[k].shape for k in out)
    # no alignment left with the honest direction (grad_noise would keep ~1.0)
    cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
    assert abs(cos) < 0.05, cos
    # ... and grad_noise at the SAME magnitude does keep it -- the contrast that
    # motivates the new threat (signal+noise vs noise-only)
    gn = make_delta_transform([1], "grad_noise", std=float(a.std()), seed=0)(1, 0, honest)
    assert float(torch.dot(a, _flat(gn)) / (a.norm() * _flat(gn).norm())) > 0.5


def test_frrand_is_norm_matched_to_the_benign_update():
    honest = _delta()
    out = make_delta_transform([1], "frrand", std=1.0, seed=0)(1, 0, honest)
    r = float(_flat(out).std() / _flat(honest).std())
    assert 0.9 < r < 1.1, r                       # per-entry std matched within 10%
    # sqrt(3) conversion: U(-s,s) with s = sqrt(3)*sigma has std sigma
    assert abs(_benign_matched_scale(honest) / float(_flat(honest).std()) - 3 ** 0.5) < 1e-5
    # the amplitude multiplier scales it linearly (the LLM leg's DOSE_MULT knob)
    out2 = make_delta_transform([1], "frrand", std=2.0, seed=0)(1, 0, honest)
    assert abs(float(_flat(out2).std() / _flat(out).std()) - 2.0) < 0.05


def test_honest_pass_through_and_reproducibility():
    tf = make_delta_transform([1, 3], "frrand", std=1.0, seed=7)
    honest = _delta(seed=5)
    assert tf(0, 0, honest) is honest                      # untouched, same object
    a = _flat(tf(1, 4, honest))
    assert torch.equal(a, _flat(make_delta_transform([1, 3], "frrand", std=1.0,
                                                     seed=7)(1, 4, honest)))
    assert not torch.equal(a, _flat(tf(1, 5, honest)))     # round-dependent
    assert not torch.equal(a, _flat(tf(3, 4, honest)))     # client-dependent


def test_legacy_threats_unchanged():
    honest = _delta()
    z = make_delta_transform([1], "free_rider", seed=0)(1, 0, honest)
    assert float(_flat(z).abs().sum()) == 0.0
    g = torch.Generator().manual_seed(0 + 1000 * 1 + 2)     # the documented per-(c,r) seed
    want = {k: v + 0.1 * torch.randn(v.shape, dtype=v.dtype, generator=g)
            for k, v in honest.items()}
    got = make_delta_transform([1], "grad_noise", std=0.1, seed=0)(1, 2, honest)
    assert all(torch.equal(got[k], want[k]) for k in want)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASS")
