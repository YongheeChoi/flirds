"""Track H R4 (gsm50k5 accuracy stage) unit tests -- the pure pieces: the GSM8K
final-answer normalizer, the score_records exact-match branch, and the grad-noise
seam (spec runs/track_h/README.md §1.6).

No GPU, no dataset/network.  From codes/:  PYTHONPATH=. python tests/test_r4.py
"""
import torch

from flirds.eval.generate import score_records
from flirds.eval.metrics import gsm8k_answer
from flirds.fl.llm_server import _add_gnoise


# --------------------------------------------------------------------------- #
# 1. gsm8k_answer -- gold convention + generation fallbacks                    #
# --------------------------------------------------------------------------- #
def test_gsm8k_answer():
    assert gsm8k_answer("...so she has 9 left.\n#### 72") == "72"
    assert gsm8k_answer("#### 1,000") == "1000"                 # comma stripped
    assert gsm8k_answer("#### 1000.0") == "1000"                # trailing .0 dropped
    assert gsm8k_answer("#### -3") == "-3"
    assert gsm8k_answer("The answer is 42") == "42"             # no #### -> last number
    assert gsm8k_answer("5 apples then 3 pears, total 8") == "8"
    assert gsm8k_answer("no numbers here") is None
    assert gsm8k_answer("step 12 ... #### oops") == "12"        # malformed tail -> whole text
    assert gsm8k_answer("half is 0.5\n#### 0.5") == "0.5"       # non-integer kept


def test_score_records_gsm8k_em():
    recs = [{"completion": " reasoning\n#### 10", "domain": "gsm8k", "answer": "10"},
            {"completion": " r\n#### 7", "domain": "gsm8k", "answer": "7"},
            {"completion": " r\n#### 3", "domain": "gsm8k", "answer": "3"}]
    gens = ["I think ... #### 10", "the total is 7", "wrong: #### 4"]
    out = score_records(gens, recs)["gsm8k"]
    assert out["n"] == 3
    assert abs(out["exact_match"] - 2 / 3) < 1e-12              # last one wrong
    assert 0.0 <= out["rouge_l"] <= 1.0


# --------------------------------------------------------------------------- #
# 2. grad-noise seam -- RMS scale, determinism, zero-delta no-op              #
# --------------------------------------------------------------------------- #
def _delta(scale=1.0):
    torch.manual_seed(0)
    return {"a": torch.randn(50, 20) * scale, "b": torch.randn(30) * scale}


def test_gnoise_scale_and_determinism():
    d = _delta()
    numel = sum(v.numel() for v in d.values())
    rms = float(sum(v.pow(2).sum() for v in d.values()) / numel) ** 0.5
    n1 = _add_gnoise({k: v.clone() for k, v in d.items()}, 1.0,
                     torch.Generator().manual_seed(7))
    n2 = _add_gnoise({k: v.clone() for k, v in d.items()}, 1.0,
                     torch.Generator().manual_seed(7))
    for k in d:                                                 # same seed -> identical
        assert torch.equal(n1[k], n2[k])
    noise = torch.cat([(n1[k] - d[k]).flatten() for k in d])
    emp = float(noise.pow(2).mean()) ** 0.5
    assert abs(emp - rms) / rms < 0.15, f"noise RMS {emp} != gamma*deltaRMS {rms}"
    half = _add_gnoise({k: v.clone() for k, v in d.items()}, 0.5,
                       torch.Generator().manual_seed(7))
    noise_h = torch.cat([(half[k] - d[k]).flatten() for k in d])
    assert abs(float(noise_h.pow(2).mean()) ** 0.5 - 0.5 * rms) / rms < 0.15


def test_gnoise_zero_delta_noop():
    d = {"a": torch.zeros(10, 4), "b": torch.zeros(6)}          # free-rider composition
    out = _add_gnoise(d, 1.0, torch.Generator().manual_seed(0))
    for k in d:
        assert torch.equal(out[k], d[k])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASS")
