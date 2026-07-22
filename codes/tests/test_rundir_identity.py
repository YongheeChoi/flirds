"""Rundir identity guard (protocol §1.7) -- plain asserts, seconds on CPU, no pytest.

The bug being fixed: RunLogger compared the WHOLE config as one string to decide "is
this the same run?".  `config` mixes identity (which experiment) with provenance (what
exactly ran), so the guard ran backwards -- it forked a phantom `<name>_<hash>` dir every
time a harmless provenance key was added (5 real cases in 2026-07), while staying silent
on beta 0.5 -> 0.3, a real semantic change that lived in a source literal and therefore
never reached `config` at all.
"""
import os
import shutil
import tempfile

import yaml

from flirds.run_logger import (RunDirIdentityError, RunLogger, _identity_diff,
                               check_identity)

IDENT = ("regime", "threat", "arm", "seed", "sfl_beta")
BASE = {"regime": "gsm50k5", "threat": "noisy", "arm": "observer", "seed": 0,
        "sfl_beta": 0.3, "rcfg": {"rounds": 200}}


def _seed_dir(root, name, config):
    os.makedirs(os.path.join(root, name), exist_ok=True)
    with open(os.path.join(root, name, "config.yaml"), "w") as f:
        f.write(yaml.safe_dump(config, sort_keys=False))


def test_provenance_growth_does_not_fork():
    """The 2026-07 false alarm: new self-describing keys must NOT change identity."""
    with tempfile.TemporaryDirectory() as root:
        _seed_dir(root, "cell", BASE)
        grown = dict(BASE, dose_mult=1.0, obs_sources=["flirds"], t2_csign=False)
        assert _identity_diff(BASE, grown, IDENT) == {}
        rl = RunLogger(root, "cell", grown, repo_root=root, identity=IDENT)
        assert rl.dir == os.path.join(root, "cell"), "re-run must reuse the canonical dir"
        assert not [d for d in os.listdir(root) if d.startswith("cell_")], "no phantom dir"


def test_identity_change_raises():
    """A different experiment under the same name = name-generation bug -> hard fail."""
    with tempfile.TemporaryDirectory() as root:
        _seed_dir(root, "cell", BASE)
        for field, value in [("seed", 1), ("threat", "frzero"), ("regime", "silo5")]:
            try:
                check_identity(root, "cell", dict(BASE, **{field: value}), IDENT)
            except RunDirIdentityError:
                continue
            raise AssertionError(f"identity change on {field!r} was not caught")


def test_beta_change_is_caught_even_when_the_old_config_lacks_the_key():
    """THE regression this exists for: a beta0.5-era rundir has no `sfl_beta` key at all,
    so an absent field must count as a mismatch -- otherwise beta0.3 silently overwrites
    the canonical run, which is exactly what happened before the fix."""
    with tempfile.TemporaryDirectory() as root:
        pre_schema = {k: v for k, v in BASE.items() if k != "sfl_beta"}
        _seed_dir(root, "cell", pre_schema)
        assert "sfl_beta" in _identity_diff(pre_schema, BASE, IDENT)
        try:
            check_identity(root, "cell", BASE, IDENT)
        except RunDirIdentityError:
            return
        raise AssertionError("absent identity field must not pass as a match")


def test_rundir_replace_env_overrides():
    with tempfile.TemporaryDirectory() as root:
        _seed_dir(root, "cell", BASE)
        os.environ["RUNDIR_REPLACE"] = "1"
        try:
            check_identity(root, "cell", dict(BASE, seed=1), IDENT)   # must not raise
        finally:
            del os.environ["RUNDIR_REPLACE"]


def test_precheck_makes_no_directory():
    """precheck runs BEFORE the compute (RunLogger is built at persist time, i.e. after a
    ~16 h arm), so it must be side-effect free."""
    with tempfile.TemporaryDirectory() as root:
        RunLogger.precheck(root, "absent", BASE, IDENT)               # no stored config
        assert os.listdir(root) == []


def test_legacy_whole_config_guard_unchanged():
    """Runners not yet migrated keep the old behaviour verbatim (identity=None)."""
    with tempfile.TemporaryDirectory() as root:
        _seed_dir(root, "cell", BASE)
        rl = RunLogger(root, "cell", dict(BASE, extra=1), repo_root=root)
        assert rl.dir != os.path.join(root, "cell"), "legacy guard still forks on any diff"
        shutil.rmtree(rl.dir)
