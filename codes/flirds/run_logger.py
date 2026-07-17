"""Local run-dir logger (protocol §6 / D2 -- NO W&B / mlflow).

Each run gets a directory holding config.yaml + meta.json (git SHA + dirty flag +
env-fingerprint hash + key package versions) + a per-client phi parquet + a metrics
json, so every reported number is linkable to a (config, env, git SHA, run dir):
PR review on a paper claim = "for claim X, show me the run dir."

`save_phi` takes a caller-shaped list of records (e.g. {round, client, phi, ...} or
flat per-client, or per-layer components from the seam-1 estimator) and parquets it
for post-hoc analysis without re-running -- the logger does not dictate the schema.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pandas as pd
import yaml

_KEY_PKGS = ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "numpy")


def _git(args, cwd):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _env_fingerprint():
    """(sha256 of `pip freeze`, key package versions) -- the reproducibility anchor."""
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, timeout=60).stdout
    except Exception:
        freeze = ""
    versions = {}
    for pkg in _KEY_PKGS:
        try:
            versions[pkg] = __import__(pkg).__version__
        except Exception:
            versions[pkg] = None
    versions["python"] = sys.version.split()[0]
    return hashlib.sha256(freeze.encode()).hexdigest()[:16], versions


class RunLogger:
    """Create `root/name/` and persist config + reproducibility meta + phi + metrics.

    `name` is the run identity (caller makes it unique, e.g. includes seed); `repo_root`
    is where the git SHA is read from (default cwd, i.e. run from codes/).
    """

    def __init__(self, root, name, config, repo_root="."):
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir, exist_ok=True)
        with open(self._p("config.yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        env_hash, versions = _env_fingerprint()
        self.meta = {
            "name": name,
            "git_sha": _git(["rev-parse", "HEAD"], repo_root),
            "git_dirty": bool(_git(["status", "--porcelain"], repo_root)),
            "env_hash": env_hash,
            "versions": versions,
        }
        with open(self._p("meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    def _p(self, fname):
        return os.path.join(self.dir, fname)

    def save_phi(self, records, fname="phi.parquet"):
        """Persist per-client (optionally per-round / per-layer) phi records to parquet."""
        pd.DataFrame(records).to_parquet(self._p(fname))
        return self._p(fname)

    def save_metrics(self, metrics, fname="metrics.json"):
        """Persist the run's headline metrics (selection-convergence / task-acc / AUROC)."""
        with open(self._p(fname), "w") as f:
            json.dump(metrics, f, indent=2)
        return self._p(fname)

    def save_timing(self, timing, fname="timing.json"):
        """Persist per-phase wall-clock + GPU-hours + peak GPU memory (protocol §15.1).

        `timing` is a flirds.timing.PhaseTimer.to_timing() dict (client-training /
        phi-estimation / oracle / eval phases + gpu_hours + peak_gib)."""
        with open(self._p(fname), "w") as f:
            json.dump(timing, f, indent=2)
        return self._p(fname)
