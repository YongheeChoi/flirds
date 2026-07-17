"""Per-phase wall-clock + GPU-hour + peak-memory accounting (protocol §15.1).

The §15.1 gap: runners time each valuation method (the `_timed` wrapper) but NOT
the FL trajectory generation (client-training), and nothing rolls the phases up
into GPU-hours / peak memory.  `PhaseTimer` closes both:

  pt = PhaseTimer(device, n_gpus=1)
  with pt.phase("client-training"):        # the previously-untimed log generation
      logs = build_trajectory(...)
  for name, _, _, t in methods:            # per-method times already measured by _timed
      pt.record(f"valuation:{name}", t)
  logger.save_timing(pt.to_timing())       # -> timing.json

`phase(name)` GPU-syncs on both ends (real wall-clock of async CUDA kernels) and
records peak GPU memory via torch.cuda.max_memory_allocated (reset at phase entry).
`record(name, s)` folds in a phase already timed elsewhere (e.g. `_timed`).
`to_timing()` emits the §15.1 dict: per-phase {s, peak_gib}, total_s,
gpu_hours = Σ_phase(s × n_gpus)/3600, peak_gib.  No new dependency, no framework
(codes/CLAUDE.md: no speculative abstraction) -- a thin timer + a dict.
"""
from __future__ import annotations

import time

import torch


class PhaseTimer:
    """Accumulate named phases' wall-clock + peak GPU memory for one run."""

    def __init__(self, device, n_gpus=1):
        self.device = device
        self.n_gpus = n_gpus
        self.phases = {}                                   # name -> {"s", "peak_gib"}
        self._cuda = str(device).startswith("cuda") and torch.cuda.is_available()

    def _sync(self):
        if self._cuda:
            torch.cuda.synchronize()

    def phase(self, name):
        """Context manager: GPU-synced wall-clock + peak mem for the enclosed block."""
        return _Phase(self, name)

    def record(self, name, seconds, peak_gib=None):
        """Fold in a phase whose seconds were measured elsewhere (e.g. `_timed`).

        Additive per name (repeated method calls accumulate); peak_gib takes the max."""
        p = self.phases.setdefault(name, {"s": 0.0, "peak_gib": 0.0})
        p["s"] += float(seconds)
        if peak_gib is not None:
            p["peak_gib"] = max(p["peak_gib"], float(peak_gib))

    def to_timing(self):
        """The §15.1 timing.json dict (per-phase + GPU-hours + peak memory)."""
        total_s = sum(p["s"] for p in self.phases.values())
        gpu_hours = sum(p["s"] * self.n_gpus for p in self.phases.values()) / 3600.0
        peak = max((p["peak_gib"] for p in self.phases.values()), default=0.0)
        return {
            "phases": {k: {"s": round(v["s"], 3), "peak_gib": round(v["peak_gib"], 3)}
                       for k, v in self.phases.items()},
            "total_s": round(total_s, 3),
            "gpu_hours": round(gpu_hours, 6),
            "n_gpus": self.n_gpus,
            "peak_gib": round(peak, 3),
        }


class _Phase:
    def __init__(self, timer, name):
        self.t = timer
        self.name = name

    def __enter__(self):
        self.t._sync()
        if self.t._cuda:
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.t._sync()
        dt = time.perf_counter() - self.t0
        peak = torch.cuda.max_memory_allocated() / 1e9 if self.t._cuda else 0.0
        self.t.record(self.name, dt, peak)
        return False
