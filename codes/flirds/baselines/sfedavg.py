"""S-FedAvg / Game of Gradients (Nagalapatti & Narayanam, AAAI 2021) -- the
client-SELECTION baseline for Track C2 (plan §3.11 decision ②, 2026-06-12).

Unlike the from-logs valuers, S-FedAvg is INHERENTLY ONLINE: its relevance
vector phi steers which clients are sampled each round, so it must run inside the
FL loop via the `fl.server._fedavg_core` intervention seam.  This module provides
a stateful `SFedAvgSelector` exposing the `select_fn` / `weights_fn` hooks.

Faithful to Algorithm 1 + 2 (page 9049):
  - relevance phi init uniformly 1/K (so round-1 selection is ~uniform).
  - select m clients ~ softmax(phi) WITHOUT replacement (plain softmax, no temp).
  - per round, over the SELECTED cohort S^t, Monte-Carlo Shapley (Alg.2):
    R=10 random permutations; marginal a_i = v(pred∪{i}) - v(pred); sv_i = mean
    over R.  utility v(X) = ACCURACY on D_V of the UNIFORM-average submodel
    theta^t + (1/|X|) Σ_{s∈X} Δ_s (Eq 3-4; native S-FedAvg metric, NOT n-weighted
    and NOT loss).  R=10 < m! is the paper's deliberate robustness choice.
  - phi_k <- alpha*phi_k + beta*sv_k for k∈S^t (Eq 5; alpha=0.75, beta=0.25),
    others carried forward.
  - real aggregation = UNIFORM mean over the selected cohort (Eq 6; not n-weighted).

Our adaptations (documented deviations): momentum=0 (project lock, vs the paper's
plain SGD which is already momentum-0) and -- since C2 is N=100, C=0.1 -- m =
round(C*N)=10 selected of K=100 (the paper's m=5/K=10).  Reuses
`shapleyfl._uniform_submodel_cnn` for the submodel and `fl.server.evaluate` for
accuracy, so the submodel convention is shared with the ShapleyFL baseline.
"""
from __future__ import annotations

import numpy as np

from ..fl.server import evaluate
from .shapleyfl import _uniform_submodel_cnn

ALPHA, BETA, R_PERM = 0.75, 0.25, 10           # Eq 5 + Alg 2 paper constants


class SFedAvgSelector:
    """Stateful S-FedAvg relevance vector + intervention hooks (CNN accuracy
    utility).  `model` / `val_loader` are the validation D_V the per-round MC
    Shapley scores against."""

    def __init__(self, n_clients, model, val_loader, device, seed=0):
        self.phi = np.full(n_clients, 1.0 / n_clients)   # init 1/K (Alg 1)
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.rng = np.random.default_rng(seed)           # MC permutations (loop rng is separate)

    def select_fn(self, r, k, rng):
        """softmax(phi) sample of k clients without replacement (Alg 1)."""
        z = self.phi - self.phi.max()
        p = np.exp(z)
        p /= p.sum()
        return rng.choice(len(self.phi), size=k, replace=False, p=p)

    def _mc_shapley(self, gb, dm, players):
        """Monte-Carlo Shapley (Alg 2) over the cohort `players` on val accuracy."""
        m = len(players)
        cache = {}

        def v(sub):                                      # sub = tuple of positions into players
            key = tuple(sorted(sub))
            if key not in cache:
                subset = [players[i] for i in key]
                st = _uniform_submodel_cnn(gb, dm, subset, self.device)
                cache[key] = evaluate(self.model, st, self.val_loader, self.device)
            return cache[key]

        sv = np.zeros(m)
        for _ in range(R_PERM):
            perm = self.rng.permutation(m)
            pred, v_pred = [], v(())
            for i in perm:
                v_new = v(pred + [i])
                sv[i] += (v_new - v_pred) / R_PERM
                pred.append(i)
                v_pred = v_new
        return sv

    def weights_fn(self, r, w_r, deltas_map):
        """Update phi over the selected cohort (Eq 5), return UNIFORM aggregation
        weights (Eq 6).  The MC Shapley reads the round-start state w_r."""
        players = sorted(deltas_map)
        sv = self._mc_shapley(w_r, deltas_map, players)
        for i, p in enumerate(players):
            self.phi[p] = ALPHA * self.phi[p] + BETA * sv[i]
        return {p: 1.0 / len(players) for p in players}
