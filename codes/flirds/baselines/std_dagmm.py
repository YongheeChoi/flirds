"""STD-DAGMM (Lin et al., 2019) -- server-side from-logs free-rider detector.

Reference: Jierui Lin, Min Du, Jian Liu, "Free-riders in Federated Learning:
Attacks and Defenses" (arXiv:1911.12560); defense = DAGMM (Zong et al., ICLR 2018)
augmented with the standard deviation of the flattened update.  Reference-guided
self-build on our `logs` contract: a DETECTION score (the GMM energy), NOT a
valuation phi -> AUROC table only (no Spearman / no marginal-contribution value).

Like FLDetector this is MODEL-FREE -- it needs ONLY the logged update vectors (no
loss_fn / model / test_loader).  Unlike FLDetector it trains its OWN small
autoencoder + GMM on the pooled updates: "model-free" means free of the FL model,
NOT free of a learned anomaly model (DAGMM is one).

DAGMM.  A deep AE compresses each update to a low-dim embedding z_c plus two
reconstruction-distance features (relative-euclidean + cosine); an estimation net
maps z = [z_c, recon, std] to soft GMM memberships; the per-sample energy
    E(z) = -log Σ_k φ_k N(z; μ_k, Σ_k)
is the anomaly score (high = anomalous).  STD augmentation: the std of the FULL
flattened update is stacked into z before the GMM -- free-riders scaled toward
zero, or that recycle an average of others' updates, have anomalously LOW std,
which the recon/cosine terms alone miss; conversely a random fake tuned to the
benign std evades std alone but reconstructs poorly, so the recon/cosine terms
catch IT.  Combining the two generalizes across learning rates (the paper's
headline result).

Adaptations for the LLM-FL from-logs setting:
  - SAMPLES = per-(client, round) POOLED updates (Σ_r |cohort_r| samples); the
    client score is the MEAN energy over the rounds it participated.  Per-client-
    mean (one averaged vector per client) is degenerate at N=5 -- only N points to
    fit a GMM; pooling gives N*R, and it absorbs partial participation for free (a
    cross-device client simply contributes one sample per round it is sampled).
  - DIM REDUCTION = signed FEATURE-HASHING random projection of the ~5.6M-dim LoRA
    update down to D~256 before the AE (the raw update is far too wide for a dense
    AE).  Feature hashing preserves inner products in expectation, so benign
    structure survives the projection while random fakes stay random.  The std
    feature is taken on the FULL update (reduction-independent) -- the magnitude
    signal is never lost to the projection.
  - UNSEEN clients (never sampled) get the minimum score: a free-rider must
    participate to reap rewards, so "never seen" is evidence of benign, not anomaly.

Cross-silo (full participation) and cross-device (K-of-N) both fall straight out
of the pooling.  Input is the standard logs[(w_r, deltas_map)] with
deltas_map[c] = (delta, n_c); output is score[n_clients], orientation-matched to
eval.metrics.detection_auroc (corrupt clients score HIGH).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ripple import _flat


# --------------------------------------------------------------------------- #
# feature-hashing random projection (deterministic given P, D, seed)          #
# --------------------------------------------------------------------------- #
def _hash_proj(P, D, seed, device):
    """Signed feature-hashing buckets/signs for a P-dim input -> D-dim output.

    Each input coordinate j maps to ONE bucket h(j) in [0,D) with a random sign
    s(j) in {-1,+1}; proj[h(j)] += s(j) * x[j].  Preserves inner products in
    expectation (Weinberger et al. 2009).  Computed once per (P, D, seed)."""
    g = torch.Generator().manual_seed(seed)
    buckets = torch.randint(0, D, (P,), generator=g).to(device)
    signs = (torch.randint(0, 2, (P,), generator=g).float() * 2 - 1).to(device)
    return buckets, signs


def _collect_samples(logs, proj_dim, seed, device):
    """Pool per-(client, round) updates -> (X[M, proj_dim], std[M], client_idx[M]).

    X = feature-hashed projections; std = std of the FULL flattened update (taken
    before projection); client_idx[i] = which client sample i belongs to."""
    keys = sorted(next(iter(logs[0][1].values()))[0].keys())
    first = next(iter(logs[0][1].values()))[0]
    P = sum(int(first[k].numel()) for k in keys)
    buckets, signs = _hash_proj(P, proj_dim, seed, device)

    X, std, client_idx = [], [], []
    for _, dm in logs:
        for c in dm:
            vec = _flat(dm[c][0], keys).to(device).float()
            std.append(float(vec.std()))
            X.append(torch.zeros(proj_dim, device=device).index_add_(0, buckets, signs * vec))
            client_idx.append(c)
    return torch.stack(X), torch.tensor(std, device=device), client_idx


# --------------------------------------------------------------------------- #
# DAGMM networks                                                              #
# --------------------------------------------------------------------------- #
class _AE(nn.Module):
    """Symmetric tanh autoencoder; forward -> (reconstruction, latent z_c)."""

    def __init__(self, d_in, hidden, latent):
        super().__init__()
        dims = [d_in, *hidden, latent]
        enc = [layer for a, b in zip(dims, dims[1:]) for layer in (nn.Linear(a, b), nn.Tanh())]
        dec_dims = [latent, *hidden[::-1], d_in]
        dec = [layer for a, b in zip(dec_dims, dec_dims[1:]) for layer in (nn.Linear(a, b), nn.Tanh())]
        self.enc = nn.Sequential(*enc[:-1])   # linear latent (no activation)
        self.dec = nn.Sequential(*dec[:-1])   # linear output (no activation)

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


class _Estim(nn.Module):
    """Estimation network: z -> softmax GMM membership (K components)."""

    def __init__(self, d_in, hidden, k, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(),
                                 nn.Dropout(dropout), nn.Linear(hidden, k))

    def forward(self, z):
        return torch.softmax(self.net(z), dim=1)


def _augment(x, x_hat, z_c, std):
    """z = [z_c, relative-euclidean, cosine, std]  (the GMM input)."""
    rel = (x - x_hat).norm(dim=1) / (x.norm(dim=1) + 1e-12)
    cos = F.cosine_similarity(x, x_hat, dim=1)
    return torch.cat([z_c, rel[:, None], cos[:, None], std[:, None]], dim=1)


def _gmm_params(z, gamma):
    """Batch GMM params from soft memberships: phi[K], mu[K,D], cov[K,D,D]."""
    sg = gamma.sum(0).clamp_min(1e-6)                       # (K,)
    phi = sg / z.shape[0]
    mu = (gamma.t() @ z) / sg[:, None]                      # (K, D)
    zc = z[:, None, :] - mu[None, :, :]                     # (M, K, D)
    cov = torch.einsum("mk,mkd,mke->kde", gamma, zc, zc) / sg[:, None, None]
    return phi, mu, cov


def _energy(z, phi, mu, cov, eps=1e-6):
    """Per-sample GMM energy E(z)=-log Σ_k φ_k N(z;μ_k,Σ_k) and the cov-diag penalty.

    Numerically stable: a Cholesky factor gives both the Mahalanobis solve and the
    log-determinant; logsumexp over components.  eps regularizes Σ_k (PD + the
    DAGMM 1/diag penalty that keeps it away from singular)."""
    M, D = z.shape
    zc = z[:, None, :] - mu[None, :, :]                     # (M, K, D)
    cov_reg = cov + eps * torch.eye(D, device=z.device)[None]
    L = torch.linalg.cholesky(cov_reg)                     # (K, D, D)
    sol = torch.cholesky_solve(zc.permute(1, 2, 0), L)     # (K, D, M) = Σ^{-1} (z-μ)
    maha = torch.einsum("mkd,kdm->mk", zc, sol)            # (M, K)
    logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)   # (K,)
    log_prob = (torch.log(phi)[None, :]
                - 0.5 * (D * math.log(2 * math.pi) + logdet[None, :])
                - 0.5 * maha)                              # (M, K)
    energy = -torch.logsumexp(log_prob, dim=1)             # (M,)
    cov_diag = (1.0 / torch.diagonal(cov_reg, dim1=-2, dim2=-1)).sum()
    return energy, cov_diag


# --------------------------------------------------------------------------- #
# entry point                                                                #
# --------------------------------------------------------------------------- #
def std_dagmm_from_logs(logs, n_clients, proj_dim=256, latent_dim=4,
                        ae_hidden=(64, 16), est_hidden=16, n_gmm=2,
                        epochs=200, lr=1e-3, lambda_energy=0.1, lambda_cov=0.005,
                        dropout=0.3, seed=0, device="cpu"):
    """STD-DAGMM suspicious score per client over a frozen FedAvg trajectory.

    Returns score[n_clients] (higher = more anomalous = more suspicious), the mean
    GMM energy over each client's per-round updates.  Model-free (no loss_fn /
    model).  Trains a small AE+GMM (seeded; the global RNG is snapshotted and
    restored so the detector never perturbs a downstream method's randomness)."""
    X, std, client_idx = _collect_samples(logs, proj_dim, seed, device)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)                 # per-dim standardize
    std = (std - std.mean()) / (std.std() + 1e-8)

    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    ae = _AE(proj_dim, list(ae_hidden), latent_dim).to(device)
    est = _Estim(latent_dim + 3, est_hidden, n_gmm, dropout).to(device)
    opt = torch.optim.Adam([*ae.parameters(), *est.parameters()], lr=lr)
    for _ in range(epochs):                                # full-batch (M is small)
        ae.train(); est.train()
        x_hat, z_c = ae(X)
        z = _augment(X, x_hat, z_c, std)
        energy, cov_diag = _energy(z, *_gmm_params(z, est(z)))
        loss = F.mse_loss(x_hat, X) + lambda_energy * energy.mean() + lambda_cov * cov_diag
        opt.zero_grad(); loss.backward(); opt.step()

    ae.eval(); est.eval()
    with torch.no_grad():
        x_hat, z_c = ae(X)
        z = _augment(X, x_hat, z_c, std)
        energy, _ = _energy(z, *_gmm_params(z, est(z)))
    torch.set_rng_state(rng_state)

    e = energy.cpu().numpy()
    idx = np.array(client_idx)
    score = np.full(n_clients, np.nan)
    for c in range(n_clients):
        if (idx == c).any():
            score[c] = e[idx == c].mean()
    unseen = np.isnan(score)
    if unseen.any():                                       # never sampled -> least suspicious
        score[unseen] = np.nanmin(score)
    return score
