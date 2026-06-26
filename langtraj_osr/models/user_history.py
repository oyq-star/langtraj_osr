"""User history module for LangTraj-OSR.

Maintains per-user prototypes (weekday/weekend x time-of-day bins) fitted
via k-means and computes a Gaussian-mixture normality energy together with
deviation features that describe *how* a new trip departs from the user's
historical pattern.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserHistoryModule(nn.Module):
    """Score a trip embedding against a user's historical prototypes.

    Parameters
    ----------
    d_model : int
        Dimensionality of trip embeddings.
    n_prototypes : int
        Number of prototypes per user (default 8 = 2 day-types x 4 time bins).
    n_deviation_features : int
        Number of scalar deviation features produced.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_prototypes: int = 8,
        n_deviation_features: int = 5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_prototypes = n_prototypes
        self.n_deviation_features = n_deviation_features

        # Learnable projection for deviation feature extraction
        self.deviation_proj = nn.Sequential(
            nn.Linear(d_model + n_prototypes, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_deviation_features),
        )

        # Small epsilon for numerical stability
        self.register_buffer("_eps", torch.tensor(1e-8))

    # ------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def fit_user(
        normal_trip_embeddings: torch.Tensor,
        n_prototypes: int = 8,
        n_iter: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """Fit K prototypes to a user's normal trip embeddings via k-means.

        This is called **offline** during training/calibration and produces
        the prototype parameters that are later fed into :meth:`forward`.

        Parameters
        ----------
        normal_trip_embeddings : Tensor
            Shape ``(N, D)`` — embeddings of the user's known-normal trips.
        n_prototypes : int
            Number of cluster centres (K).
        n_iter : int
            Number of k-means iterations.

        Returns
        -------
        dict
            ``mu``    — cluster centres, shape ``(K, D)``.
            ``sigma`` — per-cluster diagonal covariance, shape ``(K, D)``.
            ``pi``    — mixture weights, shape ``(K,)``.
        """
        N, D = normal_trip_embeddings.shape
        device = normal_trip_embeddings.device

        # Handle edge case: fewer points than prototypes
        K = min(n_prototypes, N)

        # Initialise centroids with k-means++
        indices = [torch.randint(N, (1,), device=device).item()]
        for _ in range(1, K):
            dists = torch.cdist(
                normal_trip_embeddings,
                normal_trip_embeddings[indices],
            ).min(dim=1).values  # (N,)
            probs = dists / (dists.sum() + 1e-12)
            probs = probs.nan_to_num(nan=0.0).clamp(min=0.0)
            if probs.sum() < 1e-12:
                probs = torch.ones_like(probs)
            idx = torch.multinomial(probs, 1).item()
            indices.append(idx)

        centroids = normal_trip_embeddings[indices].clone()  # (K, D)

        # K-means iterations
        for _ in range(n_iter):
            dists = torch.cdist(normal_trip_embeddings, centroids)  # (N, K)
            assignments = dists.argmin(dim=1)                       # (N,)

            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(K, device=device)
            for k in range(K):
                member_mask = assignments == k
                count = member_mask.sum()
                if count > 0:
                    new_centroids[k] = normal_trip_embeddings[member_mask].mean(dim=0)
                    counts[k] = count.float()
                else:
                    new_centroids[k] = centroids[k]

            centroids = new_centroids

        # Compute diagonal covariance and mixture weights
        dists = torch.cdist(normal_trip_embeddings, centroids)
        assignments = dists.argmin(dim=1)

        sigma = torch.ones(K, D, device=device)
        pi = torch.ones(K, device=device) / K

        for k in range(K):
            member_mask = assignments == k
            count = member_mask.sum().float()
            if count > 1:
                members = normal_trip_embeddings[member_mask]       # (n_k, D)
                sigma[k] = (members - centroids[k]).pow(2).mean(dim=0).clamp(min=1e-4)
            pi[k] = count

        pi = pi / pi.sum().clamp(min=1e-12)

        # Pad to n_prototypes if K < n_prototypes
        if K < n_prototypes:
            pad = n_prototypes - K
            centroids = F.pad(centroids, (0, 0, 0, pad))            # (n_proto, D)
            sigma = F.pad(sigma, (0, 0, 0, pad), value=1.0)
            pi_pad = torch.zeros(n_prototypes, device=device)
            pi_pad[:K] = pi
            pi = pi_pad

        return {"mu": centroids, "sigma": sigma, "pi": pi}

    # ------------------------------------------------------------------
    def _gmm_log_likelihood(
        self,
        z_x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        pi: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GMM negative log-likelihood (normality energy).

        Parameters
        ----------
        z_x : Tensor   (B, D)
        mu : Tensor     (B, K, D)
        sigma : Tensor  (B, K, D)   diagonal covariance
        pi : Tensor     (B, K)      mixture weights

        Returns
        -------
        Tensor  (B,) — normality energy  E_norm = -log sum_j pi_j N(z_x; mu_j, sigma_j)
        """
        # Cast to float32 — GMM is numerically unstable in fp16 for D=256.
        z_x = z_x.float()
        mu = mu.float()
        sigma = sigma.float()
        pi = pi.float()

        D = z_x.shape[-1]
        z_expanded = z_x.unsqueeze(1)  # (B, 1, D)

        # Log-probability under each diagonal Gaussian — normalized by D so that
        # the scale is independent of embedding dimension.  Omitting the constant
        # D*log(2π)/D = log(2π) ≈ 1.84 has no effect on relative ordering.
        diff = z_expanded - mu                                              # (B, K, D)
        log_det_per_d = sigma.clamp(min=1e-4).log().mean(dim=-1)           # (B, K)
        mahal_per_d   = (diff.pow(2) / sigma.clamp(min=1e-4)).mean(dim=-1) # (B, K)
        log_norm = -0.5 * (log_det_per_d + mahal_per_d)                    # (B, K)

        # Log mixture probability
        log_pi = pi.clamp(min=1e-12).log()                                 # (B, K)
        log_mixture = torch.logsumexp(log_pi + log_norm, dim=1)            # (B,)

        return -log_mixture  # normality energy (lower = more normal)

    # ------------------------------------------------------------------
    def forward(
        self,
        z_x: torch.Tensor,
        user_prototypes: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute normality energy and deviation features.

        Parameters
        ----------
        z_x : Tensor
            Global trip embedding, shape ``(B, D)``.
        user_prototypes : dict
            Batched prototype parameters:
            ``mu`` (B, K, D), ``sigma`` (B, K, D), ``pi`` (B, K).

        Returns
        -------
        E_norm : Tensor
            Normality energy, shape ``(B,)``.
        deviation_features : Tensor
            Deviation feature vector, shape ``(B, n_deviation_features)``.
            Channels: role_novelty, time_novelty, duration_novelty,
            transition_novelty, prototype_distance.
        """
        mu = user_prototypes["mu"]       # (B, K, D)
        sigma = user_prototypes["sigma"] # (B, K, D)
        pi = user_prototypes["pi"]       # (B, K)

        E_norm = self._gmm_log_likelihood(z_x, mu, sigma, pi)  # (B,) float32
        E_norm = E_norm.nan_to_num(nan=0.0, posinf=200.0, neginf=-200.0)

        # Per-prototype squared Mahalanobis distances → soft summary (in fp32)
        z_f = z_x.float()
        mu_f = mu.float()
        sigma_f = sigma.float()
        diff = z_f.unsqueeze(1) - mu_f                              # (B, K, D)
        mahal = (diff.pow(2) / sigma_f.clamp(min=1e-8)).sum(dim=-1) # (B, K) float32
        # Clamp before feeding into MLP to prevent float16 overflow downstream
        mahal = mahal.clamp(max=1000.0)

        # Always use float32 for the deviation MLP (avoids fp16 overflow on large mahal)
        concat = torch.cat([z_f, mahal], dim=-1)  # (B, D + K)
        deviation_features = self.deviation_proj(concat)           # (B, 5) float32

        # Return both in float32 for numerical safety in loss computation
        return E_norm, deviation_features
