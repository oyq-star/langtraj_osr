"""B7: ATROM/OSSL-style open-set detector WITHOUT language.

Learns prototypes for known anomaly types and uses reciprocal point learning
for open-set rejection. Unknown detection via distance to nearest prototype +
normality energy, calibrated with Extreme Value Theory (EVT).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.episode_encoder import EpisodeEncoder
from ..models.trajectory_encoder import TrajectoryEncoder
from ..models.user_history import UserHistoryModule


class PrototypeBank(nn.Module):
    """Learnable prototype bank for known anomaly types + reciprocal points.

    For each known class k, maintains:
    - A positive prototype p_k (representative embedding for class k)
    - A reciprocal point r_k (anti-prototype, pushed away from class k)

    Parameters
    ----------
    n_classes : int
        Number of known anomaly types.
    d_model : int
        Embedding dimensionality.
    """

    def __init__(self, n_classes: int, d_model: int = 256) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model

        # Positive prototypes
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        # Reciprocal points (anti-prototypes)
        self.reciprocal_points = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)

    def distances_to_prototypes(self, z: torch.Tensor) -> torch.Tensor:
        """Compute squared Euclidean distances to all prototypes.

        Parameters
        ----------
        z : Tensor (B, D)

        Returns
        -------
        Tensor (B, K) — distances to each prototype.
        """
        # (B, 1, D) - (1, K, D) -> (B, K, D) -> sum -> (B, K)
        return (z.unsqueeze(1) - self.prototypes.unsqueeze(0)).pow(2).sum(dim=-1)

    def distances_to_reciprocals(self, z: torch.Tensor) -> torch.Tensor:
        """Compute squared Euclidean distances to all reciprocal points.

        Parameters
        ----------
        z : Tensor (B, D)

        Returns
        -------
        Tensor (B, K)
        """
        return (z.unsqueeze(1) - self.reciprocal_points.unsqueeze(0)).pow(2).sum(dim=-1)


class EVTCalibrator:
    """Extreme Value Theory calibrator for unknown rejection thresholds.

    Fits a Weibull distribution to the tail of known-class distance scores,
    then uses the CDF to determine if a new sample is likely unknown.
    """

    def __init__(self, tail_fraction: float = 0.05) -> None:
        self.tail_fraction = tail_fraction
        self.weibull_params: Optional[Dict[int, Tuple[float, float, float]]] = None

    @torch.no_grad()
    def fit(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor,
        n_classes: int,
    ) -> None:
        """Fit per-class Weibull models on validation distances.

        Parameters
        ----------
        distances : Tensor (N, K) — distances to prototypes for N val samples.
        labels : Tensor (N,) — class labels (0-indexed known classes).
        n_classes : int
        """
        self.weibull_params = {}
        distances_np = distances.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for k in range(n_classes):
            class_mask = labels_np == k
            if not class_mask.any():
                self.weibull_params[k] = (1.0, 1.0, 0.0)
                continue

            class_dists = distances_np[class_mask, k]
            # Use the largest distances as tail
            n_tail = max(1, int(len(class_dists) * self.tail_fraction))
            tail = sorted(class_dists)[-n_tail:]

            # Simple moment-based Weibull fit (shape, scale, location)
            import numpy as np
            tail_arr = np.array(tail, dtype=np.float64)
            loc = tail_arr.min() - 1e-6
            shifted = tail_arr - loc
            shifted = shifted[shifted > 0]

            if len(shifted) < 2:
                self.weibull_params[k] = (1.0, float(tail_arr.mean()) + 1e-6, float(loc))
                continue

            log_shifted = np.log(shifted)
            # Method of moments for Weibull
            mean_log = log_shifted.mean()
            var_log = log_shifted.var()

            # Approximate shape parameter
            shape = max(0.5, 1.283 / (var_log ** 0.5 + 1e-8))
            scale = float(np.exp(mean_log + 0.5772 / shape))

            self.weibull_params[k] = (float(shape), scale, float(loc))

    def rejection_probability(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute probability that each sample is unknown.

        Parameters
        ----------
        distances : Tensor (B, K) — distances to prototypes.

        Returns
        -------
        Tensor (B,) — rejection probability (higher = more likely unknown).
        """
        if self.weibull_params is None:
            return torch.ones(distances.shape[0], device=distances.device)

        B, K = distances.shape
        # For each class, compute Weibull CDF on the distance
        rejection_scores = torch.ones(B, K, device=distances.device)

        for k in range(K):
            if k not in self.weibull_params:
                continue
            shape, scale, loc = self.weibull_params[k]
            d = distances[:, k].cpu().numpy()

            import numpy as np
            shifted = np.maximum(d - loc, 1e-12)
            # Weibull CDF: 1 - exp(-(x/scale)^shape)
            cdf = 1.0 - np.exp(-((shifted / (scale + 1e-12)) ** shape))
            rejection_scores[:, k] = torch.tensor(cdf, dtype=torch.float32, device=distances.device)

        # Min rejection across classes = probability of not belonging to any known class
        # High CDF for all classes -> likely unknown
        return rejection_scores.min(dim=1).values


class ATROMModel(nn.Module):
    """Baseline B7: ATROM/OSSL-style open-set trajectory anomaly detector.

    Uses prototype learning + reciprocal points for known anomaly detection
    and EVT for unknown rejection. No language branch.

    Parameters
    ----------
    n_known_classes : int
        Number of known anomaly types.
    d_model : int
        Hidden dimensionality.
    lambda_reciprocal : float
        Weight for reciprocal point loss.
    """

    def __init__(
        self,
        n_known_classes: int = 10,
        poi_vocab_size: int = 64,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        n_prototypes: int = 8,
        lambda_reciprocal: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_known_classes = n_known_classes
        self.lambda_reciprocal = lambda_reciprocal

        self.episode_encoder = EpisodeEncoder(
            poi_vocab_size=poi_vocab_size,
            hidden_dim=d_model,
        )
        self.trajectory_encoder = TrajectoryEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.user_history = UserHistoryModule(
            d_model=d_model,
            n_prototypes=n_prototypes,
        )
        self.prototype_bank = PrototypeBank(
            n_classes=n_known_classes,
            d_model=d_model,
        )
        self.evt_calibrator = EVTCalibrator()

        # Projection for combining normality energy with prototype distances
        self.fusion = nn.Sequential(
            nn.Linear(n_known_classes + 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def encode_trajectory(
        self,
        episodes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ep_emb = self.episode_encoder(episodes)
        return self.trajectory_encoder(ep_emb, mask=mask)

    def forward(
        self,
        episodes: Dict[str, torch.Tensor],
        user_prototypes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z_x, h_i = self.encode_trajectory(episodes, mask=mask)
        E_norm, dev_feats = self.user_history(z_x, user_prototypes)

        proto_dists = self.prototype_bank.distances_to_prototypes(z_x)  # (B, K)
        recip_dists = self.prototype_bank.distances_to_reciprocals(z_x) # (B, K)

        # Known class assignment: closest prototype
        class_logits = -proto_dists  # (B, K) — higher = closer = more likely

        return {
            "E_norm": E_norm,
            "deviation_features": dev_feats,
            "z_x": z_x,
            "proto_dists": proto_dists,
            "recip_dists": recip_dists,
            "class_logits": class_logits,
        }

    @torch.no_grad()
    def predict(
        self,
        episodes: Dict[str, torch.Tensor],
        user_prototypes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict known class, unknown rejection score, and anomaly score.

        Returns
        -------
        class_logits : Tensor (B, K) — known class scores.
        unknown_score : Tensor (B,) — EVT rejection probability.
        anomaly_score : Tensor (B,) — combined anomaly score.
        """
        self.eval()
        out = self.forward(episodes, user_prototypes, mask=mask)

        class_logits = out["class_logits"]
        proto_dists = out["proto_dists"]
        E_norm = out["E_norm"]

        # Unknown score from EVT
        unknown_score = self.evt_calibrator.rejection_probability(proto_dists)

        # Combined anomaly score: fuse prototype distance + normality energy
        min_proto_dist = proto_dists.min(dim=1).values.unsqueeze(-1)  # (B, 1)
        fusion_input = torch.cat([proto_dists, E_norm.unsqueeze(-1)], dim=-1)
        anomaly_score = self.fusion(fusion_input).squeeze(-1)  # (B,)

        return class_logits, unknown_score, anomaly_score

    def compute_loss(
        self,
        episodes: Dict[str, torch.Tensor],
        user_prototypes: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute training loss: prototype + reciprocal point + normality losses.

        Parameters
        ----------
        labels : Tensor (B,)
            0 = normal, 1..K = known anomaly classes.
        """
        out = self.forward(episodes, user_prototypes, mask=mask)
        proto_dists = out["proto_dists"]       # (B, K_known)
        recip_dists = out["recip_dists"]       # (B, K_known)
        E_norm = out["E_norm"]

        loss = torch.tensor(0.0, device=labels.device)

        # --- Prototype classification loss for known anomalies ---
        known_mask = labels > 0
        if known_mask.any():
            # Map concept IDs (1..25) to prototype indices (0..K-1)
            known_labels = (labels[known_mask] - 1) % self.n_known_classes
            known_proto_dists = proto_dists[known_mask]

            # Softmin over distances (closer = more likely)
            cls_loss = F.cross_entropy(-known_proto_dists, known_labels)
            loss = loss + cls_loss

            # Pull known samples toward their prototype
            B_known = known_labels.shape[0]
            target_dists = known_proto_dists[torch.arange(B_known), known_labels]
            pull_loss = target_dists.mean()
            loss = loss + pull_loss

        # --- Reciprocal point loss: push reciprocal points away from their class ---
        if known_mask.any():
            known_recip_dists = recip_dists[known_mask]
            known_labels_k = (labels[known_mask] - 1) % self.n_known_classes
            B_known = known_labels_k.shape[0]
            # Distance to own reciprocal point should be LARGE
            own_recip_dist = known_recip_dists[torch.arange(B_known), known_labels_k]
            # Distance to OTHER reciprocal points should be SMALL
            other_mask_mat = torch.ones_like(known_recip_dists, dtype=torch.bool)
            other_mask_mat[torch.arange(B_known), known_labels_k] = False
            other_recip_dist = known_recip_dists[other_mask_mat].reshape(B_known, -1)

            margin_recip = 10.0
            reciprocal_loss = (
                F.relu(margin_recip - own_recip_dist).mean()
                + other_recip_dist.mean() * 0.1
            )
            loss = loss + self.lambda_reciprocal * reciprocal_loss

        # --- Normality contrastive loss ---
        is_normal = (labels == 0).float()
        is_anomaly = (labels > 0).float()
        margin_norm = 10.0
        if is_normal.sum() > 0:
            loss = loss + (E_norm * is_normal).sum() / is_normal.sum()
        if is_anomaly.sum() > 0:
            loss = loss + (F.relu(margin_norm - E_norm) * is_anomaly).sum() / is_anomaly.sum()

        return loss

    @torch.no_grad()
    def fit_evt(
        self,
        episodes: Dict[str, torch.Tensor],
        user_prototypes: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Fit EVT calibrator on validation data (known classes only).

        Call after training, before evaluation on test set.
        """
        self.eval()
        out = self.forward(episodes, user_prototypes, mask=mask)
        proto_dists = out["proto_dists"]

        known_mask = labels > 0
        if known_mask.any():
            self.evt_calibrator.fit(
                proto_dists[known_mask],
                (labels[known_mask] - 1) % self.n_known_classes,
                self.n_known_classes,
            )
