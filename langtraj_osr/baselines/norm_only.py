"""B1: NormOnly baseline — anomaly detection using ONLY personalized normality model.

Uses the user history GMM to score trips. No language, no definitions.
Anomaly score = E_norm(x|u) (normality energy from GMM).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.episode_encoder import EpisodeEncoder
from ..models.trajectory_encoder import TrajectoryEncoder
from ..models.user_history import UserHistoryModule


class NormOnlyModel(nn.Module):
    """Baseline B1: anomaly detection via personalized normality energy only.

    Architecture: EpisodeEncoder -> TrajectoryEncoder -> UserHistoryModule.
    No language branch, no definition encoder.

    Parameters
    ----------
    poi_vocab_size : int
        POI role vocabulary size.
    d_model : int
        Hidden dimensionality throughout the model.
    nhead : int
        Number of attention heads in TrajectoryEncoder.
    num_layers : int
        Number of Transformer encoder layers.
    n_prototypes : int
        Number of GMM prototypes per user.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        poi_vocab_size: int = 64,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        n_prototypes: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

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

    def encode_trajectory(
        self,
        episodes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode episodes into global trip embedding and per-episode embeddings.

        Parameters
        ----------
        episodes : dict[str, Tensor]
            Episode feature tensors, each of shape ``(B, L)``.
        mask : Tensor, optional
            Padding mask ``(B, L)``, True for padded positions.

        Returns
        -------
        z_x : Tensor  (B, D)
        h_i : Tensor  (B, L, D)
        """
        ep_emb = self.episode_encoder(episodes)       # (B, L, D)
        z_x, h_i = self.trajectory_encoder(ep_emb, mask=mask)
        return z_x, h_i

    def forward(
        self,
        episodes: Dict[str, torch.Tensor],
        user_prototypes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning normality energy and deviation features.

        Parameters
        ----------
        episodes : dict[str, Tensor]
            Episode features ``(B, L)``.
        user_prototypes : dict
            Batched prototype parameters: mu (B,K,D), sigma (B,K,D), pi (B,K).
        mask : Tensor, optional
            Padding mask ``(B, L)``.

        Returns
        -------
        dict with keys:
            ``E_norm``   — normality energy (B,).
            ``deviation_features`` — deviation vector (B, 5).
            ``z_x``      — trip embedding (B, D).
        """
        z_x, h_i = self.encode_trajectory(episodes, mask=mask)
        E_norm, dev_feats = self.user_history(z_x, user_prototypes)
        return {
            "E_norm": E_norm,
            "deviation_features": dev_feats,
            "z_x": z_x,
        }

    @torch.no_grad()
    def predict(
        self,
        episodes: Dict[str, torch.Tensor],
        user_prototypes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute anomaly scores (higher = more anomalous).

        Returns
        -------
        Tensor  (B,) — normality energy as anomaly score.
        """
        self.eval()
        out = self.forward(episodes, user_prototypes, mask=mask)
        return out["E_norm"]

    def compute_loss(
        self,
        episodes: Dict[str, torch.Tensor],
        user_prototypes: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Training loss: minimize normality energy for normal trips,
        maximize for anomalous trips (contrastive margin loss).

        Parameters
        ----------
        labels : Tensor (B,)
            0 = normal, >0 = anomalous.
        """
        out = self.forward(episodes, user_prototypes, mask=mask)
        E_norm = out["E_norm"]

        is_normal = (labels == 0).float()
        is_anomaly = (labels > 0).float()

        # Normal trips: minimize energy; anomalous trips: energy should exceed margin
        margin = 10.0
        loss_normal = (E_norm * is_normal).sum() / is_normal.sum().clamp(min=1)
        loss_anomaly = (F.relu(margin - E_norm) * is_anomaly).sum() / is_anomaly.sum().clamp(min=1)

        return loss_normal + loss_anomaly
