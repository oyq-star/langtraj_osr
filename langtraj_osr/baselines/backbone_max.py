"""B8: Backbone+max — original LangTraj-OW before refinement.

Score = max(language_match_score, novelty_score).
No conformal calibration, no primitive head, simple max fusion.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.episode_encoder import EpisodeEncoder
from ..models.trajectory_encoder import TrajectoryEncoder
from ..models.user_history import UserHistoryModule

from .direct_text import FrozenTextEncoder


class BackboneMaxModel(nn.Module):
    """Baseline B8: simple max-fusion of language match and novelty scores.

    Score = max(language_match_score, novelty_score).
    No conformal calibration, no primitive head.

    Parameters
    ----------
    d_model : int
        Hidden dimensionality.
    text_vocab_size : int
        Text vocabulary size.
    novelty_weight : float
        Scaling factor for normality energy when computing novelty score.
    """

    def __init__(
        self,
        poi_vocab_size: int = 64,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        n_prototypes: int = 8,
        text_vocab_size: int = 10000,
        text_max_len: int = 128,
        text_num_layers: int = 2,
        novelty_weight: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.novelty_weight = novelty_weight

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
        self.text_encoder = FrozenTextEncoder(
            vocab_size=text_vocab_size,
            d_model=d_model,
            max_len=text_max_len,
            num_layers=text_num_layers,
            dropout=dropout,
        )

        # Simple bilinear alignment (no primitive head)
        self.W = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

    def encode_trajectory(
        self,
        episodes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ep_emb = self.episode_encoder(episodes)
        return self.trajectory_encoder(ep_emb, mask=mask)

    def language_scores(
        self,
        z_x: torch.Tensor,
        def_token_ids: torch.Tensor,
        def_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute language match scores: z_x^T W c_d.

        Parameters
        ----------
        z_x : Tensor (B, D)
        def_token_ids : Tensor (K, S)
        def_attention_mask : Tensor (K, S) optional

        Returns
        -------
        Tensor (B, K)
        """
        c_d = self.text_encoder(def_token_ids, def_attention_mask)  # (K, D)
        z_proj = z_x @ self.W  # (B, D)
        return z_proj @ c_d.T  # (B, K)

    def forward(
        self,
        episodes: Dict[str, torch.Tensor],
        def_token_ids: torch.Tensor,
        user_prototypes: Dict[str, torch.Tensor],
        def_attention_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z_x, h_i = self.encode_trajectory(episodes, mask=mask)
        E_norm, dev_feats = self.user_history(z_x, user_prototypes)

        lang_scores = self.language_scores(z_x, def_token_ids, def_attention_mask)
        # (B, K)

        # Novelty score: scaled normality energy
        novelty_score = self.novelty_weight * E_norm  # (B,)

        # Max fusion: for each trajectory, final score per definition =
        # max(language_score, novelty_score)
        # Broadcast novelty to match definitions
        novelty_expanded = novelty_score.unsqueeze(-1).expand_as(lang_scores)  # (B, K)
        fused_scores = torch.max(lang_scores, novelty_expanded)  # (B, K)

        return {
            "E_norm": E_norm,
            "deviation_features": dev_feats,
            "language_scores": lang_scores,
            "novelty_score": novelty_score,
            "fused_scores": fused_scores,
            "z_x": z_x,
        }

    @torch.no_grad()
    def predict(
        self,
        episodes: Dict[str, torch.Tensor],
        def_token_ids: torch.Tensor,
        user_prototypes: Dict[str, torch.Tensor],
        def_attention_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict fused scores and normality energy.

        Returns
        -------
        fused_scores : Tensor (B, K) — max(language, novelty) per definition.
        E_norm : Tensor (B,) — normality energy.
        """
        self.eval()
        out = self.forward(
            episodes, def_token_ids, user_prototypes,
            def_attention_mask=def_attention_mask, mask=mask,
        )
        return out["fused_scores"], out["E_norm"]

    def compute_loss(
        self,
        episodes: Dict[str, torch.Tensor],
        def_token_ids: torch.Tensor,
        user_prototypes: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        def_attention_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.forward(
            episodes, def_token_ids, user_prototypes,
            def_attention_mask=def_attention_mask, mask=mask,
        )
        fused_scores = out["fused_scores"]
        E_norm = out["E_norm"]

        # Cross-entropy on fused scores for known anomalies
        n_cls = fused_scores.shape[1]
        known_mask = (labels > 0) & (labels <= n_cls)
        if known_mask.any():
            cls_loss = F.cross_entropy(fused_scores[known_mask], (labels[known_mask] - 1).clamp(0, n_cls - 1))
        else:
            cls_loss = torch.tensor(0.0, device=labels.device)

        # Normality margin loss
        is_normal = (labels == 0).float()
        is_anomaly = (labels > 0).float()
        margin = 10.0
        norm_loss = (
            (E_norm * is_normal).sum() / is_normal.sum().clamp(min=1)
            + (F.relu(margin - E_norm) * is_anomaly).sum() / is_anomaly.sum().clamp(min=1)
        )

        return cls_loss + norm_loss
