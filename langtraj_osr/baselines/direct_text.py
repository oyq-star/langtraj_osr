"""B5: DirectText baseline — text contrastive alignment WITHOUT primitive head.

Score: s_k = z_x^T W c_d  (only global alignment, no primitive grounding).
Tests whether the primitive head adds value beyond simple text-trajectory alignment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.episode_encoder import EpisodeEncoder
from ..models.trajectory_encoder import TrajectoryEncoder
from ..models.user_history import UserHistoryModule


class FrozenTextEncoder(nn.Module):
    """Lightweight text encoder that maps tokenized text to d_model embeddings.

    Uses a learnable embedding table + small Transformer. In practice this
    could be replaced with a frozen pre-trained LM; here we use a trainable
    proxy for reproducibility without external model dependencies.

    Parameters
    ----------
    vocab_size : int
        Text vocabulary size.
    d_model : int
        Output embedding dimensionality.
    max_len : int
        Maximum token sequence length.
    nhead : int
        Attention heads.
    num_layers : int
        Transformer layers.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        max_len: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text token sequences.

        Parameters
        ----------
        token_ids : Tensor (N, S) — LongTensor of token indices.
        attention_mask : Tensor (N, S) — True for padded positions.

        Returns
        -------
        Tensor (N, d_model) — text embeddings (mean pool over non-padded tokens).
        """
        N, S = token_ids.shape
        positions = torch.arange(S, device=token_ids.device).unsqueeze(0).expand(N, -1)

        x = self.token_embed(token_ids) + self.pos_embed(positions)
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        x = self.layer_norm(x)

        # Mean pool over non-padded positions
        if attention_mask is not None:
            valid_mask = (~attention_mask).unsqueeze(-1).float()  # (N, S, 1)
            x = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return x  # (N, d_model)


class DirectTextModel(nn.Module):
    """Baseline B5: text contrastive alignment without primitive head.

    Score: s_k = z_x^T W c_d  (bilinear alignment only).

    Parameters
    ----------
    d_model : int
        Hidden dimensionality.
    text_vocab_size : int
        Text tokenizer vocabulary size.
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
        self.text_encoder = FrozenTextEncoder(
            vocab_size=text_vocab_size,
            d_model=d_model,
            max_len=text_max_len,
            num_layers=text_num_layers,
            dropout=dropout,
        )

        # Bilinear alignment matrix (no primitive head — key difference from full model)
        self.W = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

    def encode_trajectory(
        self,
        episodes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ep_emb = self.episode_encoder(episodes)
        return self.trajectory_encoder(ep_emb, mask=mask)

    def encode_definitions(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text definitions into embeddings.

        Parameters
        ----------
        token_ids : Tensor (K, S)
        attention_mask : Tensor (K, S) optional

        Returns
        -------
        Tensor (K, d_model)
        """
        return self.text_encoder(token_ids, attention_mask)

    def score_definitions(
        self,
        z_x: torch.Tensor,
        c_d: torch.Tensor,
    ) -> torch.Tensor:
        """Simple bilinear score: s_k = z_x^T W c_d.

        Parameters
        ----------
        z_x : Tensor (B, D)
        c_d : Tensor (K, D)

        Returns
        -------
        Tensor (B, K)
        """
        z_proj = z_x @ self.W   # (B, D)
        return z_proj @ c_d.T   # (B, K)

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
        c_d = self.encode_definitions(def_token_ids, def_attention_mask)
        def_scores = self.score_definitions(z_x, c_d)

        return {
            "E_norm": E_norm,
            "deviation_features": dev_feats,
            "definition_scores": def_scores,
            "z_x": z_x,
            "c_d": c_d,
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
        self.eval()
        out = self.forward(
            episodes, def_token_ids, user_prototypes,
            def_attention_mask=def_attention_mask, mask=mask,
        )
        return out["definition_scores"], out["E_norm"]

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
        def_scores = out["definition_scores"]
        E_norm = out["E_norm"]

        # Cross-entropy for known anomaly types
        n_cls = def_scores.shape[1]
        known_mask = (labels > 0) & (labels <= n_cls)
        if known_mask.any():
            cls_loss = F.cross_entropy(def_scores[known_mask], (labels[known_mask] - 1).clamp(0, n_cls - 1))
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
