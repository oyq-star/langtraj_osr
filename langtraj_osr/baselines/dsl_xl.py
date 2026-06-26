"""B2: DSL-XL — Structured Definition Language baseline.

Replaces natural-language definitions with structured 12-slot definition vectors.
Scores trajectories using bilinear alignment between trip embedding and DSL encoding.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.episode_encoder import EpisodeEncoder
from ..models.trajectory_encoder import TrajectoryEncoder
from ..models.user_history import UserHistoryModule


# 12 DSL slot names for reference
DSL_SLOT_NAMES: List[str] = [
    "target_time_range",
    "target_poi_roles",
    "forbidden_poi_roles",
    "min_dwell",
    "max_dwell",
    "expected_transitions",
    "forbidden_transitions",
    "spatial_zone_type",
    "temporal_regularity",
    "event_context",
    "companion_required",
    "negation_flag",
]

NUM_DSL_SLOTS = len(DSL_SLOT_NAMES)


class DSLEncoder(nn.Module):
    """Encode structured 12-slot definition vectors into d_model embeddings.

    Each slot value is embedded independently, then all are concatenated
    and projected through an MLP.

    Parameters
    ----------
    slot_vocab_sizes : list[int]
        Vocabulary size for each of the 12 DSL slots.
    slot_embed_dim : int
        Embedding dimension per slot.
    d_model : int
        Output dimensionality.
    """

    def __init__(
        self,
        slot_vocab_sizes: Optional[List[int]] = None,
        slot_embed_dim: int = 32,
        d_model: int = 256,
    ) -> None:
        super().__init__()
        # Default vocab sizes per slot
        if slot_vocab_sizes is None:
            slot_vocab_sizes = [
                24,   # target_time_range (hour buckets)
                64,   # target_poi_roles
                64,   # forbidden_poi_roles
                16,   # min_dwell (bins)
                16,   # max_dwell (bins)
                8,    # expected_transitions (mode combos)
                8,    # forbidden_transitions
                16,   # spatial_zone_type
                8,    # temporal_regularity
                4,    # event_context
                2,    # companion_required
                2,    # negation_flag
            ]
        assert len(slot_vocab_sizes) == NUM_DSL_SLOTS

        self.slot_embeddings = nn.ModuleList([
            nn.Embedding(vs, slot_embed_dim) for vs in slot_vocab_sizes
        ])

        concat_dim = NUM_DSL_SLOTS * slot_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(self, slot_values: torch.Tensor) -> torch.Tensor:
        """Encode DSL slot values.

        Parameters
        ----------
        slot_values : Tensor
            LongTensor of shape ``(N, 12)`` — one value per slot per definition.

        Returns
        -------
        Tensor (N, d_model) — definition embeddings.
        """
        parts: List[torch.Tensor] = []
        for i, embed_layer in enumerate(self.slot_embeddings):
            parts.append(embed_layer(slot_values[:, i]))  # (N, slot_embed_dim)
        concat = torch.cat(parts, dim=-1)  # (N, 12*slot_embed_dim)
        return self.mlp(concat)            # (N, d_model)


class DSLXLModel(nn.Module):
    """Baseline B2: trajectory scoring with structured DSL definitions.

    Score: s_k = z_x^T W c_dsl + lambda * a_dsl^T v_x

    Parameters
    ----------
    d_model : int
        Hidden dimensionality.
    n_primitives : int
        Number of primitive dimensions for the attribute alignment term.
    lambda_attr : float
        Weight for the attribute alignment term.
    """

    def __init__(
        self,
        poi_vocab_size: int = 64,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        n_prototypes: int = 8,
        n_primitives: int = 10,
        lambda_attr: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_primitives = n_primitives
        self.lambda_attr = lambda_attr

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
        self.dsl_encoder = DSLEncoder(d_model=d_model)

        # Bilinear alignment matrix
        self.W = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

        # Primitive attribute head: project trip embedding to primitive logits
        self.primitive_head = nn.Linear(d_model, n_primitives)

        # DSL attribute vector projection (from d_model to n_primitives)
        self.dsl_attr_proj = nn.Linear(d_model, n_primitives)

    def encode_trajectory(
        self,
        episodes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode episodes into global trip embedding and per-episode embeddings."""
        ep_emb = self.episode_encoder(episodes)
        return self.trajectory_encoder(ep_emb, mask=mask)

    def score_definitions(
        self,
        z_x: torch.Tensor,
        dsl_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Compute definition match scores.

        Parameters
        ----------
        z_x : Tensor (B, D)
            Trip embeddings.
        dsl_slots : Tensor (K, 12)
            DSL slot values for K definitions.

        Returns
        -------
        Tensor (B, K) — match scores for each trajectory-definition pair.
        """
        c_dsl = self.dsl_encoder(dsl_slots)            # (K, D)

        # Global bilinear alignment: z_x^T W c_dsl
        z_proj = z_x @ self.W                           # (B, D)
        global_score = z_proj @ c_dsl.T                  # (B, K)

        # Attribute alignment: a_dsl^T v_x
        v_x = self.primitive_head(z_x)                   # (B, n_prim)
        a_dsl = self.dsl_attr_proj(c_dsl)                # (K, n_prim)
        attr_score = v_x @ a_dsl.T                       # (B, K)

        return global_score + self.lambda_attr * attr_score  # (B, K)

    def forward(
        self,
        episodes: Dict[str, torch.Tensor],
        dsl_slots: torch.Tensor,
        user_prototypes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Returns dict with E_norm, definition_scores, z_x.
        """
        z_x, h_i = self.encode_trajectory(episodes, mask=mask)
        E_norm, dev_feats = self.user_history(z_x, user_prototypes)
        def_scores = self.score_definitions(z_x, dsl_slots)

        return {
            "E_norm": E_norm,
            "deviation_features": dev_feats,
            "definition_scores": def_scores,
            "z_x": z_x,
        }

    @torch.no_grad()
    def predict(
        self,
        episodes: Dict[str, torch.Tensor],
        dsl_slots: torch.Tensor,
        user_prototypes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict anomaly type scores and normality energy.

        Returns
        -------
        definition_scores : Tensor (B, K)
        E_norm : Tensor (B,)
        """
        self.eval()
        out = self.forward(episodes, dsl_slots, user_prototypes, mask=mask)
        return out["definition_scores"], out["E_norm"]

    def compute_loss(
        self,
        episodes: Dict[str, torch.Tensor],
        dsl_slots: torch.Tensor,
        user_prototypes: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-entropy on definition scores + normality margin loss.

        Parameters
        ----------
        labels : Tensor (B,)
            0 = normal, 1..K = anomaly concept index.
        """
        out = self.forward(episodes, dsl_slots, user_prototypes, mask=mask)
        def_scores = out["definition_scores"]  # (B, K)
        E_norm = out["E_norm"]

        # Classification loss for known anomaly types (labels > 0)
        n_cls = def_scores.shape[1]
        known_mask = (labels > 0) & (labels <= n_cls)
        if known_mask.any():
            # Shift labels to 0-indexed for definitions
            cls_loss = F.cross_entropy(
                def_scores[known_mask], (labels[known_mask] - 1).clamp(0, n_cls - 1)
            )
        else:
            cls_loss = torch.tensor(0.0, device=labels.device)

        # Normality contrastive loss
        is_normal = (labels == 0).float()
        is_anomaly = (labels > 0).float()
        margin = 10.0
        norm_loss = (
            (E_norm * is_normal).sum() / is_normal.sum().clamp(min=1)
            + (F.relu(margin - E_norm) * is_anomaly).sum() / is_anomaly.sum().clamp(min=1)
        )

        return cls_loss + norm_loss
