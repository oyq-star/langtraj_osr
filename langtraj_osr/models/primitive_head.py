"""Primitive evidence head with multiple-instance learning (MIL) attention.

For each of the 10 mobility primitives the head learns an attention
distribution over episodes, computes a weighted-sum representation, and
produces a scalar violation score via sigmoid.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical ordering of the 10 mobility-anomaly primitives.
PRIMITIVE_NAMES: List[str] = [
    "unusual_time_at_role",
    "unusual_dest_role",
    "unusual_dwell",
    "unusual_transition_order",
    "unusual_detour",
    "missing_routine_stop",
    "unusual_repetition",
    "unusual_long_jump",
    "event_conflict",
    "unusual_companion",
]


class PrimitiveHead(nn.Module):
    """Multi-instance learning (MIL) head that detects primitive violations.

    For each primitive *p* the module learns:
    1. An attention vector that scores the relevance of each episode.
    2. A classifier that maps the attention-pooled representation to a
       violation probability.

    Parameters
    ----------
    d_model : int
        Dimensionality of episode embeddings.
    n_primitives : int
        Number of primitive violation types.
    """

    def __init__(self, d_model: int = 256, n_primitives: int = 10) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_primitives = n_primitives

        # Per-primitive gated attention (Ilse et al., 2018)
        self.attn_V = nn.Linear(d_model, 128)
        self.attn_U = nn.Linear(d_model, 128)
        self.attn_w = nn.Linear(128, n_primitives)  # one score per primitive

        # Per-primitive classifier on the pooled representation
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_primitives),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        h_i: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute primitive violation scores with MIL attention.

        Parameters
        ----------
        h_i : Tensor
            Contextualised episode embeddings, shape ``(B, L, D)``.
        mask : Tensor, optional
            Boolean padding mask ``(B, L)``; ``True`` = padded position.

        Returns
        -------
        v_x : Tensor
            Primitive violation vector, shape ``(B, n_primitives)``,
            values in ``(0, 1)``.
        episode_attention : Tensor
            Attention weights, shape ``(B, n_primitives, L)``.
        """
        B, L, D = h_i.shape

        # Gated attention scores  (Ilse et al., 2018)
        attn_logits = self.attn_w(
            torch.tanh(self.attn_V(h_i)) * torch.sigmoid(self.attn_U(h_i))
        )  # (B, L, P)

        # Mask padded positions
        if mask is not None:
            attn_logits = attn_logits.masked_fill(
                mask.unsqueeze(-1), float("-inf")
            )

        # Softmax over episode dimension → (B, L, P) then transpose
        episode_attention = F.softmax(attn_logits, dim=1).permute(0, 2, 1)
        # episode_attention: (B, P, L)

        # Weighted sum of episode embeddings per primitive
        pooled = torch.bmm(episode_attention, h_i)  # (B, P, D)

        # Classification: we want one score per primitive.
        # Apply the shared classifier to a *mean* pooled vector and use
        # per-primitive outputs.  We also incorporate the per-primitive
        # pooled representation via element-wise gating.
        #
        # Strategy: for each primitive p, classifier maps pooled[:, p, :]
        # to a scalar.  Because classifier outputs n_primitives scores at
        # once we use the diagonal trick: pass the mean-pooled vector and
        # pick the p-th output, then refine with per-primitive evidence.
        mean_pooled = pooled.mean(dim=1)               # (B, D)
        base_scores = self.classifier(mean_pooled)      # (B, P)

        # Refine with per-primitive cosine similarity to mean
        per_prim_sim = F.cosine_similarity(
            pooled, mean_pooled.unsqueeze(1).expand_as(pooled), dim=-1
        )  # (B, P)

        v_x = torch.sigmoid(base_scores + per_prim_sim)  # (B, P)

        return v_x, episode_attention
