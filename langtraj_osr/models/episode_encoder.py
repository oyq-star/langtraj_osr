"""Episode-level encoder for LangTraj-OSR.

Encodes each episode (a stop or transition in a trajectory) into a dense
embedding by combining categorical embeddings with continuous features.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodeEncoder(nn.Module):
    """Encode raw episode features into dense episode embeddings.

    Each episode in a trajectory is described by a set of categorical and
    continuous fields.  This module embeds every field independently and
    then fuses them through a two-layer MLP.

    Parameters
    ----------
    poi_vocab_size : int
        Number of distinct POI-role categories (e.g. home, work, gym, ...).
    time_bins : int
        Number of time-of-week bins (default 168 = 24 h * 7 d).
    dwell_bins : int
        Number of discretised dwell-duration bins.
    transition_types : int
        Number of transition-mode categories (walk, drive, transit, other).
    embed_dim : int
        Dimensionality of each individual embedding table.
    hidden_dim : int
        Output dimensionality of the episode embedding.
    """

    def __init__(
        self,
        poi_vocab_size: int = 64,
        time_bins: int = 168,
        dwell_bins: int = 16,
        transition_types: int = 4,
        embed_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        # --- categorical embeddings ---
        self.poi_role_embed = nn.Embedding(poi_vocab_size, embed_dim)
        self.time_bin_embed = nn.Embedding(time_bins, embed_dim)
        self.dwell_bin_embed = nn.Embedding(dwell_bins, embed_dim)
        self.transition_type_embed = nn.Embedding(transition_types, embed_dim)

        # --- continuous / binary feature encoders ---
        self.trip_length_proj = nn.Linear(1, embed_dim)
        self.event_flag_embed = nn.Embedding(2, embed_dim)
        self.companion_flag_embed = nn.Embedding(2, embed_dim)

        # Total concatenated width: 7 fields * embed_dim
        concat_dim = 7 * embed_dim

        # --- fusion MLP ---
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Xavier-uniform for linear layers, normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def forward(self, episodes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a batch of episode sequences.

        Parameters
        ----------
        episodes : dict[str, Tensor]
            Must contain the following keys (all with shape ``(B, L)``
            unless noted):

            * ``poi_role``        — LongTensor  (B, L)
            * ``time_bin``        — LongTensor  (B, L)
            * ``dwell_bin``       — LongTensor  (B, L)
            * ``transition_type`` — LongTensor  (B, L)
            * ``trip_length_change`` — FloatTensor (B, L)
            * ``event_flag``      — LongTensor  (B, L)  values in {0, 1}
            * ``companion_flag``  — LongTensor  (B, L)  values in {0, 1}

        Returns
        -------
        Tensor
            Episode embeddings of shape ``(B, L, hidden_dim)``.
        """
        e_poi = self.poi_role_embed(episodes["poi_role"])                # (B,L,E)
        e_time = self.time_bin_embed(episodes["time_bin"])               # (B,L,E)
        e_dwell = self.dwell_bin_embed(episodes["dwell_bin"])            # (B,L,E)
        e_trans = self.transition_type_embed(episodes["transition_type"])  # (B,L,E)

        # Continuous trip-length change → project from scalar
        trip_len = episodes["trip_length_change"].unsqueeze(-1)          # (B,L,1)
        e_trip = self.trip_length_proj(trip_len)                         # (B,L,E)

        e_event = self.event_flag_embed(episodes["event_flag"])          # (B,L,E)
        e_comp = self.companion_flag_embed(episodes["companion_flag"])   # (B,L,E)

        # Concatenate along feature dimension
        concat = torch.cat(
            [e_poi, e_time, e_dwell, e_trans, e_trip, e_event, e_comp],
            dim=-1,
        )  # (B, L, 7*E)

        return self.mlp(concat)  # (B, L, hidden_dim)
