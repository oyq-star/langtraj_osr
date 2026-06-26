"""B6: LM-TAD-style baseline — sequential anomaly detector via autoregressive prediction.

Predicts the next episode given history. Anomaly score = reconstruction error
(negative log-likelihood of the actual next episode). No language, no definitions.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.episode_encoder import EpisodeEncoder


class CausalTransformerDecoder(nn.Module):
    """Causal (autoregressive) Transformer for next-episode prediction.

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of decoder layers.
    max_len : int
        Maximum sequence length.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        max_len: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.pos_embed = nn.Embedding(max_len, d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Use TransformerEncoder with causal mask for autoregressive decoding
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive encoding with causal mask.

        Parameters
        ----------
        x : Tensor (B, L, D)
        padding_mask : Tensor (B, L) optional, True = padded

        Returns
        -------
        Tensor (B, L, D) — contextualised hidden states.
        """
        B, L, D = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(positions)

        # Causal mask: prevent attending to future positions
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.layer_norm(x)


class EpisodePredictorHead(nn.Module):
    """Predict next episode features from hidden states.

    Predicts distributions over each categorical field and a Gaussian
    for the continuous field (trip_length_change).

    Parameters
    ----------
    d_model : int
        Input dimensionality.
    poi_vocab_size : int
        POI role vocabulary.
    time_bins : int
        Time bin vocabulary.
    dwell_bins : int
        Dwell bin vocabulary.
    transition_types : int
        Transition type vocabulary.
    """

    def __init__(
        self,
        d_model: int = 256,
        poi_vocab_size: int = 64,
        time_bins: int = 168,
        dwell_bins: int = 16,
        transition_types: int = 4,
    ) -> None:
        super().__init__()
        self.poi_head = nn.Linear(d_model, poi_vocab_size)
        self.time_head = nn.Linear(d_model, time_bins)
        self.dwell_head = nn.Linear(d_model, dwell_bins)
        self.transition_head = nn.Linear(d_model, transition_types)
        self.event_head = nn.Linear(d_model, 2)
        self.companion_head = nn.Linear(d_model, 2)
        # Trip length change: predict mean and log-variance
        self.trip_length_head = nn.Linear(d_model, 2)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict next-episode distributions.

        Parameters
        ----------
        hidden : Tensor (B, L, D)

        Returns
        -------
        dict of logits/parameters for each field.
        """
        return {
            "poi_role_logits": self.poi_head(hidden),           # (B, L, V_poi)
            "time_bin_logits": self.time_head(hidden),          # (B, L, V_time)
            "dwell_bin_logits": self.dwell_head(hidden),        # (B, L, V_dwell)
            "transition_logits": self.transition_head(hidden),  # (B, L, V_trans)
            "event_logits": self.event_head(hidden),            # (B, L, 2)
            "companion_logits": self.companion_head(hidden),    # (B, L, 2)
            "trip_length_params": self.trip_length_head(hidden),# (B, L, 2)
        }


class LMTADModel(nn.Module):
    """Baseline B6: LM-TAD-style sequential anomaly detector.

    Architecture: EpisodeEncoder -> CausalTransformerDecoder -> EpisodePredictorHead.
    Anomaly score = reconstruction error (NLL of actual next episode).

    Parameters
    ----------
    poi_vocab_size : int
        POI vocabulary size.
    d_model : int
        Hidden dimensionality.
    nhead : int
        Attention heads.
    num_layers : int
        Decoder layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        poi_vocab_size: int = 64,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        max_len: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.episode_encoder = EpisodeEncoder(
            poi_vocab_size=poi_vocab_size,
            hidden_dim=d_model,
        )
        self.decoder = CausalTransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
        )
        self.predictor = EpisodePredictorHead(
            d_model=d_model,
            poi_vocab_size=poi_vocab_size,
        )

    def forward(
        self,
        episodes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: encode episodes, decode autoregressively, predict next.

        Returns dict of predicted logits/params for each field.
        """
        ep_emb = self.episode_encoder(episodes)  # (B, L, D)
        hidden = self.decoder(ep_emb, padding_mask=mask)  # (B, L, D)
        predictions = self.predictor(hidden)
        predictions["hidden"] = hidden
        return predictions

    def compute_reconstruction_nll(
        self,
        predictions: Dict[str, torch.Tensor],
        episodes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-sample negative log-likelihood of actual next episodes.

        Uses shifted targets: prediction at position t should match episode at t+1.

        Parameters
        ----------
        predictions : dict
            Output from forward().
        episodes : dict
            Ground-truth episode features.
        mask : Tensor (B, L), optional

        Returns
        -------
        Tensor (B,) — total NLL per trajectory.
        """
        B, L = episodes["poi_role"].shape

        if L < 2:
            return torch.zeros(B, device=episodes["poi_role"].device)

        # Shifted: predict[t] -> target[t+1], for t in 0..L-2
        pred_slice = slice(None, L - 1)  # positions 0..L-2
        tgt_slice = slice(1, None)       # positions 1..L-1

        nll = torch.zeros(B, device=episodes["poi_role"].device)

        # Categorical fields
        for field_name, logits_key in [
            ("poi_role", "poi_role_logits"),
            ("time_bin", "time_bin_logits"),
            ("dwell_bin", "dwell_bin_logits"),
            ("transition_type", "transition_logits"),
            ("event_flag", "event_logits"),
            ("companion_flag", "companion_logits"),
        ]:
            logits = predictions[logits_key][:, pred_slice]  # (B, L-1, V)
            targets = episodes[field_name][:, tgt_slice]      # (B, L-1)
            # Per-position cross entropy
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            ).reshape(B, L - 1)

            if mask is not None:
                valid = ~mask[:, tgt_slice]
                ce = ce * valid.float()

            nll = nll + ce.sum(dim=1)

        # Continuous field: trip_length_change (Gaussian NLL)
        params = predictions["trip_length_params"][:, pred_slice]  # (B, L-1, 2)
        mu = params[:, :, 0]
        log_var = params[:, :, 1].clamp(-10, 10)
        targets_cont = episodes["trip_length_change"][:, tgt_slice]  # (B, L-1)

        gauss_nll = 0.5 * (log_var + (targets_cont - mu).pow(2) / log_var.exp().clamp(min=1e-8))
        if mask is not None:
            valid = ~mask[:, tgt_slice]
            gauss_nll = gauss_nll * valid.float()
        nll = nll + gauss_nll.sum(dim=1)

        return nll  # (B,)

    @torch.no_grad()
    def predict(
        self,
        episodes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute anomaly scores = reconstruction NLL (higher = more anomalous).

        Returns
        -------
        Tensor (B,) — anomaly scores.
        """
        self.eval()
        predictions = self.forward(episodes, mask=mask)
        return self.compute_reconstruction_nll(predictions, episodes, mask=mask)

    def compute_loss(
        self,
        episodes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Training loss: mean reconstruction NLL over batch.

        Trained on normal trips only — learns to reconstruct normal patterns.
        """
        predictions = self.forward(episodes, mask=mask)
        nll = self.compute_reconstruction_nll(predictions, episodes, mask=mask)
        return nll.mean()
