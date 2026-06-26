"""Sequence-level Transformer encoder for LangTraj-OSR.

Converts a variable-length sequence of episode embeddings into a fixed-size
global trip embedding (via a CLS token) and contextualised per-episode
representations.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Adds position information to the input embeddings so the Transformer
    can reason about ordering.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                       # (max_len, D)
        position = torch.arange(0, max_len).unsqueeze(1).float() # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding and apply dropout.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, L, D)``.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TrajectoryEncoder(nn.Module):
    """Transformer encoder that produces a global trip embedding and
    contextualised episode embeddings.

    A learnable ``[CLS]`` token is prepended to the sequence and its final
    hidden state is used as the global trip embedding ``z_x``.

    Parameters
    ----------
    d_model : int
        Model / embedding dimensionality.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of Transformer encoder layers.
    dropout : float
        Dropout rate used throughout.
    max_len : int
        Maximum supported sequence length (including CLS).
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (max_len + 1 to accommodate CLS)
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model, max_len=max_len + 1, dropout=dropout
        )

        # Build TransformerEncoderLayer with compatibility for older PyTorch
        # versions that lack batch_first / norm_first kwargs.
        _layer_kwargs: dict = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
        )
        # batch_first and norm_first were added in PyTorch >= 1.9 / 1.11
        import inspect as _inspect
        _sig = _inspect.signature(nn.TransformerEncoderLayer.__init__)
        if "batch_first" in _sig.parameters:
            _layer_kwargs["batch_first"] = True
            self._batch_first = True
        else:
            self._batch_first = False
        if "norm_first" in _sig.parameters:
            _layer_kwargs["norm_first"] = True

        encoder_layer = nn.TransformerEncoderLayer(**_layer_kwargs)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    def forward(
        self,
        episode_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of episode-embedding sequences.

        Parameters
        ----------
        episode_embeddings : Tensor
            Shape ``(B, L, D)`` — output of :class:`EpisodeEncoder`.
        mask : Tensor, optional
            Boolean padding mask of shape ``(B, L)`` where ``True``
            indicates a **padded** (invalid) position.

        Returns
        -------
        z_x : Tensor
            Global trip embedding, shape ``(B, D)``.
        h_i : Tensor
            Contextualised episode embeddings, shape ``(B, L, D)``.
        """
        B, L, D = episode_embeddings.shape

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        x = torch.cat([cls_tokens, episode_embeddings], dim=1) # (B, 1+L, D)

        # Extend mask to include CLS (never masked)
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)      # (B, 1+L)
        else:
            full_mask = None

        # Positional encoding + Transformer
        x = self.pos_encoder(x)

        if self._batch_first:
            x = self.transformer(x, src_key_padding_mask=full_mask)  # (B, 1+L, D)
        else:
            # Older PyTorch: Transformer expects (S, B, D)
            x = x.transpose(0, 1)                                    # (1+L, B, D)
            x = self.transformer(x, src_key_padding_mask=full_mask)  # (1+L, B, D)
            x = x.transpose(0, 1)                                    # (B, 1+L, D)

        x = self.layer_norm(x)

        z_x = x[:, 0, :]       # CLS → global trip embedding  (B, D)
        h_i = x[:, 1:, :]      # episode embeddings            (B, L, D)

        return z_x, h_i
