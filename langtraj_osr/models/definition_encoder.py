"""Text definition encoder for LangTraj-OSR.

Encodes natural-language anomaly concept definitions into dense concept
embeddings and primitive-attention vectors using a frozen SentenceTransformer
backbone followed by learnable projection heads.
"""

from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DefinitionEncoder(nn.Module):
    """Encode textual anomaly definitions into concept embeddings.

    Architecture
    ------------
    1. **Frozen text backbone** — a SentenceTransformer that maps each
       definition string to a fixed-size vector.
    2. **Projection MLP** — 2-layer MLP that maps the text vector into the
       shared trajectory-embedding space (``d_model``).
    3. **Primitive attention head** — 2-layer MLP that outputs a softmax
       distribution over the ``n_primitives`` mobility primitives,
       indicating which primitives are relevant for a given concept.

    Parameters
    ----------
    text_encoder_name : str
        HuggingFace model identifier for the SentenceTransformer.
    d_model : int
        Dimensionality of the shared embedding space.
    n_primitives : int
        Number of mobility primitives.
    """

    def __init__(
        self,
        text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        d_model: int = 256,
        n_primitives: int = 10,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_primitives = n_primitives
        self.text_encoder_name = text_encoder_name

        # --- Lazy-loaded frozen text encoder ---
        # We defer loading so the module can be instantiated without network
        # access; the encoder is loaded on first forward call.
        self._text_encoder: object | None = None
        self._text_dim: int | None = None

        # Placeholders — actual layers created in _ensure_encoder()
        self.projection: nn.Module | None = None
        self.prim_attn: nn.Module | None = None

    # ------------------------------------------------------------------
    def _ensure_encoder(self) -> None:
        """Lazily load the SentenceTransformer and build projection heads."""
        if self._text_encoder is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self._text_encoder = SentenceTransformer(self.text_encoder_name)

        # Freeze all parameters
        for param in self._text_encoder.parameters():  # type: ignore[union-attr]
            param.requires_grad = False

        self._text_dim = self._text_encoder.get_sentence_embedding_dimension()  # type: ignore[union-attr]

        # Build projection heads on the correct device
        device = next(self.parameters(), torch.tensor(0.0)).device
        self.projection = nn.Sequential(
            nn.Linear(self._text_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.d_model),
        ).to(device)

        self.prim_attn = nn.Sequential(
            nn.Linear(self._text_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.n_primitives),
        ).to(device)

        # Xavier init
        for m in [self.projection, self.prim_attn]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Run the frozen SentenceTransformer and return embeddings on the
        model device as a float tensor."""
        self._ensure_encoder()
        device = next(self.projection.parameters()).device  # type: ignore[union-attr]
        embeddings = self._text_encoder.encode(  # type: ignore[union-attr]
            texts, convert_to_tensor=True, show_progress_bar=False
        )
        return embeddings.to(device).float()

    # ------------------------------------------------------------------
    def forward(
        self, definition_texts: Union[List[str], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode anomaly concept definitions.

        Parameters
        ----------
        definition_texts : list[str] | Tensor
            Either a list of *B_d* definition strings **or** a pre-computed
            tensor of text embeddings with shape ``(B_d, text_dim)``.

        Returns
        -------
        c_d : Tensor
            Concept embeddings in the shared space, shape ``(B_d, d_model)``.
        a_d : Tensor
            Primitive attention vectors, shape ``(B_d, n_primitives)``.
            Each row sums to 1 (softmax over primitives).
        """
        self._ensure_encoder()

        if isinstance(definition_texts, list):
            text_emb = self._encode_texts(definition_texts)  # (B_d, text_dim)
        else:
            text_emb = definition_texts

        c_d = self.projection(text_emb)                      # (B_d, d_model)
        a_d = F.softmax(self.prim_attn(text_emb), dim=-1)    # (B_d, n_prims)

        return c_d, a_d
