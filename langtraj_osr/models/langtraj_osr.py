"""Full LangTraj-OSR model.

Integrates all sub-modules and implements the two-stage inference pipeline
(normality gating followed by conformal concept scoring).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.concepts import CONCEPT_BY_ID, get_concept_ids_for_split
from .definition_encoder import DefinitionEncoder
from .episode_encoder import EpisodeEncoder
from .primitive_head import PrimitiveHead
from .trajectory_encoder import TrajectoryEncoder
from .user_history import UserHistoryModule


@dataclass
class LangTrajConfig:
    """Configuration for the full LangTraj-OSR model."""

    # Episode encoder
    poi_vocab_size: int = 64
    time_bins: int = 168
    dwell_bins: int = 16
    transition_types: int = 4
    embed_dim: int = 64
    hidden_dim: int = 256

    # Trajectory encoder
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    max_len: int = 64

    # User history
    n_prototypes: int = 8
    n_deviation_features: int = 5

    # Primitive head
    n_primitives: int = 10

    # Definition encoder
    text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Scoring hyper-parameters
    lambda_prim: float = 0.5
    gamma_loc: float = 0.25


class LangTrajOSR(nn.Module):
    """LangTraj-OSR: Language-defined open-set recognition for mobility
    anomaly detection.

    The model scores a trip against a *set* of textual anomaly definitions
    and can also reject trips as either normal or unknown-anomaly.

    Parameters
    ----------
    config : LangTrajConfig
        Full model configuration.
    """

    def __init__(self, config: LangTrajConfig) -> None:
        super().__init__()
        self.config = config

        # Sub-modules
        self.episode_encoder = EpisodeEncoder(
            poi_vocab_size=config.poi_vocab_size,
            time_bins=config.time_bins,
            dwell_bins=config.dwell_bins,
            transition_types=config.transition_types,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
        )
        self.trajectory_encoder = TrajectoryEncoder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dropout=config.dropout,
            max_len=config.max_len,
        )
        self.user_history = UserHistoryModule(
            d_model=config.d_model,
            n_prototypes=config.n_prototypes,
            n_deviation_features=config.n_deviation_features,
        )
        self.primitive_head = PrimitiveHead(
            d_model=config.d_model,
            n_primitives=config.n_primitives,
        )
        self.definition_encoder = DefinitionEncoder(
            text_encoder_name=config.text_encoder_name,
            d_model=config.d_model,
            n_primitives=config.n_primitives,
        )

        # Learnable scoring matrices
        self.W = nn.Parameter(torch.empty(config.d_model, config.d_model))
        self.U = nn.Parameter(torch.empty(config.d_model, config.d_model))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.U)

        self.lambda_prim = config.lambda_prim
        self.gamma_loc = config.gamma_loc

    # ------------------------------------------------------------------
    @staticmethod
    def build_declared_primitive_matrix(
        concept_ids: List[int], n_primitives: int = 10,
    ) -> torch.Tensor:
        """Build a (K, P) binary matrix of declared primitives for K concepts.

        Each row corresponds to a concept (ordered by concept_ids) and has 1s
        at the indices of its declared primitives from core/concepts.py.
        """
        mat = torch.zeros(len(concept_ids), n_primitives)
        for i, cid in enumerate(concept_ids):
            if cid in CONCEPT_BY_ID:
                for p in CONCEPT_BY_ID[cid]["primitives"]:
                    mat[i, p] = 1.0
        return mat

    # ------------------------------------------------------------------
    def _compute_concept_scores(
        self,
        z_x: torch.Tensor,
        h_i: torch.Tensor,
        v_x: torch.Tensor,
        c_d: torch.Tensor,
        a_d: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        declared_primitives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute concept matching scores.

        s_k = z_x^T W c_d  +  lambda * a_d^T v_x  +  gamma * max_i(h_i^T U c_d)
              + delta * v_x @ declared_primitives^T   (when available)

        Parameters
        ----------
        z_x : (B, D)
        h_i : (B, L, D)
        v_x : (B, P)
        c_d : (B_d, D)
        a_d : (B_d, P)
        mask : (B, L) optional
        declared_primitives : (B_d, P) optional
            Binary matrix of declared primitives per concept from concepts.py.
            When provided, adds an explicit primitive-matching bonus.

        Returns
        -------
        scores : (B, B_d)
        """
        # Term 1: global bilinear  z_x^T W c_d  → (B, B_d)
        z_W = z_x @ self.W                                 # (B, D)
        term1 = z_W @ c_d.t()                              # (B, B_d)

        # Term 2: primitive alignment  lambda * a_d^T v_x  → (B, B_d)
        # v_x: (B, P),  a_d: (B_d, P)  →  (B, B_d) via v_x @ a_d^T
        term2 = self.lambda_prim * (v_x @ a_d.t())         # (B, B_d)

        # Term 3: localised evidence  gamma * max_i h_i^T U c_d
        h_U = h_i @ self.U                                 # (B, L, D)
        local_scores = torch.einsum("bld,kd->blk", h_U, c_d)  # (B, L, B_d)

        if mask is not None:
            # mask here is True=valid (from caller); fill padded positions (False=padded → ~mask)
            local_scores = local_scores.masked_fill(
                ~mask.unsqueeze(-1), float("-inf")
            )

        term3 = self.gamma_loc * local_scores.max(dim=1).values  # (B, B_d)

        scores = term1 + term2 + term3

        # Term 4 (optional): explicit primitive matching bonus
        # Uses declared primitives from concepts.py instead of learned a_d.
        # This is crucial for zero-shot concepts where the learned a_d may
        # not generalize, but the declared primitives are ground truth.
        if declared_primitives is not None:
            dp = declared_primitives.to(v_x.device)
            term4 = self.lambda_prim * (v_x @ dp.t())      # (B, B_d)
            scores = scores + term4

        return scores

    # ------------------------------------------------------------------
    def forward(
        self,
        episodes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor],
        user_prototypes: Dict[str, torch.Tensor],
        definition_texts: Union[List[str], torch.Tensor],
        declared_primitives: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Parameters
        ----------
        episodes : dict[str, Tensor]
            Raw episode features (see :class:`EpisodeEncoder`).
        mask : Tensor | None
            Padding mask ``(B, L)``, ``True`` = padded.
        user_prototypes : dict
            Batched prototype parameters (``mu``, ``sigma``, ``pi``).
        definition_texts : list[str] | Tensor
            Anomaly concept definitions.
        declared_primitives : Tensor | None
            ``(B_d, P)`` binary matrix of declared primitives per concept.
            When provided, adds explicit primitive-matching bonus to scoring.

        Returns
        -------
        dict with keys:
            ``concept_scores`` (B, B_d), ``E_norm`` (B,),
            ``v_x`` (B, P), ``deviation_features`` (B, 5),
            ``episode_attention`` (B, P, L).
        """
        # 1. Encode episodes
        ep_emb = self.episode_encoder(episodes)          # (B, L, H)
        # mask from caller is True=valid; sub-modules expect True=padded → negate
        pad_mask = ~mask if mask is not None else None
        z_x, h_i = self.trajectory_encoder(ep_emb, pad_mask) # (B, D), (B, L, D)

        # 2. User history
        E_norm, deviation_features = self.user_history(z_x, user_prototypes)

        # 3. Primitive violations (also expects True=padded)
        v_x, episode_attention = self.primitive_head(h_i, pad_mask)

        # 4. Encode definitions
        c_d, a_d = self.definition_encoder(definition_texts)

        # 5. Concept scores
        concept_scores = self._compute_concept_scores(
            z_x, h_i, v_x, c_d, a_d, mask, declared_primitives
        )

        return {
            "concept_scores": concept_scores,
            "E_norm": E_norm,
            "v_x": v_x,
            "deviation_features": deviation_features,
            "episode_attention": episode_attention,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        episodes: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor],
        user_prototypes: Dict[str, torch.Tensor],
        definition_texts: Union[List[str], torch.Tensor],
        norm_threshold: float,
        conformal_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Two-stage inference with conformal guarantees.

        Stage A
            If ``E_norm <= norm_threshold`` the trip is classified as
            **normal** (no anomaly).

        Stage B
            Compute conformal p-values from concept scores.  If any
            concept's p-value exceeds ``conformal_threshold`` the trip is
            a **known anomaly** (return top-1 or prediction set).
            Otherwise it is an **unknown anomaly**.

        Parameters
        ----------
        episodes, mask, user_prototypes, definition_texts
            Same as :meth:`forward`.
        norm_threshold : float
            Energy threshold from conformal calibration on normal trips.
        conformal_threshold : float
            Conformal p-value threshold for concept acceptance.

        Returns
        -------
        list[dict]
            One prediction per sample with keys ``label``, ``concept_scores``,
            ``prediction_set``, ``E_norm``.
        """
        outputs = self.forward(episodes, mask, user_prototypes, definition_texts)
        E_norm = outputs["E_norm"]                         # (B,)
        concept_scores = outputs["concept_scores"]         # (B, B_d)
        B = E_norm.shape[0]

        results: List[Dict[str, Any]] = []

        for i in range(B):
            energy = E_norm[i].item()
            scores = concept_scores[i]                     # (B_d,)

            if energy <= norm_threshold:
                # Stage A: normal
                results.append({
                    "label": "normal",
                    "concept_scores": scores.cpu(),
                    "prediction_set": [],
                    "E_norm": energy,
                })
                continue

            # Stage B: compute conformal p-values (higher score = more anomalous)
            # p-value approximated as sigmoid of centred score
            p_values = torch.sigmoid(scores)                # (B_d,)
            accepted = (p_values >= conformal_threshold).nonzero(as_tuple=False)

            if accepted.numel() > 0:
                pred_set = accepted.squeeze(-1).cpu().tolist()
                if isinstance(pred_set, int):
                    pred_set = [pred_set]
                top1 = scores.argmax().item()
                results.append({
                    "label": "known_anomaly",
                    "concept_scores": scores.cpu(),
                    "prediction_set": pred_set,
                    "top1_concept": top1,
                    "E_norm": energy,
                })
            else:
                # No concept passes threshold → unknown anomaly
                results.append({
                    "label": "unknown_anomaly",
                    "concept_scores": scores.cpu(),
                    "prediction_set": [],
                    "E_norm": energy,
                })

        return results
