"""Loss functions for LangTraj-OSR training.

Provides individual loss components and a combined loss that orchestrates
them with configurable weights.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """L_pair — Contrastive loss over (trip, definition) matches.

    Treats each trip-definition pair in the batch as a positive and all
    other combinations as negatives (symmetric InfoNCE).

    Parameters
    ----------
    temperature : float
        Softmax temperature for the contrastive logits.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, z_x: torch.Tensor, c_d: torch.Tensor
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE.

        Parameters
        ----------
        z_x : Tensor (B, D)
            Trip embeddings (L2-normalised inside this method).
        c_d : Tensor (B, D)
            Matched concept embeddings (same batch ordering).

        Returns
        -------
        Tensor  scalar loss.
        """
        # Force float32 to prevent overflow in mixed-precision training.
        z_x = F.normalize(z_x.float(), dim=-1)
        c_d = F.normalize(c_d.float(), dim=-1)

        logits = z_x @ c_d.t() / self.temperature       # (B, B)
        labels = torch.arange(z_x.size(0), device=z_x.device)

        loss_zc = F.cross_entropy(logits, labels)
        loss_cz = F.cross_entropy(logits.t(), labels)
        return (loss_zc + loss_cz) / 2.0


class ClassificationLoss(nn.Module):
    """L_cls — Cross-entropy for seen concept classification.

    Applied when the ground-truth concept label is known.
    """

    def forward(
        self,
        concept_scores: torch.Tensor,
        concept_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        concept_scores : Tensor (B, K)
            Raw concept matching scores.
        concept_labels : Tensor (B,)
            Ground-truth concept indices.
        """
        return F.cross_entropy(concept_scores, concept_labels)


class PrimitiveLoss(nn.Module):
    """L_prim — Binary cross-entropy on primitive labels.

    Each primitive is an independent binary indicator.
    """

    def forward(
        self,
        v_x: torch.Tensor,
        primitive_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        v_x : Tensor (B, P)
            Predicted primitive violation probabilities.
        primitive_labels : Tensor (B, P)
            Ground-truth binary primitive indicators.
        """
        # binary_cross_entropy is unsafe inside autocast; explicitly cast to float32
        # and wrap in a no-autocast context to avoid the PyTorch 2.x runtime error.
        with torch.amp.autocast("cuda", enabled=False):
            # Clamp to valid range to guard against float16 NaN propagation
            # (cosine_similarity in PrimitiveHead can produce NaN when pooled
            # embeddings are near-zero early in training).
            v_safe = v_x.float().nan_to_num(nan=0.5).clamp(1e-7, 1 - 1e-7)
            return F.binary_cross_entropy(v_safe, primitive_labels.float())


class ParaphraseConsistencyLoss(nn.Module):
    """L_para — KL divergence between concept embeddings of paraphrases.

    Encourages the model to produce similar concept-score distributions
    for semantically equivalent definitions of the same concept.
    """

    def forward(
        self,
        scores_a: torch.Tensor,
        scores_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        scores_a : Tensor (B, K)
            Concept scores using phrasing A.
        scores_b : Tensor (B, K)
            Concept scores using phrasing B (same concepts, different words).
        """
        log_p = F.log_softmax(scores_a, dim=-1)
        q = F.softmax(scores_b, dim=-1)
        kl_ab = F.kl_div(log_p, q, reduction="batchmean")

        log_q = F.log_softmax(scores_b, dim=-1)
        p = F.softmax(scores_a, dim=-1)
        kl_ba = F.kl_div(log_q, p, reduction="batchmean")

        return (kl_ab + kl_ba) / 2.0


class OrthogonalityLoss(nn.Module):
    """L_orth — Decorrelate deviation features from language evidence.

    Minimises the squared Frobenius norm of the cross-correlation matrix
    between the deviation features and the concept scores, encouraging
    complementary information.
    """

    def forward(
        self,
        deviation_features: torch.Tensor,
        concept_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        deviation_features : Tensor (B, F)
        concept_scores : Tensor (B, K)
        """
        # Centre
        dev = deviation_features - deviation_features.mean(dim=0, keepdim=True)
        cs = concept_scores - concept_scores.mean(dim=0, keepdim=True)

        # Cross-correlation  (F, K)
        B = dev.shape[0]
        cross_corr = (dev.t() @ cs) / max(B, 1)

        return cross_corr.pow(2).sum()


class NormalityLoss(nn.Module):
    """L_norm — Negative log-likelihood for the normality model.

    Encourages normal trips to have low energy and anomalous trips to have
    high energy via a margin-based objective.

    Parameters
    ----------
    margin : float
        Desired minimum gap between normal and anomalous energies.
    """

    def __init__(self, margin: float = 5.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        E_norm_normal: torch.Tensor,
        E_norm_anomalous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        E_norm_normal : Tensor (B_n,)
            Normality energies for known-normal trips (should be low).
        E_norm_anomalous : Tensor (B_a,), optional
            Normality energies for anomalous trips (should be high).
        """
        # Normal trips: minimise energy
        loss = E_norm_normal.mean()

        # Anomalous trips: maximise energy (hinge with margin)
        if E_norm_anomalous is not None and E_norm_anomalous.numel() > 0:
            hinge = F.relu(self.margin - E_norm_anomalous)
            loss = loss + hinge.mean()

        return loss


class CombinedLoss(nn.Module):
    """Combined training loss for LangTraj-OSR.

    L = L_pair + w_cls * L_cls + w_prim * L_prim
      + w_para * L_para + w_orth * L_orth + w_norm * L_norm

    Parameters
    ----------
    w_cls : float
        Weight for the classification loss.
    w_prim : float
        Weight for the primitive loss.
    w_para : float
        Weight for the paraphrase consistency loss.
    w_orth : float
        Weight for the orthogonality loss.
    w_norm : float
        Weight for the normality loss.
    """

    def __init__(
        self,
        w_cls: float = 0.5,
        w_prim: float = 1.0,
        w_para: float = 0.2,
        w_orth: float = 0.05,
        w_norm: float = 1.0,
        temperature: float = 0.07,
        normality_margin: float = 5.0,
    ) -> None:
        super().__init__()
        self.w_cls = w_cls
        self.w_prim = w_prim
        self.w_para = w_para
        self.w_orth = w_orth
        self.w_norm = w_norm

        self.infonce = InfoNCELoss(temperature=temperature)
        self.cls_loss = ClassificationLoss()
        self.prim_loss = PrimitiveLoss()
        self.para_loss = ParaphraseConsistencyLoss()
        self.orth_loss = OrthogonalityLoss()
        self.norm_loss = NormalityLoss(margin=normality_margin)

    def forward(
        self,
        z_x: torch.Tensor,
        c_d: torch.Tensor,
        concept_scores: torch.Tensor,
        concept_labels: torch.Tensor,
        v_x: torch.Tensor,
        primitive_labels: torch.Tensor,
        deviation_features: torch.Tensor,
        E_norm_normal: torch.Tensor,
        E_norm_anomalous: Optional[torch.Tensor] = None,
        scores_para_a: Optional[torch.Tensor] = None,
        scores_para_b: Optional[torch.Tensor] = None,
        concept_scores_orth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined loss and return a breakdown.

        Returns
        -------
        dict
            ``total`` — scalar combined loss.
            ``L_pair``, ``L_cls``, ``L_prim``, ``L_para``, ``L_orth``,
            ``L_norm`` — individual components for logging.
        """
        L_pair = self.infonce(z_x, c_d)
        L_cls = self.cls_loss(concept_scores, concept_labels)
        L_prim = self.prim_loss(v_x, primitive_labels)
        # Use full-batch concept_scores for orth loss so batch dims match deviation_features.
        # concept_scores may be filtered to seen-only for L_cls; concept_scores_orth is full batch.
        _orth_cs = concept_scores_orth if concept_scores_orth is not None else concept_scores
        L_orth = self.orth_loss(deviation_features, _orth_cs)
        # Skip L_norm entirely when w_norm=0 to avoid 0*NaN=NaN from unstable E_norm.
        if self.w_norm > 0:
            L_norm = self.norm_loss(E_norm_normal, E_norm_anomalous)
        else:
            L_norm = torch.zeros(1, device=z_x.device, dtype=z_x.dtype)

        # Paraphrase loss is optional (only when paraphrase pairs available)
        if scores_para_a is not None and scores_para_b is not None:
            L_para = self.para_loss(scores_para_a, scores_para_b)
        else:
            L_para = torch.tensor(0.0, device=z_x.device)

        total = (
            L_pair
            + self.w_cls * L_cls
            + self.w_prim * L_prim
            + self.w_para * L_para
            + self.w_orth * L_orth
            + self.w_norm * L_norm
        )

        return {
            "total": total,
            "L_pair": L_pair.detach(),
            "L_cls": L_cls.detach(),
            "L_prim": L_prim.detach(),
            "L_para": L_para.detach(),
            "L_orth": L_orth.detach(),
            "L_norm": L_norm.detach(),
        }
