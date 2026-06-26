"""Conformal calibration for LangTraj-OSR.

Provides distribution-free coverage guarantees for both the normality gate
(Stage A) and the concept classification (Stage B).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class ConformalCalibrator:
    """Split-conformal calibrator for two-stage anomaly detection.

    Usage
    -----
    1. Call :meth:`fit_normality` on a calibration set of normal-trip
       energies to obtain the normality threshold ``q_norm``.
    2. Call :meth:`fit_concepts` on a calibration set of concept scores
       with ground-truth labels to obtain per-concept thresholds.
    3. At test time, call :meth:`predict` or :meth:`prediction_set`.

    Attributes
    ----------
    q_norm : float | None
        Normality energy threshold (trips with E <= q_norm are normal).
    concept_thresholds : dict[int, float]
        Per-concept conformity score thresholds.
    alpha_norm : float
        Miscoverage level for the normality test.
    alpha_concept : float
        Miscoverage level for concept prediction sets.
    """

    q_norm: Optional[float] = None
    concept_thresholds: Dict[int, float] = field(default_factory=dict)
    alpha_norm: float = 0.05
    alpha_concept: float = 0.10

    # ------------------------------------------------------------------
    def fit_normality(
        self,
        normal_energies: np.ndarray,
        alpha: float = 0.05,
    ) -> float:
        """Compute the normality threshold from calibration energies.

        Uses the standard split-conformal quantile:
        ``q_norm = ceil((1-alpha)(n+1)) / n``-th order statistic of the
        calibration energies.

        Parameters
        ----------
        normal_energies : ndarray (N,)
            Normality energies of known-normal calibration trips.
        alpha : float
            Desired miscoverage level (e.g. 0.05 for 95 % coverage).

        Returns
        -------
        float
            The normality threshold ``q_norm``.
        """
        self.alpha_norm = alpha
        n = len(normal_energies)
        if n == 0:
            raise ValueError("Calibration set must be non-empty.")

        sorted_energies = np.sort(normal_energies)
        # Quantile index with finite-sample correction
        idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
        idx = min(max(idx, 0), n - 1)

        self.q_norm = float(sorted_energies[idx])
        return self.q_norm

    # ------------------------------------------------------------------
    def fit_concepts(
        self,
        concept_scores: np.ndarray,
        concept_labels: np.ndarray,
        alpha: float = 0.10,
    ) -> Dict[int, float]:
        """Compute per-concept conformal thresholds.

        For each concept *k* we collect the conformity scores of all
        calibration examples whose true label is *k* and take the
        appropriate quantile as the threshold.

        The **conformity score** for a sample is defined as
        ``1 - softmax(concept_scores)[true_label]``, i.e. a high
        conformity score means the model is *less* confident about the
        correct label.

        Parameters
        ----------
        concept_scores : ndarray (N, K)
            Raw concept matching scores for calibration samples.
        concept_labels : ndarray (N,)
            Ground-truth concept indices.
        alpha : float
            Desired miscoverage level.

        Returns
        -------
        dict[int, float]
            Per-concept thresholds.
        """
        self.alpha_concept = alpha
        N, K = concept_scores.shape

        # Softmax over concepts
        exp_scores = np.exp(concept_scores - concept_scores.max(axis=1, keepdims=True))
        softmax_scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        self.concept_thresholds = {}
        unique_labels = np.unique(concept_labels)

        for k in unique_labels:
            k = int(k)
            mask = concept_labels == k
            conformity = 1.0 - softmax_scores[mask, k]
            n_k = len(conformity)
            if n_k == 0:
                continue
            sorted_conf = np.sort(conformity)
            idx = int(np.ceil((1 - alpha) * (n_k + 1))) - 1
            idx = min(max(idx, 0), n_k - 1)
            self.concept_thresholds[k] = float(sorted_conf[idx])

        return self.concept_thresholds

    # ------------------------------------------------------------------
    def predict(
        self,
        E_norm: np.ndarray,
        concept_scores: np.ndarray,
    ) -> List[Dict]:
        """Two-stage prediction with coverage guarantees.

        Parameters
        ----------
        E_norm : ndarray (B,)
            Normality energies.
        concept_scores : ndarray (B, K)
            Raw concept matching scores.

        Returns
        -------
        list[dict]
            One entry per sample with keys:
            ``label`` in {``"normal"``, ``"known_anomaly"``,
            ``"unknown_anomaly"``},
            ``prediction_set``, ``top1_concept``, ``E_norm``.
        """
        if self.q_norm is None:
            raise RuntimeError("Call fit_normality() before predict().")

        B, K = concept_scores.shape
        results: List[Dict] = []

        # Softmax
        exp_scores = np.exp(concept_scores - concept_scores.max(axis=1, keepdims=True))
        softmax_scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        for i in range(B):
            energy = float(E_norm[i])

            # Stage A: normality gating
            if energy <= self.q_norm:
                results.append({
                    "label": "normal",
                    "prediction_set": [],
                    "top1_concept": None,
                    "E_norm": energy,
                })
                continue

            # Stage B: conformal concept prediction
            pred_set = self.prediction_set(concept_scores[i : i + 1])[0]

            if len(pred_set) > 0:
                top1 = int(np.argmax(concept_scores[i]))
                results.append({
                    "label": "known_anomaly",
                    "prediction_set": sorted(pred_set),
                    "top1_concept": top1,
                    "E_norm": energy,
                })
            else:
                results.append({
                    "label": "unknown_anomaly",
                    "prediction_set": [],
                    "top1_concept": None,
                    "E_norm": energy,
                })

        return results

    # ------------------------------------------------------------------
    def prediction_set(
        self,
        concept_scores: np.ndarray,
    ) -> List[Set[int]]:
        """Compute conformal prediction sets.

        For each sample, include concept *k* in the prediction set if its
        conformity score ``1 - softmax_score[k]`` is at most the
        calibrated threshold for concept *k*.

        Parameters
        ----------
        concept_scores : ndarray (B, K)

        Returns
        -------
        list[set[int]]
            One set of plausible concept indices per sample.
        """
        if not self.concept_thresholds:
            raise RuntimeError("Call fit_concepts() before prediction_set().")

        B, K = concept_scores.shape

        exp_scores = np.exp(concept_scores - concept_scores.max(axis=1, keepdims=True))
        softmax_scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        sets: List[Set[int]] = []
        for i in range(B):
            pred: Set[int] = set()
            for k, threshold in self.concept_thresholds.items():
                conformity = 1.0 - softmax_scores[i, k]
                if conformity <= threshold:
                    pred.add(k)
            sets.append(pred)

        return sets

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist calibrator state to a JSON file."""
        state = {
            "q_norm": self.q_norm,
            "concept_thresholds": {str(k): v for k, v in self.concept_thresholds.items()},
            "alpha_norm": self.alpha_norm,
            "alpha_concept": self.alpha_concept,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str) -> None:
        """Restore calibrator state from a JSON file."""
        with open(path, "r") as f:
            state = json.load(f)
        self.q_norm = state.get("q_norm")
        self.concept_thresholds = {int(k): v for k, v in state.get("concept_thresholds", {}).items()}
        self.alpha_norm = state.get("alpha_norm", 0.05)
        self.alpha_concept = state.get("alpha_concept", 0.10)
