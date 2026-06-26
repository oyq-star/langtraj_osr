"""Shared utilities for LangTraj-OSR.

Includes seed management, metric computation, result I/O, early stopping,
running-average tracking, and logging helpers.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from sklearn import metrics as sk_metrics


# ============================================================================
# Reproducibility
# ============================================================================


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================================
# Result I/O
# ============================================================================


def save_results(results_dict: Dict[str, Any], path: Union[str, Path]) -> None:
    """Serialise *results_dict* to a JSON file at *path*.

    Automatically creates parent directories.  NumPy and PyTorch scalar
    types are converted to native Python types for JSON compatibility.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_make_json_safe(results_dict), f, indent=2)


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a results dictionary previously saved with :func:`save_results`."""
    with open(path, "r") as f:
        return json.load(f)


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy / torch scalars to Python primitives."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ============================================================================
# Standard classification metrics
# ============================================================================


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute binary detection metrics.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Binary ground-truth labels (0 = normal, 1 = anomaly).
    y_score : array-like of shape (N,)
        Continuous anomaly scores (higher = more anomalous).
    y_pred : array-like of shape (N,) or None
        Hard binary predictions.  When *None*, predictions are derived
        from *y_score* using a threshold that yields 95 % TPR.

    Returns
    -------
    dict
        ``auroc``, ``auprc``, ``fpr_at_95tpr``, and, when *y_pred* is
        available, ``precision``, ``recall``, ``f1``.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    results: Dict[str, float] = {}

    # AUROC
    if len(np.unique(y_true)) < 2:
        results["auroc"] = float("nan")
    else:
        results["auroc"] = float(sk_metrics.roc_auc_score(y_true, y_score))

    # AUPRC (average precision)
    if len(np.unique(y_true)) < 2:
        results["auprc"] = float("nan")
    else:
        results["auprc"] = float(sk_metrics.average_precision_score(y_true, y_score))

    # FPR @ 95 % TPR
    results["fpr_at_95tpr"] = _fpr_at_tpr(y_true, y_score, target_tpr=0.95)

    # Hard-prediction metrics
    if y_pred is None:
        # Derive from threshold at 95 % TPR
        y_pred = _threshold_at_tpr(y_true, y_score, target_tpr=0.95)

    y_pred = np.asarray(y_pred, dtype=np.int32)
    results["precision"] = float(
        sk_metrics.precision_score(y_true, y_pred, zero_division=0)
    )
    results["recall"] = float(
        sk_metrics.recall_score(y_true, y_pred, zero_division=0)
    )
    results["f1"] = float(sk_metrics.f1_score(y_true, y_pred, zero_division=0))

    return results


def _fpr_at_tpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """Compute the false-positive rate at a given true-positive rate."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_score)
    # Find the largest FPR where TPR >= target_tpr
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def _threshold_at_tpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_tpr: float = 0.95,
) -> np.ndarray:
    """Return hard predictions using the threshold that achieves *target_tpr*."""
    if len(np.unique(y_true)) < 2:
        return np.zeros_like(y_true)
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_score)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        thresh = float(y_score.min()) - 1.0
    else:
        thresh = float(thresholds[idx[0]])
    return (y_score >= thresh).astype(np.int32)


# ============================================================================
# Open-set recognition metrics
# ============================================================================


def compute_open_set_metrics(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute open-set recognition metrics.

    Parameters
    ----------
    known_scores : array-like of shape (N_known,)
        Anomaly / uncertainty scores for known-class test samples.
    unknown_scores : array-like of shape (N_unknown,)
        Anomaly / uncertainty scores for unknown-class test samples.

    Returns
    -------
    dict
        ``h_score``, ``oscr_auc``, ``unknown_rejection_auroc``.
    """
    known_scores = np.asarray(known_scores, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)

    results: Dict[str, float] = {}

    # --- Unknown-rejection AUROC: can we separate known from unknown? ---
    y_true = np.concatenate(
        [np.zeros(len(known_scores)), np.ones(len(unknown_scores))]
    )
    y_score = np.concatenate([known_scores, unknown_scores])
    if len(np.unique(y_true)) < 2:
        results["unknown_rejection_auroc"] = float("nan")
    else:
        results["unknown_rejection_auroc"] = float(
            sk_metrics.roc_auc_score(y_true, y_score)
        )

    # --- H-score (harmonic mean of known accuracy and unknown rejection) ---
    # Use threshold at 95 % unknown-rejection rate.
    if len(unknown_scores) > 0:
        thresh_95 = np.percentile(unknown_scores, 5)  # reject 95 % of unknowns
        known_acc = float(np.mean(known_scores < thresh_95))
        unknown_rej = 0.95
    else:
        known_acc = 1.0
        unknown_rej = 1.0
    if known_acc + unknown_rej > 0:
        results["h_score"] = float(
            2 * known_acc * unknown_rej / (known_acc + unknown_rej)
        )
    else:
        results["h_score"] = 0.0

    # --- OSCR (Open-Set Classification Rate) AUC ---
    results["oscr_auc"] = _compute_oscr_auc(known_scores, unknown_scores)

    return results


def _compute_oscr_auc(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    num_thresholds: int = 1000,
) -> float:
    """Compute the area under the OSCR curve.

    The OSCR curve plots *correct classification rate* (CCR) of known
    samples against *false positive rate* (FPR) of unknown samples as
    the decision threshold varies.
    """
    all_scores = np.concatenate([known_scores, unknown_scores])
    if len(all_scores) == 0:
        return 0.0

    thresholds = np.linspace(
        float(all_scores.min()), float(all_scores.max()), num_thresholds
    )

    ccrs: List[float] = []
    fprs: List[float] = []

    n_known = len(known_scores)
    n_unknown = len(unknown_scores)

    for t in thresholds:
        # Known samples correctly accepted (score below threshold)
        ccr = float(np.sum(known_scores < t)) / max(n_known, 1)
        # Unknown samples incorrectly accepted (score below threshold)
        fpr = float(np.sum(unknown_scores < t)) / max(n_unknown, 1)
        ccrs.append(ccr)
        fprs.append(fpr)

    # AUC via trapezoidal rule (FPR on x-axis, CCR on y-axis)
    fprs_arr = np.array(fprs)
    ccrs_arr = np.array(ccrs)

    # Sort by FPR
    order = np.argsort(fprs_arr)
    fprs_arr = fprs_arr[order]
    ccrs_arr = ccrs_arr[order]

    return float(np.trapz(ccrs_arr, fprs_arr))


# ============================================================================
# Calibration metrics
# ============================================================================


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Compute probability calibration metrics.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Binary ground-truth labels.
    y_prob : array-like of shape (N,)
        Predicted probabilities of the positive class.
    n_bins : int
        Number of bins for ECE computation.

    Returns
    -------
    dict
        ``ece`` (expected calibration error), ``brier`` (Brier score),
        ``nll`` (negative log-likelihood).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64).clip(1e-15, 1.0 - 1e-15)

    results: Dict[str, float] = {}

    # Brier score
    results["brier"] = float(np.mean((y_prob - y_true) ** 2))

    # Negative log-likelihood (binary cross-entropy)
    nll = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    results["nll"] = float(np.mean(nll))

    # Expected Calibration Error
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        bin_count = int(mask.sum())
        if bin_count == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece += (bin_count / total) * abs(bin_acc - bin_conf)
    results["ece"] = float(ece)

    return results


# ============================================================================
# Early stopping
# ============================================================================


class EarlyStopping:
    """Early stopping tracker that monitors a validation metric.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after the last improvement.
    min_delta : float
        Minimum absolute change to qualify as an improvement.
    mode : str
        ``'min'`` (lower is better, e.g. loss) or ``'max'`` (higher is
        better, e.g. AUROC).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_epoch: int = 0

    def __call__(self, score: float, epoch: int = 0) -> bool:
        """Update with the latest validation score.

        Returns
        -------
        bool
            ``True`` if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        improved = (
            (score < self.best_score - self.min_delta)
            if self.mode == "min"
            else (score > self.best_score + self.min_delta)
        )

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        """Reset the tracker to its initial state."""
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0


# ============================================================================
# Running average meter
# ============================================================================


class AverageMeter:
    """Computes and stores a running average and current value.

    Useful for tracking training loss / metric across mini-batches.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Record a new value with batch size *n*."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter({self.name}: avg={self.avg:.4f}, count={self.count})"


# ============================================================================
# Logging
# ============================================================================


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with a console handler and standard formatting.

    Avoids duplicate handlers when called multiple times with the same name.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
