"""Complete metrics computation for LangTraj-OSR evaluation.

All metrics from the experiment plan:
- Detection: AUROC, AUPRC, FPR@95TPR
- Open-set: H-score, OSCR curve, unknown-rejection AUROC
- Concept: top-1 accuracy, macro F1, mean average precision, recall@k
- Calibration: ECE, Brier score, NLL, risk-coverage curve, conformal set size
- Localization: span IoU, pointing accuracy
- Robustness: worst paraphrase gap, city transfer drop, history-stratified metrics
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def auroc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area Under the Receiver Operating Characteristic curve.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground truth (0 = normal, 1 = anomalous).
    y_score : array-like of shape (n,)
        Continuous anomaly scores (higher = more anomalous).

    Returns
    -------
    float
        AUROC value in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    # Sort by descending score
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    n_pos = y_true_sorted.sum()
    n_neg = len(y_true_sorted) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    return float(np.trapz(tpr, fpr))


def auprc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area Under the Precision-Recall Curve.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground truth.
    y_score : array-like of shape (n,)
        Continuous anomaly scores.

    Returns
    -------
    float
        AUPRC value in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    n_pos = y_true_sorted.sum()
    if n_pos == 0:
        return float("nan")

    tps = np.cumsum(y_true_sorted)
    total = np.arange(1, len(y_true_sorted) + 1, dtype=np.float64)

    precision = tps / total
    recall = tps / n_pos

    # Append sentinel
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    return float(np.trapz(precision, recall))


def fpr_at_tpr(
    y_true: ArrayLike, y_score: ArrayLike, target_tpr: float = 0.95
) -> float:
    """False Positive Rate at a given True Positive Rate threshold.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground truth.
    y_score : array-like of shape (n,)
        Continuous anomaly scores.
    target_tpr : float
        The TPR level at which to report FPR (default 0.95).

    Returns
    -------
    float
        FPR@{target_tpr}TPR.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    thresholds = np.sort(np.unique(y_score))[::-1]

    for thresh in thresholds:
        predicted_pos = y_score >= thresh
        tp = (predicted_pos & (y_true == 1)).sum()
        fp = (predicted_pos & (y_true == 0)).sum()
        tpr = tp / n_pos
        if tpr >= target_tpr:
            return float(fp / n_neg)

    return 1.0


# ---------------------------------------------------------------------------
# Open-set recognition metrics
# ---------------------------------------------------------------------------

def h_score(
    known_accuracy: float, unknown_rejection_rate: float
) -> float:
    """Harmonic mean of known-class accuracy and unknown-rejection rate.

    Parameters
    ----------
    known_accuracy : float
        Accuracy on known (seen) classes.
    unknown_rejection_rate : float
        Rate at which unknown samples are correctly rejected.

    Returns
    -------
    float
        H-score (harmonic mean).
    """
    if known_accuracy + unknown_rejection_rate == 0:
        return 0.0
    return float(
        2.0 * known_accuracy * unknown_rejection_rate
        / (known_accuracy + unknown_rejection_rate)
    )


def oscr_curve(
    known_scores: ArrayLike,
    known_labels: ArrayLike,
    unknown_scores: ArrayLike,
    n_thresholds: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Open-Set Classification Rate (OSCR) curve.

    Sweeps a rejection threshold over anomaly scores and computes the
    correct classification rate (CCR) at each false positive rate (FPR).

    Parameters
    ----------
    known_scores : array-like of shape (n_known,)
        Max softmax or concept scores for known-class samples.
    known_labels : array-like of shape (n_known,)
        Predicted class labels for known samples (1 if correct, 0 otherwise).
    unknown_scores : array-like of shape (n_unknown,)
        Max softmax or concept scores for unknown-class samples.
    n_thresholds : int
        Number of threshold steps.

    Returns
    -------
    fpr : ndarray of shape (n_thresholds,)
        False positive rates (unknown samples accepted).
    ccr : ndarray of shape (n_thresholds,)
        Correct classification rates among accepted known samples.
    """
    known_scores = np.asarray(known_scores, dtype=np.float64)
    known_labels = np.asarray(known_labels, dtype=np.int32)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)

    n_unknown = len(unknown_scores)
    n_known = len(known_scores)

    if n_unknown == 0 or n_known == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    all_scores = np.concatenate([known_scores, unknown_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n_thresholds)

    fpr_arr = np.zeros(n_thresholds)
    ccr_arr = np.zeros(n_thresholds)

    for i, t in enumerate(thresholds):
        # FPR: fraction of unknowns accepted (score >= threshold)
        fpr_arr[i] = (unknown_scores >= t).sum() / n_unknown

        # CCR: fraction of knowns correctly classified AND accepted
        accepted_known = known_scores >= t
        if accepted_known.sum() > 0:
            ccr_arr[i] = (known_labels[accepted_known] == 1).sum() / n_known
        else:
            ccr_arr[i] = 0.0

    return fpr_arr, ccr_arr


def unknown_rejection_auroc(
    known_scores: ArrayLike, unknown_scores: ArrayLike
) -> float:
    """AUROC for distinguishing known vs. unknown samples by their scores.

    Parameters
    ----------
    known_scores : array-like of shape (n_known,)
        Confidence scores for known-class samples.
    unknown_scores : array-like of shape (n_unknown,)
        Confidence scores for unknown-class samples.

    Returns
    -------
    float
        AUROC for the known-vs-unknown discrimination task.
    """
    known_scores = np.asarray(known_scores, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)

    y_true = np.concatenate([
        np.ones(len(known_scores)),
        np.zeros(len(unknown_scores)),
    ])
    y_score = np.concatenate([known_scores, unknown_scores])

    return auroc(y_true, y_score)


# ---------------------------------------------------------------------------
# Concept classification metrics
# ---------------------------------------------------------------------------

def top1_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Top-1 classification accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground truth class labels.
    y_pred : array-like of shape (n,)
        Predicted class labels.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Macro-averaged F1 score across all classes.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground truth class labels.
    y_pred : array-like of shape (n,)
        Predicted class labels.

    Returns
    -------
    float
        Macro F1 in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1_scores: List[float] = []

    for cls in classes:
        tp = ((y_pred == cls) & (y_true == cls)).sum()
        fp = ((y_pred == cls) & (y_true != cls)).sum()
        fn = ((y_pred != cls) & (y_true == cls)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1_scores.append(2.0 * precision * recall / (precision + recall))
        else:
            f1_scores.append(0.0)

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def mean_average_precision(
    y_true: ArrayLike, y_score: ArrayLike, n_classes: Optional[int] = None
) -> float:
    """Mean Average Precision across all classes.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground truth class labels.
    y_score : array-like of shape (n, C)
        Per-class score matrix.
    n_classes : int, optional
        Number of classes. Inferred from y_score if not provided.

    Returns
    -------
    float
        mAP in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)

    if n_classes is None:
        n_classes = y_score.shape[1] if y_score.ndim == 2 else int(y_true.max()) + 1

    ap_list: List[float] = []
    for cls in range(n_classes):
        binary_true = (y_true == cls).astype(np.int32)
        if binary_true.sum() == 0:
            continue
        cls_score = y_score[:, cls] if y_score.ndim == 2 else y_score
        ap_list.append(auprc(binary_true, cls_score))

    return float(np.nanmean(ap_list)) if ap_list else 0.0


def recall_at_k(
    y_true: ArrayLike, y_score: ArrayLike, k: int = 5
) -> float:
    """Recall@k: fraction of true positives in top-k predictions per sample.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground truth class labels.
    y_score : array-like of shape (n, C)
        Per-class score matrix.
    k : int
        Number of top predictions to consider.

    Returns
    -------
    float
        Recall@k in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_score.ndim != 2:
        raise ValueError("y_score must be 2-dimensional (n, C)")

    n_samples = len(y_true)
    if n_samples == 0:
        return float("nan")

    hits = 0
    for i in range(n_samples):
        top_k_classes = np.argsort(-y_score[i])[:k]
        if y_true[i] in top_k_classes:
            hits += 1

    return float(hits / n_samples)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground truth.
    y_prob : array-like of shape (n,)
        Predicted probabilities for the positive class.
    n_bins : int
        Number of equal-width bins for calibration.

    Returns
    -------
    float
        ECE value (lower is better).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    if total == 0:
        return float("nan")

    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += (n_in_bin / total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Brier score (mean squared error of probability estimates).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground truth.
    y_prob : array-like of shape (n,)
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Brier score (lower is better).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if len(y_true) == 0:
        return float("nan")

    return float(np.mean((y_prob - y_true) ** 2))


def negative_log_likelihood(
    y_true: ArrayLike, y_prob: ArrayLike, eps: float = 1e-12
) -> float:
    """Negative log-likelihood (cross-entropy loss).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground truth.
    y_prob : array-like of shape (n,)
        Predicted probabilities for the positive class.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    float
        Mean negative log-likelihood.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if len(y_true) == 0:
        return float("nan")

    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    nll = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(nll.mean())


def risk_coverage_curve(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    y_score_confidence: ArrayLike,
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve: error rate as a function of coverage.

    Samples are sorted by decreasing confidence. At each coverage level
    (fraction of samples retained), the error rate is computed.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground truth.
    y_prob : array-like of shape (n,)
        Predicted labels or probabilities.
    y_score_confidence : array-like of shape (n,)
        Confidence scores (higher = more confident).
    n_points : int
        Number of coverage steps.

    Returns
    -------
    coverage : ndarray of shape (n_points,)
    risk : ndarray of shape (n_points,)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_score_confidence = np.asarray(y_score_confidence, dtype=np.float64)

    n = len(y_true)
    if n == 0:
        return np.array([0.0]), np.array([0.0])

    # Sort by descending confidence
    order = np.argsort(-y_score_confidence)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    errors = (y_true_sorted != (y_prob_sorted >= 0.5).astype(float))
    cum_errors = np.cumsum(errors)

    coverage_levels = np.linspace(1.0 / n, 1.0, n_points)
    coverage_arr = np.zeros(n_points)
    risk_arr = np.zeros(n_points)

    for i, cov in enumerate(coverage_levels):
        k = max(1, int(np.ceil(cov * n)))
        coverage_arr[i] = k / n
        risk_arr[i] = cum_errors[k - 1] / k

    return coverage_arr, risk_arr


def conformal_set_size(
    set_sizes: ArrayLike,
) -> Dict[str, float]:
    """Summary statistics of conformal prediction set sizes.

    Parameters
    ----------
    set_sizes : array-like of shape (n,)
        Number of labels in each sample's conformal prediction set.

    Returns
    -------
    dict
        Keys: mean, median, p90, fraction_singleton, fraction_empty.
    """
    sizes = np.asarray(set_sizes, dtype=np.float64)
    if len(sizes) == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "fraction_singleton": float("nan"),
            "fraction_empty": float("nan"),
        }
    return {
        "mean": float(sizes.mean()),
        "median": float(np.median(sizes)),
        "p90": float(np.percentile(sizes, 90)),
        "fraction_singleton": float((sizes == 1).mean()),
        "fraction_empty": float((sizes == 0).mean()),
    }


# ---------------------------------------------------------------------------
# Localization metrics
# ---------------------------------------------------------------------------

def span_iou(
    pred_spans: ArrayLike, true_spans: ArrayLike
) -> float:
    """Mean Intersection-over-Union for episode-level anomaly spans.

    Each span is represented as (start_idx, end_idx) indicating which
    episodes in the trajectory are flagged as anomalous.

    Parameters
    ----------
    pred_spans : array-like of shape (n, 2)
        Predicted anomaly spans as (start, end) pairs.
    true_spans : array-like of shape (n, 2)
        Ground truth anomaly spans.

    Returns
    -------
    float
        Mean span IoU in [0, 1].
    """
    pred_spans = np.asarray(pred_spans, dtype=np.float64)
    true_spans = np.asarray(true_spans, dtype=np.float64)

    if len(pred_spans) == 0 or len(true_spans) == 0:
        return float("nan")

    if pred_spans.ndim == 1:
        pred_spans = pred_spans.reshape(1, 2)
    if true_spans.ndim == 1:
        true_spans = true_spans.reshape(1, 2)

    ious: List[float] = []
    for i in range(len(pred_spans)):
        p_start, p_end = pred_spans[i]
        t_start, t_end = true_spans[i] if i < len(true_spans) else true_spans[-1]

        inter_start = max(p_start, t_start)
        inter_end = min(p_end, t_end)
        intersection = max(0.0, inter_end - inter_start)

        union = (p_end - p_start) + (t_end - t_start) - intersection
        if union <= 0:
            ious.append(0.0)
        else:
            ious.append(intersection / union)

    return float(np.mean(ious))


def pointing_accuracy(
    pred_episode_idx: ArrayLike,
    true_episode_idx: ArrayLike,
    tolerance: int = 0,
) -> float:
    """Pointing accuracy: fraction of predictions that point to the correct
    anomalous episode (within a tolerance window).

    Parameters
    ----------
    pred_episode_idx : array-like of shape (n,)
        Predicted episode index with highest anomaly attention.
    true_episode_idx : array-like of shape (n,)
        Ground truth episode index where anomaly occurs.
    tolerance : int
        Allow predictions within +/- tolerance episodes.

    Returns
    -------
    float
        Pointing accuracy in [0, 1].
    """
    pred_idx = np.asarray(pred_episode_idx, dtype=np.int32)
    true_idx = np.asarray(true_episode_idx, dtype=np.int32)

    if len(pred_idx) == 0:
        return float("nan")

    correct = np.abs(pred_idx - true_idx) <= tolerance
    return float(correct.mean())


# ---------------------------------------------------------------------------
# Robustness metrics
# ---------------------------------------------------------------------------

def worst_paraphrase_gap(
    paraphrase_scores: ArrayLike,
) -> float:
    """Maximum gap between best and worst paraphrase scores per concept.

    Parameters
    ----------
    paraphrase_scores : array-like of shape (n_concepts, n_paraphrases)
        Performance metric (e.g., AUROC) for each paraphrase of each concept.

    Returns
    -------
    float
        Maximum gap across all concepts.
    """
    scores = np.asarray(paraphrase_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0:
        return float("nan")

    gaps = scores.max(axis=1) - scores.min(axis=1)
    return float(gaps.max())


def city_transfer_drop(
    source_metric: float, target_metric: float
) -> float:
    """Relative performance drop when transferring across cities.

    Parameters
    ----------
    source_metric : float
        Performance on the source city.
    target_metric : float
        Performance on the target city (after transfer).

    Returns
    -------
    float
        Relative drop as a fraction (positive = degradation).
    """
    if source_metric == 0:
        return float("nan")
    return float((source_metric - target_metric) / abs(source_metric))


def history_stratified_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    n_history_trips: ArrayLike,
    bins: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute detection AUROC stratified by user history length.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground truth.
    y_score : array-like of shape (n,)
        Anomaly scores.
    n_history_trips : array-like of shape (n,)
        Number of historical trips available for each sample's user.
    bins : list[int], optional
        Bin edges for history length. Default: [0, 5, 15, 30, 50, inf].

    Returns
    -------
    dict
        Mapping from bin label to dict with 'auroc', 'n_samples'.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_hist = np.asarray(n_history_trips, dtype=np.int32)

    if bins is None:
        bins = [0, 5, 15, 30, 50, 999999]

    results: Dict[str, Dict[str, float]] = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (n_hist >= lo) & (n_hist < hi)
        n_in_bin = mask.sum()
        label = f"{lo}-{hi}" if hi < 999999 else f"{lo}+"

        if n_in_bin < 2 or len(np.unique(y_true[mask])) < 2:
            results[label] = {"auroc": float("nan"), "n_samples": int(n_in_bin)}
        else:
            results[label] = {
                "auroc": auroc(y_true[mask], y_score[mask]),
                "n_samples": int(n_in_bin),
            }

    return results


# ---------------------------------------------------------------------------
# Aggregate metrics computation
# ---------------------------------------------------------------------------

def compute_all_detection_metrics(
    y_true: ArrayLike, y_score: ArrayLike
) -> Dict[str, float]:
    """Compute all detection metrics at once.

    Returns
    -------
    dict
        Keys: auroc, auprc, fpr_at_95tpr.
    """
    return {
        "auroc": auroc(y_true, y_score),
        "auprc": auprc(y_true, y_score),
        "fpr_at_95tpr": fpr_at_tpr(y_true, y_score, target_tpr=0.95),
    }


def compute_all_calibration_metrics(
    y_true: ArrayLike, y_prob: ArrayLike
) -> Dict[str, float]:
    """Compute all calibration metrics at once.

    Returns
    -------
    dict
        Keys: ece, brier, nll.
    """
    return {
        "ece": expected_calibration_error(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
        "nll": negative_log_likelihood(y_true, y_prob),
    }


def compute_full_evaluation(
    y_true_binary: ArrayLike,
    y_anomaly_score: ArrayLike,
    y_true_classes: Optional[ArrayLike] = None,
    y_class_scores: Optional[ArrayLike] = None,
    y_prob: Optional[ArrayLike] = None,
    known_mask: Optional[ArrayLike] = None,
    pred_spans: Optional[ArrayLike] = None,
    true_spans: Optional[ArrayLike] = None,
    pred_episode_idx: Optional[ArrayLike] = None,
    true_episode_idx: Optional[ArrayLike] = None,
    conformal_sizes: Optional[ArrayLike] = None,
) -> Dict[str, Any]:
    """Run the full evaluation suite and return a results dictionary.

    Parameters
    ----------
    y_true_binary : array-like of shape (n,)
        Binary anomaly labels (0=normal, 1=anomalous).
    y_anomaly_score : array-like of shape (n,)
        Continuous anomaly scores.
    y_true_classes : array-like of shape (n,), optional
        Concept class labels for classification metrics.
    y_class_scores : array-like of shape (n, C), optional
        Per-class score matrix for classification metrics.
    y_prob : array-like of shape (n,), optional
        Predicted probabilities for calibration metrics.
    known_mask : array-like of shape (n,), optional
        Boolean mask where True = known-class sample for open-set metrics.
    pred_spans : array-like of shape (m, 2), optional
        Predicted anomaly spans for localization metrics.
    true_spans : array-like of shape (m, 2), optional
        Ground truth anomaly spans for localization metrics.
    pred_episode_idx : array-like of shape (m,), optional
        Predicted anomalous episode indices.
    true_episode_idx : array-like of shape (m,), optional
        Ground truth anomalous episode indices.
    conformal_sizes : array-like of shape (n,), optional
        Conformal prediction set sizes.

    Returns
    -------
    dict
        Nested dictionary with all computed metrics.
    """
    results: Dict[str, Any] = {}

    # Detection
    results["detection"] = compute_all_detection_metrics(
        y_true_binary, y_anomaly_score
    )

    # Open-set
    if known_mask is not None:
        known_mask = np.asarray(known_mask, dtype=bool)
        known_scores = np.asarray(y_anomaly_score)[known_mask]
        unknown_scores = np.asarray(y_anomaly_score)[~known_mask]
        y_true_classes_arr = np.asarray(y_true_classes) if y_true_classes is not None else None

        rej_auroc = unknown_rejection_auroc(known_scores, unknown_scores)

        known_acc = 0.0
        if y_true_classes_arr is not None and y_class_scores is not None:
            y_class_scores_arr = np.asarray(y_class_scores)
            known_preds = y_class_scores_arr[known_mask].argmax(axis=1)
            known_acc = top1_accuracy(y_true_classes_arr[known_mask], known_preds)

        unknown_rej_rate = (np.asarray(y_anomaly_score)[~known_mask] <
                           np.median(np.asarray(y_anomaly_score)[known_mask])).mean() \
            if (~known_mask).sum() > 0 and known_mask.sum() > 0 else 0.0

        results["open_set"] = {
            "h_score": h_score(known_acc, float(unknown_rej_rate)),
            "unknown_rejection_auroc": rej_auroc,
        }

    # Concept classification
    if y_true_classes is not None and y_class_scores is not None:
        y_true_cls = np.asarray(y_true_classes)
        y_cls_scores = np.asarray(y_class_scores)
        y_pred_cls = y_cls_scores.argmax(axis=1)
        results["concept"] = {
            "top1_accuracy": top1_accuracy(y_true_cls, y_pred_cls),
            "macro_f1": macro_f1(y_true_cls, y_pred_cls),
            "map": mean_average_precision(y_true_cls, y_cls_scores),
            "recall_at_5": recall_at_k(y_true_cls, y_cls_scores, k=5),
        }

    # Calibration
    if y_prob is not None:
        results["calibration"] = compute_all_calibration_metrics(
            y_true_binary, y_prob
        )
        if conformal_sizes is not None:
            results["calibration"]["conformal_set_size"] = conformal_set_size(
                conformal_sizes
            )

    # Localization
    if pred_spans is not None and true_spans is not None:
        results["localization"] = {
            "span_iou": span_iou(pred_spans, true_spans),
        }
    if pred_episode_idx is not None and true_episode_idx is not None:
        loc = results.get("localization", {})
        loc["pointing_accuracy"] = pointing_accuracy(
            pred_episode_idx, true_episode_idx
        )
        loc["pointing_accuracy_tol1"] = pointing_accuracy(
            pred_episode_idx, true_episode_idx, tolerance=1
        )
        results["localization"] = loc

    return results


# ---------------------------------------------------------------------------
# Aliases used by evaluate.py
# ---------------------------------------------------------------------------

compute_detection_metrics = compute_all_detection_metrics
compute_calibration_metrics = compute_all_calibration_metrics
compute_all_metrics = compute_full_evaluation


def compute_concept_metrics(
    concept_scores: ArrayLike, concept_labels: ArrayLike
) -> Dict[str, float]:
    """Compute concept classification metrics from scores and labels."""
    concept_scores = np.asarray(concept_scores, dtype=np.float64)
    concept_labels = np.asarray(concept_labels, dtype=np.int32)

    if len(concept_labels) == 0:
        return {"top1_accuracy": float("nan"), "macro_f1": float("nan"), "map": float("nan")}

    y_pred = concept_scores.argmax(axis=1) if concept_scores.ndim == 2 else concept_scores
    results_dict: Dict[str, float] = {
        "top1_accuracy": top1_accuracy(concept_labels, y_pred),
        "macro_f1": macro_f1(concept_labels, y_pred),
    }
    if concept_scores.ndim == 2:
        results_dict["map"] = mean_average_precision(concept_labels, concept_scores)
        results_dict["recall_at_5"] = recall_at_k(concept_labels, concept_scores, k=5)
    return results_dict


def compute_open_set_metrics(
    known_scores: ArrayLike, unknown_scores: ArrayLike
) -> Dict[str, float]:
    """Compute open-set recognition metrics."""
    known_scores = np.asarray(known_scores, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)

    rej_auroc = unknown_rejection_auroc(known_scores, unknown_scores)

    if len(known_scores) > 0 and len(unknown_scores) > 0:
        thresh = np.median(known_scores)
        known_acc = float(np.mean(known_scores >= thresh))
        unknown_rej = float(np.mean(unknown_scores < thresh))
    else:
        known_acc, unknown_rej = 0.0, 0.0

    return {
        "h_score": h_score(known_acc, unknown_rej),
        "unknown_rejection_auroc": rej_auroc,
    }


def compute_localization_metrics(
    pred_spans: Optional[ArrayLike] = None,
    true_spans: Optional[ArrayLike] = None,
    pred_episode_idx: Optional[ArrayLike] = None,
    true_episode_idx: Optional[ArrayLike] = None,
) -> Dict[str, float]:
    """Compute localization metrics."""
    results_dict: Dict[str, float] = {}
    if pred_spans is not None and true_spans is not None:
        results_dict["span_iou"] = span_iou(pred_spans, true_spans)
    if pred_episode_idx is not None and true_episode_idx is not None:
        results_dict["pointing_accuracy"] = pointing_accuracy(pred_episode_idx, true_episode_idx)
        results_dict["pointing_accuracy_tol1"] = pointing_accuracy(pred_episode_idx, true_episode_idx, tolerance=1)
    return results_dict


def compute_robustness_metrics(
    paraphrase_scores: Optional[ArrayLike] = None,
    source_metric: float = 0.0,
    target_metric: float = 0.0,
) -> Dict[str, float]:
    """Compute robustness metrics."""
    results_dict: Dict[str, float] = {}
    if paraphrase_scores is not None:
        results_dict["worst_paraphrase_gap"] = worst_paraphrase_gap(paraphrase_scores)
    results_dict["city_transfer_drop"] = city_transfer_drop(source_metric, target_metric)
    return results_dict
