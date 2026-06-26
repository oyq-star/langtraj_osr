"""Generate publication-quality plots for LangTraj-OSR results.

Functions:
- plot_auroc_comparison: bar chart of AUROC across methods
- plot_open_set_curves: OSCR and risk-coverage curves
- plot_calibration: reliability diagrams
- plot_ablation_waterfall: ablation contribution chart
- plot_zero_shot_heatmap: concept x method heatmap
- plot_robustness: paraphrase variance plots

All plots saved as PDF for paper quality.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

# Paper-friendly defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# Colour palette for methods
METHOD_COLOURS: Dict[str, str] = {
    "LangTraj-OSR": "#2c7bb6",
    "NormOnly": "#d7191c",
    "DSL-XL": "#fdae61",
    "Backbone+max": "#abd9e9",
    "Backbone+cosine": "#a6d96a",
    "CLIP-Mob": "#756bb1",
}


def _ensure_dir(path: str) -> None:
    """Create parent directory if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. AUROC bar chart comparison
# ---------------------------------------------------------------------------

def plot_auroc_comparison(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Detection AUROC Comparison",
    metric_key: str = "auroc",
) -> str:
    """Bar chart comparing a detection metric across methods.

    Parameters
    ----------
    results : dict
        Mapping from method name to dict containing *metric_key*.
        Example: ``{"LangTraj-OSR": {"auroc": 0.85, "std": 0.02}, ...}``.
    output_path : str
        Path to save the plot (PDF recommended).
    title : str
        Plot title.
    metric_key : str
        Key to extract from each method's result dict.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    _ensure_dir(output_path)

    methods = list(results.keys())
    values = [results[m].get(metric_key, 0.0) for m in methods]
    stds = [results[m].get("std", 0.0) for m in methods]
    colours = [METHOD_COLOURS.get(m, "#999999") for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(methods))
    bars = ax.bar(x, values, yerr=stds, capsize=4, color=colours, edgecolor="black",
                  linewidth=0.5, width=0.6)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel(metric_key.upper())
    ax.set_title(title)
    ax.set_ylim(0.0, min(1.05, max(values) + 0.1))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# 2. Open-set curves (OSCR + risk-coverage)
# ---------------------------------------------------------------------------

def plot_open_set_curves(
    results: Dict[str, Dict[str, Any]],
    output_path: str,
    title: str = "Open-Set Recognition",
) -> str:
    """Plot OSCR curves and risk-coverage curves side by side.

    Parameters
    ----------
    results : dict
        Mapping from method name to dict with keys:
        - ``oscr_fpr``: array-like FPR values
        - ``oscr_ccr``: array-like CCR values
        - ``rc_coverage``: array-like coverage values
        - ``rc_risk``: array-like risk values

    output_path : str
        Path to save the figure.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    _ensure_dir(output_path)

    fig, (ax_oscr, ax_rc) = plt.subplots(1, 2, figsize=(12, 4.5))

    for method, data in results.items():
        colour = METHOD_COLOURS.get(method, "#999999")

        # OSCR
        if "oscr_fpr" in data and "oscr_ccr" in data:
            fpr = np.asarray(data["oscr_fpr"])
            ccr = np.asarray(data["oscr_ccr"])
            ax_oscr.plot(fpr, ccr, label=method, color=colour, linewidth=1.5)

        # Risk-coverage
        if "rc_coverage" in data and "rc_risk" in data:
            cov = np.asarray(data["rc_coverage"])
            risk = np.asarray(data["rc_risk"])
            ax_rc.plot(cov, risk, label=method, color=colour, linewidth=1.5)

    ax_oscr.set_xlabel("False Positive Rate")
    ax_oscr.set_ylabel("Correct Classification Rate")
    ax_oscr.set_title("OSCR Curve")
    ax_oscr.legend(loc="lower right", framealpha=0.9)
    ax_oscr.set_xlim(0, 1)
    ax_oscr.set_ylim(0, 1)

    ax_rc.set_xlabel("Coverage")
    ax_rc.set_ylabel("Risk (Error Rate)")
    ax_rc.set_title("Risk-Coverage Curve")
    ax_rc.legend(loc="upper right", framealpha=0.9)
    ax_rc.set_xlim(0, 1)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# 3. Calibration (reliability diagram)
# ---------------------------------------------------------------------------

def plot_calibration(
    results: Dict[str, Dict[str, Any]],
    output_path: str,
    n_bins: int = 10,
    title: str = "Calibration (Reliability Diagram)",
) -> str:
    """Reliability diagram for one or more methods.

    Parameters
    ----------
    results : dict
        Mapping from method name to dict with keys:
        - ``y_true``: array-like binary labels
        - ``y_prob``: array-like predicted probabilities
        Alternatively, pre-binned data:
        - ``bin_centers``: array-like
        - ``bin_accuracies``: array-like

    output_path : str
        Path to save the figure.
    n_bins : int
        Number of calibration bins (if computing from raw data).

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    for method, data in results.items():
        colour = METHOD_COLOURS.get(method, "#999999")

        if "bin_centers" in data and "bin_accuracies" in data:
            centers = np.asarray(data["bin_centers"])
            accs = np.asarray(data["bin_accuracies"])
        elif "y_true" in data and "y_prob" in data:
            y_true = np.asarray(data["y_true"], dtype=np.float64)
            y_prob = np.asarray(data["y_prob"], dtype=np.float64)
            edges = np.linspace(0, 1, n_bins + 1)
            centers = []
            accs = []
            for i in range(n_bins):
                mask = (y_prob > edges[i]) & (y_prob <= edges[i + 1])
                if mask.sum() > 0:
                    centers.append(y_prob[mask].mean())
                    accs.append(y_true[mask].mean())
            centers = np.array(centers)
            accs = np.array(accs)
        else:
            continue

        ece_val = data.get("ece", None)
        label = method
        if ece_val is not None:
            label += f" (ECE={ece_val:.3f})"

        ax.plot(centers, accs, "o-", color=colour, label=label,
                linewidth=1.5, markersize=5)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# 4. Ablation waterfall chart
# ---------------------------------------------------------------------------

def plot_ablation_waterfall(
    results: Dict[str, float],
    output_path: str,
    baseline_value: Optional[float] = None,
    metric_name: str = "AUROC",
    title: str = "Ablation Study",
) -> str:
    """Waterfall chart showing the contribution of each component.

    Parameters
    ----------
    results : dict
        Mapping from ablation name (e.g. "A1: No language") to metric value.
    output_path : str
        Path to save the figure.
    baseline_value : float, optional
        Full model metric value. If None, uses max value in results.
    metric_name : str
        Name of the metric for axis label.
    title : str
        Plot title.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    _ensure_dir(output_path)

    if baseline_value is None:
        baseline_value = max(results.values())

    # Sort by drop magnitude
    drops = {k: baseline_value - v for k, v in results.items()}
    sorted_items = sorted(drops.items(), key=lambda kv: -kv[1])

    names = [k for k, _ in sorted_items]
    drop_vals = [v for _, v in sorted_items]
    ablation_vals = [results[k] for k in names]

    fig, ax = plt.subplots(figsize=(10, 5))

    n = len(names)
    x = np.arange(n + 1)

    # Full model bar
    colours = ["#2c7bb6"] + [
        "#d7191c" if d > 0.05 else "#fdae61" if d > 0.02 else "#a6d96a"
        for d in drop_vals
    ]
    all_vals = [baseline_value] + ablation_vals
    all_names = ["Full Model"] + names

    bars = ax.bar(x, all_vals, color=colours, edgecolor="black", linewidth=0.5,
                  width=0.7)

    # Drop annotations
    for i, (bar, val) in enumerate(zip(bars, all_vals)):
        label = f"{val:.3f}"
        if i > 0:
            drop = baseline_value - val
            label += f"\n({drop:+.3f})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            label,
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel(metric_name)
    ax.set_title(title)

    # Reference line at full model
    ax.axhline(y=baseline_value, color="#2c7bb6", linestyle=":", linewidth=0.8,
               alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# 5. Zero-shot heatmap (concept x method)
# ---------------------------------------------------------------------------

def plot_zero_shot_heatmap(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Zero-Shot Concept Detection",
    metric_name: str = "AUROC",
) -> str:
    """Heatmap of metric values across concepts and methods.

    Parameters
    ----------
    results : dict
        Mapping from method name to dict of concept_name -> metric_value.
        Example: ``{"LangTraj-OSR": {"late_night_trip": 0.82, ...}, ...}``.
    output_path : str
        Path to save the figure.
    title : str
        Plot title.
    metric_name : str
        Name of the metric for colour-bar label.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    _ensure_dir(output_path)

    methods = list(results.keys())
    all_concepts: List[str] = []
    for m in methods:
        for c in results[m]:
            if c not in all_concepts:
                all_concepts.append(c)

    # Build matrix
    matrix = np.full((len(methods), len(all_concepts)), np.nan)
    for i, method in enumerate(methods):
        for j, concept in enumerate(all_concepts):
            if concept in results[method]:
                matrix[i, j] = results[method][concept]

    fig, ax = plt.subplots(figsize=(max(8, len(all_concepts) * 0.8),
                                    max(4, len(methods) * 0.6)))

    if _HAS_SEABORN:
        sns.heatmap(
            matrix,
            annot=True, fmt=".2f",
            xticklabels=all_concepts,
            yticklabels=methods,
            cmap="RdYlBu_r",
            vmin=0.0, vmax=1.0,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": metric_name},
        )
    else:
        im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(all_concepts)))
        ax.set_xticklabels(all_concepts, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels(methods)
        # Annotate cells
        for i in range(len(methods)):
            for j in range(len(all_concepts)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color="black")
        plt.colorbar(im, ax=ax, label=metric_name)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# 6. Robustness / paraphrase variance plots
# ---------------------------------------------------------------------------

def plot_robustness(
    results: Dict[str, Dict[str, Any]],
    output_path: str,
    title: str = "Paraphrase Robustness",
) -> str:
    """Plot paraphrase variance across methods and concepts.

    Parameters
    ----------
    results : dict
        Mapping from method name to dict with keys:
        - ``concepts``: list of concept names
        - ``mean_scores``: array-like of mean paraphrase scores per concept
        - ``min_scores``: array-like of min paraphrase scores per concept
        - ``max_scores``: array-like of max paraphrase scores per concept

    output_path : str
        Path to save the figure.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    _ensure_dir(output_path)

    methods = list(results.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, method in zip(axes, methods):
        data = results[method]
        concepts = data.get("concepts", [])
        means = np.asarray(data.get("mean_scores", []))
        mins = np.asarray(data.get("min_scores", []))
        maxs = np.asarray(data.get("max_scores", []))

        n = len(concepts)
        if n == 0:
            ax.set_title(method)
            continue

        y = np.arange(n)
        colour = METHOD_COLOURS.get(method, "#999999")

        ax.barh(y, means, color=colour, alpha=0.7, edgecolor="black",
                linewidth=0.5, height=0.6)
        ax.errorbar(
            means, y,
            xerr=[means - mins, maxs - means],
            fmt="none", ecolor="black", capsize=3, linewidth=1,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(concepts, fontsize=9)
        ax.set_xlabel("Score")
        ax.set_title(method)
        ax.set_xlim(0, 1.05)

        # Worst gap annotation
        if len(maxs) > 0 and len(mins) > 0:
            gaps = maxs - mins
            worst_idx = np.argmax(gaps)
            ax.annotate(
                f"max gap: {gaps[worst_idx]:.3f}",
                xy=(maxs[worst_idx], worst_idx),
                xytext=(maxs[worst_idx] + 0.05, worst_idx),
                fontsize=8, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            )

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return os.path.abspath(output_path)
