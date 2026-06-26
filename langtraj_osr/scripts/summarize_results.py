"""Aggregate all experiment results into paper-ready tables.

Reads JSON result files from all phases and produces:
  - results/SUMMARY.md        — full markdown report
  - results/tables/           — individual CSV tables
  - results/summary.json      — machine-readable summary

Usage:
    python -m langtraj_osr.scripts.summarize_results --results_dir results/
    python -m langtraj_osr.scripts.summarize_results --results_dir results/ --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- Metric display names ----
METRIC_DISPLAY = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "fpr95": "FPR@95",
    "h_score": "H-score",
    "oscr": "OSCR",
    "unknown_auroc": "Unk-AUROC",
    "top1_acc": "Top-1 Acc",
    "macro_f1": "Macro-F1",
    "map": "mAP",
    "recall_at_10": "R@10",
    "ece": "ECE",
    "brier": "Brier",
    "nll": "NLL",
    "set_size": "Set Size",
    "span_iou": "Span IoU",
    "pointing_acc": "Pointing Acc",
}

METHOD_ORDER = [
    "norm_only", "backbone_max", "dsl_xl", "nl2dsl", "canonical_json",
    "direct_text", "lm_tad", "atrom_ossl", "langtraj_osr",
]

METHOD_DISPLAY = {
    "norm_only": "NormOnly (B1)",
    "dsl_xl": "DSL-XL (B2)",
    "nl2dsl": "NL→DSL (B3)",
    "canonical_json": "Canonical-JSON (B4)",
    "direct_text": "DirectText (B5)",
    "lm_tad": "LM-TAD (B6)",
    "atrom_ossl": "ATROM/OSSL (B7)",
    "backbone_max": "Backbone+max (B8)",
    "langtraj_osr": "**LangTraj-OSR (Ours)**",
}

DATASET_DISPLAY = {
    "numosim": "NUMOSIM",
    "geolife": "GeoLife",
    "porto": "Porto",
    "foursquare_nyc": "FS-NYC",
    "foursquare_tokyo": "FS-Tokyo",
}

ABLATION_DISPLAY = {
    "no_language": "A1: No language",
    "no_primitive": "A2: No primitive head",
    "dsl_instead": "A3: DSL instead of NL",
    "no_user_history": "A4: No user history",
    "cohort_history": "A5: Cohort history",
    "no_para_loss": "A6: No L_para",
    "no_conformal": "A7: No conformal reject",
    "opaque_tokens": "A8: Opaque tokens",
    "no_orth": "A9: No L_orth",
    "random_text": "A10: Random text embeds",
    "bow_text": "A11: BoW/TF-IDF text",
    "full_model": "Full LangTraj-OSR",
}


# ---------------------------------------------------------------------------
# JSON loading helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Optional[Dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def find_result_files(results_dir: Path, pattern: str) -> List[Path]:
    return sorted(results_dir.rglob(pattern))


def safe_get(d: Any, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is None:
            return default
    return d


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def aggregate_seeds(values: List[float]) -> Tuple[float, float]:
    """Return (mean, std) over seeds."""
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values)
    return float(arr.mean()), float(arr.std()) if len(arr) > 1 else 0.0


def fmt(mean: float, std: float, bold: bool = False) -> str:
    if np.isnan(mean):
        return "-"
    s = f"{mean:.3f}" if std == 0.0 else f"{mean:.3f}±{std:.3f}"
    return f"**{s}**" if bold else s


# ---------------------------------------------------------------------------
# Result loaders per phase
# ---------------------------------------------------------------------------

def load_phase2_results(results_dir: Path, seeds: List[int]) -> Dict:
    """Phase 2: sanity baselines on NUMOSIM."""
    out = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        for method in ["norm_only", "dsl_xl", "backbone_max"]:
            path = results_dir / "phase2" / method / "numosim" / f"seed_{seed}" / "results.json"
            data = load_json(path)
            if data:
                for metric, val in data.get("test_metrics", {}).items():
                    out[method][metric].append(val)
    return dict(out)


def load_phase3_results(results_dir: Path, seeds: List[int]) -> Dict:
    """Phase 3: full model on NUMOSIM + GeoLife."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for seed in seeds:
        for dataset in ["numosim", "geolife"]:
            path = results_dir / "phase3" / dataset / f"seed_{seed}" / "results.json"
            data = load_json(path)
            if data:
                for metric, val in data.get("test_metrics", {}).items():
                    out["langtraj_osr"][dataset][metric].append(val)
                # Per-split metrics
                for split, sm in data.get("split_metrics", {}).items():
                    for metric, val in sm.items():
                        out["langtraj_osr"][f"{dataset}/{split}"][metric].append(val)
    return dict(out)


def load_phase4_5_results(results_dir: Path, seeds: List[int], phase: int) -> Dict:
    """Phase 4/5: zero-shot eval."""
    phase_name = "phase4" if phase == 4 else "phase5"
    split_name = "zs_comp" if phase == 4 else "zs_family"
    out = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        for dataset in ["numosim", "geolife"]:
            path = results_dir / phase_name / dataset / f"seed_{seed}" / "results.json"
            data = load_json(path)
            if data:
                split_key = f"A_{split_name}"
                sm = safe_get(data, "split_metrics", split_key) or {}
                for metric, val in sm.items():
                    out[dataset][metric].append(val)
    return dict(out)


def load_phase6_results(results_dir: Path, seeds: List[int]) -> Dict:
    """Phase 6: full baseline suite on all datasets."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    datasets = ["numosim", "geolife", "porto", "foursquare_nyc", "foursquare_tokyo"]
    baselines = ["norm_only", "dsl_xl", "nl2dsl", "canonical_json",
                 "direct_text", "lm_tad", "atrom_ossl", "backbone_max"]

    for seed in seeds:
        # Baselines
        for method in baselines:
            for dataset in datasets:
                path = results_dir / "phase6" / method / dataset / f"seed_{seed}" / "results.json"
                data = load_json(path)
                if data:
                    for metric, val in data.get("test_metrics", {}).items():
                        out[method][dataset][metric].append(val)
        # Full model
        for dataset in datasets:
            path = results_dir / "phase6" / "langtraj_osr" / dataset / f"seed_{seed}" / "results.json"
            data = load_json(path)
            if data:
                for metric, val in data.get("test_metrics", {}).items():
                    out["langtraj_osr"][dataset][metric].append(val)

    return dict(out)


def load_phase7_results(results_dir: Path, seeds: List[int]) -> Dict:
    """Phase 7: cross-city transfer."""
    out = {}
    for seed in seeds:
        for src, tgt in [("foursquare_nyc", "foursquare_tokyo"),
                         ("foursquare_tokyo", "foursquare_nyc")]:
            key = f"{src}→{tgt}"
            path = (results_dir / "phase7" / f"{src}_to_{tgt}" /
                    "transfer_eval" / f"seed_{seed}" / "results.json")
            data = load_json(path)
            if data:
                if key not in out:
                    out[key] = defaultdict(list)
                for metric, val in data.get("test_metrics", {}).items():
                    out[key][metric].append(val)
    return out


def load_phase8_results(results_dir: Path, seeds: List[int]) -> Dict:
    """Phase 8: robustness studies."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for seed in seeds:
        for dataset in ["numosim", "geolife"]:
            path = results_dir / "phase8" / dataset / f"seed_{seed}" / "results.json"
            data = load_json(path)
            if data:
                for split, sm in data.get("split_metrics", {}).items():
                    for metric, val in sm.items():
                        out[dataset][split][metric].append(val)
    return dict(out)


def load_phase9_results(results_dir: Path, seeds: List[int]) -> Dict:
    """Phase 9: ablations."""
    out = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        ablation_dir = results_dir / "phase9"
        for ablation_path in ablation_dir.rglob(f"seed_{seed}/results.json"):
            # Path pattern: phase9/<ablation_name>/numosim/seed_42/results.json
            parts = ablation_path.parts
            try:
                phase9_idx = next(i for i, p in enumerate(parts) if p == "phase9")
                ablation_name = parts[phase9_idx + 1]
            except (StopIteration, IndexError):
                continue
            data = load_json(ablation_path)
            if data:
                for metric, val in data.get("test_metrics", {}).items():
                    out[ablation_name][metric].append(val)
    return dict(out)


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------

def make_main_table(phase6: Dict, seeds: List[int], primary_metrics: List[str]) -> str:
    """Main results table: methods × datasets × metrics."""
    datasets = [d for d in ["numosim", "geolife", "porto", "foursquare_nyc", "foursquare_tokyo"]
                if any(d in phase6.get(m, {}) for m in phase6)]
    methods = [m for m in METHOD_ORDER if m in phase6]

    if not datasets or not methods:
        return "_No phase 6 results found yet._\n"

    # Pick one primary metric to show per dataset (AUROC)
    primary = "auroc"
    lines = []
    header = "| Method | " + " | ".join(DATASET_DISPLAY.get(d, d) for d in datasets) + " |"
    sep = "|" + "--------|" * (len(datasets) + 1)
    lines += [header, sep]

    ours_vals = {}
    for dataset in datasets:
        vals = phase6.get("langtraj_osr", {}).get(dataset, {}).get(primary, [])
        ours_vals[dataset] = aggregate_seeds(vals)[0] if vals else float("nan")

    for method in methods:
        row_parts = [METHOD_DISPLAY.get(method, method)]
        for dataset in datasets:
            vals = phase6.get(method, {}).get(dataset, {}).get(primary, [])
            mean, std = aggregate_seeds(vals)
            bold = (not np.isnan(mean)) and (not np.isnan(ours_vals[dataset])) and (mean >= ours_vals[dataset])
            row_parts.append(fmt(mean, std, bold=bold and method == "langtraj_osr"))
        lines.append("| " + " | ".join(row_parts) + " |")

    return "\n".join(lines) + "\n"


def make_zeroshot_table(zs_comp: Dict, zs_family: Dict) -> str:
    datasets = ["numosim", "geolife"]
    lines = []
    header = "| Dataset | A_zs-comp AUROC | A_zs-family AUROC |"
    sep = "|---------|----------------|------------------|"
    lines += [header, sep]
    for dataset in datasets:
        comp_vals = zs_comp.get(dataset, {}).get("auroc", [])
        fam_vals = zs_family.get(dataset, {}).get("auroc", [])
        comp_str = fmt(*aggregate_seeds(comp_vals))
        fam_str = fmt(*aggregate_seeds(fam_vals))
        lines.append(f"| {DATASET_DISPLAY.get(dataset, dataset)} | {comp_str} | {fam_str} |")
    return "\n".join(lines) + "\n"


def make_ablation_table(ablations: Dict) -> str:
    if not ablations:
        return "_No ablation results found yet._\n"

    metrics = ["auroc", "h_score", "top1_acc"]
    lines = []
    header = "| Ablation | " + " | ".join(METRIC_DISPLAY.get(m, m) for m in metrics) + " |"
    sep = "|----------|" + "---------|" * len(metrics)
    lines += [header, sep]

    # Full model first
    for abl_name in ["full_model"] + [k for k in ablations if k != "full_model"]:
        display = ABLATION_DISPLAY.get(abl_name, abl_name)
        data = ablations.get(abl_name, {})
        row_parts = [display]
        for metric in metrics:
            vals = data.get(metric, [])
            row_parts.append(fmt(*aggregate_seeds(vals)))
        lines.append("| " + " | ".join(row_parts) + " |")

    return "\n".join(lines) + "\n"


def make_transfer_table(transfer: Dict) -> str:
    if not transfer:
        return "_No cross-city transfer results found yet._\n"
    metrics = ["auroc", "auprc"]
    lines = []
    header = "| Direction | " + " | ".join(METRIC_DISPLAY.get(m, m) for m in metrics) + " |"
    sep = "|-----------|" + "---------|" * len(metrics)
    lines += [header, sep]
    for direction, data in transfer.items():
        row_parts = [direction]
        for metric in metrics:
            vals = data.get(metric, [])
            row_parts.append(fmt(*aggregate_seeds(vals)))
        lines.append("| " + " | ".join(row_parts) + " |")
    return "\n".join(lines) + "\n"


def make_sanity_table(phase2: Dict, phase3_ours: Dict) -> str:
    """Sanity check table for Phase 2 + early full-model results."""
    lines = []
    header = "| Method | Dataset | AUROC | AUPRC | FPR@95 |"
    sep = "|--------|---------|-------|-------|--------|"
    lines += [header, sep]

    for method in ["norm_only", "dsl_xl", "backbone_max"]:
        data = phase2.get(method, {})
        mean_auroc, std_auroc = aggregate_seeds(data.get("auroc", []))
        mean_auprc, std_auprc = aggregate_seeds(data.get("auprc", []))
        mean_fpr, std_fpr = aggregate_seeds(data.get("fpr95", []))
        lines.append(
            f"| {METHOD_DISPLAY.get(method, method)} | NUMOSIM | "
            f"{fmt(mean_auroc, std_auroc)} | {fmt(mean_auprc, std_auprc)} | {fmt(mean_fpr, std_fpr)} |"
        )

    for dataset in ["numosim", "geolife"]:
        data = phase3_ours.get("langtraj_osr", {}).get(dataset, {})
        mean_auroc, std_auroc = aggregate_seeds(data.get("auroc", []))
        mean_auprc, std_auprc = aggregate_seeds(data.get("auprc", []))
        mean_fpr, std_fpr = aggregate_seeds(data.get("fpr95", []))
        lines.append(
            f"| **LangTraj-OSR** | {DATASET_DISPLAY.get(dataset, dataset)} | "
            f"{fmt(mean_auroc, std_auroc)} | {fmt(mean_auprc, std_auprc)} | {fmt(mean_fpr, std_fpr)} |"
        )

    return "\n".join(lines) + "\n"


def check_gonogo_gates(phase2: Dict, phase6: Dict) -> List[str]:
    """Check go/no-go gates and return list of status strings."""
    gates = []

    # Gate 1: NormOnly AUROC < 0.90 on NUMOSIM
    norm_auroc_vals = phase2.get("norm_only", {}).get("auroc", [])
    if norm_auroc_vals:
        mean_norm = np.mean(norm_auroc_vals)
        passed = mean_norm < 0.90
        gates.append(
            f"Gate 1 (Phase 2) — NormOnly AUROC = {mean_norm:.3f} "
            f"{'✓ PASS (language claim viable)' if passed else '⚠ WARN: normality alone is strong'}"
        )
    else:
        gates.append("Gate 1 (Phase 2) — NormOnly AUROC: NO DATA YET")

    # Gate 2: Language > DSL-XL on compositional concepts (Phase 4)
    # Approximated from phase 6 if phase 4 not available
    ours_auroc = np.mean(phase6.get("langtraj_osr", {}).get("numosim", {}).get("auroc", [float("nan")]))
    dsl_auroc = np.mean(phase6.get("dsl_xl", {}).get("numosim", {}).get("auroc", [float("nan")]))
    if not np.isnan(ours_auroc) and not np.isnan(dsl_auroc):
        passed = ours_auroc > dsl_auroc
        gates.append(
            f"Gate 2 (Phase 4) — LangTraj-OSR AUROC ({ours_auroc:.3f}) vs DSL-XL ({dsl_auroc:.3f}): "
            f"{'✓ PASS' if passed else '✗ FAIL — consider fallback'}"
        )
    else:
        gates.append("Gate 2 (Phase 4) — Language vs DSL-XL: NO DATA YET")

    # Gate 3: Full > ATROM on unknown rejection (Phase 6)
    ours_h = np.mean(phase6.get("langtraj_osr", {}).get("numosim", {}).get("h_score", [float("nan")]))
    atrom_h = np.mean(phase6.get("atrom_ossl", {}).get("numosim", {}).get("h_score", [float("nan")]))
    if not np.isnan(ours_h) and not np.isnan(atrom_h):
        passed = ours_h > atrom_h
        gates.append(
            f"Gate 3 (Phase 6) — H-score: Ours ({ours_h:.3f}) vs ATROM ({atrom_h:.3f}): "
            f"{'✓ PASS' if passed else '✗ FAIL — consider dropping open-set claim'}"
        )
    else:
        gates.append("Gate 3 (Phase 6) — Full vs ATROM on H-score: NO DATA YET")

    return gates


def save_csv(data: Dict, path: Path, row_key: str = "method") -> None:
    """Save a flat dict of dicts to CSV."""
    if not data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = sorted({k for row in data.values() for k in (row if isinstance(row, dict) else {})})
    lines = [f"{row_key}," + ",".join(all_keys)]
    for name, row in data.items():
        if isinstance(row, dict):
            vals = [str(row.get(k, "")) for k in all_keys]
        else:
            vals = [str(row)]
        lines.append(f"{name}," + ",".join(vals))
    path.write_text("\n".join(lines))
    logger.info("  Saved CSV: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LangTraj-OSR experiment results")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: results_dir/SUMMARY.md)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    output_path = Path(args.output) if args.output else results_dir / "SUMMARY.md"
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LangTraj-OSR Results Summarizer")
    logger.info("Results dir: %s", results_dir)
    logger.info("Seeds: %s", args.seeds)
    logger.info("=" * 60)

    # ---- Load all results ----
    logger.info("\nLoading phase results...")
    phase2 = load_phase2_results(results_dir, args.seeds)
    phase3 = load_phase3_results(results_dir, args.seeds)
    zs_comp = load_phase4_5_results(results_dir, args.seeds, phase=4)
    zs_family = load_phase4_5_results(results_dir, args.seeds, phase=5)
    phase6 = load_phase6_results(results_dir, args.seeds)
    transfer = load_phase7_results(results_dir, args.seeds)
    robustness = load_phase8_results(results_dir, args.seeds)
    ablations = load_phase9_results(results_dir, args.seeds)

    # ---- Go/No-Go gates ----
    gates = check_gonogo_gates(phase2, phase6)

    # ---- Build SUMMARY.md ----
    lines = [
        "# LangTraj-OSR — Experiment Results Summary",
        f"\n**Seeds**: {args.seeds}  |  **Results dir**: `{results_dir}`\n",
        "---\n",
        "## Go/No-Go Gate Status\n",
    ]
    for g in gates:
        lines.append(f"- {g}")
    lines.append("")

    lines += [
        "---\n",
        "## Table 1: Sanity Check (Phase 2 + Phase 3)\n",
        "> Detection metrics on NUMOSIM. Phase 2 = sanity baselines; Phase 3 = full model.\n",
        make_sanity_table(phase2, phase3),
    ]

    lines += [
        "---\n",
        "## Table 2: Main Results — All Methods × All Datasets (AUROC)\n",
        "> Phase 6. Primary metric: AUROC. Bold = best in column.\n",
        make_main_table(phase6, args.seeds, primary_metrics=["auroc"]),
    ]

    lines += [
        "---\n",
        "## Table 3: Zero-Shot Generalization\n",
        "> Phase 4 (compositional) and Phase 5 (family). LangTraj-OSR only.\n",
        make_zeroshot_table(zs_comp, zs_family),
    ]

    lines += [
        "---\n",
        "## Table 4: Cross-City Transfer (Phase 7)\n",
        "> LangTraj-OSR trained on one city, evaluated on another.\n",
        make_transfer_table(transfer),
    ]

    lines += [
        "---\n",
        "## Table 5: Ablation Study (Phase 9, NUMOSIM)\n",
        make_ablation_table(ablations),
    ]

    # ---- Robustness summary ----
    lines += [
        "---\n",
        "## Table 6: Robustness (Phase 8)\n",
    ]
    if robustness:
        for dataset, splits in robustness.items():
            lines.append(f"\n### {DATASET_DISPLAY.get(dataset, dataset)}\n")
            rows = []
            for split, metrics in splits.items():
                auroc_mean, auroc_std = aggregate_seeds(metrics.get("auroc", []))
                rows.append(f"| {split} | {fmt(auroc_mean, auroc_std)} |")
            if rows:
                lines += ["| Condition | AUROC |", "|-----------|-------|"] + rows
    else:
        lines.append("_No robustness results found yet._\n")

    # ---- Target metrics checklist ----
    lines += [
        "---\n",
        "## Target Metrics Checklist\n",
        "| Metric | Target | Status |",
        "|--------|--------|--------|",
    ]

    targets = [
        ("AUROC (seen concepts)", ">0.90",
         phase6.get("langtraj_osr", {}).get("numosim", {}).get("auroc", [])),
        ("AUROC (A_zs-comp)", ">0.80",
         zs_comp.get("numosim", {}).get("auroc", [])),
        ("H-score (open-set)", ">0.70",
         phase6.get("langtraj_osr", {}).get("numosim", {}).get("h_score", [])),
        ("Language vs DSL-XL gap (AUROC, compositional)", ">5%", []),
        ("Cross-city transfer drop", "<10%", []),
        ("Conformal set size", "<3",
         phase6.get("langtraj_osr", {}).get("numosim", {}).get("set_size", [])),
    ]

    for metric_name, target, vals in targets:
        if vals:
            mean, std = aggregate_seeds(vals)
            status = f"{fmt(mean, std)}"
        else:
            status = "—"
        lines.append(f"| {metric_name} | {target} | {status} |")

    lines.append("\n---\n")
    lines.append(f"_Generated by `summarize_results.py` | Results dir: `{results_dir}`_\n")

    # ---- Write output ----
    summary_text = "\n".join(lines)
    output_path.write_text(summary_text, encoding="utf-8")
    logger.info("\nSummary written to: %s", output_path)

    # ---- Save CSVs ----
    # Main results: method → dataset → auroc mean
    main_csv = {}
    for method in METHOD_ORDER:
        row = {}
        for dataset in ["numosim", "geolife", "porto", "foursquare_nyc", "foursquare_tokyo"]:
            vals = phase6.get(method, {}).get(dataset, {}).get("auroc", [])
            mean, _ = aggregate_seeds(vals)
            row[dataset] = f"{mean:.4f}" if not np.isnan(mean) else ""
        main_csv[method] = row
    save_csv(main_csv, tables_dir / "main_results.csv")

    # Ablations CSV
    abl_csv = {}
    for name, data in ablations.items():
        row = {}
        for metric in ["auroc", "h_score", "top1_acc"]:
            vals = data.get(metric, [])
            mean, _ = aggregate_seeds(vals)
            row[metric] = f"{mean:.4f}" if not np.isnan(mean) else ""
        abl_csv[name] = row
    save_csv(abl_csv, tables_dir / "ablations.csv", row_key="ablation")

    # Machine-readable summary JSON
    summary_json = {
        "seeds": args.seeds,
        "gates": gates,
        "phase2_sanity": {m: {k: aggregate_seeds(v)[0] for k, v in d.items()}
                          for m, d in phase2.items()},
        "phase6_main": {m: {ds: {k: aggregate_seeds(v)[0] for k, v in dm.items()}
                             for ds, dm in d.items()}
                        for m, d in phase6.items()},
        "zeroshot_comp": {ds: {k: aggregate_seeds(v)[0] for k, v in dm.items()}
                          for ds, dm in zs_comp.items()},
        "zeroshot_family": {ds: {k: aggregate_seeds(v)[0] for k, v in dm.items()}
                            for ds, dm in zs_family.items()},
        "ablations": {name: {k: aggregate_seeds(v)[0] for k, v in d.items()}
                      for name, d in ablations.items()},
    }
    summary_json_path = results_dir / "summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_json, f, indent=2, default=str)
    logger.info("Machine-readable summary: %s", summary_json_path)

    logger.info("\nDone. Output files:")
    logger.info("  %s  (full markdown report)", output_path)
    logger.info("  %s  (CSV tables)", tables_dir)
    logger.info("  %s  (JSON summary)", summary_json_path)


if __name__ == "__main__":
    main()
