"""Run a specific experiment phase from the LangTraj-OSR plan.

Usage:
    python -m langtraj_osr.scripts.run_phase --phase 1 --gpu 0 --output_dir results/
    python -m langtraj_osr.scripts.run_phase --phase 2 --gpu 0 --use_synthetic
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..core.utils import get_logger, save_results

logger = get_logger(__name__)

# Phase definitions matching EXPERIMENT_PLAN.md
PHASES = {
    1: {
        "name": "Benchmark Construction",
        "description": "Build MobDef-Bench: generate anomalies, collect descriptions",
        "estimated_time": "1-2 CPU days",
    },
    2: {
        "name": "Sanity Baselines on NUMOSIM",
        "description": "Train NormOnly, DSL-XL, Backbone+max on NUMOSIM",
        "estimated_time": "2-3 GPU-hours",
        "gate": "Normality model alone < 90% AUROC",
    },
    3: {
        "name": "Full Model on NUMOSIM + GeoLife",
        "description": "Train LangTraj-OSR on NUMOSIM and GeoLife",
        "estimated_time": "4-6 GPU-hours",
    },
    4: {
        "name": "Zero-Shot Composition",
        "description": "Evaluate on A_zs_comp split",
        "estimated_time": "3-4 GPU-hours",
        "gate": "Language > DSL-XL on compositional concepts",
    },
    5: {
        "name": "Zero-Shot Family",
        "description": "Evaluate on A_zs_family split",
        "estimated_time": "3-4 GPU-hours",
    },
    6: {
        "name": "Full Baseline Suite",
        "description": "All baselines on all datasets",
        "estimated_time": "15-20 GPU-hours",
        "gate": "Full > ATROM on unknown rejection",
    },
    7: {
        "name": "Cross-City Transfer",
        "description": "NYC↔Tokyo transfer experiments",
        "estimated_time": "4-6 GPU-hours",
    },
    8: {
        "name": "Robustness Studies",
        "description": "Paraphrase + analyst noise robustness",
        "estimated_time": "3-4 GPU-hours",
    },
    9: {
        "name": "Ablations",
        "description": "All 11 ablation experiments",
        "estimated_time": "8-12 GPU-hours",
    },
}


def run_command(cmd: List[str], desc: str) -> int:
    """Run a subprocess command and log output."""
    logger.info("Running: %s", desc)
    logger.info("  Command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error("  FAILED with return code %d", result.returncode)
    else:
        logger.info("  COMPLETED successfully")
    return result.returncode


def _text_encoder_flag(args: argparse.Namespace) -> List[str]:
    """Return ['--text_encoder', path] if non-default, else []."""
    default = "sentence-transformers/all-MiniLM-L6-v2"
    if hasattr(args, "text_encoder") and args.text_encoder != default:
        return ["--text_encoder", args.text_encoder]
    return []


def run_phase_1(args: argparse.Namespace) -> Dict[str, Any]:
    """Phase 1: Benchmark construction."""
    logger.info("=== Phase 1: Benchmark Construction ===")

    cmd = [
        sys.executable, "-m", "langtraj_osr.benchmark.benchmark_builder",
        "--output_dir", str(Path(args.output_dir) / "benchmark"),
        "--seed", str(args.seed),
    ]
    if args.use_synthetic:
        cmd.append("--synthetic")

    rc = run_command(cmd, "Build MobDef-Bench")
    return {"phase": 1, "status": "PASSED" if rc == 0 else "FAILED"}


def run_phase_2(args: argparse.Namespace) -> Dict[str, Any]:
    """Phase 2: Sanity baselines on NUMOSIM."""
    logger.info("=== Phase 2: Sanity Baselines (NUMOSIM) ===")
    results = {}
    dataset = "numosim"
    synthetic_flag = ["--use_synthetic"] if args.use_synthetic else []

    te_flag = _text_encoder_flag(args)
    for baseline in ["norm_only", "dsl_xl", "backbone_max"]:
        cmd = [
            sys.executable, "-m", "langtraj_osr.baselines.run_baseline",
            "--baseline", baseline,
            "--dataset", dataset,
            "--seed", str(args.seed),
            "--output_dir", str(Path(args.output_dir) / "phase2"),
            "--gpu", str(args.gpu),
            "--epochs", "30",
        ] + synthetic_flag + te_flag

        rc = run_command(cmd, f"Baseline: {baseline} on {dataset}")
        results[baseline] = "DONE" if rc == 0 else "FAILED"

    # Check go/no-go gate
    norm_result_path = Path(args.output_dir) / "phase2" / "norm_only" / dataset / f"seed_{args.seed}" / "results.json"
    if norm_result_path.exists():
        with open(norm_result_path) as f:
            norm_results = json.load(f)
        norm_auroc = norm_results.get("test_metrics", {}).get("auroc", 0)
        gate_passed = norm_auroc < 0.90
        logger.info("Go/No-Go: NormOnly AUROC = %.4f (gate: < 0.90 → %s)",
                     norm_auroc, "PASS" if gate_passed else "WARN: normality alone is strong")
        results["gate"] = {"norm_auroc": norm_auroc, "passed": gate_passed}

    return {"phase": 2, "results": results}


def run_phase_3(args: argparse.Namespace) -> Dict[str, Any]:
    """Phase 3: Full model on NUMOSIM + GeoLife."""
    logger.info("=== Phase 3: Full Model ===")
    results = {}
    synthetic_flag = ["--use_synthetic"] if args.use_synthetic else []
    te_flag = _text_encoder_flag(args)

    for dataset in ["numosim", "geolife"]:
        cmd = [
            sys.executable, "-m", "langtraj_osr.train",
            "--dataset", dataset,
            "--seed", str(args.seed),
            "--output_dir", str(Path(args.output_dir) / "phase3"),
            "--gpu", str(args.gpu),
        ] + synthetic_flag + te_flag

        rc = run_command(cmd, f"LangTraj-OSR on {dataset}")
        results[dataset] = "DONE" if rc == 0 else "FAILED"

    return {"phase": 3, "results": results}


def run_phase_4(args: argparse.Namespace) -> Dict[str, Any]:
    """Phase 4: Zero-shot composition split."""
    logger.info("=== Phase 4: Zero-Shot Composition ===")
    results = {}
    synthetic_flag = ["--use_synthetic"] if args.use_synthetic else []
    te_flag = _text_encoder_flag(args)

    for dataset in ["numosim", "geolife"]:
        ckpt = Path(args.output_dir) / "phase3" / dataset / f"seed_{args.seed}" / "best_model.pt"
        if not ckpt.exists():
            logger.warning("Checkpoint not found: %s, skipping", ckpt)
            results[dataset] = "SKIPPED"
            continue

        cmd = [
            sys.executable, "-m", "langtraj_osr.evaluate",
            "--checkpoint", str(ckpt),
            "--dataset", dataset,
            "--seed", str(args.seed),
            "--output_dir", str(Path(args.output_dir) / "phase4" / dataset),
            "--gpu", str(args.gpu),
        ] + synthetic_flag + te_flag

        rc = run_command(cmd, f"Zero-shot eval on {dataset}")
        results[dataset] = "DONE" if rc == 0 else "FAILED"

    return {"phase": 4, "results": results}


def run_phase_6(args: argparse.Namespace) -> Dict[str, Any]:
    """Phase 6: Full baseline suite on all datasets."""
    logger.info("=== Phase 6: Full Baseline Suite ===")
    results = {}
    baselines = ["norm_only", "dsl_xl", "nl2dsl", "canonical_json",
                 "direct_text", "lm_tad", "atrom_ossl", "backbone_max"]
    datasets = ["numosim", "geolife", "porto", "foursquare_nyc", "foursquare_tokyo"]
    synthetic_flag = ["--use_synthetic"] if args.use_synthetic else []
    te_flag = _text_encoder_flag(args)

    if args.use_synthetic:
        datasets = ["numosim"]  # Only synthetic for dev

    for dataset in datasets:
        for baseline in baselines:
            cmd = [
                sys.executable, "-m", "langtraj_osr.baselines.run_baseline",
                "--baseline", baseline,
                "--dataset", dataset,
                "--seed", str(args.seed),
                "--output_dir", str(Path(args.output_dir) / "phase6"),
                "--gpu", str(args.gpu),
            ] + synthetic_flag + te_flag

            rc = run_command(cmd, f"{baseline} on {dataset}")
            results[f"{baseline}/{dataset}"] = "DONE" if rc == 0 else "FAILED"

        # Also train full model
        cmd = [
            sys.executable, "-m", "langtraj_osr.train",
            "--dataset", dataset,
            "--seed", str(args.seed),
            "--output_dir", str(Path(args.output_dir) / "phase6" / "langtraj_osr"),
            "--gpu", str(args.gpu),
        ] + synthetic_flag + te_flag

        rc = run_command(cmd, f"LangTraj-OSR on {dataset}")
        results[f"langtraj_osr/{dataset}"] = "DONE" if rc == 0 else "FAILED"

    return {"phase": 6, "results": results}


def run_phase_5(args: argparse.Namespace) -> Dict[str, Any]:
    """Phase 5: Zero-shot family split."""
    logger.info("=== Phase 5: Zero-Shot Family ===")
    results = {}
    synthetic_flag = ["--use_synthetic"] if args.use_synthetic else []
    te_flag = _text_encoder_flag(args)

    for dataset in ["numosim", "geolife"]:
        ckpt = Path(args.output_dir) / "phase3" / dataset / f"seed_{args.seed}" / "best_model.pt"
        if not ckpt.exists():
            logger.warning("Checkpoint not found: %s, skipping", ckpt)
            results[dataset] = "SKIPPED"
            continue

        cmd = [
            sys.executable, "-m", "langtraj_osr.evaluate",
            "--checkpoint", str(ckpt),
            "--dataset", dataset,
            "--seed", str(args.seed),
            "--output_dir", str(Path(args.output_dir) / "phase5" / dataset),
            "--gpu", str(args.gpu),
        ] + synthetic_flag + te_flag

        rc = run_command(cmd, f"Zero-shot family eval on {dataset}")
        results[dataset] = "DONE" if rc == 0 else "FAILED"

    return {"phase": 5, "results": results}


def run_phase_7(args: argparse.Namespace) -> Dict[str, Any]:
    """Phase 7: Cross-city transfer (NYC <-> Tokyo)."""
    logger.info("=== Phase 7: Cross-City Transfer ===")
    results = {}
    synthetic_flag = ["--use_synthetic"] if args.use_synthetic else []

    if args.use_synthetic:
        logger.info("Skipping cross-city transfer in synthetic mode (only 1 dataset)")
        return {"phase": 7, "results": {"status": "SKIPPED (synthetic mode)"}}

    te_flag = _text_encoder_flag(args)
    # Train on NYC, test on Tokyo
    for src, tgt in [("foursquare_nyc", "foursquare_tokyo"), ("foursquare_tokyo", "foursquare_nyc")]:
        # Train
        cmd = [
            sys.executable, "-m", "langtraj_osr.train",
            "--dataset", src,
            "--seed", str(args.seed),
            "--output_dir", str(Path(args.output_dir) / "phase7" / f"{src}_to_{tgt}"),
            "--gpu", str(args.gpu),
        ] + te_flag
        rc = run_command(cmd, f"Train on {src}")
        if rc != 0:
            results[f"{src}_to_{tgt}"] = "TRAIN_FAILED"
            continue

        # Evaluate on target city
        ckpt = Path(args.output_dir) / "phase7" / f"{src}_to_{tgt}" / src / f"seed_{args.seed}" / "best_model.pt"
        if not ckpt.exists():
            results[f"{src}_to_{tgt}"] = "CKPT_MISSING"
            continue

        cmd = [
            sys.executable, "-m", "langtraj_osr.evaluate",
            "--checkpoint", str(ckpt),
            "--dataset", tgt,
            "--seed", str(args.seed),
            "--output_dir", str(Path(args.output_dir) / "phase7" / f"{src}_to_{tgt}" / "transfer_eval"),
            "--gpu", str(args.gpu),
        ] + te_flag
        rc = run_command(cmd, f"Evaluate {src} model on {tgt}")
        results[f"{src}_to_{tgt}"] = "DONE" if rc == 0 else "FAILED"

    return {"phase": 7, "results": results}


def run_phase_8(args: argparse.Namespace) -> Dict[str, Any]:
    """Phase 8: Robustness studies (paraphrase + analyst noise)."""
    logger.info("=== Phase 8: Robustness Studies ===")
    results = {}
    synthetic_flag = ["--use_synthetic"] if args.use_synthetic else []

    te_flag = _text_encoder_flag(args)
    datasets = ["numosim"] if args.use_synthetic else ["numosim", "geolife"]

    for dataset in datasets:
        ckpt = Path(args.output_dir) / "phase3" / dataset / f"seed_{args.seed}" / "best_model.pt"
        if not ckpt.exists():
            logger.warning("Checkpoint not found: %s, skipping", ckpt)
            results[dataset] = "SKIPPED"
            continue

        cmd = [
            sys.executable, "-m", "langtraj_osr.evaluate",
            "--checkpoint", str(ckpt),
            "--dataset", dataset,
            "--seed", str(args.seed),
            "--output_dir", str(Path(args.output_dir) / "phase8" / dataset),
            "--gpu", str(args.gpu),
            "--eval_paraphrase",
        ] + synthetic_flag + te_flag

        rc = run_command(cmd, f"Paraphrase robustness on {dataset}")
        results[dataset] = "DONE" if rc == 0 else "FAILED"

    return {"phase": 8, "results": results}


def run_phase_9(args: argparse.Namespace) -> Dict[str, Any]:
    """Phase 9: All ablations."""
    logger.info("=== Phase 9: Ablations ===")
    synthetic_flag = ["--use_synthetic"] if args.use_synthetic else []

    te_flag = _text_encoder_flag(args)
    cmd = [
        sys.executable, "-m", "langtraj_osr.scripts.ablation_runner",
        "--dataset", "numosim",
        "--seed", str(args.seed),
        "--output_dir", str(Path(args.output_dir) / "phase9"),
        "--gpu", str(args.gpu),
    ] + synthetic_flag + te_flag

    rc = run_command(cmd, "All ablations")
    return {"phase": 9, "status": "DONE" if rc == 0 else "FAILED"}


PHASE_RUNNERS = {
    1: run_phase_1,
    2: run_phase_2,
    3: run_phase_3,
    4: run_phase_4,
    5: run_phase_5,
    6: run_phase_6,
    7: run_phase_7,
    8: run_phase_8,
    9: run_phase_9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment phase")
    parser.add_argument("--phase", type=int, required=True, choices=list(PHASES.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--text_encoder", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Text encoder model name or local path (for offline use)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    phase_info = PHASES[args.phase]
    logger.info("=" * 60)
    logger.info("Phase %d: %s", args.phase, phase_info["name"])
    logger.info("Description: %s", phase_info["description"])
    logger.info("Estimated time: %s", phase_info["estimated_time"])
    if "gate" in phase_info:
        logger.info("Go/No-Go gate: %s", phase_info["gate"])
    logger.info("=" * 60)

    runner = PHASE_RUNNERS.get(args.phase)
    if runner is None:
        logger.error("Phase %d runner not implemented", args.phase)
        return

    results = runner(args)

    # Save phase results
    output_path = Path(args.output_dir) / f"phase{args.phase}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_path))
    logger.info("Phase %d results saved to %s", args.phase, output_path)

    # Print summary
    logger.info("\n--- Phase %d Summary ---", args.phase)
    logger.info(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
