"""Comprehensive evaluation script for LangTraj-OSR.

Usage:
    python -m langtraj_osr.evaluate --checkpoint results/numosim/seed_42/best_model.pt --dataset numosim
    python -m langtraj_osr.evaluate --checkpoint results/geolife/seed_42/best_model.pt --dataset geolife --output_dir results/eval/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core.concepts import (
    ANOMALY_CONCEPTS,
    get_all_definitions,
    get_concept_ids_for_split,
)
from .core.dataset import MobDefBenchDataModule
from .core.utils import get_logger, save_results, set_seed
from .evaluation.metrics import (
    compute_all_metrics,
    compute_calibration_metrics,
    compute_concept_metrics,
    compute_detection_metrics,
    compute_localization_metrics,
    compute_open_set_metrics,
    compute_robustness_metrics,
)
from .models.conformal import ConformalCalibrator
from .models.langtraj_osr import LangTrajConfig, LangTrajOSR

logger = get_logger(__name__)


def _tensor_to_episode_dict(ep_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "zone_id": ep_tensor[:, :, 0].long(),
        "poi_role": ep_tensor[:, :, 1].long(),
        "time_bin": ep_tensor[:, :, 2].long(),
        "dwell_bin": ep_tensor[:, :, 3].long(),
        "transition_type": ep_tensor[:, :, 4].long(),
        "trip_length_change": ep_tensor[:, :, 5].float(),
        "event_flag": ep_tensor[:, :, 6].long(),
        "companion_flag": ep_tensor[:, :, 7].long(),
    }


def _batch_user_prototypes(
    user_ids: List[str],
    user_prototypes: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    default_proto = {
        "mu": torch.zeros(8, 256),
        "sigma": torch.ones(8, 256),
        "pi": torch.ones(8) / 8,
    }
    mus, sigmas, pis = [], [], []
    for uid in user_ids:
        proto = user_prototypes.get(uid, default_proto)
        mus.append(proto["mu"].to(device))
        sigmas.append(proto["sigma"].to(device))
        pis.append(proto["pi"].to(device))
    return {
        "mu": torch.stack(mus),
        "sigma": torch.stack(sigmas),
        "pi": torch.stack(pis),
    }


@torch.no_grad()
def collect_predictions(
    model: LangTrajOSR,
    loader: DataLoader,
    device: torch.device,
    user_prototypes: Dict[str, Dict[str, torch.Tensor]],
    definition_bank: List[str],
) -> Dict[str, np.ndarray]:
    """Collect model outputs for a full dataset split."""
    model.eval()
    all_scores, all_labels, all_energies = [], [], []
    all_v_x, all_prim_labels = [], []
    all_episode_attn = []

    for batch in loader:
        ep_tensor = batch["episode_tensor"].to(device)
        pad_mask = batch["mask"].to(device)
        labels = batch["label"]
        prim_labels = batch["primitive_labels"]
        user_ids = batch["user_id"]

        episodes = _tensor_to_episode_dict(ep_tensor)
        proto_batch = _batch_user_prototypes(user_ids, user_prototypes, device)
        outputs = model(episodes, ~pad_mask, proto_batch, definition_bank)

        all_scores.append(outputs["concept_scores"].cpu().numpy())
        all_labels.append(labels.numpy())
        all_energies.append(outputs["E_norm"].cpu().numpy())
        all_v_x.append(outputs["v_x"].cpu().numpy())
        all_prim_labels.append(prim_labels.max(dim=1).values.numpy())
        all_episode_attn.append(outputs["episode_attention"].cpu().numpy())

    return {
        "concept_scores": np.concatenate(all_scores),
        "labels": np.concatenate(all_labels),
        "energies": np.concatenate(all_energies),
        "v_x": np.concatenate(all_v_x),
        "primitive_labels": np.concatenate(all_prim_labels),
        "episode_attention": np.concatenate(all_episode_attn),
    }


def evaluate_split(
    preds: Dict[str, np.ndarray],
    split_name: str,
    calibrator: Optional[ConformalCalibrator] = None,
) -> Dict[str, Any]:
    """Compute all metrics for a single evaluation split."""
    labels = preds["labels"]
    energies = preds["energies"]
    concept_scores = preds["concept_scores"]
    v_x = preds["v_x"]
    prim_labels = preds["primitive_labels"]

    results: Dict[str, Any] = {"split": split_name, "n_samples": len(labels)}

    # 1. Detection metrics (binary: normal vs anomaly)
    y_true_binary = (labels > 0).astype(int)
    results["detection"] = compute_detection_metrics(y_true_binary, energies)

    # 2. Open-set metrics
    # Unknown concepts have IDs 23-25 in MobDef-Bench (not -1).
    # known_mask covers ALL positives; unknown_mask is the subset with no definition.
    unknown_concept_ids = set(get_concept_ids_for_split("unknown"))
    known_mask = labels > 0
    unknown_mask = np.isin(labels, list(unknown_concept_ids))
    if known_mask.any() and unknown_mask.any():
        results["open_set"] = compute_open_set_metrics(
            energies[known_mask], energies[unknown_mask]
        )

    # 3. Concept recognition metrics (for known anomalies)
    if known_mask.any():
        results["concept"] = compute_concept_metrics(
            concept_scores[known_mask], labels[known_mask]
        )

    # 4. Calibration metrics
    if calibrator is not None:
        y_prob = 1.0 / (1.0 + np.exp(-energies))  # sigmoid
        results["calibration"] = compute_calibration_metrics(
            y_true_binary, y_prob
        )
        # Conformal set sizes
        if hasattr(calibrator, "concept_thresholds") and calibrator.concept_thresholds:
            set_sizes = []
            # Use the mean threshold across all concepts as a global fallback
            thresholds = list(calibrator.concept_thresholds.values())
            global_thresh = float(np.mean(thresholds)) if thresholds else 0.0
            for i in range(len(concept_scores)):
                n_accepted = sum(
                    1 for k, s in enumerate(concept_scores[i])
                    if s >= calibrator.concept_thresholds.get(k, global_thresh)
                )
                set_sizes.append(n_accepted)
            results["calibration"]["mean_set_size"] = float(np.mean(set_sizes))
            results["calibration"]["median_set_size"] = float(np.median(set_sizes))

    # 5. Primitive prediction quality
    if prim_labels.sum() > 0:
        from sklearn.metrics import average_precision_score
        prim_aps = []
        for j in range(v_x.shape[1]):
            if prim_labels[:, j].sum() > 0:
                prim_aps.append(average_precision_score(prim_labels[:, j], v_x[:, j]))
        if prim_aps:
            results["primitive_map"] = float(np.mean(prim_aps))

    return results


def evaluate_paraphrase_robustness(
    model: LangTrajOSR,
    loader: DataLoader,
    device: torch.device,
    user_prototypes: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    """Evaluate robustness to definition paraphrases."""
    all_defs = get_all_definitions(include_paraphrases=True)

    # For each concept, evaluate with each paraphrase
    concept_results: Dict[int, List[float]] = {}

    for concept_id, definitions in all_defs.items():
        if concept_id <= 0 or len(definitions) < 2:
            continue
        aurocs = []
        for def_text in definitions:
            def_bank = [def_text]
            preds = collect_predictions(model, loader, device, user_prototypes, def_bank)
            y_true = (preds["labels"] == concept_id).astype(int)
            if y_true.sum() == 0:
                continue
            from sklearn.metrics import roc_auc_score
            try:
                auroc = roc_auc_score(y_true, preds["concept_scores"][:, 0])
                aurocs.append(auroc)
            except ValueError:
                pass
        if aurocs:
            concept_results[concept_id] = aurocs

    # Compute worst-paraphrase gap
    worst_gaps = []
    for cid, aurocs in concept_results.items():
        if len(aurocs) >= 2:
            worst_gaps.append(max(aurocs) - min(aurocs))

    return {
        "per_concept_aurocs": {str(k): v for k, v in concept_results.items()},
        "worst_paraphrase_gap": float(np.mean(worst_gaps)) if worst_gaps else 0.0,
        "max_paraphrase_gap": float(max(worst_gaps)) if worst_gaps else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LangTraj-OSR")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="numosim")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval_paraphrase", action="store_true")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--text_encoder", type=str, default=None,
                        help="Override text encoder path (for offline use). "
                             "If not set, uses the value saved in checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = LangTrajConfig(**checkpoint["config"])
    # Override text encoder path for offline use
    if args.text_encoder is not None:
        config.text_encoder_name = args.text_encoder
    model = LangTrajOSR(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded model from %s (epoch %d)", checkpoint_path, checkpoint["epoch"])

    # Load calibrator
    calibrator = None
    cal_path = checkpoint_path.parent / "calibrator.json"
    if cal_path.exists():
        calibrator = ConformalCalibrator()
        calibrator.load(str(cal_path))
        logger.info("Loaded conformal calibrator")

    # Load data
    if args.use_synthetic:
        from .benchmark.synthetic_data import SyntheticMobilityGenerator
        gen = SyntheticMobilityGenerator(n_users=100, seed=args.seed)
        trajs = gen.generate(trips_per_user=50)
        n = len(trajs)
        concept_defs = get_all_definitions(include_paraphrases=True)
        user_histories: Dict[str, list] = {}
        for t in trajs[:int(0.7 * n)]:
            if t.label == 0:
                user_histories.setdefault(t.user_id, []).append(t)
        data_module = MobDefBenchDataModule(
            trajectories={"train": trajs[:int(0.7*n)], "val": trajs[int(0.7*n):int(0.85*n)], "test": trajs[int(0.85*n):]},
            concept_definitions=concept_defs,
            user_histories=user_histories,
            batch_size=args.batch_size,
        )
    else:
        data_module = MobDefBenchDataModule.load_dataset(
            args.dataset, batch_size=args.batch_size
        )

    # Build definition bank
    all_defs = get_all_definitions(include_paraphrases=False)
    def_bank = [all_defs[cid][0] for cid in sorted(all_defs.keys()) if all_defs[cid]]

    # Fit user prototypes (from training data)
    from .train import fit_user_routines
    train_loader = data_module.train_dataloader()
    user_prototypes = fit_user_routines(model, train_loader, device)

    # ---- Overall test evaluation ----
    logger.info("=== Overall Test Evaluation ===")
    test_loader = data_module.test_dataloader()
    test_preds = collect_predictions(model, test_loader, device, user_prototypes, def_bank)
    test_results = evaluate_split(test_preds, "test_overall", calibrator)
    logger.info("Test AUROC: %.4f | AUPRC: %.4f",
                test_results["detection"].get("auroc", 0),
                test_results["detection"].get("auprc", 0))

    # ---- Per concept-split evaluation ----
    logger.info("=== Per-Split Evaluation ===")
    concept_loaders = data_module.concept_split_dataloaders()
    split_results = {}
    for split_name, loader in concept_loaders.items():
        preds = collect_predictions(model, loader, device, user_prototypes, def_bank)
        split_results[split_name] = evaluate_split(preds, split_name, calibrator)
        det = split_results[split_name].get("detection", {})
        logger.info("  %s — AUROC: %.4f | AUPRC: %.4f | FPR@95: %.4f",
                     split_name, det.get("auroc", 0), det.get("auprc", 0), det.get("fpr_at_95tpr", 0))

    # ---- Paraphrase robustness ----
    para_results = {}
    if args.eval_paraphrase:
        logger.info("=== Paraphrase Robustness ===")
        para_results = evaluate_paraphrase_robustness(
            model, test_loader, device, user_prototypes
        )
        logger.info("Worst paraphrase gap: %.4f", para_results.get("worst_paraphrase_gap", 0))

    # ---- Save all results ----
    full_results = {
        "checkpoint": str(checkpoint_path),
        "dataset": args.dataset,
        "seed": args.seed,
        "test_overall": test_results,
        "per_split": split_results,
        "paraphrase_robustness": para_results,
    }
    save_results(full_results, str(output_dir / "eval_results.json"))

    # Save summary CSV
    import csv
    csv_path = output_dir / "eval_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "auroc", "auprc", "fpr_at_95tpr"])
        for name, res in [("overall", test_results)] + list(split_results.items()):
            det = res.get("detection", {})
            writer.writerow([
                name,
                f"{det.get('auroc', 0):.4f}",
                f"{det.get('auprc', 0):.4f}",
                f"{det.get('fpr_at_95tpr', 0):.4f}",
            ])

    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
