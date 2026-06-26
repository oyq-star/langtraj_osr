"""Unified training/evaluation script for all LangTraj-OSR baselines.

Usage:
  python -m langtraj_osr.baselines.run_baseline --baseline norm_only --dataset numosim --seed 42
  python -m langtraj_osr.baselines.run_baseline --baseline dsl_xl --dataset geolife --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

# ---------------------------------------------------------------------------
# Baseline imports
# ---------------------------------------------------------------------------
from .norm_only import NormOnlyModel
from .dsl_xl import DSLXLModel
from .nl2dsl import NL2DSLModel
from .canonical_json import CanonicalJSONModel
from .direct_text import DirectTextModel
from .lm_tad import LMTADModel
from .atrom_ossl import ATROMModel
from .backbone_max import BackboneMaxModel

logger = logging.getLogger(__name__)

BASELINE_REGISTRY: Dict[str, type] = {
    "norm_only": NormOnlyModel,
    "dsl_xl": DSLXLModel,
    "nl2dsl": NL2DSLModel,
    "canonical_json": CanonicalJSONModel,
    "direct_text": DirectTextModel,
    "lm_tad": LMTADModel,
    "atrom_ossl": ATROMModel,
    "backbone_max": BackboneMaxModel,
}

SPLIT_NAMES: List[str] = ["seen", "zs_comp", "zs_family", "unknown"]


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class MobDefBenchDataset(Dataset):
    """Lightweight dataset wrapper for MobDefBench trajectory data.

    Expects a directory with:
      - episodes.pt: dict of tensors, each (N, L)
      - labels.pt: LongTensor (N,)
      - user_prototypes.pt: dict with mu (N,K,D), sigma (N,K,D), pi (N,K)
      - split_indices.pt: dict mapping split name -> LongTensor of indices
      - (optional) definitions.json: list of text definitions
      - (optional) dsl_slots.pt: LongTensor (K, 12)
      - (optional) def_token_ids.pt: LongTensor (K, S)
      - (optional) def_attention_mask.pt: BoolTensor (K, S)
    """

    def __init__(self, data_dir: str, split: str = "train") -> None:
        self.data_dir = Path(data_dir)
        self.split = split

        self.episodes: Dict[str, torch.Tensor] = torch.load(
            self.data_dir / "episodes.pt", weights_only=False
        )
        self.labels: torch.Tensor = torch.load(
            self.data_dir / "labels.pt", weights_only=False
        )
        self.user_prototypes: Dict[str, torch.Tensor] = torch.load(
            self.data_dir / "user_prototypes.pt", weights_only=False
        )

        split_indices = torch.load(
            self.data_dir / "split_indices.pt", weights_only=False
        )
        self.indices: torch.Tensor = split_indices[split]

        # Optional assets
        self.definitions: Optional[List[str]] = None
        def_path = self.data_dir / "definitions.json"
        if def_path.exists():
            with open(def_path, "r") as f:
                self.definitions = json.load(f)

        self.dsl_slots: Optional[torch.Tensor] = None
        dsl_path = self.data_dir / "dsl_slots.pt"
        if dsl_path.exists():
            self.dsl_slots = torch.load(dsl_path, weights_only=False)

        self.def_token_ids: Optional[torch.Tensor] = None
        self.def_attention_mask: Optional[torch.Tensor] = None
        tok_path = self.data_dir / "def_token_ids.pt"
        if tok_path.exists():
            self.def_token_ids = torch.load(tok_path, weights_only=False)
            mask_path = self.data_dir / "def_attention_mask.pt"
            if mask_path.exists():
                self.def_attention_mask = torch.load(mask_path, weights_only=False)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self.indices[idx].item()
        sample: Dict[str, Any] = {}

        # Episode features
        sample["episodes"] = {
            key: val[real_idx] for key, val in self.episodes.items()
        }
        sample["label"] = self.labels[real_idx]
        sample["user_prototypes"] = {
            key: val[real_idx] for key, val in self.user_prototypes.items()
        }
        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of samples into batched tensors."""
    episodes: Dict[str, List[torch.Tensor]] = {}
    labels: List[torch.Tensor] = []
    protos: Dict[str, List[torch.Tensor]] = {}

    for sample in batch:
        for key, val in sample["episodes"].items():
            episodes.setdefault(key, []).append(val)
        labels.append(sample["label"])
        for key, val in sample["user_prototypes"].items():
            protos.setdefault(key, []).append(val)

    return {
        "episodes": {k: torch.stack(v) for k, v in episodes.items()},
        "labels": torch.stack(labels),
        "user_prototypes": {k: torch.stack(v) for k, v in protos.items()},
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute AUROC, AUPR, and F1 for binary anomaly detection.

    Parameters
    ----------
    scores : Tensor (N,) — anomaly scores (higher = more anomalous).
    labels : Tensor (N,) — 0 = normal, >0 = anomaly.

    Returns
    -------
    dict with auroc, aupr, f1.
    """
    scores_np = scores.cpu().numpy()
    binary_labels = (labels > 0).cpu().numpy().astype(int)

    # Handle degenerate cases
    if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
        return {"auroc": 0.0, "aupr": 0.0, "f1": 0.0}

    # AUROC (manual trapezoid — avoids sklearn dependency)
    sorted_indices = np.argsort(-scores_np)
    sorted_labels = binary_labels[sorted_indices]

    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    n_pos = binary_labels.sum()
    n_neg = len(binary_labels) - n_pos

    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)

    # Trapezoid AUROC
    auroc = float(np.trapz(tpr, fpr))

    # AUPR (precision-recall)
    precision = tp / (tp + fp + 1e-12)
    recall = tpr
    aupr = float(np.trapz(precision, recall))

    # F1 at best threshold
    if threshold is None:
        f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
        best_idx = int(np.argmax(f1_scores))
        f1 = float(f1_scores[best_idx])
    else:
        preds = (scores_np >= threshold).astype(int)
        tp_f1 = ((preds == 1) & (binary_labels == 1)).sum()
        fp_f1 = ((preds == 1) & (binary_labels == 0)).sum()
        fn_f1 = ((preds == 0) & (binary_labels == 1)).sum()
        prec = tp_f1 / max(tp_f1 + fp_f1, 1)
        rec = tp_f1 / max(tp_f1 + fn_f1, 1)
        f1 = float(2 * prec * rec / max(prec + rec, 1e-12))

    return {"auroc": auroc, "aupr": aupr, "f1": f1}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def create_model(
    baseline_name: str,
    dataset: MobDefBenchDataset,
    d_model: int = 256,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Instantiate a baseline model.

    Parameters
    ----------
    baseline_name : str
        One of the keys in BASELINE_REGISTRY.
    dataset : MobDefBenchDataset
        Used to extract definitions, DSL slots, etc.
    d_model : int
        Model dimensionality.
    device : torch.device

    Returns
    -------
    nn.Module
    """
    common_kwargs: Dict[str, Any] = {
        "poi_vocab_size": 64,
        "d_model": d_model,
        "nhead": 4,
        "num_layers": 4,
        "n_prototypes": 8,
        "dropout": 0.1,
    }

    if baseline_name == "norm_only":
        model = NormOnlyModel(**common_kwargs)

    elif baseline_name == "dsl_xl":
        model = DSLXLModel(**common_kwargs, n_primitives=10, lambda_attr=0.5)

    elif baseline_name == "nl2dsl":
        model = NL2DSLModel(
            definitions=dataset.definitions,
            **common_kwargs,
            n_primitives=10,
            lambda_attr=0.5,
        )

    elif baseline_name == "canonical_json":
        model = CanonicalJSONModel(**common_kwargs)

    elif baseline_name == "direct_text":
        model = DirectTextModel(**common_kwargs, text_vocab_size=10000)

    elif baseline_name == "lm_tad":
        model = LMTADModel(
            poi_vocab_size=common_kwargs["poi_vocab_size"],
            d_model=d_model,
            nhead=common_kwargs["nhead"],
            num_layers=common_kwargs["num_layers"],
            dropout=common_kwargs["dropout"],
        )

    elif baseline_name == "atrom_ossl":
        model = ATROMModel(
            n_known_classes=10,
            **common_kwargs,
            lambda_reciprocal=1.0,
        )

    elif baseline_name == "backbone_max":
        model = BackboneMaxModel(**common_kwargs, text_vocab_size=10000)

    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    return model.to(device)


# ---------------------------------------------------------------------------
# Training step dispatchers
# ---------------------------------------------------------------------------
def train_step(
    model: nn.Module,
    batch: Dict[str, Any],
    baseline_name: str,
    dataset: MobDefBenchDataset,
    device: torch.device,
) -> torch.Tensor:
    """Compute loss for one training batch, dispatching to the right model API."""
    episodes = {k: v.to(device) for k, v in batch["episodes"].items()}
    labels = batch["labels"].to(device)
    user_protos = {k: v.to(device) for k, v in batch["user_prototypes"].items()}

    if baseline_name == "norm_only":
        return model.compute_loss(episodes, user_protos, labels)

    elif baseline_name == "dsl_xl":
        dsl_slots = dataset.dsl_slots
        if dsl_slots is None:
            dsl_slots = torch.zeros(25, 12, dtype=torch.long)
        dsl_slots = dsl_slots.to(device)
        return model.compute_loss(episodes, dsl_slots, user_protos, labels)

    elif baseline_name == "nl2dsl":
        return model.compute_loss(episodes, user_protos, labels)

    elif baseline_name == "canonical_json":
        return model.compute_loss(episodes, user_protos, labels)

    elif baseline_name == "direct_text":
        if dataset.def_token_ids is None:
            B = labels.shape[0]
            return torch.tensor(0.0, device=device, requires_grad=True)
        def_tok = dataset.def_token_ids.to(device)
        def_mask = dataset.def_attention_mask.to(device) if dataset.def_attention_mask is not None else None
        return model.compute_loss(episodes, def_tok, user_protos, labels, def_attention_mask=def_mask)

    elif baseline_name == "lm_tad":
        # LM-TAD trains on normal data only (reconstruction)
        normal_mask = labels == 0
        if not normal_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        normal_episodes = {k: v[normal_mask] for k, v in episodes.items()}
        return model.compute_loss(normal_episodes)

    elif baseline_name == "atrom_ossl":
        return model.compute_loss(episodes, user_protos, labels)

    elif baseline_name == "backbone_max":
        if dataset.def_token_ids is None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        def_tok = dataset.def_token_ids.to(device)
        def_mask = dataset.def_attention_mask.to(device) if dataset.def_attention_mask is not None else None
        return model.compute_loss(episodes, def_tok, user_protos, labels, def_attention_mask=def_mask)

    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")


@torch.no_grad()
def eval_step(
    model: nn.Module,
    batch: Dict[str, Any],
    baseline_name: str,
    dataset: MobDefBenchDataset,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute anomaly scores and labels for one evaluation batch.

    Returns
    -------
    scores : Tensor (B,)
    labels : Tensor (B,)
    """
    model.eval()
    episodes = {k: v.to(device) for k, v in batch["episodes"].items()}
    labels = batch["labels"].to(device)
    user_protos = {k: v.to(device) for k, v in batch["user_prototypes"].items()}

    if baseline_name == "norm_only":
        scores = model.predict(episodes, user_protos)

    elif baseline_name == "dsl_xl":
        dsl_slots = dataset.dsl_slots
        if dsl_slots is None:
            dsl_slots = torch.zeros(25, 12, dtype=torch.long)
        dsl_slots = dsl_slots.to(device)
        def_scores, E_norm = model.predict(episodes, dsl_slots, user_protos)
        scores = def_scores.max(dim=1).values + 0.1 * E_norm

    elif baseline_name == "nl2dsl":
        def_scores, E_norm = model.predict(episodes, user_protos)
        scores = def_scores.max(dim=1).values + 0.1 * E_norm

    elif baseline_name == "canonical_json":
        def_scores, E_norm = model.predict(episodes, user_protos)
        scores = def_scores.max(dim=1).values + 0.1 * E_norm

    elif baseline_name == "direct_text":
        if dataset.def_token_ids is None:
            scores = E_norm = model.predict_norm(episodes, user_protos) if hasattr(model, 'predict_norm') else torch.zeros(labels.shape[0], device=device)
        else:
            def_tok = dataset.def_token_ids.to(device)
            def_mask = dataset.def_attention_mask.to(device) if dataset.def_attention_mask is not None else None
            def_scores, E_norm = model.predict(episodes, def_tok, user_protos, def_attention_mask=def_mask)
            scores = def_scores.max(dim=1).values + 0.1 * E_norm

    elif baseline_name == "lm_tad":
        scores = model.predict(episodes)

    elif baseline_name == "atrom_ossl":
        class_logits, unknown_score, anomaly_score = model.predict(episodes, user_protos)
        scores = anomaly_score

    elif baseline_name == "backbone_max":
        if dataset.def_token_ids is None:
            scores = torch.zeros(labels.shape[0], device=device)
        else:
            def_tok = dataset.def_token_ids.to(device)
            def_mask = dataset.def_attention_mask.to(device) if dataset.def_attention_mask is not None else None
            fused, E_norm = model.predict(episodes, def_tok, user_protos, def_attention_mask=def_mask)
            scores = fused.max(dim=1).values

    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    return scores.cpu(), labels.cpu()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train_and_evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """Full training and evaluation pipeline for a single baseline run.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments.

    Returns
    -------
    dict — results including metrics per split.
    """
    # --- Setup ---
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Baseline: {args.baseline} | Dataset: {args.dataset} | Seed: {args.seed}")
    logger.info(f"Device: {device}")

    # --- Data ---
    test_dataset_synthetic = None
    if getattr(args, "use_synthetic", False):
        train_dataset, val_dataset, test_dataset_synthetic = _build_synthetic_datasets(args)
    else:
        data_dir = Path(args.data_dir) / args.dataset
        train_dataset = MobDefBenchDataset(str(data_dir), split="train")
        val_dataset = MobDefBenchDataset(str(data_dir), split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # --- Model ---
    model = create_model(args.baseline, train_dataset, d_model=256, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # --- Training with early stopping ---
    best_val_auroc: float = -1.0
    patience_counter: int = 0
    patience: int = 10
    best_checkpoint_path = output_dir / f"{args.baseline}_{args.dataset}_s{args.seed}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            loss = train_step(model, batch, args.baseline, train_dataset, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # --- Validation ---
        all_scores: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for batch in val_loader:
            scores, labels = eval_step(model, batch, args.baseline, val_dataset, device)
            all_scores.append(scores)
            all_labels.append(labels)

        if all_scores:
            val_scores = torch.cat(all_scores)
            val_labels = torch.cat(all_labels)
            val_metrics = compute_metrics(val_scores, val_labels)
            val_auroc = val_metrics["auroc"]
        else:
            val_auroc = 0.0
            val_metrics = {"auroc": 0.0, "aupr": 0.0, "f1": 0.0}

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f} | "
            f"Val AUPR: {val_metrics['aupr']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Early stopping
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"  -> New best model saved (AUROC={val_auroc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  -> Early stopping at epoch {epoch}")
                break

    # --- Load best model ---
    if best_checkpoint_path.exists():
        model.load_state_dict(torch.load(best_checkpoint_path, weights_only=False))
        logger.info(f"Loaded best checkpoint (val AUROC={best_val_auroc:.4f})")

    # --- EVT calibration for ATROM ---
    if args.baseline == "atrom_ossl":
        logger.info("Fitting EVT calibrator on validation data...")
        all_val_episodes: Dict[str, List[torch.Tensor]] = {}
        all_val_protos: Dict[str, List[torch.Tensor]] = {}
        all_val_labels_list: List[torch.Tensor] = []
        for batch in val_loader:
            for k, v in batch["episodes"].items():
                all_val_episodes.setdefault(k, []).append(v)
            for k, v in batch["user_prototypes"].items():
                all_val_protos.setdefault(k, []).append(v)
            all_val_labels_list.append(batch["labels"])

        if all_val_labels_list:
            cat_episodes = {k: torch.cat(v).to(device) for k, v in all_val_episodes.items()}
            cat_protos = {k: torch.cat(v).to(device) for k, v in all_val_protos.items()}
            cat_labels = torch.cat(all_val_labels_list).to(device)
            model.fit_evt(cat_episodes, cat_protos, cat_labels)

    # --- Test evaluation on all splits ---
    results: Dict[str, Any] = {
        "baseline": args.baseline,
        "dataset": args.dataset,
        "seed": args.seed,
        "best_val_auroc": best_val_auroc,
        "metrics_per_split": {},
    }

    # For synthetic mode, evaluate on the full test set and also per-split
    if getattr(args, "use_synthetic", False) and test_dataset_synthetic is not None:
        from ..core.concepts import get_concept_ids_for_split

        split_id_map = {
            "seen": sorted(get_concept_ids_for_split("seen")),
            "zs_comp": sorted(get_concept_ids_for_split("zs_comp")),
            "zs_family": sorted(get_concept_ids_for_split("zs_family")),
            "unknown": sorted(get_concept_ids_for_split("unknown")),
        }

        test_loader = DataLoader(
            test_dataset_synthetic, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

        all_scores = []
        all_labels = []
        for batch in test_loader:
            scores, labels = eval_step(model, batch, args.baseline, test_dataset_synthetic, device)
            all_scores.append(scores)
            all_labels.append(labels)

        if all_scores:
            test_scores = torch.cat(all_scores)
            test_labels = torch.cat(all_labels)

            # Overall test metrics
            overall = compute_metrics(test_scores, test_labels)
            results["test_metrics"] = overall
            logger.info(
                f"Test [overall     ] | "
                f"AUROC: {overall['auroc']:.4f} | "
                f"AUPR: {overall['aupr']:.4f} | "
                f"F1: {overall['f1']:.4f}"
            )

            # Per-split metrics: normal vs anomalies of that split
            for split_name, split_ids in split_id_map.items():
                normal_mask = test_labels == 0
                split_anom_mask = torch.zeros_like(test_labels, dtype=torch.bool)
                for cid in split_ids:
                    split_anom_mask |= (test_labels == cid)
                mask = normal_mask | split_anom_mask
                if mask.sum() > 0 and split_anom_mask.sum() > 0:
                    metrics = compute_metrics(test_scores[mask], test_labels[mask])
                else:
                    metrics = {"auroc": 0.0, "aupr": 0.0, "f1": 0.0}
                results["metrics_per_split"][split_name] = metrics
                logger.info(
                    f"Test [{split_name:12s}] | "
                    f"AUROC: {metrics['auroc']:.4f} | "
                    f"AUPR: {metrics['aupr']:.4f} | "
                    f"F1: {metrics['f1']:.4f}"
                )
    else:
        for split_name in SPLIT_NAMES:
            try:
                test_dataset = MobDefBenchDataset(str(data_dir), split=f"test_{split_name}")
            except (FileNotFoundError, KeyError):
                logger.warning(f"Split test_{split_name} not found, skipping.")
                continue

            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=0,
            )

            all_scores = []
            all_labels = []
            for batch in test_loader:
                scores, labels = eval_step(model, batch, args.baseline, test_dataset, device)
                all_scores.append(scores)
                all_labels.append(labels)

            if all_scores:
                test_scores = torch.cat(all_scores)
                test_labels = torch.cat(all_labels)
                metrics = compute_metrics(test_scores, test_labels)
            else:
                metrics = {"auroc": 0.0, "aupr": 0.0, "f1": 0.0}

            results["metrics_per_split"][split_name] = metrics
            logger.info(
                f"Test [{split_name:12s}] | "
                f"AUROC: {metrics['auroc']:.4f} | "
                f"AUPR: {metrics['aupr']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )

    # --- Save results ---
    results_path = output_dir / f"{args.baseline}_{args.dataset}_s{args.seed}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# Synthetic data support
# ---------------------------------------------------------------------------

class SyntheticBaselineDataset(Dataset):
    """Wraps SemanticTrajectory lists into the format expected by baselines."""

    def __init__(
        self,
        trajectories: list,
        concept_definitions: dict,
        max_len: int = 64,
    ) -> None:
        self.trajectories = trajectories
        self.concept_definitions = concept_definitions
        self.max_len = max_len
        self.definitions: Optional[List[str]] = []
        for cid in sorted(concept_definitions.keys()):
            if concept_definitions[cid]:
                self.definitions.append(concept_definitions[cid][0])
        self.dsl_slots: Optional[torch.Tensor] = None
        self.def_token_ids: Optional[torch.Tensor] = None
        self.def_attention_mask: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj = self.trajectories[idx]
        # Convert SemanticEpisode list to tensor
        eps_lists = [ep.to_list() for ep in traj.episodes[:self.max_len]]
        while len(eps_lists) < self.max_len:
            eps_lists.append([0.0] * 8)
        ep_tensor = torch.tensor(eps_lists, dtype=torch.float32)

        episodes = {
            "zone_id": ep_tensor[:, 0].long(),
            "poi_role": ep_tensor[:, 1].long().clamp(0, 63),
            "time_bin": ep_tensor[:, 2].long().clamp(0, 167),
            "dwell_bin": ep_tensor[:, 3].long().clamp(0, 15),
            "transition_type": ep_tensor[:, 4].long().clamp(0, 3),
            "trip_length_change": ep_tensor[:, 5].float(),
            "event_flag": ep_tensor[:, 6].long().clamp(0, 1),
            "companion_flag": ep_tensor[:, 7].long().clamp(0, 1),
        }

        label = torch.tensor(traj.label, dtype=torch.long)
        user_prototypes = {
            "mu": torch.zeros(8, 256),
            "sigma": torch.ones(8, 256),
            "pi": torch.ones(8) / 8,
        }

        return {
            "episodes": episodes,
            "label": label,
            "user_prototypes": user_prototypes,
        }


def _build_synthetic_datasets(args):
    """Build synthetic train/val/test datasets using MobDefBenchBuilder."""
    import tempfile
    from ..benchmark.benchmark_builder import MobDefBenchBuilder
    from ..core.concepts import get_all_definitions

    with tempfile.TemporaryDirectory() as tmpdir:
        builder = MobDefBenchBuilder(
            data_dir=tmpdir,        # will fall back to synthetic generation
            output_dir=tmpdir,
            seed=args.seed,
        )
        benchmarks = builder.build(datasets=["numosim"])
        bench = benchmarks["numosim"]

    concept_defs = get_all_definitions(include_paraphrases=True)

    # Combine normal + anomaly trajectories for each split
    train_trajs = bench.train.normal + bench.train.anomalous
    val_trajs = bench.val.normal + bench.val.anomalous
    test_trajs = bench.test.normal + bench.test.anomalous

    # Shuffle
    import random as _rnd
    _rng = _rnd.Random(args.seed)
    _rng.shuffle(train_trajs)
    _rng.shuffle(val_trajs)
    _rng.shuffle(test_trajs)

    train_ds = SyntheticBaselineDataset(train_trajs, concept_defs)
    val_ds = SyntheticBaselineDataset(val_trajs, concept_defs)
    test_ds = SyntheticBaselineDataset(test_trajs, concept_defs)

    logger.info("Synthetic data: train=%d (norm=%d, anom=%d), val=%d, test=%d",
                len(train_trajs), len(bench.train.normal), len(bench.train.anomalous),
                len(val_trajs), len(test_trajs))

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate LangTraj-OSR baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        choices=list(BASELINE_REGISTRY.keys()),
        help="Baseline method to run.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="numosim",
        help="Dataset name (subdirectory of data_dir).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/baselines",
        help="Directory for checkpoints and results.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mobdefbench",
        help="Root data directory containing dataset subdirectories.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id.")
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Use synthetic data instead of pre-processed files.",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Text encoder model name or local path (for offline use).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
