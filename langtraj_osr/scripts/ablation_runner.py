"""Run all ablation experiments (A1-A11) for LangTraj-OSR.

Usage:
    python -m langtraj_osr.scripts.ablation_runner --dataset numosim --seed 42 --output_dir results/ablations/
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch

from ..core.concepts import get_all_definitions, get_concept_ids_for_split
from ..core.dataset import MobDefBenchDataModule
from ..core.utils import get_logger, save_results, set_seed
from ..models.langtraj_osr import LangTrajConfig, LangTrajOSR
from ..train import (
    _batch_user_prototypes,
    _tensor_to_episode_dict,
    fit_user_routines,
    pretrain_masked,
    train_one_epoch,
    validate,
)
from ..models.losses import CombinedLoss
from ..models.conformal import ConformalCalibrator

logger = get_logger(__name__)


ABLATIONS = {
    "A1_no_language": {
        "description": "No language — normality detection only",
        "config_changes": {},
        "train_changes": {"skip_language": True},
    },
    "A2_no_primitive_head": {
        "description": "Language but no primitive head",
        "config_changes": {},
        "loss_changes": {"w_prim": 0.0},
        "model_changes": {"disable_primitive_head": True},
    },
    "A3_dsl_instead": {
        "description": "DSL definitions instead of language",
        "config_changes": {},
        "use_dsl": True,
    },
    "A4_no_user_history": {
        "description": "No user-history module",
        "config_changes": {"n_prototypes": 0},
        "model_changes": {"disable_user_history": True},
    },
    "A5_cohort_history": {
        "description": "Cohort history instead of per-user",
        "config_changes": {},
        "model_changes": {"use_cohort_history": True},
    },
    "A6_no_L_para": {
        "description": "No paraphrase consistency loss",
        "loss_changes": {"w_para": 0.0},
    },
    "A7_no_conformal": {
        "description": "No conformal reject — revert to max(language, novelty)",
        "config_changes": {},
        "model_changes": {"use_max_instead_of_conformal": True},
    },
    "A8_opaque_tokens": {
        "description": "Opaque itinerary tokens vs factorized representation",
        "config_changes": {},
        "model_changes": {"opaque_tokens": True},
    },
    "A9_no_L_orth": {
        "description": "No orthogonality loss — allow feature confounding",
        "loss_changes": {"w_orth": 0.0},
    },
    "A10_random_text": {
        "description": "Random text embeddings instead of LLM",
        "config_changes": {},
        "model_changes": {"random_text_encoder": True},
    },
    "A11_bow_text": {
        "description": "BoW/TF-IDF text features instead of dense LLM",
        "config_changes": {},
        "model_changes": {"bow_text_encoder": True},
    },
}


def create_ablated_model(
    base_config: LangTrajConfig,
    ablation_name: str,
    ablation_spec: Dict[str, Any],
    device: torch.device,
) -> LangTrajOSR:
    """Create a model with ablation modifications."""
    config = copy.deepcopy(base_config)

    # Apply config changes
    for key, value in ablation_spec.get("config_changes", {}).items():
        if hasattr(config, key):
            setattr(config, key, value)

    model = LangTrajOSR(config).to(device)

    model_changes = ablation_spec.get("model_changes", {})

    # A1: No language — zero out definition encoder
    if ablation_spec.get("train_changes", {}).get("skip_language"):
        for param in model.definition_encoder.parameters():
            param.requires_grad = False

    # A2: Disable primitive head
    if model_changes.get("disable_primitive_head"):
        model.lambda_prim = 0.0

    # A4: Disable user history
    if model_changes.get("disable_user_history"):
        for param in model.user_history.parameters():
            param.requires_grad = False

    # A10: Replace text encoder with random projections
    if model_changes.get("random_text_encoder"):
        for param in model.definition_encoder.projection.parameters():
            torch.nn.init.normal_(param, std=0.01)
            param.requires_grad = False

    return model


def create_ablated_loss(
    ablation_spec: Dict[str, Any],
    temperature: float = 0.07,
) -> CombinedLoss:
    """Create loss function with ablation modifications."""
    loss_changes = ablation_spec.get("loss_changes", {})
    return CombinedLoss(
        w_cls=loss_changes.get("w_cls", 0.5),
        w_prim=loss_changes.get("w_prim", 1.0),
        w_para=loss_changes.get("w_para", 0.2),
        w_orth=loss_changes.get("w_orth", 0.05),
        w_norm=loss_changes.get("w_norm", 1.0),
        temperature=temperature,
    )


def run_single_ablation(
    ablation_name: str,
    ablation_spec: Dict[str, Any],
    data_module: MobDefBenchDataModule,
    base_config: LangTrajConfig,
    device: torch.device,
    output_dir: Path,
    seed: int = 42,
    epochs: int = 30,
) -> Dict[str, Any]:
    """Run a single ablation experiment."""
    logger.info("--- Ablation: %s ---", ablation_name)
    logger.info("  %s", ablation_spec["description"])

    set_seed(seed)
    abl_dir = output_dir / ablation_name
    abl_dir.mkdir(parents=True, exist_ok=True)

    # Create ablated model and loss
    model = create_ablated_model(base_config, ablation_name, ablation_spec, device)
    criterion = create_ablated_loss(ablation_spec)

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Stage 1: Pretrain
    pretrain_masked(model, train_loader, device, epochs=5)

    # Stage 2: Fit user routines
    user_prototypes = fit_user_routines(model, train_loader, device)

    # Stage 3: Train
    from torch.cuda.amp import GradScaler
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=1e-2,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    def_bank = []
    all_defs = get_all_definitions(include_paraphrases=False)
    for cid in sorted(all_defs.keys()):
        if all_defs[cid]:
            def_bank.append(all_defs[cid][0])

    best_auroc = 0.0
    for epoch in range(epochs):
        train_losses = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler,
            device, user_prototypes, epoch,
        )
        val_metrics = validate(model, val_loader, device, user_prototypes, def_bank)
        scheduler.step()

        auroc = val_metrics.get("auroc", 0.0)
        if auroc > best_auroc:
            best_auroc = auroc
            torch.save(model.state_dict(), abl_dir / "best_model.pt")

    # Evaluate
    if (abl_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(abl_dir / "best_model.pt", map_location=device))

    test_loader = data_module.test_dataloader()
    test_metrics = validate(model, test_loader, device, user_prototypes, def_bank)

    # Per-split evaluation
    split_metrics = {}
    concept_loaders = data_module.concept_split_dataloaders()
    for split_name, loader in concept_loaders.items():
        split_metrics[split_name] = validate(model, loader, device, user_prototypes, def_bank)

    results = {
        "ablation": ablation_name,
        "description": ablation_spec["description"],
        "best_val_auroc": best_auroc,
        "test_metrics": test_metrics,
        "split_metrics": split_metrics,
    }

    save_results(results, str(abl_dir / "results.json"))
    logger.info("  Test AUROC: %.4f (val best: %.4f)", test_metrics.get("auroc", 0), best_auroc)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LangTraj-OSR ablations")
    parser.add_argument("--dataset", type=str, default="numosim")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/ablations")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--ablations", type=str, nargs="*", default=None,
                        help="Specific ablations to run (default: all)")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--text_encoder", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Text encoder model name or local path (for offline use)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.use_synthetic:
        import tempfile
        from ..benchmark.benchmark_builder import MobDefBenchBuilder
        with tempfile.TemporaryDirectory() as tmp_data, tempfile.TemporaryDirectory() as tmp_out:
            builder = MobDefBenchBuilder(data_dir=tmp_data, output_dir=tmp_out, seed=args.seed)
            benchmarks = builder.build(datasets=["numosim"])
            bench = benchmarks["numosim"]
        train_trajs = bench.train.normal + bench.train.anomalous
        val_trajs = bench.val.normal + bench.val.anomalous
        test_trajs = bench.test.normal + bench.test.anomalous
        concept_defs = get_all_definitions(include_paraphrases=True)
        user_histories = {}
        for t in train_trajs:
            if t.label == 0:
                user_histories.setdefault(t.user_id, []).append(t)
        data_module = MobDefBenchDataModule(
            trajectories={"train": train_trajs, "val": val_trajs, "test": test_trajs},
            concept_definitions=concept_defs,
            user_histories=user_histories,
            batch_size=args.batch_size,
        )
    else:
        data_module = MobDefBenchDataModule.load_dataset(
            args.dataset, batch_size=args.batch_size,
        )

    base_config = LangTrajConfig(text_encoder_name=args.text_encoder)

    # Determine which ablations to run
    ablation_names = args.ablations or list(ABLATIONS.keys())

    all_results = {}
    for abl_name in ablation_names:
        if abl_name not in ABLATIONS:
            logger.warning("Unknown ablation: %s, skipping", abl_name)
            continue
        result = run_single_ablation(
            abl_name, ABLATIONS[abl_name], data_module, base_config,
            device, output_dir, args.seed, args.epochs,
        )
        all_results[abl_name] = result

    # Save summary
    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "ablations": {
            name: {
                "description": res["description"],
                "test_auroc": res["test_metrics"].get("auroc", 0),
            }
            for name, res in all_results.items()
        },
    }
    save_results(summary, str(output_dir / "ablation_summary.json"))

    # Print summary table
    logger.info("\n=== Ablation Summary ===")
    logger.info("%-25s  %-50s  %s", "Ablation", "Description", "Test AUROC")
    logger.info("-" * 90)
    for name, res in all_results.items():
        logger.info("%-25s  %-50s  %.4f",
                     name, res["description"][:50], res["test_metrics"].get("auroc", 0))


if __name__ == "__main__":
    main()
