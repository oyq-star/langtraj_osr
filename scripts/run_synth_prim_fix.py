#!/usr/bin/env python3
"""Run synthetic zero-shot typing experiment with primitive alignment fixes.

This script:
1. Uses existing checkpoints trained on synthetic data
2. Evaluates typed metrics (Top-1, M-F1) for seen and zs_comp concepts
3. Compares with/without declared primitive matching bonus at inference
"""
import sys
sys.path.insert(0, "/home/hello/ouyangqi")

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

from langtraj_osr.core.concepts import (
    ANOMALY_CONCEPTS, CONCEPT_BY_ID, get_all_definitions,
    get_concept_ids_for_split,
)
from langtraj_osr.core.dataset import MobDefBenchDataModule
from langtraj_osr.core.utils import set_seed
from langtraj_osr.models.langtraj_osr import LangTrajConfig, LangTrajOSR

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _tensor_to_episode_dict(ep_tensor):
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


def _batch_user_prototypes(user_ids, user_prototypes, device):
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


def collect_scores(model, loader, device, user_prototypes, def_bank, declared_prims):
    model.eval()
    all_scores, all_labels, all_energies, all_vx = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            ep_tensor = batch["episode_tensor"].to(device)
            pad_mask = batch["mask"].to(device)
            labels = batch["label"]
            user_ids = batch["user_id"]

            episodes = _tensor_to_episode_dict(ep_tensor)
            proto_batch = _batch_user_prototypes(user_ids, user_prototypes, device)
            outputs = model(
                episodes, ~pad_mask, proto_batch, def_bank,
                declared_primitives=declared_prims,
            )

            all_scores.append(outputs["concept_scores"].cpu().numpy())
            all_labels.append(labels.numpy())
            all_energies.append(outputs["E_norm"].cpu().numpy())
            all_vx.append(outputs["v_x"].cpu().numpy())

    return {
        "concept_scores": np.concatenate(all_scores),
        "labels": np.concatenate(all_labels),
        "energies": np.concatenate(all_energies),
        "v_x": np.concatenate(all_vx),
    }


def typed_metrics(concept_scores, labels, concept_ids_in_bank):
    """Compute Top-1 accuracy and Macro-F1 for concept assignment."""
    from sklearn.metrics import f1_score

    valid_mask = np.isin(labels, concept_ids_in_bank)
    if not valid_mask.any():
        return {"top1": 0.0, "macro_f1": 0.0, "n_samples": 0}

    scores = concept_scores[valid_mask]
    true_labels = labels[valid_mask]

    id_to_idx = {cid: i for i, cid in enumerate(concept_ids_in_bank)}
    true_indices = np.array([id_to_idx.get(l, -1) for l in true_labels])

    ok = true_indices >= 0
    if not ok.any():
        return {"top1": 0.0, "macro_f1": 0.0, "n_samples": 0}

    scores = scores[ok]
    true_indices = true_indices[ok]
    pred_indices = scores.argmax(axis=1)
    top1 = float((pred_indices == true_indices).mean())

    try:
        mf1 = f1_score(true_indices, pred_indices, average="macro", zero_division=0)
    except Exception:
        mf1 = 0.0

    return {"top1": top1, "macro_f1": float(mf1), "n_samples": int(ok.sum())}


def run_single_seed(seed, output_dir):
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("=== Seed %d ===", seed)

    # Always train from scratch with corrected primitive alignment
    logger.info("Training from scratch with fixed primitives...")
    from langtraj_osr.train import main as train_main
    seed_dir = output_dir / f"seed_{seed}"
    old_argv = sys.argv
    sys.argv = [
        "train",
        "--use_synthetic", "--seed", str(seed),
        "--epochs", "50", "--batch_size", "256",
        "--output_dir", str(seed_dir),
        "--no_amp",
        "--text_encoder",
        "/home/hello/.cache/huggingface/hub/models--sentence-transformers--"
        "all-MiniLM-L6-v2/snapshots/"
        "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
    ]
    train_main()
    sys.argv = old_argv

    ckpt_path = seed_dir / "numosim" / f"seed_{seed}" / "best_model.pt"
    if not ckpt_path.exists():
        # Try alternate path
        import glob
        candidates = glob.glob(str(seed_dir / "**" / "best_model.pt"), recursive=True)
        if candidates:
            ckpt_path = Path(candidates[0])
        else:
            raise FileNotFoundError(f"No best_model.pt found under {seed_dir}")

    # Load model
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = LangTrajConfig(**checkpoint["config"])
    config.text_encoder_name = (
        "/home/hello/.cache/huggingface/hub/models--sentence-transformers--"
        "all-MiniLM-L6-v2/snapshots/"
        "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    )
    model = LangTrajOSR(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    logger.info("Loaded freshly trained model from %s", ckpt_path)

    # Build data (synthetic with anomaly injection)
    import tempfile
    from langtraj_osr.benchmark.benchmark_builder import MobDefBenchBuilder

    with tempfile.TemporaryDirectory() as tmp_data_dir:
        with tempfile.TemporaryDirectory() as tmp_out_dir:
            builder = MobDefBenchBuilder(
                data_dir=tmp_data_dir,
                output_dir=tmp_out_dir,
                seed=seed,
            )
            benchmarks = builder.build(datasets=["numosim"])
            bench = benchmarks["numosim"]

    def _combine(split):
        return split.normal + split.anomalous

    train_trajs = _combine(bench.train)
    val_trajs = _combine(bench.val)
    test_trajs = _combine(bench.test)

    logger.info(
        "  Data: train=%d, val=%d, test=%d",
        len(train_trajs), len(val_trajs), len(test_trajs),
    )

    concept_defs = get_all_definitions(include_paraphrases=True)
    user_histories = {}
    for t in train_trajs:
        if t.label == 0:
            user_histories.setdefault(t.user_id, []).append(t)

    data_module = MobDefBenchDataModule(
        trajectories={"train": train_trajs, "val": val_trajs, "test": test_trajs},
        concept_definitions=concept_defs,
        user_histories=user_histories,
        batch_size=256,
    )

    # Fit user prototypes
    from langtraj_osr.train import fit_user_routines
    train_loader = data_module.train_dataloader()
    user_prototypes = fit_user_routines(model, train_loader, device)

    # Build definition bank -- all concepts with definitions (IDs 1-22)
    all_defs = get_all_definitions(include_paraphrases=False)
    concept_ids_in_bank = sorted([cid for cid in all_defs if all_defs[cid]])
    def_bank = [all_defs[cid][0] for cid in concept_ids_in_bank]

    # Build declared primitive matrix
    declared_prims = LangTrajOSR.build_declared_primitive_matrix(
        concept_ids_in_bank, n_primitives=10,
    ).to(device)

    logger.info(
        "Definition bank: %d concepts, ids=%s", len(def_bank), concept_ids_in_bank
    )
    logger.info("Declared primitives matrix shape: %s", declared_prims.shape)

    # Collect scores WITH declared primitives
    test_loader = data_module.test_dataloader()
    preds_dp = collect_scores(
        model, test_loader, device, user_prototypes, def_bank, declared_prims
    )

    # Collect scores WITHOUT declared primitives
    preds_no = collect_scores(
        model, test_loader, device, user_prototypes, def_bank, None
    )

    # Compute typed metrics per split
    seen_ids = get_concept_ids_for_split("seen")
    zs_comp_ids = get_concept_ids_for_split("zs_comp")
    zs_fam_ids = get_concept_ids_for_split("zs_family")

    results = {}
    for split_name, split_ids in [
        ("seen", seen_ids), ("zs_comp", zs_comp_ids), ("zs_family", zs_fam_ids),
    ]:
        mask = np.isin(preds_dp["labels"], split_ids)
        m_dp = typed_metrics(
            preds_dp["concept_scores"][mask],
            preds_dp["labels"][mask],
            concept_ids_in_bank,
        )
        m_no = typed_metrics(
            preds_no["concept_scores"][mask],
            preds_no["labels"][mask],
            concept_ids_in_bank,
        )
        results[split_name] = {"with_dp": m_dp, "without_dp": m_no}
        logger.info(
            "  %s: Top-1 w/ DP=%.3f, w/o DP=%.3f, MF1 w/ DP=%.3f, w/o DP=%.3f, n=%d",
            split_name, m_dp["top1"], m_no["top1"],
            m_dp["macro_f1"], m_no["macro_f1"], m_dp["n_samples"],
        )

    # Binary detection AUROC
    from sklearn.metrics import roc_auc_score
    y_true_bin = (preds_dp["labels"] > 0).astype(int)
    if 0 < y_true_bin.sum() < len(y_true_bin):
        auroc = roc_auc_score(y_true_bin, preds_dp["energies"])
        results["binary_auroc"] = float(auroc)
        logger.info("  Binary AUROC: %.4f", auroc)

    # v_x stats
    logger.info(
        "  v_x stats: mean=%.4f, std=%.4f",
        preds_dp["v_x"].mean(), preds_dp["v_x"].std(),
    )

    # Per-concept confusion analysis for zs_comp
    zs_mask = np.isin(preds_dp["labels"], zs_comp_ids)
    if zs_mask.any():
        zs_scores = preds_dp["concept_scores"][zs_mask]
        zs_labels = preds_dp["labels"][zs_mask]
        id_to_idx = {cid: i for i, cid in enumerate(concept_ids_in_bank)}

        for cid in zs_comp_ids:
            c_mask = zs_labels == cid
            if not c_mask.any():
                continue
            avg_scores = zs_scores[c_mask].mean(axis=0)
            top3_idx = np.argsort(avg_scores)[-3:][::-1]
            top3_concepts = [
                (concept_ids_in_bank[i], float(avg_scores[i])) for i in top3_idx
            ]
            correct_idx = id_to_idx.get(cid, -1)
            rank_order = np.argsort(avg_scores)[::-1]
            correct_rank = int(np.where(rank_order == correct_idx)[0][0]) if correct_idx in rank_order else -1
            from langtraj_osr.core.concepts import CONCEPT_BY_ID as CB
            logger.info(
                "  Concept %d (%s): correct_rank=%d, top3=%s",
                cid, CB[cid]["name"], correct_rank, top3_concepts,
            )

    return results


def main():
    output_dir = Path("/home/hello/ouyangqi/results/v8_synth_prim_fix")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for seed in [42]:
        try:
            all_results[str(seed)] = run_single_seed(seed, output_dir)
        except Exception as e:
            logger.error("Seed %d failed: %s", seed, e)
            import traceback
            traceback.print_exc()

    # Save results
    with open(output_dir / "typed_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    logger.info("=== Summary ===")
    for split in ["seen", "zs_comp", "zs_family"]:
        top1s_dp = [
            all_results[s][split]["with_dp"]["top1"]
            for s in all_results if split in all_results[s]
        ]
        top1s_no = [
            all_results[s][split]["without_dp"]["top1"]
            for s in all_results if split in all_results[s]
        ]
        if top1s_dp:
            logger.info(
                "%s Top-1: w/ DP=%.3f+/-%.3f, w/o DP=%.3f+/-%.3f",
                split,
                np.mean(top1s_dp), np.std(top1s_dp),
                np.mean(top1s_no), np.std(top1s_no),
            )


if __name__ == "__main__":
    main()
