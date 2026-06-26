#!/usr/bin/env python3
"""Verify Porto typed metrics don't regress with primitive alignment fixes.

Uses existing v6_porto checkpoints (trained with old primitives) and
re-evaluates with the fixed code. Also tests declared primitive bonus.

Reuses build_porto and run_one from run_typed_foursquare_and_centroid.py.
"""
import sys
sys.path.insert(0, "/home/hello/ouyangqi")

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.chdir("/home/hello/ouyangqi")

import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from langtraj_osr.core.concepts import (
    CONCEPT_BY_ID, get_all_definitions, get_concept_ids_for_split,
)
from langtraj_osr.core.dataset import MobDefBenchDataset, collate_mobdef
from langtraj_osr.core.utils import set_seed
from langtraj_osr.models.langtraj_osr import LangTrajConfig, LangTrajOSR

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEMPERATURE = 0.07

TEXT_ENCODER_PATH = (
    "/home/hello/.cache/huggingface/hub/models--sentence-transformers--"
    "all-MiniLM-L6-v2/snapshots/"
    "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)


def ep_dict(t, device):
    return {
        k: t[:, :, i].long().to(device)
        if i < 5 or i >= 6
        else t[:, :, i].float().to(device)
        for i, k in enumerate([
            "zone_id", "poi_role", "time_bin", "dwell_bin",
            "transition_type", "trip_length_change", "event_flag",
            "companion_flag",
        ])
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


@torch.no_grad()
def collect_cosine_scores(model, loader, c_bank):
    """Cosine similarity scoring (same as v6 eval)."""
    model.eval()
    all_scores, all_labels = [], []
    for b in loader:
        pad_mask = b["mask"].to(DEVICE)
        ep = ep_dict(b["episode_tensor"], DEVICE)
        ep_emb = model.episode_encoder(ep)
        z_x, _ = model.trajectory_encoder(ep_emb, ~pad_mask)
        z_norm = F.normalize(z_x.float(), dim=-1)
        scores = (z_norm @ c_bank.T) / TEMPERATURE
        all_scores.append(scores.cpu().float().numpy())
        all_labels.append(b["label"].numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


@torch.no_grad()
def collect_full_model_scores(model, loader, def_bank, user_prototypes,
                              declared_prims=None):
    """Full model forward with all 4 terms (including optional DP bonus)."""
    model.eval()
    all_scores, all_labels, all_energies = [], [], []
    for b in loader:
        ep_tensor = b["episode_tensor"].to(DEVICE)
        pad_mask = b["mask"].to(DEVICE)
        labels = b["label"]
        user_ids = b["user_id"]

        episodes = ep_dict(ep_tensor, DEVICE)
        proto_batch = _batch_user_prototypes(user_ids, user_prototypes, DEVICE)
        outputs = model(
            episodes, ~pad_mask, proto_batch, def_bank,
            declared_primitives=declared_prims,
        )
        all_scores.append(outputs["concept_scores"].cpu().numpy())
        all_labels.append(labels.numpy())
        all_energies.append(outputs["E_norm"].cpu().numpy())

    return (
        np.concatenate(all_scores),
        np.concatenate(all_labels),
        np.concatenate(all_energies),
    )


def typed_metrics(scores, labels, subset_ids, all_ids):
    from sklearn.metrics import f1_score

    mask = np.isin(labels, subset_ids)
    if mask.sum() == 0:
        return {"top1": 0.0, "macro_f1": 0.0, "n": 0}
    s, l = scores[mask], labels[mask]
    pred_idx = s.argmax(axis=1)
    pred_ids = np.array([all_ids[p] if p < len(all_ids) else -1 for p in pred_idx])
    top1 = float((pred_ids == l).mean())
    mf1 = float(f1_score(l, pred_ids, average="macro", zero_division=0))
    return {"top1": round(top1, 4), "macro_f1": round(mf1, 4), "n": int(mask.sum())}


def eval_seed(seed):
    """Evaluate one Porto v6 checkpoint with fixed code."""
    set_seed(seed)
    logger.info("=== Porto v6 checkpoint, seed %d ===", seed)

    ckpt_path = Path(
        f"/home/hello/ouyangqi/results/v6_porto/numosim/seed_{seed}/best_model.pt"
    )
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return None

    # Load model
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = LangTrajConfig(**checkpoint["config"])
    config.text_encoder_name = TEXT_ENCODER_PATH
    model = LangTrajOSR(config).to(DEVICE)
    model.definition_encoder(["init"])  # lazy init
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    logger.info("Loaded model from %s", ckpt_path)

    # Build Porto data using same approach as run_typed_foursquare_and_centroid.py
    from run_typed_foursquare_and_centroid import build_porto
    bench = build_porto(seed)

    train_all = bench.train.normal + bench.train.anomalous
    test_trajs = bench.test.normal + bench.test.anomalous
    logger.info("Porto data: train=%d, test=%d", len(train_all), len(test_trajs))

    # Build concept bank
    all_defs = get_all_definitions(include_paraphrases=True)
    seen_ids = sorted(get_concept_ids_for_split("seen"))
    zsc_ids = sorted(get_concept_ids_for_split("zs_comp"))
    zsf_ids = sorted(get_concept_ids_for_split("zs_family"))
    full_ids = seen_ids + zsc_ids + zsf_ids
    def_texts = [all_defs[cid][0] for cid in full_ids]

    user_hist = {}
    for t in train_all:
        if t.label == 0:
            user_hist.setdefault(t.user_id, []).append(t)

    # Build data loaders
    from torch.utils.data import DataLoader
    test_ds = MobDefBenchDataset(
        test_trajs, concept_definitions=all_defs, user_histories=user_hist
    )
    train_ds = MobDefBenchDataset(
        train_all, concept_definitions=all_defs, user_histories=user_hist
    )
    test_loader = DataLoader(
        test_ds, batch_size=128, shuffle=False, collate_fn=collate_mobdef
    )
    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=False, collate_fn=collate_mobdef
    )

    # === Method 1: Cosine similarity (same as v6 eval) ===
    with torch.no_grad():
        c_bank, _ = model.definition_encoder(def_texts)
        c_bank = F.normalize(c_bank.float(), dim=-1).to(DEVICE)

    cos_scores, cos_labels = collect_cosine_scores(model, test_loader, c_bank)
    logger.info("Cosine scores shape: %s", cos_scores.shape)

    results = {"cosine": {}}
    for split_name, split_ids in [
        ("seen", seen_ids), ("zs_comp", zsc_ids), ("zs_family", zsf_ids),
    ]:
        m = typed_metrics(cos_scores, cos_labels, split_ids, full_ids)
        results["cosine"][split_name] = m
        logger.info(
            "  [cosine] %s: Top-1=%.3f, MF1=%.3f, n=%d",
            split_name, m["top1"], m["macro_f1"], m["n"],
        )

    # === Method 2: Full model forward (with user prototypes + primitives) ===
    from langtraj_osr.train import fit_user_routines
    user_prototypes = fit_user_routines(model, train_loader, DEVICE)

    # Without declared primitives
    full_scores_no, full_labels, full_energies = collect_full_model_scores(
        model, test_loader, def_texts, user_prototypes, declared_prims=None,
    )
    results["full_no_dp"] = {}
    for split_name, split_ids in [
        ("seen", seen_ids), ("zs_comp", zsc_ids), ("zs_family", zsf_ids),
    ]:
        m = typed_metrics(full_scores_no, full_labels, split_ids, full_ids)
        results["full_no_dp"][split_name] = m
        logger.info(
            "  [full no DP] %s: Top-1=%.3f, MF1=%.3f, n=%d",
            split_name, m["top1"], m["macro_f1"], m["n"],
        )

    # With declared primitives
    declared_prims = LangTrajOSR.build_declared_primitive_matrix(
        full_ids, n_primitives=10,
    ).to(DEVICE)
    logger.info("Declared primitives matrix shape: %s", declared_prims.shape)

    full_scores_dp, _, _ = collect_full_model_scores(
        model, test_loader, def_texts, user_prototypes, declared_prims=declared_prims,
    )
    results["full_with_dp"] = {}
    for split_name, split_ids in [
        ("seen", seen_ids), ("zs_comp", zsc_ids), ("zs_family", zsf_ids),
    ]:
        m = typed_metrics(full_scores_dp, full_labels, split_ids, full_ids)
        results["full_with_dp"][split_name] = m
        logger.info(
            "  [full with DP] %s: Top-1=%.3f, MF1=%.3f, n=%d",
            split_name, m["top1"], m["macro_f1"], m["n"],
        )

    # Binary AUROC
    from sklearn.metrics import roc_auc_score
    y_true_bin = (cos_labels > 0).astype(int)
    if 0 < y_true_bin.sum() < len(y_true_bin):
        auroc = roc_auc_score(y_true_bin, cos_scores.max(axis=1))
        results["binary_auroc_cosine"] = round(float(auroc), 4)
        logger.info("  Binary AUROC (cosine): %.4f", auroc)
    if 0 < y_true_bin.sum() < len(y_true_bin):
        auroc = roc_auc_score(y_true_bin, full_energies)
        results["binary_auroc_energy"] = round(float(auroc), 4)
        logger.info("  Binary AUROC (energy): %.4f", auroc)

    return results


def main():
    output_dir = Path("/home/hello/ouyangqi/results/v8_porto_prim_fix")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for seed in [42, 123, 456]:
        try:
            r = eval_seed(seed)
            if r is not None:
                all_results[str(seed)] = r
        except Exception as e:
            logger.error("Seed %d failed: %s", seed, e)
            import traceback
            traceback.print_exc()

    # Save results
    out_path = output_dir / "typed_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # Summary comparison
    logger.info("=== Summary ===")
    for method in ["cosine", "full_no_dp", "full_with_dp"]:
        logger.info("--- %s ---", method)
        for split in ["seen", "zs_comp", "zs_family"]:
            vals = [
                all_results[s][method][split]["top1"]
                for s in all_results
                if method in all_results[s] and split in all_results[s][method]
            ]
            mf1s = [
                all_results[s][method][split]["macro_f1"]
                for s in all_results
                if method in all_results[s] and split in all_results[s][method]
            ]
            if vals:
                logger.info(
                    "  %s: Top-1=%.3f+/-%.3f, MF1=%.3f+/-%.3f",
                    split, np.mean(vals), np.std(vals),
                    np.mean(mf1s), np.std(mf1s),
                )


if __name__ == "__main__":
    main()
