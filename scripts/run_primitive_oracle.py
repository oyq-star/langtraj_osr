#!/usr/bin/env python3
"""Primitive-only oracle baseline: scores concepts using ONLY the primitive head.

Tests circularity concern: if primitive-only scoring beats language-based
scoring on zero-shot concepts, it confirms the benchmark is circular.

Score = v_x @ declared_primitives.T  (no language, no contrastive alignment)
"""
import sys
sys.path.insert(0, "/home/hello/ouyangqi")

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.chdir("/home/hello/ouyangqi")

import json, logging, tempfile
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
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


def build_declared_primitives(concept_ids, n_primitives=10):
    """Build (K, P) binary matrix from concepts.py."""
    from langtraj_osr.core.concepts import CONCEPT_BY_ID
    K = len(concept_ids)
    P = torch.zeros(K, n_primitives)
    for i, cid in enumerate(concept_ids):
        c = CONCEPT_BY_ID[cid]
        for p in c["primitives"]:
            P[i, p] = 1.0
    return P


def typed_metrics(scores, labels, subset_ids, all_ids):
    mask = np.isin(labels, subset_ids)
    if mask.sum() == 0:
        return {"top1": 0.0, "macro_f1": 0.0, "n": 0}
    s, l = scores[mask], labels[mask]
    pred_idx = s.argmax(axis=1)
    pred_ids = np.array([all_ids[p] if p < len(all_ids) else -1 for p in pred_idx])
    top1 = float((pred_ids == l).mean())
    mf1 = float(f1_score(l, pred_ids, average="macro", zero_division=0))
    return {"top1": round(top1, 4), "macro_f1": round(mf1, 4), "n": int(mask.sum())}


@torch.no_grad()
def collect_primitive_scores(model, loader, declared_prims):
    """Score using ONLY primitive head: v_x @ P.T"""
    model.eval()
    all_prim_scores, all_cosine_scores, all_labels = [], [], []

    # Build concept bank for cosine comparison
    from langtraj_osr.core.concepts import get_all_definitions, get_concept_ids_for_split
    seen_ids = sorted(get_concept_ids_for_split("seen"))
    zsc_ids = sorted(get_concept_ids_for_split("zs_comp"))
    zsf_ids = sorted(get_concept_ids_for_split("zs_family"))
    full_ids = seen_ids + zsc_ids + zsf_ids
    all_defs = get_all_definitions(include_paraphrases=True)
    def_texts = [all_defs[cid][0] for cid in full_ids]
    c_bank, _ = model.definition_encoder(def_texts)
    c_bank = F.normalize(c_bank.float(), dim=-1).to(DEVICE)

    for b in loader:
        pad_mask = b["mask"].to(DEVICE)
        ep = ep_dict(b["episode_tensor"], DEVICE)
        labels = b["label"].numpy()

        # Forward through backbone
        ep_emb = model.episode_encoder(ep)
        z_x, h_i = model.trajectory_encoder(ep_emb, ~pad_mask)

        # Primitive head: v_x (B, 10), attn_weights
        v_x, _ = model.primitive_head(h_i, ~pad_mask)

        # Primitive-only scores: v_x @ P.T  (B, K)
        prim_scores = v_x @ declared_prims.T  # (B, K)

        # Cosine scores for comparison
        z_norm = F.normalize(z_x.float(), dim=-1)
        cos_scores = (z_norm @ c_bank.T) / TEMPERATURE

        all_prim_scores.append(prim_scores.cpu().numpy())
        all_cosine_scores.append(cos_scores.cpu().numpy())
        all_labels.append(labels)

    return (
        np.concatenate(all_prim_scores),
        np.concatenate(all_cosine_scores),
        np.concatenate(all_labels),
    )


def run_dataset(dataset_name, seed=42):
    """Run primitive oracle on one dataset."""
    from langtraj_osr.core.concepts import get_all_definitions, get_concept_ids_for_split
    from langtraj_osr.core.dataset import MobDefBenchDataset, collate_mobdef
    from langtraj_osr.models.langtraj_osr import LangTrajOSR, LangTrajConfig
    from torch.utils.data import DataLoader

    np.random.seed(seed)
    torch.manual_seed(seed)

    seen_ids = sorted(get_concept_ids_for_split("seen"))
    zsc_ids = sorted(get_concept_ids_for_split("zs_comp"))
    zsf_ids = sorted(get_concept_ids_for_split("zs_family"))
    full_ids = seen_ids + zsc_ids + zsf_ids
    all_defs = get_all_definitions(include_paraphrases=True)

    # Load checkpoint
    if dataset_name == "synthetic":
        ckpt_path = f"/home/hello/ouyangqi/results/v6_synthetic/numosim/seed_{seed}/best_model.pt"
    elif dataset_name == "porto":
        ckpt_path = f"/home/hello/ouyangqi/results/v6_porto/numosim/seed_{seed}/best_model.pt"
    elif dataset_name == "tokyo":
        ckpt_path = f"/home/hello/ouyangqi/results/foursquare_tokyo/seed_{seed}/numosim/seed_{seed}/best_model.pt"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not Path(ckpt_path).exists():
        logger.warning("Checkpoint not found: %s", ckpt_path)
        return None

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = LangTrajConfig(**ckpt["config"])
    config.text_encoder_name = TEXT_ENCODER_PATH
    model = LangTrajOSR(config).to(DEVICE)
    model.definition_encoder(["init"])
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    logger.info("Loaded %s seed %d from %s", dataset_name, seed, ckpt_path)

    # Build data
    if dataset_name == "synthetic":
        from langtraj_osr.benchmark.benchmark_builder import MobDefBenchBuilder
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as to2:
            builder = MobDefBenchBuilder(data_dir=td, output_dir=to2, seed=seed)
            benchmarks = builder.build(datasets=["numosim"])
        bench = benchmarks["numosim"]
    elif dataset_name == "porto":
        from run_typed_foursquare_and_centroid import build_porto
        bench = build_porto(seed)
    elif dataset_name == "tokyo":
        from run_typed_foursquare_and_centroid import build_foursquare
        bench = build_foursquare("tokyo", seed)

    test_trajs = bench.test.normal + bench.test.anomalous
    train_all = bench.train.normal + bench.train.anomalous
    user_hist = {}
    for t in train_all:
        if t.label == 0:
            user_hist.setdefault(t.user_id, []).append(t)

    test_ds = MobDefBenchDataset(test_trajs, concept_definitions=all_defs, user_histories=user_hist)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_mobdef)

    # Build declared primitives
    declared_prims = build_declared_primitives(full_ids).to(DEVICE)
    logger.info("Declared primitives matrix: %s", declared_prims.shape)

    # Collect scores
    prim_scores, cos_scores, labels = collect_primitive_scores(model, test_loader, declared_prims)
    logger.info("%s: prim_scores shape=%s, labels shape=%s", dataset_name, prim_scores.shape, labels.shape)

    # Typed metrics
    results = {"dataset": dataset_name, "seed": seed}
    for method_name, scores in [("primitive_oracle", prim_scores), ("cosine_language", cos_scores)]:
        results[method_name] = {}
        for split_name, split_ids in [("seen", seen_ids), ("zs_comp", zsc_ids), ("zs_family", zsf_ids)]:
            m = typed_metrics(scores, labels, split_ids, full_ids)
            results[method_name][split_name] = m
            logger.info("  [%s] %s: Top-1=%.3f, MF1=%.3f, n=%d",
                        method_name, split_name, m["top1"], m["macro_f1"], m["n"])

    # Primitive head stats
    anom_mask = labels > 0
    norm_mask = labels == 0
    if anom_mask.sum() > 0:
        # v_x stats: how well-calibrated are primitive predictions?
        mean_anom = prim_scores[anom_mask].mean()
        mean_norm = prim_scores[norm_mask].mean()
        results["prim_score_stats"] = {
            "mean_anom": round(float(mean_anom), 4),
            "mean_norm": round(float(mean_norm), 4),
        }

    return results


def main():
    output_dir = Path("/home/hello/ouyangqi/results/primitive_oracle")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for dataset in ["synthetic", "porto", "tokyo"]:
        for seed in [42]:
            try:
                r = run_dataset(dataset, seed)
                if r is not None:
                    all_results[f"{dataset}_s{seed}"] = r
            except Exception as e:
                logger.error("%s seed %d failed: %s", dataset, seed, e)
                import traceback
                traceback.print_exc()

    out = output_dir / "primitive_oracle_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved to %s", out)

    # Summary
    print("\n" + "=" * 70)
    print("  PRIMITIVE ORACLE vs COSINE (LANGUAGE)")
    print("=" * 70)
    for key, r in all_results.items():
        print(f"\n  {key}:")
        for method in ["primitive_oracle", "cosine_language"]:
            print(f"    {method}:")
            for split in ["seen", "zs_comp", "zs_family"]:
                m = r.get(method, {}).get(split, {})
                if m:
                    print(f"      {split:12s}: Top-1={m['top1']:.3f}  MF1={m['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
