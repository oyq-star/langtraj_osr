#!/usr/bin/env python3
"""Run component ablation experiments with typed metrics.

Ablation variants (all synthetic, seed 42, 50 epochs):
  full        - existing checkpoint (no retraining needed)
  no_prim     - w_prim=0.0
  no_cls      - w_cls=0.0
  no_para     - w_para=0.0
  no_orth     - w_orth=0.0
  no_repel    - w_repel=0.0
  random_bank - random fixed embeddings instead of language
  shuffle     - existing shuffled checkpoint
"""
import sys
sys.path.insert(0, "/home/hello/ouyangqi")

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.chdir("/home/hello/ouyangqi")

import json, subprocess, logging, tempfile
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

PY = "/home/hello/miniconda3/envs/oyq_v01/bin/python"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 50
BATCH_SIZE = 64
OUTPUT_BASE = Path("/home/hello/ouyangqi/results/ablation_typed_v2")
TEXT_ENCODER_PATH = (
    "/home/hello/.cache/huggingface/hub/models--sentence-transformers--"
    "all-MiniLM-L6-v2/snapshots/"
    "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)

# Ablation variants: name -> extra train.py args
VARIANTS = {
    "full":        [],  # use existing checkpoint
    "no_prim":     ["--w_prim", "0.0"],
    "no_cls":      ["--w_cls", "0.0"],
    "no_para":     ["--w_para", "0.0"],
    "no_orth":     ["--w_orth", "0.0"],
    "no_repel":    ["--w_repel", "0.0"],
    "random_bank": ["--random_bank"],
    "shuffle":     [],  # use existing shuffled checkpoint
}


def find_checkpoint(out_dir):
    """Find best_model.pt in out_dir or nested subdirs."""
    direct = out_dir / "best_model.pt"
    if direct.exists():
        return str(direct)
    # train.py creates nested: out_dir/numosim/seed_X/best_model.pt
    for p in out_dir.rglob("best_model.pt"):
        return str(p)
    return None


def train_variant(name, extra_args):
    """Train one ablation variant. Returns checkpoint path."""
    out_dir = OUTPUT_BASE / name / f"seed_{SEED}"

    # Skip if checkpoint exists (check nested paths too)
    existing = find_checkpoint(out_dir)
    if existing:
        logger.info("[%s] Checkpoint exists, skipping training", name)
        return existing

    # For 'full', use existing v6 checkpoint
    if name == "full":
        src = f"/home/hello/ouyangqi/results/v6_synthetic/numosim/seed_{SEED}/best_model.pt"
        if Path(src).exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(src, str(ckpt))
            logger.info("[full] Copied existing v6 checkpoint")
            return str(ckpt)
        else:
            logger.error("[full] v6 checkpoint not found: %s", src)
            return None

    # For 'shuffle', use existing shuffled checkpoint
    if name == "shuffle":
        src = f"/home/hello/ouyangqi/results/v7_ablation/shuffled/numosim/seed_{SEED}/best_model.pt"
        if Path(src).exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(src, str(ckpt))
            logger.info("[shuffle] Copied existing shuffled checkpoint")
            return str(ckpt)
        else:
            logger.warning("[shuffle] Shuffled checkpoint not found, training fresh")
            extra_args = ["--shuffle_embeddings"]

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PY, "-m", "langtraj_osr.train",
        "--use_synthetic",
        "--seed", str(SEED),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--no_amp",
        "--output_dir", str(out_dir),
    ] + extra_args

    logger.info("[%s] Training: %s", name, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        logger.error("[%s] Training FAILED:\n%s", name, result.stderr[-2000:])
        return None

    found = find_checkpoint(out_dir)
    if found:
        logger.info("[%s] Training complete, checkpoint saved", name)
        return found
    else:
        logger.error("[%s] Training finished but no checkpoint found", name)
        return None


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
def evaluate_variant(ckpt_path, name):
    """Evaluate one checkpoint with typed metrics."""
    from langtraj_osr.core.concepts import get_all_definitions, get_concept_ids_for_split
    from langtraj_osr.core.dataset import MobDefBenchDataset, collate_mobdef
    from langtraj_osr.models.langtraj_osr import LangTrajOSR, LangTrajConfig
    from langtraj_osr.benchmark.benchmark_builder import MobDefBenchBuilder
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    seen_ids = sorted(get_concept_ids_for_split("seen"))
    zsc_ids = sorted(get_concept_ids_for_split("zs_comp"))
    zsf_ids = sorted(get_concept_ids_for_split("zs_family"))
    full_ids = seen_ids + zsc_ids + zsf_ids
    all_defs = get_all_definitions(include_paraphrases=True)

    # Load model
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = LangTrajConfig(**ckpt["config"])
    config.text_encoder_name = TEXT_ENCODER_PATH
    model = LangTrajOSR(config).to(DEVICE)
    model.definition_encoder(["init"])
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Build benchmark
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as to2:
        builder = MobDefBenchBuilder(data_dir=td, output_dir=to2, seed=SEED)
        benchmarks = builder.build(datasets=["numosim"])
    bench = benchmarks["numosim"]

    test_trajs = bench.test.normal + bench.test.anomalous
    train_all = bench.train.normal + bench.train.anomalous
    user_hist = {}
    for t in train_all:
        if t.label == 0:
            user_hist.setdefault(t.user_id, []).append(t)

    test_ds = MobDefBenchDataset(test_trajs, concept_definitions=all_defs, user_histories=user_hist)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_mobdef)

    # Build concept bank
    def_texts = [all_defs[cid][0] for cid in full_ids]
    c_bank, _ = model.definition_encoder(def_texts)
    c_bank = F.normalize(c_bank.float(), dim=-1).to(DEVICE)

    # For shuffle variant, shuffle the bank at eval time too
    if name == "shuffle":
        perm = torch.randperm(c_bank.size(0))
        c_bank = c_bank[perm]

    # For random_bank variant, replace with random
    if name == "random_bank":
        torch.manual_seed(9999)
        c_bank = F.normalize(torch.randn_like(c_bank), dim=-1)

    TEMPERATURE = 0.07
    all_scores, all_labels = [], []

    for b in test_loader:
        pad_mask = b["mask"].to(DEVICE)
        ep = ep_dict(b["episode_tensor"], DEVICE)
        labels = b["label"].numpy()

        ep_emb = model.episode_encoder(ep)
        z_x, h_i = model.trajectory_encoder(ep_emb, ~pad_mask)
        z_norm = F.normalize(z_x.float(), dim=-1)
        scores = (z_norm @ c_bank.T) / TEMPERATURE

        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    # Binary AUROC
    binary_labels = (labels > 0).astype(int)
    max_scores = scores.max(axis=1)
    try:
        auroc = float(roc_auc_score(binary_labels, max_scores))
    except:
        auroc = 0.0

    # Typed metrics
    results = {"variant": name, "binary_auroc": round(auroc, 4)}
    for split_name, split_ids in [("seen", seen_ids), ("zs_comp", zsc_ids), ("zs_family", zsf_ids)]:
        m = typed_metrics(scores, labels, split_ids, full_ids)
        results[split_name] = m
        logger.info("  [%s] %s: Top-1=%.3f, MF1=%.3f, n=%d",
                     name, split_name, m["top1"], m["macro_f1"], m["n"])

    return results


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for name, extra_args in VARIANTS.items():
        logger.info("\n=== Ablation: %s ===", name)

        # Train (or copy existing checkpoint)
        ckpt_path = train_variant(name, extra_args)
        if ckpt_path is None:
            logger.error("[%s] Skipping evaluation (no checkpoint)", name)
            continue

        # Evaluate
        try:
            r = evaluate_variant(ckpt_path, name)
            all_results[name] = r
        except Exception as e:
            logger.error("[%s] Evaluation failed: %s", name, e)
            import traceback
            traceback.print_exc()

    # Save results
    out = OUTPUT_BASE / "ablation_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved to %s", out)

    # Print summary table
    print("\n" + "=" * 85)
    print(f"  {'Variant':<14} {'AUROC':>7} {'Seen Top1':>10} {'ZSC Top1':>10} {'ZSF Top1':>10} {'Seen MF1':>10} {'ZSC MF1':>10}")
    print("=" * 85)
    for name in VARIANTS:
        r = all_results.get(name)
        if r is None:
            print(f"  {name:<14} {'FAILED':>7}")
            continue
        print(f"  {name:<14} {r['binary_auroc']:>7.3f}"
              f" {r['seen']['top1']:>10.3f} {r['zs_comp']['top1']:>10.3f} {r['zs_family']['top1']:>10.3f}"
              f" {r['seen']['macro_f1']:>10.3f} {r['zs_comp']['macro_f1']:>10.3f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
