#!/usr/bin/env python3
"""Evaluate all foursquare checkpoints (Tokyo + NYC, seeds 42/123/456) with typed metrics."""
import sys
sys.path.insert(0, "/home/hello/ouyangqi")

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.chdir("/home/hello/ouyangqi")

import json, logging
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
def eval_checkpoint(city, seed):
    from langtraj_osr.core.concepts import get_all_definitions, get_concept_ids_for_split
    from langtraj_osr.core.dataset import MobDefBenchDataset, collate_mobdef
    from langtraj_osr.models.langtraj_osr import LangTrajOSR, LangTrajConfig
    from run_typed_foursquare_and_centroid import build_foursquare
    from torch.utils.data import DataLoader

    np.random.seed(seed)
    torch.manual_seed(seed)

    seen_ids = sorted(get_concept_ids_for_split("seen"))
    zsc_ids = sorted(get_concept_ids_for_split("zs_comp"))
    zsf_ids = sorted(get_concept_ids_for_split("zs_family"))
    full_ids = seen_ids + zsc_ids + zsf_ids
    all_defs = get_all_definitions(include_paraphrases=True)

    ckpt_path = f"/home/hello/ouyangqi/results/foursquare_{city}/seed_{seed}/numosim/seed_{seed}/best_model.pt"
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
    logger.info("Loaded %s seed %d", city, seed)

    bench = build_foursquare(city, seed)
    test_trajs = bench.test.normal + bench.test.anomalous
    train_all = bench.train.normal + bench.train.anomalous
    user_hist = {}
    for t in train_all:
        if t.label == 0:
            user_hist.setdefault(t.user_id, []).append(t)

    test_ds = MobDefBenchDataset(test_trajs, concept_definitions=all_defs, user_histories=user_hist)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_mobdef)

    def_texts = [all_defs[cid][0] for cid in full_ids]
    c_bank, _ = model.definition_encoder(def_texts)
    c_bank = F.normalize(c_bank.float(), dim=-1).to(DEVICE)

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

    results = {"city": city, "seed": seed, "binary_auroc": round(auroc, 4)}
    for split_name, split_ids in [("seen", seen_ids), ("zs_comp", zsc_ids), ("zs_family", zsf_ids)]:
        m = typed_metrics(scores, labels, split_ids, full_ids)
        results[split_name] = m
        logger.info("  [%s s%d] %s: Top-1=%.3f, MF1=%.3f, n=%d",
                     city, seed, split_name, m["top1"], m["macro_f1"], m["n"])

    return results


def main():
    output_dir = Path("/home/hello/ouyangqi/results/multiseed_foursquare_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for city in ["tokyo", "nyc"]:
        for seed in [42, 123, 456]:
            key = f"{city}_s{seed}"
            try:
                r = eval_checkpoint(city, seed)
                if r is not None:
                    all_results[key] = r
            except Exception as e:
                logger.error("%s failed: %s", key, e)
                import traceback
                traceback.print_exc()

    out = output_dir / "multiseed_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved to %s", out)

    # Summary table
    print("\n" + "=" * 90)
    print(f"  {'City':<8} {'Seed':>5} {'AUROC':>7} {'Seen Top1':>10} {'ZSC Top1':>10} {'ZSF Top1':>10} {'Seen MF1':>10}")
    print("=" * 90)
    for city in ["tokyo", "nyc"]:
        for seed in [42, 123, 456]:
            key = f"{city}_s{seed}"
            r = all_results.get(key)
            if r is None:
                print(f"  {city:<8} {seed:>5} {'MISSING':>7}")
                continue
            print(f"  {city:<8} {seed:>5} {r['binary_auroc']:>7.3f}"
                  f" {r['seen']['top1']:>10.3f} {r['zs_comp']['top1']:>10.3f} {r['zs_family']['top1']:>10.3f}"
                  f" {r['seen']['macro_f1']:>10.3f}")
        # Compute mean/std for this city
        city_results = [all_results[f"{city}_s{s}"] for s in [42, 123, 456] if f"{city}_s{s}" in all_results]
        if len(city_results) >= 2:
            aurocs = [r["binary_auroc"] for r in city_results]
            seen_t1 = [r["seen"]["top1"] for r in city_results]
            zsc_t1 = [r["zs_comp"]["top1"] for r in city_results]
            print(f"  {'':8} {'mean':>5} {np.mean(aurocs):>7.3f}"
                  f" {np.mean(seen_t1):>10.3f} {np.mean(zsc_t1):>10.3f}")
            print(f"  {'':8} {'std':>5} {np.std(aurocs):>7.3f}"
                  f" {np.std(seen_t1):>10.3f} {np.std(zsc_t1):>10.3f}")
        print("-" * 90)
    print("=" * 90)


if __name__ == "__main__":
    main()
