"""Main training script for LangTraj-OSR.

Usage:
    python -m langtraj_osr.train --dataset numosim --seed 42 --output_dir results/
    python -m langtraj_osr.train --dataset geolife --seed 42 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .core.concepts import get_all_definitions, get_concept_ids_for_split
from .core.dataset import MobDefBenchDataModule, collate_mobdef
from .core.utils import (
    AverageMeter,
    EarlyStopping,
    compute_metrics,
    compute_open_set_metrics,
    get_logger,
    save_results,
    set_seed,
)
from .evaluation.metrics import compute_all_metrics
from .models.conformal import ConformalCalibrator
from .models.langtraj_osr import LangTrajConfig, LangTrajOSR
from .models.losses import CombinedLoss

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: Self-supervised pretraining (masked attribute modeling)
# ---------------------------------------------------------------------------


class MaskedAttributeModel(nn.Module):
    """Wrapper for self-supervised pretraining via masked attribute prediction."""

    def __init__(self, episode_encoder: nn.Module, trajectory_encoder: nn.Module,
                 n_fields: int = 8, hidden_dim: int = 256) -> None:
        super().__init__()
        self.episode_encoder = episode_encoder
        self.trajectory_encoder = trajectory_encoder
        # Prediction heads for each masked field
        self.pred_heads = nn.ModuleDict({
            "poi_role": nn.Linear(hidden_dim, 64),
            "time_bin": nn.Linear(hidden_dim, 168),
            "dwell_bin": nn.Linear(hidden_dim, 16),
            "transition_type": nn.Linear(hidden_dim, 4),
        })

    def forward(self, episodes: Dict[str, torch.Tensor], mask: torch.Tensor,
                mask_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        ep_emb = self.episode_encoder(episodes)
        _, h_i = self.trajectory_encoder(ep_emb, mask)
        # Gather masked position embeddings
        masked_emb = torch.gather(
            h_i, 1, mask_indices.unsqueeze(-1).expand(-1, -1, h_i.size(-1))
        )
        preds = {}
        for name, head in self.pred_heads.items():
            preds[name] = head(masked_emb)
        return preds


def pretrain_masked(model: LangTrajOSR, train_loader: DataLoader,
                    device: torch.device, epochs: int = 10,
                    lr: float = 1e-3, mask_ratio: float = 0.15) -> None:
    """Stage 1: Self-supervised pretraining with masked attribute modeling."""
    logger.info("Stage 1: Self-supervised pretraining (%d epochs)", epochs)

    mam = MaskedAttributeModel(
        model.episode_encoder, model.trajectory_encoder
    ).to(device)
    optimizer = AdamW(mam.parameters(), lr=lr, weight_decay=1e-2)

    mam.train()
    for epoch in range(epochs):
        loss_meter = AverageMeter()
        for batch in train_loader:
            ep_tensor = batch["episode_tensor"].to(device)
            pad_mask = batch["mask"].to(device)
            B, L, _ = ep_tensor.shape

            # Random masking
            n_mask = max(1, int(L * mask_ratio))
            mask_indices = torch.stack([
                torch.randperm(L, device=device)[:n_mask] for _ in range(B)
            ])

            # Build episodes dict from tensor
            episodes = _tensor_to_episode_dict(ep_tensor)

            # Mask selected positions (zero out)
            ep_masked = ep_tensor.clone()
            for b in range(B):
                ep_masked[b, mask_indices[b]] = 0.0
            episodes_masked = _tensor_to_episode_dict(ep_masked)

            preds = mam(episodes_masked, ~pad_mask, mask_indices)

            # Compute losses for each predicted field
            loss = torch.tensor(0.0, device=device)
            targets = {
                "poi_role": ep_tensor[:, :, 1].long(),
                "time_bin": ep_tensor[:, :, 2].long(),
                "dwell_bin": ep_tensor[:, :, 3].long(),
                "transition_type": ep_tensor[:, :, 4].long(),
            }
            for name, pred in preds.items():
                target = torch.gather(targets[name], 1, mask_indices)
                loss = loss + F.cross_entropy(
                    pred.reshape(-1, pred.size(-1)), target.reshape(-1)
                )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mam.parameters(), 1.0)
            optimizer.step()
            loss_meter.update(loss.item(), B)

        logger.info("  Pretrain epoch %d/%d — loss: %.4f", epoch + 1, epochs, loss_meter.avg)


# ---------------------------------------------------------------------------
# Stage 2: Fit user routine bank
# ---------------------------------------------------------------------------


def fit_user_routines(model: LangTrajOSR, train_loader: DataLoader,
                      device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
    """Stage 2: Build per-user prototype banks from normal training trips."""
    logger.info("Stage 2: Fitting user routine banks")
    model.eval()

    user_embeddings: Dict[str, List[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in train_loader:
            ep_tensor = batch["episode_tensor"].to(device)
            pad_mask = batch["mask"].to(device)
            labels = batch["label"]
            user_ids = batch["user_id"]

            # Only normal trips
            normal_mask = labels == 0
            if not normal_mask.any():
                continue

            episodes = _tensor_to_episode_dict(ep_tensor[normal_mask])
            ep_emb = model.episode_encoder(episodes)
            z_x, _ = model.trajectory_encoder(ep_emb, ~pad_mask[normal_mask])

            for i, uid in enumerate(
                [u for u, m in zip(user_ids, normal_mask.tolist()) if m]
            ):
                if uid not in user_embeddings:
                    user_embeddings[uid] = []
                user_embeddings[uid].append(z_x[i].cpu())

    # Fit prototypes per user
    user_prototypes: Dict[str, Dict[str, torch.Tensor]] = {}
    for uid, embs in user_embeddings.items():
        emb_stack = torch.stack(embs)
        user_prototypes[uid] = model.user_history.fit_user(emb_stack.to(device))

    logger.info("  Fitted routines for %d users", len(user_prototypes))
    return user_prototypes


# ---------------------------------------------------------------------------
# Stage 3: Main concept alignment training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: LangTrajOSR,
    criterion: CombinedLoss,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    user_prototypes: Dict[str, Dict[str, torch.Tensor]],
    epoch: int,
    concept_bank: Optional[torch.Tensor] = None,
    seen_ids_sorted: Optional[List[int]] = None,
    temperature: float = 0.07,
    concept_bank_full: Optional[torch.Tensor] = None,
    w_repel: float = 0.3,
) -> Dict[str, float]:
    """Run one training epoch."""
    model.train()
    meters = {
        "total": AverageMeter(), "L_pair": AverageMeter(),
        "L_cls": AverageMeter(), "L_cls_bank": AverageMeter(),
        "L_prim": AverageMeter(),
        "L_para": AverageMeter(), "L_orth": AverageMeter(),
        "L_norm": AverageMeter(), "L_repel": AverageMeter(),
    }

    for batch_idx, batch in enumerate(train_loader):
        ep_tensor = batch["episode_tensor"].to(device)
        pad_mask = batch["mask"].to(device)
        labels = batch["label"].to(device)
        prim_labels = batch["primitive_labels"].to(device)
        definition_texts = batch["definition_text"]
        user_ids = batch["user_id"]
        B = ep_tensor.size(0)

        # Build episode dict and user prototype batch
        episodes = _tensor_to_episode_dict(ep_tensor)
        proto_batch = _batch_user_prototypes(user_ids, user_prototypes, device)

        with autocast(enabled=scaler.is_enabled()):
            outputs = model(episodes, ~pad_mask, proto_batch, definition_texts)

            # Separate normal and anomalous energies
            normal_idx = (labels == 0)
            anom_idx = (labels > 0)
            E_norm_normal = outputs["E_norm"][normal_idx] if normal_idx.any() else torch.zeros(1, device=device)
            E_norm_anom = outputs["E_norm"][anom_idx] if anom_idx.any() else None

            # Trip-level primitive labels (max across episodes)
            trip_prim = prim_labels.max(dim=1).values  # (B, 10)

            # Get z_x and c_d for InfoNCE (need matched pairs)
            # z_x must be computed before concept bank scoring below.
            ep_emb = model.episode_encoder(episodes)
            z_x, _ = model.trajectory_encoder(ep_emb, ~pad_mask)
            c_d, _ = model.definition_encoder(definition_texts)

            # For InfoNCE (L_pair): only use SEEN anomalous trajectories.
            # Excludes A_zs_comp (13-18) and A_zs_family (19-22) for true
            # zero-shot evaluation. Normal trajectories excluded because they
            # share definition_text="" → identical c_d → gradient explosion.
            _zs_comp_ids = set(get_concept_ids_for_split("zs_comp"))
            _zs_fam_ids = set(get_concept_ids_for_split("zs_family"))
            _exclude_ids = _zs_comp_ids | _zs_fam_ids
            anom_pair_mask = (labels > 0) & torch.tensor(
                [l.item() not in _exclude_ids for l in labels],
                device=device, dtype=torch.bool,
            )
            _n_anom_pair = anom_pair_mask.sum().item()
            if batch_idx == 0 and epoch in (0, 1):
                _label_counts = {}
                for _l in labels.cpu().tolist():
                    _label_counts[_l] = _label_counts.get(_l, 0) + 1
                logger.info("DEBUG batch0: labels=%s, anom_pair_mask=%d, B=%d",
                            _label_counts, _n_anom_pair, B)
                logger.info("DEBUG batch0: z_x dtype=%s, c_d dtype=%s, z_x[:3]=%s, c_d[:3]=%s",
                            z_x.dtype, c_d.dtype, z_x[:3, :4].detach().cpu().tolist(),
                            c_d[:3, :4].detach().cpu().tolist())
            if _n_anom_pair >= 2:
                z_x_pair = z_x[anom_pair_mask]
                c_d_pair = c_d[anom_pair_mask]
            else:
                z_x_pair = z_x
                c_d_pair = c_d

            # Issue B fix: use fixed concept bank for L_cls instead of
            # batch-diagonal indices.  The bank holds frozen embeddings for all
            # K=12 seen concepts; each seen trajectory scores against all K
            # entries, giving 12× denser gradient signal than the diagonal.
            seen_ids = set(get_concept_ids_for_split("seen"))
            cls_mask = torch.tensor([l.item() in seen_ids for l in labels], device=device)
            if cls_mask.any() and concept_bank is not None and seen_ids_sorted is not None:
                # scores_bank: (N_seen, K) — trajectory vs every seen concept.
                # L2-normalize both sides before dot product to bound cosine similarity
                # to [-1, 1], preventing overflow when dividing by temperature=0.07.
                z_seen = F.normalize(z_x[cls_mask].float(), dim=-1)   # (N_seen, D)
                bank_norm = F.normalize(concept_bank.float(), dim=-1)  # (K, D)
                scores_bank = torch.matmul(z_seen, bank_norm.T) / temperature
                # Map global concept IDs to bank indices 0..K-1
                concept_labels_mapped = torch.tensor(
                    [seen_ids_sorted.index(l.item()) for l in labels[cls_mask].cpu()],
                    dtype=torch.long, device=device,
                )
            # Issue B: compute bank-based L_cls separately — do NOT pass bank
            # scores into CombinedLoss (its ClassificationLoss expects batch-
            # relative format and would receive wrong-shaped tensors).
            if concept_bank is not None and cls_mask.any():
                l_cls_bank = F.cross_entropy(scores_bank, concept_labels_mapped)
            else:
                l_cls_bank = torch.tensor(0.0, device=device)

            # Pass batch-diagonal format to CombinedLoss as before (kept for
            # L_pair, L_prim, L_para, L_orth, L_norm — not for L_cls which we
            # now handle via the bank above).
            cls_indices = torch.where(cls_mask)[0] if cls_mask.any() else torch.zeros(1, dtype=torch.long, device=device)
            concept_scores_orig = outputs["concept_scores"][cls_mask] if cls_mask.any() else outputs["concept_scores"][:1]
            concept_labels_diag = cls_indices.clamp(0, concept_scores_orig.size(1) - 1)

            losses = criterion(
                z_x=z_x_pair,
                c_d=c_d_pair,
                concept_scores=concept_scores_orig,
                concept_labels=concept_labels_diag,
                v_x=outputs["v_x"],
                primitive_labels=trip_prim,
                deviation_features=outputs["deviation_features"],
                E_norm_normal=E_norm_normal,
                E_norm_anomalous=E_norm_anom,
                concept_scores_orth=outputs["concept_scores"],
            )

            # L_repel: push normal trip embeddings away from ALL concept definitions.
            # Without this, normal trips can drift toward concept embeddings (since
            # they're excluded from L_pair), causing AUROC to collapse after warmup.
            # Uses the full 22-concept bank so normals are repelled from ZS concepts too.
            _repel_bank = concept_bank_full if concept_bank_full is not None else concept_bank
            if _repel_bank is not None and normal_idx.any():
                z_norm_trips = F.normalize(z_x[normal_idx].float(), dim=-1)
                bank_repel = F.normalize(_repel_bank.float(), dim=-1)
                # Cosine similarity of each normal trip to every concept embedding
                sim_normal = torch.matmul(z_norm_trips, bank_repel.T) / temperature  # (N_n, K)
                # Penalise the maximum similarity — normals should score low vs ALL concepts
                l_repel = F.softplus(sim_normal.max(dim=1).values).mean()
            else:
                l_repel = torch.zeros(1, device=device)

            losses = dict(losses)
            losses["L_cls_bank"] = l_cls_bank
            losses["L_repel"] = l_repel
            losses["total"] = losses["total"] + 0.5 * l_cls_bank + w_repel * l_repel

        if batch_idx < 5 and epoch in (0, 1):
            logger.info("DEBUG batch%d losses: total=%.6f L_pair=%.6f L_cls_bank=%.6f L_repel=%.6f L_prim=%.6f L_orth=%.6f finite=%s",
                        batch_idx, losses["total"].item(), losses["L_pair"].item(),
                        losses["L_cls_bank"].item(), losses["L_repel"].item(),
                        losses["L_prim"].item(), losses["L_orth"].item(),
                        torch.isfinite(losses["total"]).item())
        # Safety: skip batch if loss is NaN/Inf to prevent weight corruption
        if not torch.isfinite(losses["total"]):
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)

        # Check for Inf/NaN in gradients BEFORE clipping.
        # clip_grad_norm_ converts Inf→NaN via (Inf * 0 = NaN) which bypasses
        # GradScaler's found_inf detection, corrupting weights.
        grad_ok = all(
            p.grad.isfinite().all()
            for p in model.parameters() if p.grad is not None
        )
        if not grad_ok:
            optimizer.zero_grad()
            scaler.update()  # reduce scale to prevent future overflow
            continue

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        for key in meters:
            if key in losses:
                meters[key].update(losses[key].item() if isinstance(losses[key], torch.Tensor) else losses[key], B)

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def validate(
    model: LangTrajOSR,
    val_loader: DataLoader,
    device: torch.device,
    user_prototypes: Dict[str, Dict[str, torch.Tensor]],
    definition_texts_bank: List[str],
    concept_bank: Optional[torch.Tensor] = None,
    temperature: float = 0.07,
) -> Dict[str, float]:
    """Run validation and return metrics."""
    model.eval()
    all_scores, all_labels, all_energies, all_z = [], [], [], []

    for batch in val_loader:
        ep_tensor = batch["episode_tensor"].to(device)
        pad_mask = batch["mask"].to(device)
        labels = batch["label"]
        user_ids = batch["user_id"]

        episodes = _tensor_to_episode_dict(ep_tensor)
        proto_batch = _batch_user_prototypes(user_ids, user_prototypes, device)

        outputs = model(episodes, ~pad_mask, proto_batch, definition_texts_bank)
        all_scores.append(outputs["concept_scores"].cpu())
        all_labels.append(labels)
        all_energies.append(outputs["E_norm"].cpu())

        # Also collect trajectory embeddings for bank-based scoring
        if concept_bank is not None:
            ep_emb = model.episode_encoder(episodes)
            z_x, _ = model.trajectory_encoder(ep_emb, ~pad_mask)
            all_z.append(z_x.cpu())

    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    energies = torch.cat(all_energies, dim=0)

    # Binary detection: normal (0) vs anomaly (>0)
    y_true = (labels > 0).numpy().astype(int)

    # Use fixed concept bank scores when available (Issue B fix): bank-based
    # scores are more consistent than batch-relative concept_scores.
    # L2-normalize before dot product to bound cosine similarity to [-1, 1]
    # (prevents NaN from fp16 overflow when dividing by temperature=0.07).
    if concept_bank is not None and all_z:
        z_all = torch.cat(all_z, dim=0)
        z_norm = F.normalize(z_all.float(), dim=-1).to(concept_bank.device)
        bank_norm = F.normalize(concept_bank.float(), dim=-1)
        bank_scores = torch.matmul(z_norm, bank_norm.T) / temperature  # (N, K)
        y_score = bank_scores.max(dim=1).values.cpu().numpy()
        # Guard against residual NaN (e.g. all-zero embeddings early in training)
        y_score = np.nan_to_num(y_score, nan=0.0)
    else:
        y_score = scores.max(dim=1).values.numpy()
        y_score = np.nan_to_num(y_score, nan=0.0)

    metrics = compute_metrics(y_true, y_score)
    # Also report E_norm-based AUROC for diagnostic purposes
    e_scores = np.nan_to_num(energies.numpy(), nan=0.0)
    e_metrics = compute_metrics(y_true, e_scores)
    metrics["auroc_enorm"] = e_metrics.get("auroc", 0.0)
    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tensor_to_episode_dict(ep_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Convert (B, L, 8) tensor to episode feature dict."""
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
    """Build batched prototype tensors."""
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LangTraj-OSR")
    parser.add_argument("--dataset", type=str, default="numosim",
                        choices=["numosim", "geolife", "porto", "foursquare_nyc", "foursquare_tokyo"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_heads", type=float, default=1e-4)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed-precision training (use float32)")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use synthetic data for development/debugging")
    parser.add_argument("--use_porto_real", action="store_true",
                        help="Use real Porto taxi parquet data")
    parser.add_argument("--porto_parquet", type=str,
                        default="data/porto/data/raw/porto_taxi.parquet",
                        help="Path to Porto raw parquet file")
    parser.add_argument("--use_foursquare", action="store_true",
                        help="Use Foursquare NYC or Tokyo check-in data")
    parser.add_argument("--foursquare_train", type=str, default="",
                        help="Path to Foursquare train.parquet")
    parser.add_argument("--foursquare_test", type=str, default="",
                        help="Path to Foursquare test.parquet (optional)")
    parser.add_argument("--foursquare_name", type=str, default="foursquare",
                        help="Dataset name tag (e.g. nyc or tokyo)")
    parser.add_argument("--text_encoder", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--shuffle_embeddings", action="store_true",
                        help="Ablation: shuffle concept bank embeddings to break language alignment")
    # Ablation: loss weight overrides
    parser.add_argument("--w_cls", type=float, default=0.5, help="Classification loss weight")
    parser.add_argument("--w_prim", type=float, default=1.0, help="Primitive loss weight")
    parser.add_argument("--w_para", type=float, default=0.2, help="Paraphrase loss weight")
    parser.add_argument("--w_orth", type=float, default=0.05, help="Orthogonality loss weight")
    parser.add_argument("--w_repel", type=float, default=0.3, help="Repel loss weight")
    parser.add_argument("--random_bank", action="store_true",
                        help="Ablation: replace concept bank with random fixed embeddings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir) / args.dataset / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Data ----
    if args.use_synthetic:
        logger.info("Using synthetic data for development")
        import tempfile
        from .benchmark.benchmark_builder import MobDefBenchBuilder

        # Use MobDefBenchBuilder which injects actual anomalous trajectories
        # via ConceptGenerator (not just all-normal trips).
        with tempfile.TemporaryDirectory() as tmp_data_dir:
            with tempfile.TemporaryDirectory() as tmp_out_dir:
                builder = MobDefBenchBuilder(
                    data_dir=tmp_data_dir,   # no real data → falls back to synthetic
                    output_dir=tmp_out_dir,
                    seed=args.seed,
                )
                benchmarks = builder.build(datasets=["numosim"])
                bench = benchmarks["numosim"]

        # Combine normal + anomalous trajectories per split
        def _combine(split):
            return split.normal + split.anomalous

        train_trajs = _combine(bench.train)
        val_trajs = _combine(bench.val)
        test_trajs = _combine(bench.test)

        logger.info(
            "Synthetic benchmark: train=%d (norm=%d, anom=%d), "
            "val=%d (norm=%d, anom=%d), test=%d (norm=%d, anom=%d)",
            len(train_trajs), len(bench.train.normal), len(bench.train.anomalous),
            len(val_trajs), len(bench.val.normal), len(bench.val.anomalous),
            len(test_trajs), len(bench.test.normal), len(bench.test.anomalous),
        )

        concept_defs = get_all_definitions(include_paraphrases=True)
        user_histories: Dict[str, list] = {}
        for t in train_trajs:
            if t.label == 0:
                user_histories.setdefault(t.user_id, []).append(t)

        data_module = MobDefBenchDataModule(
            trajectories={"train": train_trajs, "val": val_trajs, "test": test_trajs},
            concept_definitions=concept_defs,
            user_histories=user_histories,
            batch_size=args.batch_size,
        )
    elif args.use_porto_real:
        logger.info("Using real Porto taxi data: %s", args.porto_parquet)
        import math as _math, hashlib as _hashlib
        from .benchmark.benchmark_builder import (
            MobDefBenchBuilder, Benchmark, BenchmarkSplit,
            CONCEPT_DEFS, SPLIT_SEEN, SPLIT_ZS_COMP, SPLIT_ZS_FAMILY, SPLIT_UNKNOWN,
        )
        from .core.tokenizer import TrajectoryTokenizer as _TT
        from .core.episode import SemanticEpisode as _SE, SemanticTrajectory as _ST

        def _tokenize_porto_parquet(parquet_path, n_taxis=200, min_trips=10,
                                     max_trips=80, min_points=4):
            import numpy as _np, pandas as _pd
            _tok = _TT()

            def _hav(lat1, lon1, lat2, lon2):
                R = 6_371_000.0
                p1, p2 = _math.radians(lat1), _math.radians(lat2)
                a = (_math.sin(_math.radians(lat2-lat1)/2)**2
                     + _math.cos(p1)*_math.cos(p2)*_math.sin(_math.radians(lon2-lon1)/2)**2)
                return 2*R*_math.atan2(_math.sqrt(a), _math.sqrt(1-a))

            def _zone(lat, lon, res=0.005):
                h = _hashlib.md5(f"{int(round(lat/res))},{int(round(lon/res))}".encode()).hexdigest()
                return int(h[:8], 16) % (2**31)

            def _trans(spd): return 0 if spd < 2 else (2 if spd < 10 else 1)

            if parquet_path.endswith('.pkl'):
                import pickle as _pickle
                with open(parquet_path, 'rb') as _f:
                    df = _pickle.load(_f)
                # pickle is already trip-level (TAXI_ID, TRIP_ID, TIMESTAMP, POLYLINE)
                # skip the groupby reconstruction step
                pdf = df.copy()
                pdf = (pdf.groupby('TAXI_ID', group_keys=False)
                          .apply(lambda g: g.sample(min(len(g), max_trips), random_state=42))
                          .reset_index(drop=True))
                logger.info("Porto (pickle): %d trips from %d taxis",
                            len(pdf), pdf['TAXI_ID'].nunique())
                trajs = []
                for _, row in pdf.iterrows():
                    poly = row['POLYLINE']
                    # Filter out NaN coordinates from Porto raw data
                    poly = [(lon, lat) for lon, lat in poly
                            if lon == lon and lat == lat]  # NaN != NaN
                    if len(poly) < min_points:
                        continue
                    dists = [_hav(poly[i-1][1], poly[i-1][0], poly[i][1], poly[i][0])
                             for i in range(1, len(poly))]
                    # Filter NaN distances (corrupt GPS jumps)
                    dists = [d for d in dists if d == d and _math.isfinite(d)]
                    avg_d = max(float(_np.mean(dists)) if dists else 1.0, 1.0)
                    sub = max(1, len(poly) // 20)
                    eps = []
                    for idx in range(0, len(poly), sub):
                        lon, lat = poly[idx]
                        ts = _pd.Timestamp(int(row['TIMESTAMP']) + idx*15, unit='s')
                        db = _tok._discretize_dwell(sub * 15 / 60.0)
                        if idx > 0:
                            pl, pa = poly[max(0, idx-sub)]
                            sd = _hav(pa, pl, lat, lon)
                            if not _math.isfinite(sd):
                                sd = 0.0
                            spd = sd / (sub*15) if sub*15 > 0 else 0
                            tr = _trans(spd); tlc = min(sd/avg_d, 20.0)
                        else:
                            tr = 1; tlc = 1.0
                        eps.append(_SE(zone_id=_zone(lat, lon), poi_role=0,
                                       time_bin=ts.hour*7+ts.dayofweek, dwell_bin=db,
                                       transition_type=tr,
                                       trip_length_change=round(float(tlc), 4),
                                       event_flag=0, companion_flag=0))
                    if eps:
                        trajs.append(_ST(episodes=eps, user_id=row['TAXI_ID'],
                                         trip_id=row['TRIP_ID'], label=0))
                return trajs
            df = _pd.read_parquet(parquet_path)
            df = df.sort_values(['taxi_id', 'timestamp']).reset_index(drop=True)
            tc = df.groupby('taxi_id')['trip_id'].nunique()
            top = tc[tc >= min_trips].nlargest(n_taxis).index
            df = df[df['taxi_id'].isin(top)].copy()

            records = []
            for (taxi_id, trip_id), grp in df.groupby(['taxi_id', 'trip_id']):
                grp = grp.sort_values('timestamp')
                pts = list(zip(grp['longitude'].tolist(), grp['latitude'].tolist()))
                if len(pts) < min_points:
                    continue
                records.append({'TAXI_ID': str(taxi_id), 'TRIP_ID': str(trip_id),
                                 'TIMESTAMP': int(grp['timestamp'].iloc[0]), 'POLYLINE': pts})
            pdf = _pd.DataFrame(records)
            pdf = (pdf.groupby('TAXI_ID', group_keys=False)
                      .apply(lambda g: g.sample(min(len(g), max_trips), random_state=42))
                      .reset_index(drop=True))
            logger.info("Porto: %d trips from %d taxis", len(pdf), pdf['TAXI_ID'].nunique())

            trajs = []
            for _, row in pdf.iterrows():
                poly = row['POLYLINE']
                dists = [_hav(poly[i-1][1], poly[i-1][0], poly[i][1], poly[i][0])
                         for i in range(1, len(poly))]
                avg_d = float(_np.mean(dists)) if dists else 1.0
                sub = max(1, len(poly) // 20)
                eps = []
                for idx in range(0, len(poly), sub):
                    lon, lat = poly[idx]
                    ts = _pd.Timestamp(int(row['TIMESTAMP']) + idx*15, unit='s')
                    db = _tok._discretize_dwell(sub * 15 / 60.0)
                    if idx > 0:
                        pl, pa = poly[max(0, idx-sub)]
                        sd = _hav(pa, pl, lat, lon)
                        spd = sd / (sub*15) if sub*15 > 0 else 0
                        tr = _trans(spd); tlc = sd/avg_d if avg_d > 0 else 1.0
                    else:
                        tr = 1; tlc = 1.0
                    eps.append(_SE(zone_id=_zone(lat, lon), poi_role=0,
                                   time_bin=ts.hour*7+ts.dayofweek, dwell_bin=db,
                                   transition_type=tr,
                                   trip_length_change=round(float(tlc), 4),
                                   event_flag=0, companion_flag=0))
                if eps:
                    trajs.append(_ST(episodes=eps, user_id=row['TAXI_ID'],
                                     trip_id=row['TRIP_ID'], label=0))
            return trajs

        raw_trajs = _tokenize_porto_parquet(args.porto_parquet)
        logger.info("Porto: tokenized %d trajectories", len(raw_trajs))

        import tempfile as _tmpfile
        with _tmpfile.TemporaryDirectory() as _td, _tmpfile.TemporaryDirectory() as _to:
            _builder = MobDefBenchBuilder(data_dir=_td, output_dir=_to, seed=args.seed)
            raw_trajs = _builder._tokenize(raw_trajs)
            _tr, _va, _te = _builder._split_by_user(raw_trajs)
            _bench = Benchmark(dataset_name='porto')
            _bench.train.normal = _tr
            _bench.val.normal   = _va
            _bench.test.normal  = _te
            _sc = [c for c in CONCEPT_DEFS if c.split == SPLIT_SEEN]
            _zc = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_COMP]
            _zf = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_FAMILY]
            _uc = [c for c in CONCEPT_DEFS if c.split == SPLIT_UNKNOWN]
            # True zero-shot: A_zs_comp and A_zs_family are EXCLUDED from
            # training entirely (no L_pair, no L_cls). Only seen concepts in
            # train; zs_comp in val for monitoring; all concepts in test.
            # A_unknown (23-25) remains truly unseen — no definition, no training data.
            _builder._inject_anomalies(_bench.train, _tr, _sc)
            _builder._inject_anomalies(_bench.val,   _va, _sc + _zc)
            _builder._inject_anomalies(_bench.test,  _te, _sc + _zc + _zf + _uc)

        def _comb(s): return s.normal + s.anomalous
        train_trajs = _comb(_bench.train)
        val_trajs   = _comb(_bench.val)
        test_trajs  = _comb(_bench.test)
        logger.info("Porto benchmark: train=%d (norm=%d anom=%d), val=%d, test=%d",
                    len(train_trajs), len(_bench.train.normal), len(_bench.train.anomalous),
                    len(val_trajs), len(test_trajs))

        concept_defs = get_all_definitions(include_paraphrases=True)
        user_histories: Dict[str, list] = {}
        for t in train_trajs:
            if t.label == 0:
                user_histories.setdefault(t.user_id, []).append(t)

        data_module = MobDefBenchDataModule(
            trajectories={"train": train_trajs, "val": val_trajs, "test": test_trajs},
            concept_definitions=concept_defs,
            user_histories=user_histories,
            batch_size=args.batch_size,
        )
    elif args.use_foursquare:
        # ------------------------------------------------------------------ #
        # Foursquare NYC / Tokyo check-in data                                #
        # Parses LLM-style text records into SemanticEpisode sequences.       #
        # Each "trip" = one user's check-ins on a single calendar day.        #
        # ------------------------------------------------------------------ #
        import re as _re
        import pandas as _pd
        import hashlib as _hashlib
        from collections import defaultdict as _dd
        from .benchmark.benchmark_builder import (
            MobDefBenchBuilder, Benchmark,
            CONCEPT_DEFS, SPLIT_SEEN, SPLIT_ZS_COMP, SPLIT_ZS_FAMILY, SPLIT_UNKNOWN,
        )
        from .core.episode import SemanticEpisode as _SE, SemanticTrajectory as _ST
        from .core.tokenizer import TrajectoryTokenizer as _TT

        _tok = _TT()
        _checkin_re = _re.compile(
            r'At (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), user (\d+) visited POI id (\d+)'
            r' which is a .+? and has Category id (\d+)\.'
        )

        def _read_df(path: str):
            return _pd.read_csv(path) if path.endswith('.csv') else _pd.read_parquet(path)

        def _parse_foursquare_parquet(path: str) -> List:
            dfs = [_read_df(path)]
            if args.foursquare_test:
                dfs.append(_read_df(args.foursquare_test))
            raw = _pd.concat(dfs, ignore_index=True)

            # Extract per-user check-in lists
            user_checkins: Dict[str, list] = _dd(list)
            for txt in raw['inputs']:
                for m in _checkin_re.finditer(txt):
                    ts_str, uid, poi_id, cat_id = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
                    ts = _pd.Timestamp(ts_str)
                    user_checkins[uid].append((ts, poi_id, cat_id))

            # Sort check-ins per user and group by calendar date → trips
            trajs = []
            for uid, checkins in user_checkins.items():
                checkins.sort(key=lambda x: x[0])
                # Group by date
                by_date: Dict[str, list] = _dd(list)
                for ts, poi, cat in checkins:
                    by_date[ts.date().isoformat()].append((ts, poi, cat))

                for date_str, day_checkins in by_date.items():
                    if len(day_checkins) < 3:
                        continue   # skip days with < 3 check-ins
                    eps = []
                    for i, (ts, poi, cat) in enumerate(day_checkins):
                        # dwell: minutes until next check-in (or 30 min default)
                        if i + 1 < len(day_checkins):
                            dwell_min = (day_checkins[i+1][0] - ts).total_seconds() / 60.0
                        else:
                            dwell_min = 30.0
                        dwell_min = min(max(dwell_min, 1.0), 480.0)
                        # transition: based on gap to next
                        trans = 0 if dwell_min < 20 else (2 if dwell_min < 60 else 1)
                        # zone = hash of poi id
                        zone = int(_hashlib.md5(str(poi).encode()).hexdigest()[:8], 16) % (2**31)
                        time_bin = ts.hour * 7 + ts.dayofweek
                        dwell_bin = _tok._discretize_dwell(dwell_min)
                        tlc = float(len(day_checkins)) / max(
                            float(sum(len(v) for v in by_date.values()) / max(len(by_date), 1)), 1.0
                        )
                        tlc = min(tlc, 20.0)
                        eps.append(_SE(
                            zone_id=zone, poi_role=int(cat) % 64,
                            time_bin=time_bin, dwell_bin=dwell_bin,
                            transition_type=trans,
                            trip_length_change=round(tlc, 4),
                            event_flag=0, companion_flag=0,
                        ))
                    trajs.append(_ST(episodes=eps, user_id=uid,
                                     trip_id=f"{uid}_{date_str}", label=0))
            return trajs

        logger.info("Parsing Foursquare check-in data: %s", args.foursquare_train)
        raw_trajs = _parse_foursquare_parquet(args.foursquare_train)
        logger.info("Foursquare: tokenized %d trips from %d users",
                    len(raw_trajs), len({t.user_id for t in raw_trajs}))

        import tempfile as _tmpfile
        with _tmpfile.TemporaryDirectory() as _td, _tmpfile.TemporaryDirectory() as _to:
            _builder = MobDefBenchBuilder(data_dir=_td, output_dir=_to, seed=args.seed)
            raw_trajs = _builder._tokenize(raw_trajs)
            _tr, _va, _te = _builder._split_by_user(raw_trajs)
            _bench = Benchmark(dataset_name=args.foursquare_name)
            _bench.train.normal = _tr
            _bench.val.normal   = _va
            _bench.test.normal  = _te
            _sc = [c for c in CONCEPT_DEFS if c.split == SPLIT_SEEN]
            _zc = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_COMP]
            _zf = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_FAMILY]
            _uc = [c for c in CONCEPT_DEFS if c.split == SPLIT_UNKNOWN]
            # True zero-shot: only seen concepts in train
            _builder._inject_anomalies(_bench.train, _tr, _sc)
            _builder._inject_anomalies(_bench.val,   _va, _sc + _zc)
            _builder._inject_anomalies(_bench.test,  _te, _sc + _zc + _zf + _uc)

        def _comb(s): return s.normal + s.anomalous
        train_trajs = _comb(_bench.train)
        val_trajs   = _comb(_bench.val)
        test_trajs  = _comb(_bench.test)
        logger.info("Foursquare benchmark: train=%d (norm=%d anom=%d), val=%d, test=%d",
                    len(train_trajs), len(_bench.train.normal), len(_bench.train.anomalous),
                    len(val_trajs), len(test_trajs))

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
            args.dataset, batch_size=args.batch_size
        )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # ---- Model ----
    config = LangTrajConfig(
        text_encoder_name=args.text_encoder,
    )
    model = LangTrajOSR(config).to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # ---- Stage 1: Pretraining ----
    pretrain_masked(model, train_loader, device, epochs=args.pretrain_epochs)

    # ---- Stage 2: Fit user routines ----
    user_prototypes = fit_user_routines(model, train_loader, device)

    # ---- Stage 3: Concept alignment training ----
    logger.info("Stage 3: Concept alignment training (%d epochs)", args.epochs)

    # Separate parameter groups
    backbone_params = list(model.episode_encoder.parameters()) + \
                      list(model.trajectory_encoder.parameters())
    head_params = [p for n, p in model.named_parameters()
                   if not any(sub in n for sub in ["episode_encoder", "trajectory_encoder", "definition_encoder"])]

    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_heads},
    ], weight_decay=args.weight_decay)

    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                             milestones=[args.warmup_epochs])

    # Bug #12 fix: L_norm causes embedding collapse during Stage 3 (w_norm=1.0 loss
    # overwhelms L_pair=5.6 at epoch 1, pulling all embeddings to prototype centres).
    # Solution: disable L_norm during Stage 3 (w_norm=0.0). GMM prototypes are
    # re-fitted with the final trained model before evaluation, so E_norm is still
    # meaningful for inference. The concept-organised embedding space (via L_pair/
    # L_cls) naturally separates normal trips from anomalies: normal trips cluster
    # near their user prototype centres; anomalous trips embed near concept definitions.
    criterion = CombinedLoss(temperature=args.temperature, w_norm=0.0, w_cls=args.w_cls, w_prim=args.w_prim, w_para=args.w_para, w_orth=args.w_orth)
    use_amp = (device.type == "cuda") and (not getattr(args, 'no_amp', False))
    scaler = GradScaler(enabled=use_amp)
    early_stop = EarlyStopping(patience=args.patience, mode="max")

    # Definition bank for validation
    def_bank = []
    all_defs = get_all_definitions(include_paraphrases=False)
    for cid in sorted(all_defs.keys()):
        if all_defs[cid]:
            def_bank.append(all_defs[cid][0])

    # Issue B fix: build fixed concept bank for L_cls.
    # definition_encoder is a frozen SentenceTransformer — embeddings are
    # constant, so compute once before Stage 3.
    seen_ids_sorted = sorted(get_concept_ids_for_split("seen"))
    seen_def_texts = [all_defs[cid][0] for cid in seen_ids_sorted if all_defs.get(cid)]
    model.eval()
    with torch.no_grad():
        c_bank, _ = model.definition_encoder(seen_def_texts)
        c_bank = c_bank.to(device)  # (K=12, D), for training L_cls only
    model.train()
    logger.info("Built fixed concept bank: %d seen concepts, dim=%d", c_bank.size(0), c_bank.size(1))

    # Build full inference bank: all 22 known concepts (seen + zs_comp + zs_family).
    # ZS concepts have text definitions but no training examples. Using them at
    # inference is the standard zero-shot protocol — the analyst provides definitions
    # for every concept type they want to detect, and the model scores against all of
    # them. A_unknown (23-25) have no definitions and are detected via E_norm.
    zs_comp_ids   = sorted(get_concept_ids_for_split("zs_comp"))
    zs_family_ids = sorted(get_concept_ids_for_split("zs_family"))
    full_bank_ids  = seen_ids_sorted + zs_comp_ids + zs_family_ids
    full_bank_texts = [all_defs[cid][0] for cid in full_bank_ids if all_defs.get(cid)]
    model.eval()
    with torch.no_grad():
        c_bank_full, _ = model.definition_encoder(full_bank_texts)
        c_bank_full = c_bank_full.to(device)  # (22, D), for validation & test scoring
    model.train()
    logger.info(
        "Built full inference bank: %d concepts (seen=%d, zs_comp=%d, zs_family=%d), dim=%d",
        c_bank_full.size(0), len(seen_ids_sorted), len(zs_comp_ids), len(zs_family_ids),
        c_bank_full.size(1),
    )

    # Ablation: shuffle concept embeddings to break language ↔ trajectory alignment.
    # Keeps same embedding vectors but assigns them to wrong concepts.
    if args.shuffle_embeddings:
        perm = torch.randperm(c_bank.size(0))
        c_bank = c_bank[perm]
        perm_full = torch.randperm(c_bank_full.size(0))
        c_bank_full = c_bank_full[perm_full]
        logger.info("ABLATION: shuffled concept bank embeddings (language alignment broken)")

    # Ablation: replace concept bank with random fixed embeddings
    if args.random_bank:
        torch.manual_seed(9999)
        c_bank = F.normalize(torch.randn_like(c_bank), dim=-1)
        c_bank_full = F.normalize(torch.randn_like(c_bank_full), dim=-1)
        logger.info("ABLATION: replaced concept bank with random fixed embeddings")

    best_auroc = 0.0
    train_history: List[Dict[str, float]] = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_losses = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler,
            device, user_prototypes, epoch,
            concept_bank=c_bank, seen_ids_sorted=seen_ids_sorted,
            temperature=args.temperature,
            concept_bank_full=c_bank_full,
            w_repel=args.w_repel,
        )
        # Re-fit prototypes every 5 epochs (not every epoch) to track embedding
        # drift without adding noise from per-epoch GMM reinitialisation.
        if epoch % 5 == 0:
            user_prototypes = fit_user_routines(model, train_loader, device)
        # Use full 22-concept bank for validation so that ZS anomalies in the
        # val set are scored correctly — this gives a better early-stopping signal.
        val_metrics = validate(model, val_loader, device, user_prototypes, def_bank,
                               concept_bank=c_bank_full, temperature=args.temperature)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d — total: %.4f | L_pair: %.4f | L_cls_bank: %.4f | "
            "val AUROC(bank): %.4f | val AUROC(E_norm): %.4f | time: %.1fs",
            epoch + 1, args.epochs, train_losses["total"], train_losses["L_pair"],
            train_losses.get("L_cls_bank", 0.0),
            val_metrics.get("auroc", 0.0), val_metrics.get("auroc_enorm", 0.0), elapsed,
        )

        train_history.append({
            "epoch": epoch + 1,
            **{f"train_{k}": v for k, v in train_losses.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        auroc = val_metrics.get("auroc", 0.0)
        if auroc > best_auroc:
            best_auroc = auroc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "val_metrics": val_metrics,
            }, output_dir / "best_model.pt")
            logger.info("  Saved best model (AUROC=%.4f)", best_auroc)

        if early_stop(auroc):
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    # ---- Calibration ----
    logger.info("Calibrating conformal thresholds on validation set")
    calibrator = ConformalCalibrator()

    # Collect validation energies and scores
    model.eval()
    val_energies_normal, val_scores_all, val_labels_all = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            ep_tensor = batch["episode_tensor"].to(device)
            pad_mask = batch["mask"].to(device)
            labels = batch["label"]
            user_ids = batch["user_id"]

            episodes = _tensor_to_episode_dict(ep_tensor)
            proto_batch = _batch_user_prototypes(user_ids, user_prototypes, device)
            outputs = model(episodes, ~pad_mask, proto_batch, def_bank)

            normal_idx = labels == 0
            if normal_idx.any():
                val_energies_normal.append(outputs["E_norm"][normal_idx].cpu())
            val_scores_all.append(outputs["concept_scores"].cpu())
            val_labels_all.append(labels)

    if val_energies_normal:
        all_normal_e = torch.cat(val_energies_normal).numpy()
        calibrator.fit_normality(all_normal_e)

    all_val_scores = torch.cat(val_scores_all).numpy()
    all_val_labels = torch.cat(val_labels_all).numpy()
    calibrator.fit_concepts(all_val_scores, all_val_labels)

    # Save calibrator
    calibrator.save(str(output_dir / "calibrator.json"))

    # ---- Final evaluation on test set ----
    logger.info("Final evaluation on test set")
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Bug #11 fix: re-fit user prototypes with the best trained model so that
    # the GMM clusters reflect the final embedding space, not Stage-2 embeddings.
    logger.info("Re-fitting user routine banks with best trained model")
    user_prototypes = fit_user_routines(model, train_loader, device)

    test_loader = data_module.test_dataloader()
    concept_loaders = data_module.concept_split_dataloaders()

    # Overall test metrics — use full 22-concept bank so ZS anomalies are scored
    # against their own definitions (standard zero-shot inference protocol).
    test_metrics = validate(model, test_loader, device, user_prototypes, def_bank,
                            concept_bank=c_bank_full, temperature=args.temperature)
    logger.info("Test AUROC: %.4f | AUPRC: %.4f", test_metrics.get("auroc", 0), test_metrics.get("auprc", 0))

    # Per-split metrics
    split_metrics = {}
    for split_name, loader in concept_loaders.items():
        split_m = validate(model, loader, device, user_prototypes, def_bank,
                           concept_bank=c_bank_full, temperature=args.temperature)
        split_metrics[split_name] = split_m
        logger.info("  %s — AUROC: %.4f", split_name, split_m.get("auroc", 0))

    # ---- Save results ----
    results = {
        "dataset": args.dataset,
        "seed": args.seed,
        "best_epoch": checkpoint["epoch"],
        "best_val_auroc": best_auroc,
        "test_metrics": test_metrics,
        "split_metrics": split_metrics,
        "training_history": train_history,
        "args": vars(args),
    }
    save_results(results, str(output_dir / "results.json"))
    logger.info("Results saved to %s", output_dir / "results.json")

    # Print compact final summary for easy reading in logs
    print("\n=== FINAL RESULTS ===")
    print(f"Test AUROC: {test_metrics.get('auroc', 0):.4f}")
    print(f"Test AUPRC: {test_metrics.get('auprc', 0):.4f}")
    print(f"Best val AUROC: {best_auroc:.4f}  (epoch {checkpoint['epoch']})")
    for sn, sm in split_metrics.items():
        print(f"  {sn}: AUROC={sm.get('auroc', 0):.4f}")
    print("\nTraining history:")
    for h in train_history:
        ep = h['epoch']
        tl = h.get('train_total', 0)
        lp = h.get('train_L_pair', 0)
        va = h.get('val_auroc', 0)
        ve = h.get('val_auroc_enorm', 0)
        print(f"  E{ep}: total={tl:.3f} | L_pair={lp:.3f} | val_auroc(concept)={va:.4f} | val_auroc(E_norm)={ve:.4f}")


if __name__ == "__main__":
    main()
