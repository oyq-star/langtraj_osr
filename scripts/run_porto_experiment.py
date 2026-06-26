"""
End-to-end Porto real-data experiment for LangTraj-OSR.
Pipeline: Porto parquet -> tokenize -> MobDef-Bench build -> train (same as train.py)
"""
import sys, os, math, hashlib, logging, json, time, argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/hello/ouyangqi')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ============================================================
# Porto tokenization (inline, avoids column name issues)
# ============================================================

def tokenize_porto_parquet(
    parquet_path: str,
    n_taxis: int = 200,
    min_trips: int = 10,
    max_trips: int = 80,
    min_points: int = 4,
) -> List:
    from langtraj_osr.core.episode import SemanticEpisode, SemanticTrajectory
    from langtraj_osr.core.tokenizer import TrajectoryTokenizer

    tokenizer = TrajectoryTokenizer()

    def haversine(lat1, lon1, lat2, lon2):
        R = 6_371_000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
             + math.cos(phi1) * math.cos(phi2)
             * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def zone_id(lat, lon, res=0.005):
        gl = int(round(lat / res))
        gn = int(round(lon / res))
        h = hashlib.md5(f"{gl},{gn}".encode()).hexdigest()
        return int(h[:8], 16) % (2 ** 31)

    def infer_trans(spd):
        if spd < 2.0:  return 0
        if spd < 10.0: return 2
        return 1

    logger.info("Loading Porto parquet: %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    logger.info("  rows=%d  taxis=%d  trips=%d",
                len(df), df['taxi_id'].nunique(), df['trip_id'].nunique())

    df = df.sort_values(['taxi_id', 'timestamp']).reset_index(drop=True)
    taxi_trips = df.groupby('taxi_id')['trip_id'].nunique()
    eligible = taxi_trips[taxi_trips >= min_trips].index
    top_taxis = taxi_trips[eligible].nlargest(n_taxis).index
    df = df[df['taxi_id'].isin(top_taxis)].copy()
    logger.info("  Using %d taxis, %d rows", df['taxi_id'].nunique(), len(df))

    # Build trip-level POLYLINE
    logger.info("  Reconstructing POLYLINE per trip...")
    trip_records = []
    for (taxi_id, trip_id), grp in df.groupby(['taxi_id', 'trip_id']):
        grp = grp.sort_values('timestamp')
        pts = list(zip(grp['longitude'].tolist(), grp['latitude'].tolist()))
        if len(pts) < min_points:
            continue
        trip_records.append({
            'TAXI_ID':   str(taxi_id),
            'TRIP_ID':   str(trip_id),
            'TIMESTAMP': int(grp['timestamp'].iloc[0]),
            'POLYLINE':  pts,
        })

    porto_df = pd.DataFrame(trip_records)

    def sample_trips(g):
        return g.sample(min(len(g), max_trips), random_state=42)

    porto_df = (porto_df.groupby('TAXI_ID', group_keys=False)
                         .apply(sample_trips)
                         .reset_index(drop=True))
    logger.info("  %d trips across %d taxis",
                len(porto_df), porto_df['TAXI_ID'].nunique())

    # Tokenize each trip
    trajectories = []
    for _, row in porto_df.iterrows():
        polyline = row['POLYLINE']
        base_ts  = int(row['TIMESTAMP'])
        step_s   = 15
        taxi_id  = row['TAXI_ID']
        trip_id  = row['TRIP_ID']

        dists = [haversine(polyline[i-1][1], polyline[i-1][0],
                           polyline[i][1],   polyline[i][0])
                 for i in range(1, len(polyline))]
        avg_dist = float(np.mean(dists)) if dists else 1.0

        episodes  = []
        subsample = max(1, len(polyline) // 20)
        for idx in range(0, len(polyline), subsample):
            lon, lat = polyline[idx]
            ts = pd.Timestamp(base_ts + idx * step_s, unit='s')
            z  = zone_id(lat, lon)
            tb = ts.hour * 7 + ts.dayofweek
            db = tokenizer._discretize_dwell(subsample * step_s / 60.0)
            if idx > 0:
                pi   = max(0, idx - subsample)
                plon, plat = polyline[pi]
                sd   = haversine(plat, plon, lat, lon)
                spd  = sd / (subsample * step_s) if subsample * step_s > 0 else 0.0
                trans = infer_trans(spd)
                tlc   = sd / avg_dist if avg_dist > 0 else 1.0
            else:
                trans = 1
                tlc   = 1.0
            episodes.append(SemanticEpisode(
                zone_id=z, poi_role=0, time_bin=tb, dwell_bin=db,
                transition_type=trans,
                trip_length_change=round(float(tlc), 4),
                event_flag=0, companion_flag=0,
            ))
        if episodes:
            from langtraj_osr.core.episode import SemanticTrajectory
            trajectories.append(SemanticTrajectory(
                episodes=episodes, user_id=taxi_id,
                trip_id=trip_id, label=0,
            ))

    logger.info("  Tokenized %d trajectories", len(trajectories))
    return trajectories


# ============================================================
# Import training components (same as train.py)
# ============================================================
from langtraj_osr.core.concepts import get_all_definitions, get_concept_ids_for_split
from langtraj_osr.core.dataset  import MobDefBenchDataModule, collate_mobdef
from langtraj_osr.core.utils    import (AverageMeter, EarlyStopping,
                                         compute_metrics, get_logger,
                                         save_results, set_seed)
from langtraj_osr.evaluation.metrics import compute_all_metrics
from langtraj_osr.models.conformal    import ConformalCalibrator
from langtraj_osr.models.langtraj_osr import LangTrajConfig, LangTrajOSR
from langtraj_osr.models.losses       import CombinedLoss
from langtraj_osr.benchmark.benchmark_builder import (
    MobDefBenchBuilder, CONCEPT_DEFS, SPLIT_SEEN,
    SPLIT_ZS_COMP, SPLIT_ZS_FAMILY, SPLIT_UNKNOWN,
)

# ============================================================
# Import train.py helper functions (copy minimal ones here)
# ============================================================
# We import them directly from the train module
from langtraj_osr.train import (
    pretrain_masked, fit_user_routines, train_one_epoch,
    validate, calibrate_conformal, evaluate_test,
    parse_args,
)


def main():
    args = parse_args()
    # Override dataset to porto-real
    args.use_synthetic = False  # will be overridden by in-memory data

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Data: Porto real ----
    porto_path = '/home/hello/ouyangqi/data/porto/data/raw/porto_taxi.parquet'
    raw_trajs = tokenize_porto_parquet(porto_path, n_taxis=200, min_trips=10,
                                        max_trips=80, min_points=4)

    # Build benchmark
    logger.info("Building MobDef-Bench with anomaly injection (Porto real data)...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_data:
        with tempfile.TemporaryDirectory() as tmp_out:
            builder = MobDefBenchBuilder(data_dir=tmp_data, output_dir=tmp_out, seed=args.seed)

            raw_trajs = builder._tokenize(raw_trajs)
            train_trajs_base, val_trajs_base, test_trajs_base = builder._split_by_user(raw_trajs)

            from langtraj_osr.benchmark.benchmark_builder import Benchmark, BenchmarkSplit
            benchmark = Benchmark(dataset_name='porto')
            benchmark.train.normal = train_trajs_base
            benchmark.val.normal   = val_trajs_base
            benchmark.test.normal  = test_trajs_base

            seen_concepts     = [c for c in CONCEPT_DEFS if c.split == SPLIT_SEEN]
            zs_comp_concepts  = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_COMP]
            zs_family_concepts= [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_FAMILY]
            unknown_concepts  = [c for c in CONCEPT_DEFS if c.split == SPLIT_UNKNOWN]

            builder._inject_anomalies(benchmark.train, train_trajs_base, seen_concepts)
            builder._inject_anomalies(benchmark.val,   val_trajs_base,   seen_concepts + zs_comp_concepts)
            builder._inject_anomalies(benchmark.test,  test_trajs_base,
                                      seen_concepts + zs_comp_concepts + zs_family_concepts + unknown_concepts)

    def combine(s): return s.normal + s.anomalous
    train_trajs = combine(benchmark.train)
    val_trajs   = combine(benchmark.val)
    test_trajs  = combine(benchmark.test)

    logger.info("Porto benchmark: train=%d (norm=%d anom=%d), val=%d, test=%d",
                len(train_trajs), len(benchmark.train.normal), len(benchmark.train.anomalous),
                len(val_trajs), len(test_trajs))

    concept_defs = get_all_definitions(include_paraphrases=True)
    user_histories: Dict[str, list] = {}
    for t in train_trajs:
        if t.label == 0:
            user_histories.setdefault(t.user_id, []).append(t)

    data_module = MobDefBenchDataModule(
        trajectories={'train': train_trajs, 'val': val_trajs, 'test': test_trajs},
        concept_definitions=concept_defs,
        user_histories=user_histories,
        batch_size=args.batch_size,
    )

    train_loader = data_module.train_dataloader()
    val_loader   = data_module.val_dataloader()

    # ---- Model ----
    config = LangTrajConfig(text_encoder_name=args.text_encoder)
    model  = LangTrajOSR(config).to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # ---- Stage 1 ----
    pretrain_masked(model, train_loader, device, epochs=args.pretrain_epochs)

    # ---- Stage 2 ----
    user_prototypes = fit_user_routines(model, train_loader, device)

    # ---- Stage 3 ----
    logger.info("Stage 3: Concept alignment training (%d epochs)", args.epochs)

    backbone_params = (list(model.episode_encoder.parameters()) +
                       list(model.trajectory_encoder.parameters()))
    head_params     = (list(model.definition_encoder.parameters()) +
                       list(model.user_history.parameters()) +
                       list(model.primitive_head.parameters()))

    optimizer = AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': head_params,     'lr': args.lr},
    ], weight_decay=1e-2)

    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
    scheduler    = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                                milestones=[args.warmup_epochs])

    criterion  = CombinedLoss()
    early_stop = EarlyStopping(patience=args.patience, mode='max')
    scaler     = GradScaler(enabled=device.type == "cuda")

    out_dir = Path(args.output_dir) / 'porto' / f'seed_{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build fixed concept bank (12 seen concepts, frozen MiniLM embeddings)
    logger.info("Building fixed concept bank for validation...")
    from langtraj_osr.core.concepts import ANOMALY_CONCEPTS
    seen_ids   = get_concept_ids_for_split("seen")
    seen_texts = [ANOMALY_CONCEPTS[cid]['definition'] for cid in seen_ids]
    with torch.no_grad():
        model.definition_encoder._ensure_encoder()
        bank_embs = model.definition_encoder._encode_texts(seen_texts)
        concept_bank = model.definition_encoder.projection(bank_embs).detach()
    concept_bank = concept_bank.to(device)
    logger.info("Built fixed concept bank: %d seen concepts, dim=%d",
                len(seen_ids), concept_bank.shape[1])

    best_auroc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Re-fit user prototypes every 5 epochs to track embedding drift
        if epoch % 5 == 1:
            user_prototypes = fit_user_routines(model, train_loader, device)

        train_losses = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, user_prototypes, concept_bank, seen_ids,
        )
        scheduler.step(epoch)

        val_auroc, val_enorm_auroc = validate(
            model, val_loader, device, user_prototypes, concept_bank
        )

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d -- total: %.4f | L_pair: %.4f | L_cls_bank: %.4f | "
            "val AUROC(bank): %.4f | val AUROC(E_norm): %.4f | time: %.1fs",
            epoch, args.epochs,
            train_losses.get('total', 0), train_losses.get('L_pair', 0),
            train_losses.get('L_cls_bank', 0),
            val_auroc, val_enorm_auroc, elapsed,
        )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'val_auroc': val_auroc},
                       str(out_dir / 'best_model.pt'))
            logger.info("  Saved best model (AUROC=%.4f)", val_auroc)

        if early_stop(val_auroc):
            logger.info("Early stopping at epoch %d", epoch)
            break

    # ---- Final eval ----
    ckpt = torch.load(str(out_dir / 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model'])
    user_prototypes = fit_user_routines(model, train_loader, device)
    logger.info("Re-fitting user routine banks with best trained model")

    logger.info("Calibrating conformal thresholds on validation set")
    calibrate_conformal(model, val_loader, device, user_prototypes)

    logger.info("Final evaluation on test set")
    test_loader = data_module.test_dataloader()
    results = evaluate_test(model, test_loader, data_module.concept_split_dataloaders(),
                            device, user_prototypes, concept_bank)

    logger.info("Test AUROC: %.4f | AUPRC: %.4f", results['auroc'], results['auprc'])
    for split_name in ['A_seen', 'A_zs_comp', 'A_zs_family', 'A_unknown']:
        if split_name in results:
            logger.info("  %s -- AUROC: %.4f", split_name, results[split_name])

    results_path = out_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    # Print summary
    print("\n=== PORTO REAL DATA RESULTS ===")
    print(f"Test AUROC: {results['auroc']:.4f}")
    print(f"Test AUPRC: {results['auprc']:.4f}")
    print(f"Best val AUROC: {best_auroc:.4f}")
    for k in ['A_seen', 'A_zs_comp', 'A_zs_family', 'A_unknown']:
        if k in results:
            print(f"  {k}: AUROC={results[k]:.4f}")


if __name__ == '__main__':
    main()
