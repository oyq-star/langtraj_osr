#!/usr/bin/env python3
"""W2 + W3 fixes:
  W2: Typed evaluation (Top-1, M-F1, OSCR) for Tokyo and NYC foursquare data.
  W3: Nearest-centroid baseline (B10) — supervised prototype in trajectory space.

B10: For each seen concept, compute the centroid of its training trajectories.
     At test time, assign to nearest centroid. This is a MUCH stronger baseline
     than B9 (random-projection kNN against raw text embeddings) because it uses
     actual trajectory supervision — but it cannot handle zero-shot concepts
     (no training data for them).

Output: JSON with all typed metrics per split for Tokyo, NYC, and the centroid baseline.
"""
import sys, json, logging, tempfile, os
sys.path.insert(0, '/home/hello/ouyangqi')
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.chdir("/home/hello/ouyangqi")

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

SEED = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TEMPERATURE = 0.07
OUT = Path('/home/hello/ouyangqi/results/w2w3_fixes')
OUT.mkdir(exist_ok=True, parents=True)

np.random.seed(SEED); torch.manual_seed(SEED)


def ep_dict(t, device):
    return {k: t[:,:,i].long().to(device) if i < 5 or i >= 6 else t[:,:,i].float().to(device)
            for i, k in enumerate(['zone_id','poi_role','time_bin','dwell_bin',
                                    'transition_type','trip_length_change','event_flag','companion_flag'])}


def build_foursquare(city, seed):
    """Build foursquare benchmark for Tokyo or NYC.
    Parses LLM-style text records (same approach as train.py --use_foursquare).
    """
    import re, hashlib, pandas as pd
    from collections import defaultdict
    from langtraj_osr.core.tokenizer import TrajectoryTokenizer
    from langtraj_osr.core.episode import SemanticEpisode, SemanticTrajectory
    from langtraj_osr.benchmark.benchmark_builder import (
        MobDefBenchBuilder, Benchmark, CONCEPT_DEFS,
        SPLIT_SEEN, SPLIT_ZS_COMP, SPLIT_ZS_FAMILY, SPLIT_UNKNOWN,
    )

    tok = TrajectoryTokenizer()
    data_dir = f'data/foursquare_{city}'

    # Load CSV/parquet data (LLM-style text records)
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    raw = pd.concat([train_df, test_df], ignore_index=True)
    logger.info(f'{city}: {len(raw)} LLM-style text records')

    # Parse check-ins from text using same regex as train.py
    checkin_re = re.compile(
        r'At (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), user (\d+) visited POI id (\d+)'
        r' which is a .+? and has Category id (\d+)\.'
    )

    user_checkins = defaultdict(list)
    for txt in raw['inputs']:
        for m in checkin_re.finditer(txt):
            ts_str, uid, poi_id, cat_id = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            ts = pd.Timestamp(ts_str)
            user_checkins[uid].append((ts, poi_id, cat_id))

    logger.info(f'{city}: parsed {sum(len(v) for v in user_checkins.values())} checkins from {len(user_checkins)} users')

    # Sort and group by calendar date -> trips
    trajs = []
    for uid, checkins in user_checkins.items():
        checkins.sort(key=lambda x: x[0])
        by_date = defaultdict(list)
        for ts, poi, cat in checkins:
            by_date[ts.date().isoformat()].append((ts, poi, cat))

        for date_str, day_checkins in by_date.items():
            if len(day_checkins) < 3:
                continue
            eps = []
            for i, (ts, poi, cat) in enumerate(day_checkins):
                if i + 1 < len(day_checkins):
                    dwell_min = (day_checkins[i+1][0] - ts).total_seconds() / 60.0
                else:
                    dwell_min = 30.0
                dwell_min = min(max(dwell_min, 1.0), 480.0)
                trans = 0 if dwell_min < 20 else (2 if dwell_min < 60 else 1)
                zone = int(hashlib.md5(str(poi).encode()).hexdigest()[:8], 16) % (2**31)
                time_bin = ts.hour * 7 + ts.dayofweek
                dwell_bin = tok._discretize_dwell(dwell_min)
                tlc = float(len(day_checkins)) / max(
                    float(sum(len(v) for v in by_date.values()) / max(len(by_date), 1)), 1.0
                )
                tlc = min(tlc, 20.0)
                eps.append(SemanticEpisode(
                    zone_id=zone, poi_role=int(cat) % 64,
                    time_bin=time_bin, dwell_bin=dwell_bin,
                    transition_type=trans,
                    trip_length_change=round(tlc, 4),
                    event_flag=0, companion_flag=0,
                ))
            trajs.append(SemanticTrajectory(episodes=eps, user_id=uid,
                         trip_id=f"{uid}_{date_str}", label=0))

    logger.info(f'{city}: tokenized {len(trajs)} trajectories from {len(user_checkins)} users')

    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as to2:
        builder = MobDefBenchBuilder(data_dir=td, output_dir=to2, seed=seed)
        trajs = builder._tokenize(trajs)
        tr, va, te = builder._split_by_user(trajs)
        bench = Benchmark(dataset_name=f'foursquare_{city}')
        bench.train.normal, bench.val.normal, bench.test.normal = tr, va, te
        sc = [c for c in CONCEPT_DEFS if c.split == SPLIT_SEEN]
        zc = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_COMP]
        zf = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_FAMILY]
        uc = [c for c in CONCEPT_DEFS if c.split == SPLIT_UNKNOWN]
        builder._inject_anomalies(bench.train, tr, sc)
        builder._inject_anomalies(bench.val, va, sc + zc)
        builder._inject_anomalies(bench.test, te, sc + zc + zf + uc)
    return bench


def build_synthetic(seed):
    """Build synthetic benchmark."""
    from langtraj_osr.benchmark.benchmark_builder import MobDefBenchBuilder
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as to2:
        builder = MobDefBenchBuilder(data_dir=td, output_dir=to2, seed=seed)
        benchmarks = builder.build(datasets=['numosim'])
    return benchmarks['numosim']


def build_porto(seed):
    """Build Porto benchmark."""
    import math, hashlib, pandas as pd, pickle
    from langtraj_osr.core.tokenizer import TrajectoryTokenizer
    from langtraj_osr.core.episode import SemanticEpisode, SemanticTrajectory
    from langtraj_osr.benchmark.benchmark_builder import (
        MobDefBenchBuilder, Benchmark, CONCEPT_DEFS,
        SPLIT_SEEN, SPLIT_ZS_COMP, SPLIT_ZS_FAMILY, SPLIT_UNKNOWN,
    )
    tok = TrajectoryTokenizer()
    def _hav(lat1, lon1, lat2, lon2):
        R = 6_371_000.0; p1, p2 = math.radians(lat1), math.radians(lat2)
        a = (math.sin(math.radians(lat2-lat1)/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(math.radians(lon2-lon1)/2)**2)
        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))
    def _zone(lat, lon, res=0.005):
        h = hashlib.md5(f"{int(round(lat/res))},{int(round(lon/res))}".encode()).hexdigest()
        return int(h[:8], 16) % (2**31)
    def _trans(spd): return 0 if spd < 2 else (2 if spd < 10 else 1)

    with open('data/porto_trips.pkl', 'rb') as f: df = pickle.load(f)
    pdf = (df.groupby('TAXI_ID', group_keys=False)
             .apply(lambda g: g.sample(min(len(g), 80), random_state=seed))
             .reset_index(drop=True))
    trajs = []
    for _, row in pdf.iterrows():
        poly = [(lon, lat) for lon, lat in row['POLYLINE'] if lon == lon and lat == lat]
        if len(poly) < 4: continue
        dists = [_hav(poly[i-1][1], poly[i-1][0], poly[i][1], poly[i][0]) for i in range(1, len(poly))]
        dists = [d for d in dists if d == d and math.isfinite(d)]
        avg_d = max(float(np.mean(dists)) if dists else 1.0, 1.0)
        sub = max(1, len(poly) // 20)
        eps = []
        for idx in range(0, len(poly), sub):
            lon, lat = poly[idx]; ts = pd.Timestamp(int(row['TIMESTAMP']) + idx*15, unit='s')
            db = tok._discretize_dwell(sub * 15 / 60.0)
            if idx > 0:
                pl, pa = poly[max(0, idx-sub)]; sd = _hav(pa, pl, lat, lon)
                if not math.isfinite(sd): sd = 0.0
                spd = sd / (sub*15) if sub*15 > 0 else 0; tr = _trans(spd); tlc = min(sd/avg_d, 20.0)
            else: tr = 1; tlc = 1.0
            eps.append(SemanticEpisode(zone_id=_zone(lat, lon), poi_role=0,
                time_bin=ts.hour*7+ts.dayofweek, dwell_bin=db, transition_type=tr,
                trip_length_change=round(float(tlc), 4), event_flag=0, companion_flag=0))
        if eps:
            trajs.append(SemanticTrajectory(episodes=eps, user_id=row['TAXI_ID'],
                trip_id=row['TRIP_ID'], label=0))
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as to2:
        builder = MobDefBenchBuilder(data_dir=td, output_dir=to2, seed=seed)
        trajs = builder._tokenize(trajs)
        tr, va, te = builder._split_by_user(trajs)
        bench = Benchmark(dataset_name='porto')
        bench.train.normal, bench.val.normal, bench.test.normal = tr, va, te
        sc = [c for c in CONCEPT_DEFS if c.split == SPLIT_SEEN]
        zc = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_COMP]
        zf = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_FAMILY]
        uc = [c for c in CONCEPT_DEFS if c.split == SPLIT_UNKNOWN]
        builder._inject_anomalies(bench.train, tr, sc)
        builder._inject_anomalies(bench.val, va, sc + zc)
        builder._inject_anomalies(bench.test, te, sc + zc + zf + uc)
    return bench


@torch.no_grad()
def collect_scores_and_embeddings(model, loader, c_bank):
    """Collect concept bank scores, trajectory embeddings, and labels."""
    model.eval()
    all_scores, all_labels, all_z = [], [], []
    for b in loader:
        pad_mask = b['mask'].to(DEVICE)
        ep = ep_dict(b['episode_tensor'], DEVICE)
        ep_emb = model.episode_encoder(ep)
        z_x, _ = model.trajectory_encoder(ep_emb, ~pad_mask)
        z_norm = F.normalize(z_x.float(), dim=-1)
        scores = (z_norm @ c_bank.T) / TEMPERATURE
        all_scores.append(scores.cpu().float().numpy())
        all_labels.append(b['label'].numpy())
        all_z.append(z_norm.cpu().numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels), np.concatenate(all_z)


def typed_metrics(scores, labels, subset_ids, all_ids):
    """Top-1, Macro-F1, and OSCR for a subset of concept IDs."""
    mask = np.isin(labels, subset_ids)
    if mask.sum() == 0:
        return {'top1': 0.0, 'macro_f1': 0.0, 'oscr': 0.0, 'n': 0}
    s, l = scores[mask], labels[mask]
    pred_idx = s.argmax(axis=1)
    pred_ids = np.array([all_ids[p] if p < len(all_ids) else -1 for p in pred_idx])
    top1 = float((pred_ids == l).mean())
    mf1 = float(f1_score(l, pred_ids, average='macro', zero_division=0))
    # OSCR: fraction where correct AND above threshold
    max_scores = s.max(axis=1)
    correct = pred_ids == l
    # Simple OSCR: area under coverage-accuracy curve
    order = np.argsort(-max_scores)
    correct_ordered = correct[order]
    cum_acc = np.cumsum(correct_ordered) / np.arange(1, len(correct_ordered) + 1)
    coverages = np.arange(1, len(correct_ordered) + 1) / len(correct_ordered)
    oscr = float(np.trapz(cum_acc, coverages))
    return {'top1': round(top1, 4), 'macro_f1': round(mf1, 4), 'oscr': round(oscr, 4), 'n': int(mask.sum())}


def compute_centroid_baseline(train_z, train_labels, test_z, test_labels, seen_ids, all_ids):
    """B10: Nearest-centroid baseline.
    Compute centroids from training seen-concept trajectories.
    Assign test trajectories to nearest centroid (cosine similarity).
    Can only handle seen concepts (no centroids for zero-shot).
    """
    # Compute per-concept centroids from training data
    centroids = {}
    for cid in seen_ids:
        mask = train_labels == cid
        if mask.sum() > 0:
            centroid = train_z[mask].mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroids[cid] = centroid

    if not centroids:
        return {'seen': {'top1': 0.0, 'macro_f1': 0.0, 'n': 0}}

    # Build centroid matrix
    centroid_ids = sorted(centroids.keys())
    centroid_mat = np.stack([centroids[cid] for cid in centroid_ids])  # (K_seen, D)

    # Score test trajectories against centroids
    scores = (test_z @ centroid_mat.T) / TEMPERATURE  # (N, K_seen)

    # Typed metrics on seen concepts only
    seen_mask = np.isin(test_labels, seen_ids)
    if seen_mask.sum() == 0:
        return {'seen': {'top1': 0.0, 'macro_f1': 0.0, 'n': 0}}

    s_labels = test_labels[seen_mask]
    s_scores = scores[seen_mask]
    pred_idx = s_scores.argmax(axis=1)
    pred_ids = np.array([centroid_ids[p] for p in pred_idx])
    top1 = float((pred_ids == s_labels).mean())
    mf1 = float(f1_score(s_labels, pred_ids, average='macro', zero_division=0))

    return {
        'seen': {'top1': round(top1, 4), 'macro_f1': round(mf1, 4), 'n': int(seen_mask.sum())},
        'n_centroids': len(centroid_ids),
        'avg_samples_per_centroid': int(np.mean([int((train_labels == cid).sum()) for cid in centroid_ids])),
    }


def run_one(ckpt_path, bench, dataset_name, seed):
    """Run typed evaluation + centroid baseline for one checkpoint/dataset."""
    from langtraj_osr.core.concepts import get_all_definitions, get_concept_ids_for_split
    from langtraj_osr.core.dataset import MobDefBenchDataset, collate_mobdef
    from langtraj_osr.models.langtraj_osr import LangTrajOSR, LangTrajConfig
    from langtraj_osr.train import fit_user_routines
    from torch.utils.data import DataLoader

    np.random.seed(seed); torch.manual_seed(seed)

    test_trajs = bench.test.normal + bench.test.anomalous
    train_all = bench.train.normal + bench.train.anomalous
    logger.info(f'{dataset_name}: test={len(test_trajs)} (norm={len(bench.test.normal)} anom={len(bench.test.anomalous)})')

    all_defs = get_all_definitions(include_paraphrases=True)
    seen_ids = sorted(get_concept_ids_for_split('seen'))
    zsc_ids = sorted(get_concept_ids_for_split('zs_comp'))
    zsf_ids = sorted(get_concept_ids_for_split('zs_family'))
    unk_ids = sorted(get_concept_ids_for_split('unknown'))
    full_ids = seen_ids + zsc_ids + zsf_ids
    def_texts = [all_defs[cid][0] for cid in full_ids]
    K = len(full_ids)

    user_hist = {}
    for t in train_all:
        if t.label == 0:
            user_hist.setdefault(t.user_id, []).append(t)

    # Load model
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = LangTrajConfig(**(ckpt.get('config', {})))
    model = LangTrajOSR(config).to(DEVICE)
    model.definition_encoder(["init"])  # CRITICAL: lazy init
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    # Concept bank
    with torch.no_grad():
        c_bank, _ = model.definition_encoder(def_texts)
        c_bank = F.normalize(c_bank.float(), dim=-1).to(DEVICE)

    # Test data
    test_ds = MobDefBenchDataset(test_trajs, concept_definitions=all_defs, user_histories=user_hist)
    bs = 128
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_mobdef)

    # Train data (for centroid baseline)
    train_ds = MobDefBenchDataset(train_all, concept_definitions=all_defs, user_histories=user_hist)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, collate_fn=collate_mobdef)

    # Collect test scores and embeddings
    test_scores, test_labels, test_z = collect_scores_and_embeddings(model, test_loader, c_bank)
    logger.info(f'{dataset_name}: scores shape={test_scores.shape}, labels shape={test_labels.shape}')

    # Collect train embeddings (for centroid baseline)
    _, train_labels, train_z = collect_scores_and_embeddings(model, train_loader, c_bank)
    logger.info(f'{dataset_name}: train embeddings shape={train_z.shape}')

    # === W2: Full typed metrics per split ===
    results = {
        'dataset': dataset_name,
        'seed': seed,
        'n_test': len(test_labels),
        'typed_full_model': {
            'seen': typed_metrics(test_scores, test_labels, seen_ids, full_ids),
            'zs_comp': typed_metrics(test_scores, test_labels, zsc_ids, full_ids),
            'zs_family': typed_metrics(test_scores, test_labels, zsf_ids, full_ids),
        },
    }

    # Detection AUROC per split
    for split_name, split_ids in [('seen', seen_ids), ('zs_comp', zsc_ids), ('zs_family', zsf_ids), ('unknown', unk_ids)]:
        normal_mask = test_labels == 0
        split_mask = np.isin(test_labels, split_ids)
        combined = normal_mask | split_mask
        if combined.sum() > 0:
            y = (test_labels[combined] > 0).astype(int)
            e = test_scores[combined].max(axis=1)
            if y.sum() > 0 and y.sum() < len(y):
                results[f'auroc_{split_name}'] = round(float(roc_auc_score(y, e)), 4)

    # Overall detection AUROC
    binary = (test_labels > 0).astype(int)
    if binary.sum() > 0 and binary.sum() < len(binary):
        results['auroc_overall'] = round(float(roc_auc_score(binary, test_scores.max(axis=1))), 4)

    # === W3: Nearest-centroid baseline ===
    centroid_results = compute_centroid_baseline(train_z, train_labels, test_z, test_labels, seen_ids, full_ids)
    results['centroid_baseline'] = centroid_results

    # Also compute B9 (kNN-Raw) for comparison if not foursquare
    model.definition_encoder._ensure_encoder()
    with torch.no_grad():
        raw_embs = model.definition_encoder._encode_texts(def_texts)
        raw_embs = F.normalize(raw_embs.float(), dim=-1).cpu().numpy()

    rng = np.random.RandomState(42)
    W = rng.randn(test_z.shape[1], raw_embs.shape[1]).astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    z_proj = test_z @ W
    z_proj /= (np.linalg.norm(z_proj, axis=1, keepdims=True) + 1e-8)
    b9_scores = (z_proj @ raw_embs.T) / TEMPERATURE
    results['b9_knn_raw'] = {
        'seen': typed_metrics(b9_scores, test_labels, seen_ids, full_ids),
        'zs_comp': typed_metrics(b9_scores, test_labels, zsc_ids, full_ids),
        'zs_family': typed_metrics(b9_scores, test_labels, zsf_ids, full_ids),
    }

    return results


# =================================================================

if __name__ == "__main__":
    # Main execution
    # =================================================================
    print("=" * 70)
    print("  W2+W3: Typed Eval (Tokyo/NYC) + Centroid Baseline")
    print("=" * 70)

    CONFIGS = [
        # Tokyo and NYC only (synthetic/porto already completed in prior run)
        ('/home/hello/ouyangqi/results/foursquare_tokyo/seed_42/numosim/seed_42/best_model.pt',
         lambda s: build_foursquare('tokyo', s), 'tokyo', 42),
        ('/home/hello/ouyangqi/results/foursquare_nyc/seed_42/numosim/seed_42/best_model.pt',
         lambda s: build_foursquare('nyc', s), 'nyc', 42),
    ]

    all_results = {}
    for ckpt_path, build_fn, dname, seed in CONFIGS:
        key = f'{dname}_seed{seed}'
        print(f"\n{'='*60}\n  {key}\n{'='*60}", flush=True)
        if not Path(ckpt_path).exists():
            print(f'  SKIP: {ckpt_path} not found'); continue
        try:
            bench = build_fn(seed)
            r = run_one(ckpt_path, bench, dname, seed)
            all_results[key] = r

            # Print summary
            print(f"\n  Full model typed metrics:")
            for split in ['seen', 'zs_comp', 'zs_family']:
                m = r['typed_full_model'].get(split, {})
                print(f"    {split:12s}: Top-1={m.get('top1',0):.3f}  MF1={m.get('macro_f1',0):.3f}  OSCR={m.get('oscr',0):.3f}  n={m.get('n',0)}")

            print(f"\n  Centroid baseline (B10):")
            cm = r.get('centroid_baseline', {}).get('seen', {})
            print(f"    seen       : Top-1={cm.get('top1',0):.3f}  MF1={cm.get('macro_f1',0):.3f}")

            print(f"\n  B9 kNN-Raw:")
            for split in ['seen', 'zs_comp', 'zs_family']:
                m = r.get('b9_knn_raw', {}).get(split, {})
                print(f"    {split:12s}: Top-1={m.get('top1',0):.3f}  MF1={m.get('macro_f1',0):.3f}")

            if 'auroc_overall' in r:
                print(f"\n  Detection AUROC: {r['auroc_overall']:.4f}")

        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback; traceback.print_exc()
            all_results[key] = {'error': str(e)}

    # Save all results
    with open(OUT / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f'Saved to {OUT / "all_results.json"}')

    # === Summary tables ===
    print('\n' + '='*80)
    print('  SUMMARY: TYPED METRICS BY DATASET')
    print('='*80)

    for dname in ['synthetic', 'porto', 'tokyo', 'nyc']:
        seeds = [42, 123, 456] if dname in ['synthetic', 'porto'] else [42]
        vals = {s: {'t': [], 'f': [], 'o': []} for s in ['seen', 'zs_comp', 'zs_family']}
        centroid_vals = {'t': [], 'f': []}
        b9_vals = {s: {'t': [], 'f': []} for s in ['seen', 'zs_comp', 'zs_family']}

        for seed in seeds:
            d = all_results.get(f'{dname}_seed{seed}', {})
            if 'error' in d or 'typed_full_model' not in d: continue
            for s in vals:
                m = d['typed_full_model'].get(s, {})
                if m.get('n', 0) > 0:
                    vals[s]['t'].append(m['top1']); vals[s]['f'].append(m['macro_f1']); vals[s]['o'].append(m.get('oscr', 0))
            cm = d.get('centroid_baseline', {}).get('seen', {})
            if cm.get('n', 0) > 0:
                centroid_vals['t'].append(cm['top1']); centroid_vals['f'].append(cm['macro_f1'])
            for s in b9_vals:
                m = d.get('b9_knn_raw', {}).get(s, {})
                if m.get('n', 0) > 0:
                    b9_vals[s]['t'].append(m['top1']); b9_vals[s]['f'].append(m['macro_f1'])

        print(f'\n  === {dname.upper()} ===')
        print(f'  Full LangTraj-OSR:')
        for s in ['seen', 'zs_comp', 'zs_family']:
            t, f, o = vals[s]['t'], vals[s]['f'], vals[s]['o']
            if t:
                if len(t) > 1:
                    print(f"    {s:12s}: Top-1={np.mean(t):.3f}+/-{np.std(t):.3f}  MF1={np.mean(f):.3f}+/-{np.std(f):.3f}  OSCR={np.mean(o):.3f}+/-{np.std(o):.3f}")
                else:
                    print(f"    {s:12s}: Top-1={t[0]:.3f}  MF1={f[0]:.3f}  OSCR={o[0]:.3f}")

        if centroid_vals['t']:
            t, f = centroid_vals['t'], centroid_vals['f']
            if len(t) > 1:
                print(f'  B10 Centroid:')
                print(f"    seen       : Top-1={np.mean(t):.3f}+/-{np.std(t):.3f}  MF1={np.mean(f):.3f}+/-{np.std(f):.3f}")
            else:
                print(f'  B10 Centroid:')
                print(f"    seen       : Top-1={t[0]:.3f}  MF1={f[0]:.3f}")

        for s in ['seen', 'zs_comp', 'zs_family']:
            t, f = b9_vals[s]['t'], b9_vals[s]['f']
            if t and s == 'seen':
                print(f'  B9 kNN-Raw:')
            if t:
                if len(t) > 1:
                    print(f"    {s:12s}: Top-1={np.mean(t):.3f}+/-{np.std(t):.3f}  MF1={np.mean(f):.3f}+/-{np.std(f):.3f}")
                else:
                    print(f"    {s:12s}: Top-1={t[0]:.3f}  MF1={f[0]:.3f}")

    print(f'\nAll results saved to {OUT / "all_results.json"}')
