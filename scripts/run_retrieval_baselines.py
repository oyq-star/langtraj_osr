#!/usr/bin/env python3
"""Detector + Nearest-Definition Retrieval baseline (B9).

B9: cosine kNN against RAW frozen SentenceTransformer embeddings (384-dim),
    with random linear projection for dimension alignment (256→384).
Full: cosine kNN against PROJECTED embeddings (256-dim, from trained def encoder).

Both use the same trained trajectory encoder and cosine bank scoring.
Purpose: Contrastive alignment (projection MLP + training losses) is necessary.
"""
import sys, json, logging, tempfile
sys.path.insert(0, '/home/hello/ouyangqi')

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.chdir("/home/hello/ouyangqi")

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

SEED = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TEMPERATURE = 0.07


def ep_dict(t, device):
    return {k: t[:,:,i].long().to(device) if i < 5 or i >= 6 else t[:,:,i].float().to(device)
            for i, k in enumerate(['zone_id','poi_role','time_bin','dwell_bin',
                                    'transition_type','trip_length_change','event_flag','companion_flag'])}


@torch.no_grad()
def collect_scores(model, loader, c_bank, temperature=0.07):
    """Collect cosine bank scores (z_norm @ c_bank.T / tau)."""
    model.eval()
    all_scores, all_labels = [], []
    for b in loader:
        pad_mask = b['mask'].to(DEVICE)  # True=valid
        ep = ep_dict(b['episode_tensor'], DEVICE)
        ep_emb = model.episode_encoder(ep)
        z_x, _ = model.trajectory_encoder(ep_emb, ~pad_mask)  # ~pad_mask = True=padded
        z_norm = F.normalize(z_x.float(), dim=-1)
        scores = (z_norm @ c_bank.T) / temperature
        all_scores.append(scores.cpu().float().numpy())
        all_labels.append(b['label'].numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


@torch.no_grad()
def collect_embeddings(model, loader):
    """Collect L2-normalized trajectory embeddings and labels."""
    model.eval()
    all_z, all_labels = [], []
    for b in loader:
        pad_mask = b['mask'].to(DEVICE)
        ep = ep_dict(b['episode_tensor'], DEVICE)
        ep_emb = model.episode_encoder(ep)
        z_x, _ = model.trajectory_encoder(ep_emb, ~pad_mask)
        z_norm = F.normalize(z_x.float(), dim=-1)
        all_z.append(z_norm.cpu())
        all_labels.append(b['label'].numpy())
    return torch.cat(all_z).numpy(), np.concatenate(all_labels)


def typed_metrics(scores, labels, subset_ids, all_ids):
    mask = np.isin(labels, subset_ids)
    if mask.sum() == 0:
        return {'top1': 0.0, 'macro_f1': 0.0, 'n': 0}
    s, l = scores[mask], labels[mask]
    pred_idx = s.argmax(axis=1)
    pred_ids = np.array([all_ids[p] if p < len(all_ids) else -1 for p in pred_idx])
    top1 = float((pred_ids == l).mean())
    mf1 = float(f1_score(l, pred_ids, average='macro', zero_division=0))
    return {'top1': round(top1, 4), 'macro_f1': round(mf1, 4), 'n': int(mask.sum())}


def build_benchmark(seed, use_porto):
    """Build MobDefBench data (synthetic or Porto)."""
    from langtraj_osr.benchmark.benchmark_builder import (
        MobDefBenchBuilder, Benchmark, CONCEPT_DEFS,
        SPLIT_SEEN, SPLIT_ZS_COMP, SPLIT_ZS_FAMILY, SPLIT_UNKNOWN,
    )
    if use_porto:
        from langtraj_osr.core.tokenizer import TrajectoryTokenizer
        from langtraj_osr.core.episode import SemanticEpisode, SemanticTrajectory
        import math, hashlib, pandas as pd, pickle

        def _hav(lat1, lon1, lat2, lon2):
            R = 6_371_000.0
            p1, p2 = math.radians(lat1), math.radians(lat2)
            a = (math.sin(math.radians(lat2-lat1)/2)**2
                 + math.cos(p1)*math.cos(p2)*math.sin(math.radians(lon2-lon1)/2)**2)
            return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

        def _zone(lat, lon, res=0.005):
            h = hashlib.md5(f"{int(round(lat/res))},{int(round(lon/res))}".encode()).hexdigest()
            return int(h[:8], 16) % (2**31)

        def _trans(spd): return 0 if spd < 2 else (2 if spd < 10 else 1)

        tok = TrajectoryTokenizer()
        with open('data/porto_trips.pkl', 'rb') as f:
            df = pickle.load(f)
        pdf = (df.groupby('TAXI_ID', group_keys=False)
                 .apply(lambda g: g.sample(min(len(g), 80), random_state=42))
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
                lon, lat = poly[idx]
                ts = pd.Timestamp(int(row['TIMESTAMP']) + idx*15, unit='s')
                db = tok._discretize_dwell(sub * 15 / 60.0)
                if idx > 0:
                    pl, pa = poly[max(0, idx-sub)]
                    sd = _hav(pa, pl, lat, lon)
                    if not math.isfinite(sd): sd = 0.0
                    spd = sd / (sub*15) if sub*15 > 0 else 0
                    tr = _trans(spd); tlc = min(sd/avg_d, 20.0)
                else:
                    tr = 1; tlc = 1.0
                eps.append(SemanticEpisode(zone_id=_zone(lat, lon), poi_role=0,
                                           time_bin=ts.hour*7+ts.dayofweek, dwell_bin=db,
                                           transition_type=tr,
                                           trip_length_change=round(float(tlc), 4),
                                           event_flag=0, companion_flag=0))
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
    else:
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as to2:
            builder = MobDefBenchBuilder(data_dir=td, output_dir=to2, seed=seed)
            return builder.build(datasets=['numosim'])['numosim']


def run_one(ckpt_path, seed, use_porto):
    """Run retrieval baseline for one checkpoint."""
    from langtraj_osr.core.concepts import get_all_definitions, get_concept_ids_for_split
    from langtraj_osr.core.dataset import MobDefBenchDataset, collate_mobdef
    from langtraj_osr.models.langtraj_osr import LangTrajOSR, LangTrajConfig
    from torch.utils.data import DataLoader

    np.random.seed(seed); torch.manual_seed(seed)

    # Build data
    bench = build_benchmark(seed, use_porto)
    test_trajs = bench.test.normal + bench.test.anomalous
    logger.info('test=%d (norm=%d anom=%d)', len(test_trajs), len(bench.test.normal), len(bench.test.anomalous))

    all_defs = get_all_definitions(include_paraphrases=True)
    full_ids = sorted(get_concept_ids_for_split('seen')) + \
               sorted(get_concept_ids_for_split('zs_comp')) + \
               sorted(get_concept_ids_for_split('zs_family'))
    def_texts = [all_defs[cid][0] for cid in full_ids]
    K = len(full_ids)

    test_ds = MobDefBenchDataset(test_trajs, concept_definitions=all_defs,
                                  user_histories={})
    bs = 128 if use_porto else 256
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_mobdef)

    # Load model (CRITICAL: trigger lazy init before load_state_dict)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get('config', {})
    config = LangTrajConfig(**cfg) if cfg else LangTrajConfig()
    model = LangTrajOSR(config).to(DEVICE)
    model.definition_encoder(["init"])  # Trigger projection head init
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    # FULL MODEL: projected text embeddings (256-dim, trained projection MLP)
    with torch.no_grad():
        c_proj, _ = model.definition_encoder(def_texts)
        c_proj = F.normalize(c_proj.float(), dim=-1).to(DEVICE)  # (22, 256)

    # RAW: frozen SentenceTransformer embeddings (384-dim, no projection MLP)
    model.definition_encoder._ensure_encoder()
    with torch.no_grad():
        raw_embs = model.definition_encoder._encode_texts(def_texts)  # (22, 384)
        raw_embs = F.normalize(raw_embs.float(), dim=-1).cpu().numpy()

    # Collect trajectory embeddings
    z_all, labels = collect_embeddings(model, test_loader)  # (N, 256), (N,)

    # --- Full model scores (cosine bank scoring, matching paper) ---
    full_scores = (z_all @ c_proj.cpu().numpy().T) / TEMPERATURE

    # --- B9: kNN with raw text embeddings ---
    # Random projection (fixed seed) to match dims: 256 → 384
    rng = np.random.RandomState(42)
    W = rng.randn(256, 384).astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    z_proj = z_all @ W
    z_proj /= (np.linalg.norm(z_proj, axis=1, keepdims=True) + 1e-8)
    b9_scores = (z_proj @ raw_embs.T) / TEMPERATURE

    # --- Metrics ---
    seen_ids = sorted(get_concept_ids_for_split('seen'))
    zsc_ids = sorted(get_concept_ids_for_split('zs_comp'))
    zsf_ids = sorted(get_concept_ids_for_split('zs_family'))

    results = {}
    for method, scores in [('B9_knn_raw', b9_scores), ('Full_LangTrajOSR', full_scores)]:
        results[method] = {
            'seen': typed_metrics(scores, labels, seen_ids, full_ids),
            'zs_comp': typed_metrics(scores, labels, zsc_ids, full_ids),
            'zs_fam': typed_metrics(scores, labels, zsf_ids, full_ids),
        }

    # Detection AUROC (using max concept bank score, matching paper)
    binary = (labels > 0).astype(int)
    try:
        results['det_auroc_full'] = float(roc_auc_score(binary, full_scores.max(axis=1)))
        results['det_auroc_b9'] = float(roc_auc_score(binary, b9_scores.max(axis=1)))
    except:
        results['det_auroc_full'] = None
        results['det_auroc_b9'] = None

    # Random baseline (1/K)
    results['Random'] = {s: {'top1': round(1.0/K, 4), 'macro_f1': round(1.0/K, 4), 'n': 0}
                         for s in ['seen', 'zs_comp', 'zs_fam']}

    return results


# =================================================================
BASE = '/home/hello/ouyangqi/results'
OUT = Path('/home/hello/ouyangqi/results/retrieval_baselines')
OUT.mkdir(exist_ok=True)

runs = [
    (f'{BASE}/v6_synthetic/numosim/seed_42/best_model.pt',  42,  False, 'synthetic'),
    (f'{BASE}/v6_synthetic/numosim/seed_123/best_model.pt', 123, False, 'synthetic'),
    (f'{BASE}/v6_synthetic/numosim/seed_456/best_model.pt', 456, False, 'synthetic'),
    (f'{BASE}/v6_porto/numosim/seed_42/best_model.pt',      42,  True,  'porto'),
    (f'{BASE}/v6_porto/numosim/seed_123/best_model.pt',     123, True,  'porto'),
    (f'{BASE}/v6_porto/numosim/seed_456/best_model.pt',     456, True,  'porto'),
]

all_results = {}
for ckpt, seed, use_porto, dname in runs:
    key = f'{dname}_seed{seed}'
    print(f"\n{'='*60}\n  {key}\n{'='*60}", flush=True)
    if not Path(ckpt).exists():
        print('  SKIP: no checkpoint'); continue
    try:
        r = run_one(ckpt, seed, use_porto)
        all_results[key] = r
        print(f"  Det AUROC: full={r.get('det_auroc_full','?'):.4f}  b9={r.get('det_auroc_b9','?'):.4f}")
        for m in ['B9_knn_raw', 'Full_LangTrajOSR']:
            print(f'  {m}:')
            for s in ['seen', 'zs_comp', 'zs_fam']:
                d = r[m][s]
                print(f"    {s:10s}: Top-1={d['top1']:.3f}  F1={d['macro_f1']:.3f}  n={d['n']}")
    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback; traceback.print_exc()
        all_results[key] = {'error': str(e)}

with open(OUT / 'all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Summary
print('\n' + '='*80 + '\n  RETRIEVAL BASELINE SUMMARY\n' + '='*80)
for dname in ['synthetic', 'porto']:
    print(f'\n  === {dname.upper()} (3-seed mean +/- std) ===')
    for method in ['B9_knn_raw', 'Full_LangTrajOSR', 'Random']:
        vals = {s: {'t': [], 'f': []} for s in ['seen', 'zs_comp', 'zs_fam']}
        for seed in [42, 123, 456]:
            d = all_results.get(f'{dname}_seed{seed}', {})
            if not isinstance(d, dict) or method not in d: continue
            for s in vals:
                m = d[method].get(s, {})
                if m.get('top1') is not None and m['top1'] > 0:
                    vals[s]['t'].append(m['top1']); vals[s]['f'].append(m['macro_f1'])
        print(f'  {method}:')
        for s in ['seen', 'zs_comp', 'zs_fam']:
            t, f = vals[s]['t'], vals[s]['f']
            if t:
                print(f"    {s:10s}: Top-1={np.mean(t):.3f}+/-{np.std(t):.3f}  F1={np.mean(f):.3f}+/-{np.std(f):.3f}")
            elif method == 'Random':
                K = 22
                print(f"    {s:10s}: Top-1={1/K:.3f}  F1={1/K:.3f}  (1/{K} uniform)")
